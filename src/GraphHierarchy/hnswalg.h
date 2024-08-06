#pragma once

#include <assert.h>
#include <stdlib.h>

#include <atomic>
#include <list>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "hnswlib.h"
#include "visited_list_pool.h"

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

struct Level {
    size_t num_pivots_{0};
    std::vector<tableint> pivots_{};
    std::unordered_map<tableint, tableint> map_{};
    std::vector<std::vector<tableint>> partitions_{};
    std::vector<std::vector<tableint>> graph_{};

    void add_pivot(tableint const pivot) {
        size_t idx = num_pivots_;
        num_pivots_++;
        pivots_.push_back(pivot);
        map_[pivot] = idx;
    }
    void initialize_partitions() {
        partitions_.resize(num_pivots_);
    }
    void initialize_graph() {
        graph_.resize(num_pivots_);
    }
    void set_neighbors(tableint const pivot, std::vector<tableint> const &neighbors) {
        graph_[map_[pivot]] = neighbors;
    }
    void add_member(tableint const pivot, tableint const member) { partitions_[map_[pivot]].push_back(member); }
    std::vector<tableint> const &get_neighbors(tableint pivot) const { return graph_[map_.at(pivot)]; }
    std::vector<tableint> const &get_partition(tableint pivot) const { return partitions_[map_.at(pivot)]; }
};

template <typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
   public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    size_t dataset_size_{0};                           // friendly notation
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements

    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};

    // hierarchy considerations
    size_t scaling_{0};  // scaling of pivots in hierarchy
    double mult_{0.0};
    std::vector<Level> hierarchy_{};
    int num_levels_{0};

    // creating approximate hsp graph
    size_t max_neighbors_{0};         // max neighbors for all points
    size_t ahsp_num_partitions_ = 10;
    size_t ahsp_beam_size_ = 10;
    size_t ahsp_max_region_size_ = 10000;

    // search parameters
    size_t beam_size_{0};               // control beam size for search
    size_t max_neighbors_to_check_{0};  // control max number of neighbors to consider

    // visited list pool is for tabu search with multiple threads
    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    // bottom level size
    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

    char *data_level0_memory_{nullptr};
    size_t data_size_{0};

    DISTFUNC<dist_t> dist_func_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    // statistics
    std::chrono::high_resolution_clock::time_point tStart, tEnd;
    std::chrono::high_resolution_clock::time_point tStart1, tEnd1;
    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    /**
     * ====================================================================================================
     *
     *              CONSTRUCTORS / DESTRUCTORS
     *
     * ====================================================================================================
     */

    HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, size_t random_seed = 100) {
        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);
        loadIndex(location, s);
    }

    HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t max_neighbors = 64, size_t random_seed = 100)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS), link_list_locks_(max_elements) {

        // initializing hierarchy
        max_elements_ = max_elements;
        dataset_size_ = max_elements;

        // initialize distance function
        data_size_ = s->get_data_size();
        dist_func_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        // approximate hsp parameters
        max_neighbors_ = max_neighbors;
        max_neighbors_to_check_ = max_neighbors_;
        ahsp_num_partitions_ = 10;
        ahsp_beam_size_ = 10;
        ahsp_max_region_size_ = 10000;

        // initializing beam search
        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        // adjusted for our approach
        size_links_level0_ = max_neighbors_ * sizeof(tableint) + sizeof(linklistsizeint);  // memory for graph
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);      // memory for each point
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        // allocating all memory
        data_level0_memory_ = (char *)malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr) throw std::runtime_error("Not enough memory");
        cur_element_count = 0;

        // initializing the visited list for search
        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));
    }
    ~HierarchicalNSW() { clear(); }
    void clear() {
        hierarchy_.clear();
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }

    /**
     * ====================================================================================================
     *
     *              I/O
     *
     * ====================================================================================================
     */

    void saveIndex(const std::string &location) {
        printf("Saving dataset to: %s\n", location.c_str());
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        // write the few parameters
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, max_neighbors_);

        // write the bottom layer graph and datapoints
        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        // that's it!
        output.close();
        return;
    }

    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s) {
        std::ifstream input(location, std::ios::binary);
        if (!input.is_open()) throw std::runtime_error("Cannot open file");
        clear();

        // read the important parameters
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);
        readBinaryPOD(input, max_neighbors_);

        // derive the rest of the parameters, as done in constructor
        dataset_size_ = max_elements_;
        ahsp_num_partitions_ = 10;
        ahsp_beam_size_ = 10;
        ahsp_max_region_size_ = 10000;

        data_size_ = s->get_data_size();
        dist_func_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        size_links_level0_ = max_neighbors_ * sizeof(tableint) + sizeof(linklistsizeint);  // memory for graph
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);      // memory for each point
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;
        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements_));

        // load memory for bottom level
        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        input.close();
        return;
    }

    /**
     * ====================================================================================================
     *
     *              HELPER FUNCTIONS
     *
     * ====================================================================================================
     */

    dist_t const compute_distance(char *index1_ptr, char *index2_ptr) {
        return dist_func_(index1_ptr, index2_ptr, dist_func_param_);
    }
    dist_t const compute_distance(char *index1_ptr, tableint index2) {
        return dist_func_(index1_ptr, getDataByInternalId(index2), dist_func_param_);
    }
    dist_t const compute_distance(tableint index1, char *index2_ptr) {
        return dist_func_(getDataByInternalId(index1), index2_ptr, dist_func_param_);
    }
    dist_t const compute_distance(tableint index1, tableint index2) {
        return dist_func_(getDataByInternalId(index1), getDataByInternalId(index2), dist_func_param_);
    }

    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                  std::pair<dist_t, tableint> const &b) const noexcept {
            return a.first < b.first;
        }
    };
    inline std::mutex &getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }
    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_),
               sizeof(labeltype));
        return return_label;
    }
    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }
    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }
    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }
    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }
    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
    }
    size_t getMaxElements() { return max_elements_; }
    size_t getCurrentElementCount() { return cur_element_count; }
    linklistsizeint *get_neighbors(tableint internal_id) const {
        return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }

    unsigned short int getListCount(linklistsizeint *ptr) const { return *((unsigned short int *)ptr); }
    void setListCount(linklistsizeint *ptr, unsigned short int size) const {
        *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
    }
    template <typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char *data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *)dist_func_param_);
        std::vector<data_t> data;
        data_t *data_ptr = (data_t *)data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }

    /**
     * ====================================================================================================
     *
     *              INDEX CONSTRUCTION
     *
     * ====================================================================================================
     */

    // add each data point and initialize the bottom layer graph
    void addPoint(const void *data_point, labeltype label) {
        if (cur_element_count >= max_elements_) {
            throw std::runtime_error("The number of elements exceeds the specified limit");
        }
        tableint const node = cur_element_count;
        cur_element_count++;
        label_lookup_[label] = node;

        // - initializing and setting data/graph memory for bottom level
        memset(data_level0_memory_ + node * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
        memcpy(getExternalLabeLp(node), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(node), data_point, data_size_);
        return;
    }


    /**
     * @brief Index Construction
     *  add num_maps
     *  int M, int ef_construct, int ef_assign
     */
    void build(int scaling = 100, int num_partitions = 10) {
        printf("Begin Index Construction...\n");

        // initilaize the hierarchy for approximate hsp
        //  - probabilistically select nodes and add to each level
        initialize_hierarchy(scaling); 

        /**
         *
         *     TOP-DOWN CONSTRUCTION OF THE GRAPH HIERARCHY
         *
         */

        //> Build each level
        for (int ell = 1; ell < num_levels_; ell++) {
            printf(" * Begin Level-%d Construction\n", ell);
            tStart = std::chrono::high_resolution_clock::now();
            bool flag_bottom_level = (ell == num_levels_ - 1);

            // - initializations
            size_t num_pivots = dataset_size_;
            if (!flag_bottom_level) 
                num_pivots = hierarchy_[ell].num_pivots_;
            printf("    - num_pivots = %u\n", (uint) num_pivots);
            hierarchy_[ell-1].initialize_partitions();
            std::vector<tableint> pivot_assignments(num_pivots);

            //> Begin Partitioning the Dataset
            printf("    - begin partitioning of the level\n");
            tStart1 = std::chrono::high_resolution_clock::now();
            if (!flag_bottom_level) { // for upper levels

                //> Perform the partitioning of this current layer         
                // - perform this task using all threads
                #pragma omp parallel for
                for (size_t itp = 0; itp < num_pivots; itp++) {
                    tableint const fine_pivot = hierarchy_[ell].pivots_[itp];
                    char *fine_pivot_ptr = getDataByInternalId(fine_pivot);
                    tableint closest_pivot;

                    // - top-down assignment to a coarse pivot
                    for (int c = 0; c < ell; c++) {

                        // - define the candidate pivots in the layer
                        std::vector<tableint> candidate_coarse_pivots{};
                        if (c == 0) {
                            candidate_coarse_pivots = hierarchy_[c].pivots_;
                        } else {
                            candidate_coarse_pivots = hierarchy_[c - 1].get_partition(closest_pivot);
                        }

                        // - find and record the closest coarse pivot
                        dist_t closest_dist = HUGE_VAL;
                        for (tableint coarse_pivot : candidate_coarse_pivots) {
                            dist_t const dist =
                                dist_func_(fine_pivot_ptr, getDataByInternalId(coarse_pivot), dist_func_param_);
                            if (dist < closest_dist) {
                                closest_dist = dist;
                                closest_pivot = coarse_pivot;
                            }
                        }
                    }

                    // - record the closest coarse pivot found (thread safe)
                    pivot_assignments[itp] = closest_pivot;
                }

                // - assign to the partitions (thread safe)
                for (size_t itp = 0; itp < num_pivots; itp++) {
                    tableint const fine_pivot = hierarchy_[ell].pivots_[itp];
                    hierarchy_[ell - 1].add_member(pivot_assignments[itp], fine_pivot);
                }
            } else { // for bottom level

                //> Perform the partitioning of this current layer
                #pragma omp parallel for
                for (tableint node = 0; node < dataset_size_; node++) {
                    char *node_ptr = getDataByInternalId(node);
                    tableint closest_pivot;

                    // - top-down assignment to a coarse pivot
                    for (int c = 0; c < ell; c++) {
                        // - define the candidate pivots in the layer
                        std::vector<tableint> candidate_coarse_pivots{};
                        if (c == 0) {
                            candidate_coarse_pivots = hierarchy_[c].pivots_;
                        } else {
                            candidate_coarse_pivots = hierarchy_[c - 1].get_partition(closest_pivot);
                        }

                        // - find and record the closest coarse pivot
                        dist_t closest_dist = HUGE_VAL;
                        for (tableint coarse_pivot : candidate_coarse_pivots) {
                            dist_t const dist = dist_func_(node_ptr, getDataByInternalId(coarse_pivot), dist_func_param_);
                            if (dist < closest_dist) {
                                closest_dist = dist;
                                closest_pivot = coarse_pivot;
                            }
                        }
                    }

                    // - record the closest coarse pivot found (thread safe)
                    pivot_assignments[node] = closest_pivot;
                }

                // - assign to the partitions (thread safe)
                for (tableint node = 0; node < dataset_size_; node++) {
                    hierarchy_[ell - 1].add_member(pivot_assignments[node], node);
                }
            }
            tEnd1 = std::chrono::high_resolution_clock::now();
            double time_part = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd1 - tStart1).count();
            printf("    - partitioning time: %.4f\n", time_part);


            //> Construct the Locally Monotonic Graph on the Level
            printf("    - begin construction of locally monotonic graph\n");
            tStart1 = std::chrono::high_resolution_clock::now();
            if (!flag_bottom_level) { // for upper levels
                hierarchy_[ell].initialize_graph();

                // - find the hsp neighbors of each in parallel
                #pragma omp parallel for
                for (size_t itp = 0; itp < num_pivots; itp++) {
                    tableint const fine_pivot = hierarchy_[ell].pivots_[itp];
                    char *fine_pivot_ptr = getDataByInternalId(fine_pivot);

                    // - define the candidate list of neighbors
                    std::vector<tableint> candidate_fine_pivots{};
                    if (num_pivots < 20000) {      // brute force 
                        candidate_fine_pivots = hierarchy_[ell].pivots_;
                    } else {

                        //> Define the candidate region by a set of nearby partitions
                        // - starting from the closest pivot
                        tableint start_node = pivot_assignments[itp]; 

                        // - beam search to collect the b closest pivots (MODIFY)
                        int ahsp_num_partitions_ = 10;
                        std::vector<tableint> closest_coarse_pivots = beamSearchConstruction(ell-1, fine_pivot_ptr, ahsp_num_partitions_, start_node);

                        // - collect partitions associated with the closest coarse pivots
                        for (tableint coarse_pivot : closest_coarse_pivots) {
                            std::vector<tableint> const &coarse_partition = hierarchy_[ell - 1].get_partition(coarse_pivot);
                            candidate_fine_pivots.insert(candidate_fine_pivots.end(), coarse_partition.begin(), coarse_partition.end());
                        }
                    }

                    // - Perform the HSP test on this neighborhood
                    // - modify: max region size
                    std::vector<uint> neighbors = hsp_test(fine_pivot, candidate_fine_pivots);
                    hierarchy_[ell].set_neighbors(fine_pivot, neighbors);
                }

                // get ave. degree
                double ave_degree = 0.0f;
                for (size_t itp = 0; itp < num_pivots; itp++) {
                    uint fine_pivot = hierarchy_[ell].pivots_[itp];
                    ave_degree += (double) hierarchy_[ell].get_neighbors(fine_pivot).size();
                }
                ave_degree /= (double) num_pivots;
                printf("    - ave degree: %.2f\n", ave_degree);

            } else { // bottom level graph

                //> Construct the Locally Monotonic Graph on the bottom level
                #pragma omp parallel for
                for (tableint node = 0; node < dataset_size_; node++) {
                    char *node_ptr = getDataByInternalId(node);

                    //> Define the candidate region by a set of nearby partitions
                    // - starting from the closest pivot
                    tableint start_node = pivot_assignments[node]; 

                    // - beam search to collect the b closest pivots (MODIFY)
                    int ahsp_num_partitions_ = 10;
                    std::vector<tableint> closest_pivots = beamSearchConstruction(ell-1, node_ptr, ahsp_num_partitions_, start_node);

                    // - collect partitions associated with the closest coarse pivots
                    std::vector<tableint> candidate_nodes{};
                    for (tableint pivot : closest_pivots) {
                        std::vector<tableint> const &pivot_partition = hierarchy_[ell-1].get_partition(pivot);
                        candidate_nodes.insert(candidate_nodes.end(), pivot_partition.begin(),
                                                        pivot_partition.end());
                    }

                    // - Perform the HSP test on this neighborhood
                    // - modify: max region size
                    std::vector<uint> neighbors = hsp_test(node, candidate_nodes);

                    // set the neighbors on the bottom level
                    linklistsizeint *ll_node = get_linklist0(node);  // get start of list (num_neighbors, n1, n2, n3,...)
                    setListCount(ll_node, neighbors.size());    // set the new number of neighbors
                    tableint *neighbors_data = (tableint *)(ll_node + 1);  // pointer to first neighbor
                    for (size_t n = 0; n < neighbors.size(); n++) {
                        neighbors_data[n] = neighbors[n];
                    }
                }

                // - get the average degree
                double ave_degree = 0.0f;
                for (tableint node = 0; node < dataset_size_; node++) {
                    linklistsizeint *data = (linklistsizeint *)get_linklist0(node);
                    ave_degree += (double)getListCount((linklistsizeint *)data);
                }
                ave_degree /= (double)dataset_size_;
                printf("    - ave degree: %.2f\n", ave_degree);
            }
            tEnd1 = std::chrono::high_resolution_clock::now();
            double time_graph = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd1 - tStart1).count();
            printf("    - graph time: %.4f\n", time_graph);

            tEnd = std::chrono::high_resolution_clock::now();
            double time_level = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
            printf("    - total level construction time: %.4f\n", time_level);
        }
        printf(" * Completed the graph construction!\n");
        return;
    }

    /**
     * @brief Initialize the hierarchy
     * 
     * @param scaling 
     */
    void initialize_hierarchy(int const scaling) {
        hierarchy_.clear();

        // estimate the number of levels
        num_levels_ = 0;
        double start_value = (double) dataset_size_;
        while (start_value >= 10) {
            num_levels_++;
            start_value /= (double) scaling;
        }
        hierarchy_.resize(num_levels_ - 1);
        printf(" * Number of levels: %d\n", num_levels_);

        // select pivots probabilistically on each level, same as hnsw
        mult_ = 1 / log( (double) scaling); 
        std::default_random_engine level_generator;
        level_generator.seed(3); 
        for (tableint index = 0; index < dataset_size_; index++) {
            int level_assignment = getRandomLevel(mult_);
            if (level_assignment > num_levels_ - 1) level_assignment = num_levels_ - 1;

            // add to each level
            // want top to be level 0, bottom to be level num_levels_-1
            for (int l = 1; l <= level_assignment; l++) {
                int level_of_interest = num_levels_ - l - 1;
                hierarchy_[level_of_interest].add_pivot(index);
            }
        }
        for (int l = 0; l < hierarchy_.size(); l++) {
            printf("    - %d --> %u\n", l, (uint) hierarchy_[l].num_pivots_);
        }
        return;
    }

    // performing a beam search to obtain the closest partitions for approximate hsp
    std::vector<tableint> beamSearchConstruction(int level, const void *query_ptr, int k, tableint start_node) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        // initialize lists
        int beam_size = ahsp_beam_size_;
        if (beam_size < k) beam_size = k;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            candidateSet;

        dist_t dist = dist_func_(query_ptr, getDataByInternalId(start_node), dist_func_param_);
        top_candidates.emplace(dist, start_node);
        dist_t lowerBound = dist;
        candidateSet.emplace(-dist, start_node);
        visited_array[start_node] = visited_array_tag;

        // perform the beam search
        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> current_pair = candidateSet.top();
            if ((-current_pair.first) > lowerBound && top_candidates.size() == beam_size) {
                break;
            }
            candidateSet.pop();

            // - fetch neighbors of current node
            tableint const current_node = current_pair.second;
            std::vector<uint> const &current_node_neighbors = hierarchy_[level].get_neighbors(current_node);
            size_t num_neighbors = current_node_neighbors.size();

            // - iterate through the neighbors
            for (size_t j = 0; j < num_neighbors; j++) {
                tableint const neighbor_node = current_node_neighbors[j];

                // - skip if already visisted
                if (visited_array[neighbor_node] == visited_array_tag) continue;
                visited_array[neighbor_node] = visited_array_tag;

                // - update data structures if applicable
                dist_t dist = dist_func_(query_ptr, getDataByInternalId(neighbor_node), dist_func_param_);
                if (top_candidates.size() < beam_size || lowerBound > dist) {
                    candidateSet.emplace(-dist, neighbor_node);

                    top_candidates.emplace(dist, neighbor_node);
                    if (top_candidates.size() > beam_size) top_candidates.pop();
                    if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        // return simply the neighbors
        std::vector<tableint> neighbors(k);
        while (top_candidates.size() > k) top_candidates.pop();
        for (int i = k - 1; i >= 0; i--) {
            neighbors[i] = top_candidates.top().second;
            top_candidates.pop();
        }
        return neighbors;
    }

    // perform the hsp test to get the hsp neighbors of the node
    std::vector<tableint> hsp_test(tableint const query_node, std::vector<tableint> const &set, int max_k = 0) {
        std::vector<tableint> neighbors{};
        char *query_ptr = getDataByInternalId(query_node);

        // - initialize the active list A
        std::vector<std::pair<dist_t, tableint>> active_list{};
        active_list.reserve(set.size());

        // - initialize the list with all points and distances, find nearest neighbor
        tableint index1;
        dist_t distance_Q1 = HUGE_VAL;
        for (tableint index : set) {
            if (index == query_node) continue;
            dist_t distance = dist_func_(query_ptr, getDataByInternalId(index), dist_func_param_);
            if (distance < distance_Q1) {
                distance_Q1 = distance;
                index1 = index;
            }
            active_list.emplace_back(distance, index);
        }

        // - limit the number of points to consider
        if (max_k > 0 && active_list.size() > max_k) {
            // - nth_element sort: bring the kth closest elements to the front, but not sorted, O(N)
            std::nth_element(active_list.begin(), active_list.begin()+ max_k, active_list.end());
            active_list.resize(max_k); // keep only the top k points
        }

        // - perform the hsp loop witin some maximum number of neighbors/iterations
        for (size_t n = 0; n < max_neighbors_; n++) {
            if (active_list.size() <= 0) break;

            // - next neighbor as closest valid point
            neighbors.push_back(index1);
            char *index1_ptr = getDataByInternalId(index1);

            // - set up for the next hsp neighbor
            tableint index1_next;
            dist_t distance_Q1_next = HUGE_VAL;

            // - initialize the active_list for next iteration
            // - make new list: push_back O(1) faster than deletion O(N)
            std::vector<std::pair<dist_t, tableint>> active_list_copy = active_list;
            active_list.clear();

            // - check each point for elimination
            for (int it2 = 0; it2 < (int)active_list_copy.size(); it2++) {
                tableint const index2 = active_list_copy[it2].second;
                dist_t const distance_Q2 = active_list_copy[it2].first;
                if (index2 == index1) continue;
                dist_t const distance_12 = dist_func_(index1_ptr, getDataByInternalId(index2), dist_func_param_);

                // - check the hsp inequalities: add back if not satisfied
                if (distance_12 >= distance_Q2) {  // distance_Q1 >= distance_Q2 ||
                    active_list.emplace_back(distance_Q2, index2);

                    // - update neighbor for next iteration
                    if (distance_Q2 < distance_Q1_next) {
                        distance_Q1_next = distance_Q2;
                        index1_next = index2;
                    }
                }
            }

            // - setup the next hsp neighbor
            index1 = index1_next;
            distance_Q1 = distance_Q1_next;
        }

        return neighbors;
    }

    void print_ave_degree() {
        if (cur_element_count <= 0) return;

        // - get the average degree
        double ave_degree = 0.0f;
        for (tableint node = 0; node < dataset_size_; node++) {
            linklistsizeint *data = (linklistsizeint *)get_linklist0(node);
            ave_degree += (double)getListCount((linklistsizeint *)data);
        }
        ave_degree /= (double) dataset_size_;
        printf("ave_degree = %.2f\n", ave_degree);
        return;
    }

    // Hierarchical partitioning for search time
    void createHierarchicalPartitioning(int scaling) {
        printf("Creating hierarchical partitioning\n");

        // initilaize the hierarchy for approximate hsp
        initialize_hierarchy(scaling); 

        //> Build each level top-down
        tStart = std::chrono::high_resolution_clock::now();
        for (int ell = 1; ell < num_levels_ - 1; ell++) {
            printf(" * Partitioning level-%d\n", ell);
            tStart1 = std::chrono::high_resolution_clock::now();

            // - initializations
            size_t num_pivots = hierarchy_[ell].num_pivots_;
            printf("    - num_pivots = %u\n", (uint) num_pivots);
            hierarchy_[ell-1].initialize_partitions();
            std::vector<tableint> pivot_assignments(num_pivots);

            //> Perform the partitioning of this current layer         
            // - perform this task using all threads
            #pragma omp parallel for
            for (size_t itp = 0; itp < num_pivots; itp++) {
                tableint const fine_pivot = hierarchy_[ell].pivots_[itp];
                char *fine_pivot_ptr = getDataByInternalId(fine_pivot);
                tableint closest_pivot;

                // - top-down assignment to a coarse pivot
                for (int c = 0; c < ell; c++) {

                    // - define the candidate pivots in the layer
                    std::vector<tableint> candidate_coarse_pivots{};
                    if (c == 0) {
                        candidate_coarse_pivots = hierarchy_[c].pivots_;
                    } else {
                        candidate_coarse_pivots = hierarchy_[c - 1].get_partition(closest_pivot);
                    }

                    // - find and record the closest coarse pivot
                    dist_t closest_dist = HUGE_VAL;
                    for (tableint coarse_pivot : candidate_coarse_pivots) {
                        dist_t const dist = dist_func_(fine_pivot_ptr, getDataByInternalId(coarse_pivot), dist_func_param_);
                        if (dist < closest_dist) {
                            closest_dist = dist;
                            closest_pivot = coarse_pivot;
                        }
                    }
                }

                // - record the closest coarse pivot found
                pivot_assignments[itp] = closest_pivot;
            }
            // - assign to the partitions (single-threaded)
            for (size_t itp = 0; itp < num_pivots; itp++) {
                tableint const fine_pivot = hierarchy_[ell].pivots_[itp];
                hierarchy_[ell - 1].add_member(pivot_assignments[itp], fine_pivot);
            }
            tEnd1 = std::chrono::high_resolution_clock::now();
            double time_part = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd1 - tStart1).count();
            printf("    - partitioning time: %.4f\n", time_part);
        }

        //> Perform partitioning of the bottom level (if necessary)
        {
            int ell = num_levels_ - 1;
            printf(" * Partitioning level-%d (bottom-level)\n", ell);
            tStart1 = std::chrono::high_resolution_clock::now();

            // - initializations
            printf("    - dataset_size_ = %u\n", (uint) dataset_size_);
            hierarchy_[ell-1].initialize_partitions();
            std::vector<tableint> pivot_assignments(dataset_size_);

            //> Perform the partitioning of this layer
            #pragma omp parallel for
            for (tableint node = 0; node < dataset_size_; node++) {
                char *node_ptr = getDataByInternalId(node);
                tableint closest_pivot;

                // - top-down assignment to a coarse pivot
                for (int c = 0; c < ell; c++) {

                    // - define the candidate pivots in the layer
                    std::vector<tableint> candidate_coarse_pivots{};
                    if (c == 0) {
                        candidate_coarse_pivots = hierarchy_[c].pivots_;
                    } else {
                        candidate_coarse_pivots = hierarchy_[c - 1].get_partition(closest_pivot);
                    }

                    // - find and record the closest coarse pivot
                    dist_t closest_dist = HUGE_VAL;
                    for (tableint coarse_pivot : candidate_coarse_pivots) {
                        dist_t const dist = dist_func_(node_ptr, getDataByInternalId(coarse_pivot), dist_func_param_);
                        if (dist < closest_dist) {
                            closest_dist = dist;
                            closest_pivot = coarse_pivot;
                        }
                    }
                }

                // - record the closest coarse pivot found (thread safe)
                pivot_assignments[node] = closest_pivot;
            }

            // - assign to the partitions (thread safe)
            for (tableint node = 0; node < dataset_size_; node++) {
                hierarchy_[ell - 1].add_member(pivot_assignments[node], node);
            }
        }
        tEnd1 = std::chrono::high_resolution_clock::now();
        double time_part = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd1 - tStart1).count();
        printf("    - partitioning time: %.4f\n", time_part);

        tEnd = std::chrono::high_resolution_clock::now();
        double time_total = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
        printf(" * Total Hierarchical Partitioning Time (s): %.4f\n", time_total);
        return;
    }

/**
 * ====================================================================================================
 *
 *                  SEARCH!
 *
 * ====================================================================================================
 */

    // setting hyper-parameters
    void setBeamSize(int beam_size) {
        beam_size_ = beam_size;
    }
    void setMaxNeighborsToCheck(int max_neighbors_to_check) {
        max_neighbors_to_check_ = max_neighbors_to_check;
    }

    // use partitioning to get a starting point, then perform beam search
    std::priority_queue<std::pair<dist_t, labeltype >> searchKnn(const void *query_ptr, size_t k) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        // obtain a starting point with top-down partitioning
        tableint start_node = 0; 
        dist_t start_node_distance = HUGE_VAL;
        for (int ell = 0; ell < num_levels_; ell++) { // include bottom level

            // - get the pivots of interest
            std::vector<tableint> candidate_pivots{};
            if (ell == 0) {
                candidate_pivots = hierarchy_[ell].pivots_;
            } else {
                candidate_pivots = hierarchy_[ell-1].get_partition(start_node);
            }

            // find the closest point in this partition
            for (uint pivot : candidate_pivots) {
                dist_t const dist = dist_func_(query_ptr, getDataByInternalId(pivot), dist_func_param_);
                // distance_count_++;
                if (dist < start_node_distance) {
                    start_node_distance = dist;
                    start_node = pivot;
                }
            }
        }


        // perform a beam_search from this point forward
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        int beam_size = std::max(beam_size_, k);
        top_candidates = beamSearch(query_ptr, start_node, beam_size);

        // initialize for output
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }



    // using a basic
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    beamSearch(const void *query_ptr, tableint start_node, size_t beam_size) const {

        // initialize visiting list
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        // initialize the beam list and candidate lists
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;
        char* start_node_ptr = getDataByInternalId(start_node);
        dist_t dist = dist_func_(query_ptr, start_node_ptr, dist_func_param_);
        dist_t lowerBound = dist;
        top_candidates.emplace(dist, start_node);
        candidate_set.emplace(-dist, start_node);
        visited_array[start_node] = visited_array_tag;

        // perform the search
        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            // stopping condition: when no points in beam are unexplored
            if (candidate_dist > lowerBound) 
                break;
            candidate_set.pop();

            // fetch the neighbors
            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            // iterate through all neighbors
            for (size_t j = 1; j <= size; j++) {
                if (j > max_neighbors_to_check_) continue;
                int candidate_id = *(data + j);

#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                // only consider if unexplored
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    // compute distance to neighbor
                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = dist_func_(query_ptr, currObj1, dist_func_param_);

                    // update data structures if necessary
                    if (top_candidates.size() < beam_size || lowerBound > dist) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif
                        top_candidates.emplace(dist, candidate_id);
                        while (top_candidates.size() > beam_size) 
                            top_candidates.pop();
                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }






};

}  // namespace hnswlib
