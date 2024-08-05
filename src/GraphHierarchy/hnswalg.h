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
        partitions_.push_back(std::vector<tableint>{});
        graph_.push_back(std::vector<tableint>{});
    }
    void set_neighbors(tableint const pivot, std::vector<tableint> const &neighbors) {
        graph_[map_[pivot]] = neighbors;
    }
    void add_member(tableint const pivot, tableint const member) { partitions_[map_[pivot]].push_back(member); }
    std::vector<tableint> const &get_neighbors(tableint pivot) { return graph_[map_.at(pivot)]; }
    std::vector<tableint> const &get_partition(tableint pivot) { return partitions_[map_.at(pivot)]; }
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
    size_t ahsp_num_partitions_{0};   // number of partitions to define region for hsp test
    size_t ahsp_beam_size_{0};        // beam size used to search for closest partitions
    size_t ahsp_max_region_size_{0};  // largest region size for hsp test

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

    HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, size_t max_elements = 0) {
        loadIndex(location, s, max_elements);
    }

    HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t scaling = 10, size_t max_neighbors = 64,
                    size_t random_seed = 100)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS), link_list_locks_(max_elements) {
        // initializing hierarchy
        max_elements_ = max_elements;
        dataset_size_ = max_elements;

        // initialize distance function
        data_size_ = s->get_data_size();
        dist_func_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        // initialize the hierarchy
        scaling_ = scaling;
        mult_ = 1 / log(1.0 * scaling_);
        num_levels_ = 0;
        double start_value = (double)dataset_size_;
        while (start_value >= 10) {
            num_levels_++;
            start_value /= (double)scaling;
        }
        hierarchy_.resize(num_levels_ - 1);  // not bottom level!
        printf("Total number of levels: %d\n", num_levels_);

        // reserve space for all those data structures

        // approximate hsp parameters
        max_neighbors_ = max_neighbors;
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

        // basic data parameters
        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);

        // hierarchy parameters
        writeBinaryPOD(output, dataset_size_);
        writeBinaryPOD(output, num_levels_);
        writeBinaryPOD(output, scaling_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, max_neighbors_);
        writeBinaryPOD(output, ahsp_num_partitions_);
        writeBinaryPOD(output, ahsp_beam_size_);
        writeBinaryPOD(output, ahsp_max_region_size_);

        // search parameters
        writeBinaryPOD(output, beam_size_);
        writeBinaryPOD(output, max_neighbors_to_check_);

        // save the bottom level memory
        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        // save the top level information
        for (int ell = 0; ell < num_levels_ - 1; ell++) {
            unsigned int num_pivots = hierarchy_[ell].num_pivots_;
            writeBinaryPOD(output, num_pivots);

            // the pivots on each level
            output.write((char *)hierarchy_[ell].pivots_.data(), num_pivots * sizeof(tableint));

            // the partitions on each level
            for (size_t itp = 0; itp < num_pivots; itp++) {
                unsigned int num_members = (unsigned int)hierarchy_[ell].partitions_[itp].size();
                writeBinaryPOD(output, num_members);

                // the pivots on each level
                output.write((char *)hierarchy_[ell].partitions_[itp].data(), num_members * sizeof(tableint));
            }
        }
        output.close();
        return;
    }

    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        printf("Loading dataset from: %s\n", location.c_str());
        std::ifstream input(location, std::ios::binary);
        if (!input.is_open()) throw std::runtime_error("Cannot open file");
        clear();

        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);
        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count) max_elements = max_elements_;
        max_elements_ = max_elements;

        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);

        // hierarchy parameters
        readBinaryPOD(input, dataset_size_);
        readBinaryPOD(input, num_levels_);
        readBinaryPOD(input, scaling_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, max_neighbors_);
        readBinaryPOD(input, ahsp_num_partitions_);
        readBinaryPOD(input, ahsp_beam_size_);
        readBinaryPOD(input, ahsp_max_region_size_);

        // get distance metric information
        data_size_ = s->get_data_size();
        dist_func_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        // load memory for bottom level
        data_level0_memory_ = (char *)malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        // load the hierarchical information
        hierarchy_.resize(num_levels_ - 1);
        for (int ell = 0; ell < num_levels_ - 1; ell++) {
            unsigned int num_pivots;
            readBinaryPOD(input, num_pivots);
            hierarchy_[ell].num_pivots_ = (size_t)num_pivots;

            // read the pivots
            hierarchy_[ell].pivots_.resize(num_pivots);
            input.read((char *)hierarchy_[ell].pivots_.data(), num_pivots * sizeof(tableint));

            // read the partitions
            hierarchy_[ell].partitions_.resize(num_pivots);
            for (size_t itp = 0; itp < num_pivots; itp++) {
                unsigned int num_members;
                readBinaryPOD(input, num_members);
                hierarchy_[ell].partitions_[itp].resize(num_members);
                input.read((char *)hierarchy_[ell].partitions_[itp].data(), num_members * sizeof(tableint));
            }
        }

        size_links_level0_ = max_neighbors_ * sizeof(tableint) + sizeof(linklistsizeint);  // memory for graph
        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

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

    // simply adding the data and initializing the bottom layer graph
    void addPoint(const void *data_point, labeltype label) {
        if (cur_element_count >= max_elements_) {
            throw std::runtime_error("The number of elements exceeds the specified limit");
        }

        // adding point to the data structure
        tableint pivot = cur_element_count;
        cur_element_count++;
        label_lookup_[label] = pivot;

        // get a random level
        int level_assignment = getRandomLevel(mult_);
        if (level_assignment > num_levels_ - 1) level_assignment = num_levels_ - 1;

        // - initializing and setting data/graph memory for bottom level
        memset(data_level0_memory_ + pivot * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
        memcpy(getExternalLabeLp(pivot), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(pivot), data_point, data_size_);

        // if belongs to a higher level, add it and initialize data structures
        for (int l = 1; l <= level_assignment; l++) {
            int level_of_interest = num_levels_ - l - 1;
            hierarchy_[level_of_interest].add_pivot(pivot);
        }

        return;
    }

    void printHierarchyStats() {
        printf("Printing hierarchy stats:\n");

        for (int ell = 0; ell < num_levels_ - 1; ell++) {
            printf(" * level %d: num_pivots = %u\n", ell, hierarchy_[ell].num_pivots_);
        }
        printf(" * bottom: %u\n", dataset_size_);

        return;
    }

    /**
     * @brief Index Construction
     *  add num_maps
     *  int M, int ef_construct, int ef_assign
     */
    void build(int num_partitions = 10) {
        printf("Begin Index Construction...\n");

        /**
         *
         *     TOP-DOWN CONSTRUCTION OF THE GRAPH HIERARCHY
         *
         */

        //> BUILD THE TOP LEVELS FIRST
        for (int ell = 1; ell < num_levels_ - 1; ell++) {
            printf("Begin Level-%d Construction\n", ell);
            size_t const num_pivots = hierarchy_[ell].num_pivots_;
            printf(" * num_pivots = %u\n", num_pivots);

            //> Perform the partitioning of this current layer
            tStart = std::chrono::high_resolution_clock::now();
            tStart1 = std::chrono::high_resolution_clock::now();
            std::vector<tableint> pivot_assignments(num_pivots);
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
                tableint fine_pivot = hierarchy_[ell].pivots_[itp];
                hierarchy_[ell - 1].add_member(pivot_assignments[itp], fine_pivot);
            }
            tEnd1 = std::chrono::high_resolution_clock::now();
            double time_part = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd1 - tStart1).count();
            printf(" * partitioning time: %.4f\n", time_part);

            //> Graph Construction
            // - building graph on every other level (num_levels_ = 7: 0,1,2,3,4,5,6 numbering)
            // - why not all? better approximate hsp performance by using larger partitions, thus using partitions
            //   defined by two layers above.
            //      * if 7 levels (odd number), then construction graph on level 2,4,6
            //      * if 8 levels (even number), then construct graph on level 1,3,5,7
            int const num_levels_odd_ = num_levels_ % 2;
            int const ell_odd = ell % 2;
            if ((num_levels_odd_ && !ell_odd) || (!num_levels_odd_ && ell_odd)) {
                printf(" * constructing graph\n");

                //> Construct the Locally Monotonic Graph on the Level
                tStart1 = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<tableint>> graph(num_pivots);
#pragma omp parallel for
                for (size_t itp = 0; itp < num_pivots; itp++) {
                    tableint const fine_pivot = hierarchy_[ell].pivots_[itp];
                    char *fine_pivot_ptr = getDataByInternalId(fine_pivot);

                    // - define the candidate list of neighbors
                    std::vector<tableint> candidate_fine_pivots{};
                    if (ell == 1 || ell == 2) {
                        candidate_fine_pivots = hierarchy_[ell].pivots_;  // using all pivots
                    } else {
                        //> Define the set of pivots by the closest partitions
                        // - use partitions to find starting node in graph of level ell - 2
                        tableint start_node;
                        for (int c = 0; c < ell - 1; c++) {
                            // - define the candidate pivots in the layer
                            std::vector<tableint> candidate_coarse_pivots{};
                            if (c == 0) {
                                candidate_coarse_pivots = hierarchy_[c].pivots_;
                            } else {
                                candidate_coarse_pivots = hierarchy_[c - 1].get_partition(start_node);
                            }

                            // - find and record the closest coarse pivot
                            dist_t start_node_distance = HUGE_VAL;
                            for (tableint coarse_pivot : candidate_coarse_pivots) {
                                dist_t const dist =
                                    dist_func_(fine_pivot_ptr, getDataByInternalId(coarse_pivot), dist_func_param_);
                                if (dist < start_node_distance) {
                                    start_node_distance = dist;
                                    start_node = coarse_pivot;
                                }
                            }
                        }

                        // - perform graph-based search to obtain the closest coarse pivots in level l-2
                        // - set beam_size with variable
                        std::vector<tableint> closest_coarse_pivots2 =
                            beamSearchOnLevel(fine_pivot_ptr, num_partitions, ell - 2, start_node);

                        // - collect all the partitions associated with the l-2 coarse pivots
                        std::vector<tableint> closest_coarse_pivots1{};
                        for (tableint coarse_pivot2 : closest_coarse_pivots2) {
                            std::vector<tableint> const &coarse_partition2 =
                                hierarchy_[ell - 2].get_partition(coarse_pivot2);
                            closest_coarse_pivots1.insert(closest_coarse_pivots1.end(), coarse_partition2.begin(),
                                                          coarse_partition2.end());
                        }

                        // - collect all the partitions associated with the l-1 coarse pivots
                        for (tableint coarse_pivot1 : closest_coarse_pivots1) {
                            std::vector<tableint> const &coarse_partition1 =
                                hierarchy_[ell - 1].get_partition(coarse_pivot1);
                            candidate_fine_pivots.insert(candidate_fine_pivots.end(), coarse_partition1.begin(),
                                                         coarse_partition1.end());
                        }
                    }

                    // - Perform the HSP test on this neighborhood
                    int max_region_size = 10000;
                    int max_neighbors = 128;
                    std::vector<uint> fine_pivot_neighbors = hsp_test(fine_pivot, candidate_fine_pivots);
                    hierarchy_[ell].set_neighbors(fine_pivot, fine_pivot_neighbors);
                }

                // - assign neighbors to the graph (thread safe)
                double ave_degree = 0.0f;
                for (size_t itp = 0; itp < num_pivots; itp++) {
                    uint fine_pivot = hierarchy_[ell].pivots_[itp];
                    ave_degree += (double)hierarchy_[ell].get_neighbors(fine_pivot).size();
                }
                ave_degree /= (double)num_pivots;
                tEnd1 = std::chrono::high_resolution_clock::now();
                double time_graph = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd1 - tStart1).count();
                printf(" * graph time: %.4f\n", time_graph);
                printf(" * ave degree: %.2f\n", ave_degree);
            } else {
                printf(" * skipping graph\n");
            }
            tEnd = std::chrono::high_resolution_clock::now();
            double time_level = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
            printf(" * total level construction time: %.4f\n", time_level);
        }

        //> BUILD THE BOTTOM LEVEL NOW
        {
            int ell = num_levels_ - 1;
            printf("Begin Level-%d Construction (bottom level)\n", ell);
            printf(" * dataset_size_ = %u\n", dataset_size_);

            //> Perform the partitioning of this current layer
            tStart = std::chrono::high_resolution_clock::now();
            tStart1 = std::chrono::high_resolution_clock::now();
            std::vector<tableint> pivot_assignments(dataset_size_);
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
            tEnd1 = std::chrono::high_resolution_clock::now();
            double time_part = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd1 - tStart1).count();
            printf(" * partitioning time: %.4f\n", time_part);

            //> Construct the Locally Monotonic Graph on the bottom level
            printf(" * constructing graph\n");
            tStart1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
            for (tableint node = 0; node < dataset_size_; node++) {
                char *node_ptr = getDataByInternalId(node);

                // - define the candidate list of neighbors
                if (ell <= 2) {
                    throw std::runtime_error("Only 2 or 3 layers... expecting more");
                }

                //> Define the set of pivots by the closest partitions
                // - use partitions to find starting node in graph of level ell - 2
                tableint start_node;
                for (int c = 0; c < ell - 1; c++) {
                    // - define the candidate pivots in the layer
                    std::vector<tableint> candidate_coarse_pivots{};
                    if (c == 0) {
                        candidate_coarse_pivots = hierarchy_[c].pivots_;
                    } else {
                        candidate_coarse_pivots = hierarchy_[c - 1].get_partition(start_node);
                    }

                    // - find and record the closest coarse pivot
                    dist_t start_node_distance = HUGE_VAL;
                    for (tableint coarse_pivot : candidate_coarse_pivots) {
                        dist_t const dist = dist_func_(node_ptr, getDataByInternalId(coarse_pivot), dist_func_param_);
                        if (dist < start_node_distance) {
                            start_node_distance = dist;
                            start_node = coarse_pivot;
                        }
                    }
                }

                // - perform graph-based search to obtain the closest coarse pivots in level l-2
                std::vector<tableint> closest_coarse_pivots2 =
                    beamSearchOnLevel(node_ptr, num_partitions, ell - 2, start_node);

                // - collect all the partitions associated with the l-2 coarse pivots
                std::vector<tableint> closest_coarse_pivots1{};
                for (tableint coarse_pivot2 : closest_coarse_pivots2) {
                    std::vector<tableint> const &coarse_partition2 = hierarchy_[ell - 2].get_partition(coarse_pivot2);
                    closest_coarse_pivots1.insert(closest_coarse_pivots1.end(), coarse_partition2.begin(),
                                                  coarse_partition2.end());
                }

                // - collect all the partitions associated with the l-1 coarse pivots
                std::vector<tableint> candidate_nodes{};
                for (tableint coarse_pivot1 : closest_coarse_pivots1) {
                    std::vector<tableint> const &coarse_partition1 = hierarchy_[ell - 1].get_partition(coarse_pivot1);
                    candidate_nodes.insert(candidate_nodes.end(), coarse_partition1.begin(), coarse_partition1.end());
                }

                // - Perform the HSP test on this neighborhood
                int max_region_size = 10000;
                int max_neighbors = 128;
                std::vector<uint> node_neighbors = hsp_test(node, candidate_nodes);

                // set the neighbors of this bottom layer node
                linklistsizeint *ll_node = get_linklist0(node);  // get start of list (num_neighbors, n1, n2, n3,...)
                setListCount(ll_node, node_neighbors.size());    // set the new number of neighbors
                tableint *node_neighbors_data =
                    (tableint *)(ll_node + 1);  // get the pointer to beginning of the neighbors
                for (size_t n = 0; n < node_neighbors.size(); n++) {
                    tableint const neighbor = node_neighbors[n];
                    node_neighbors_data[n] = neighbor;
                }
            }

            // - assign neighbors to the graph (thread safe)
            double ave_degree = 0.0f;
            for (tableint node = 0; node < dataset_size_; node++) {
                linklistsizeint *data = (linklistsizeint *)get_linklist0(node);
                ave_degree += (double)getListCount((linklistsizeint *)data);
            }
            ave_degree /= (double)dataset_size_;
            tEnd1 = std::chrono::high_resolution_clock::now();
            double time_graph = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd1 - tStart1).count();
            printf(" * graph time: %.4f\n", time_graph);
            printf(" * ave degree: %.2f\n", ave_degree);
        }
    }

    // performing a beam search to obtain the closest partitions for approximate hsp
    std::vector<tableint> beamSearchOnLevel(const void *query_ptr, int k, int level, tableint start_node) {
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

        // now update the
        std::vector<tableint> neighbors(k);
        while (top_candidates.size() > k) top_candidates.pop();
        for (int i = k - 1; i >= 0; i--) {
            neighbors[i] = top_candidates.top().second;
            top_candidates.pop();
        }
        return neighbors;
    }

    // perform the hsp test to get the hsp neighbors of the node
    std::vector<tableint> hsp_test(tableint const query_node, std::vector<tableint> const &set) {
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

        // can trim the test here if desired
        // nth_select()

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
};
}  // namespace hnswlib
