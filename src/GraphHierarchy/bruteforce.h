#pragma once
#include <unordered_map>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <assert.h>

namespace hnswlib {
template<typename dist_t>
class BruteforceSearch : public AlgorithmInterface<dist_t> {
 public:
    char *data_;
    size_t maxelements_;
    size_t cur_element_count;
    size_t size_per_element_;

    size_t data_size_;
    DISTFUNC <dist_t> fstdistfunc_;
    void *dist_func_param_;
    std::mutex index_lock;

    std::unordered_map<labeltype, size_t > dict_external_to_internal;


    BruteforceSearch(SpaceInterface <dist_t> *s)
        : data_(nullptr),
            maxelements_(0),
            cur_element_count(0),
            size_per_element_(0),
            data_size_(0),
            dist_func_param_(nullptr) {
    }


    BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location)
        : data_(nullptr),
            maxelements_(0),
            cur_element_count(0),
            size_per_element_(0),
            data_size_(0),
            dist_func_param_(nullptr) {
        loadIndex(location, s);
    }


    BruteforceSearch(SpaceInterface <dist_t> *s, size_t maxElements) {
        maxelements_ = maxElements;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        size_per_element_ = data_size_ + sizeof(labeltype);
        data_ = (char *) malloc(maxElements * size_per_element_);
        if (data_ == nullptr)
            throw std::runtime_error("Not enough memory: BruteforceSearch failed to allocate data");
        cur_element_count = 0;
    }

    ~BruteforceSearch() {
        free(data_);
    }


    void addPoint(const void *datapoint, labeltype label) {
        int idx;
        {
            std::unique_lock<std::mutex> lock(index_lock);

            auto search = dict_external_to_internal.find(label);
            if (search != dict_external_to_internal.end()) {
                idx = search->second;
            } else {
                if (cur_element_count >= maxelements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit\n");
                }
                idx = cur_element_count;
                dict_external_to_internal[label] = idx;
                cur_element_count++;
            }
        }
        memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(labeltype));
        memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
    }

    std::priority_queue<std::pair<dist_t, labeltype >> searchKnn(const void *query_data, size_t k) const {
        assert(k <= cur_element_count);

        std::priority_queue<std::pair<dist_t, labeltype >> topResults;
        if (cur_element_count == 0) return topResults;
        for (int i = 0; i < k; i++) {
            dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
            labeltype label = *((labeltype*) (data_ + size_per_element_ * i + data_size_));
            topResults.emplace(dist, label);
        }

        dist_t lastdist = topResults.empty() ? std::numeric_limits<dist_t>::max() : topResults.top().first;
        for (int i = k; i < cur_element_count; i++) {
            dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
            if (dist <= lastdist) {
                labeltype label = *((labeltype *) (data_ + size_per_element_ * i + data_size_));
                topResults.emplace(dist, label);
                if (topResults.size() > k)
                    topResults.pop();
                if (!topResults.empty()) {
                    lastdist = topResults.top().first;
                }
            }
        }
        return topResults;
    }
};
}  // namespace hnswlib
