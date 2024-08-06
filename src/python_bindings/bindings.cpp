#include <assert.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdlib.h>

#include <atomic>
#include <iostream>
#include <thread>

#include "hnswlib.h"

namespace py = pybind11;
using namespace pybind11::literals;  // needed to bring in _a literal

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

inline void assert_true(bool expr, const std::string& msg) {
    if (expr == false) throw std::runtime_error("Unpickle Error: " + msg);
    return;
}

class CustomFilterFunctor : public hnswlib::BaseFilterFunctor {
    std::function<bool(hnswlib::labeltype)> filter;

   public:
    explicit CustomFilterFunctor(const std::function<bool(hnswlib::labeltype)>& f) { filter = f; }

    bool operator()(hnswlib::labeltype id) { return filter(id); }
};

inline void get_input_array_shapes(const py::buffer_info& buffer, size_t* rows, size_t* features) {
    if (buffer.ndim != 2 && buffer.ndim != 1) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "Input vector data wrong shape. Number of dimensions %d. Data must be a 1D or 2D array.", buffer.ndim);
        throw std::runtime_error(msg);
    }
    if (buffer.ndim == 2) {
        *rows = buffer.shape[0];
        *features = buffer.shape[1];
    } else {
        *rows = 1;
        *features = buffer.shape[0];
    }
}

inline std::vector<size_t> get_input_ids_and_check_shapes(const py::object& ids_, size_t feature_rows) {
    std::vector<size_t> ids;
    if (!ids_.is_none()) {
        py::array_t<size_t, py::array::c_style | py::array::forcecast> items(ids_);
        auto ids_numpy = items.request();
        // check shapes
        if (!((ids_numpy.ndim == 1 && ids_numpy.shape[0] == feature_rows) ||
              (ids_numpy.ndim == 0 && feature_rows == 1))) {
            char msg[256];
            snprintf(msg, sizeof(msg), "The input label shape %d does not match the input data vector shape %d",
                     ids_numpy.ndim, feature_rows);
            throw std::runtime_error(msg);
        }
        // extract data
        if (ids_numpy.ndim == 1) {
            std::vector<size_t> ids1(ids_numpy.shape[0]);
            for (size_t i = 0; i < ids1.size(); i++) {
                ids1[i] = items.data()[i];
            }
            ids.swap(ids1);
        } else if (ids_numpy.ndim == 0) {
            ids.push_back(*items.data());
        }
    }

    return ids;
}




template <typename dist_t, typename data_t = float>
class Index {
   public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    size_t seed;
    size_t default_ef;

    bool index_inited;
    bool normalize;
    int num_threads_default;
    hnswlib::labeltype cur_l;  // element count
    hnswlib::HierarchicalNSW<dist_t>* appr_alg;
    hnswlib::SpaceInterface<float>* l2space;

    Index(const std::string& space_name, const int dim) : space_name(space_name), dim(dim) {
        normalize = false;
        if (space_name == "l2") {
            l2space = new hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            l2space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            l2space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            throw std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        appr_alg = NULL;
        index_inited = false;
        num_threads_default = std::thread::hardware_concurrency();
    }

    ~Index() {
        delete l2space;
        if (appr_alg) delete appr_alg;
    }

    void init_new_index(size_t maxElements, size_t max_neighbors, size_t random_seed) {
        if (appr_alg) {
            throw std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, maxElements, max_neighbors, random_seed);
        index_inited = true;
        seed = random_seed;
    }

    void set_beam_size(size_t beam_size) {
      if (appr_alg)
          appr_alg->setBeamSize(beam_size);
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }

    void loadIndex(const std::string &path_to_index) {
      if (appr_alg) {
          std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." <<
          std::endl; delete appr_alg;
      }
      appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, path_to_index); 
      cur_l = appr_alg->cur_element_count; 
      index_inited = true;
    }

    void addItems(py::object input, py::object ids_ = py::none(), int num_threads = -1) {
        py::array_t<dist_t, py::array::c_style | py::array::forcecast> items(input);
        auto buffer = items.request();
        if (num_threads <= 0) num_threads = num_threads_default;

        // check the dimensions of the input
        size_t rows, features;
        get_input_array_shapes(buffer, &rows, &features);
        if (features != dim) throw std::runtime_error("Wrong dimensionality of the vectors");
        std::vector<size_t> ids = get_input_ids_and_check_shapes(ids_, rows);

        // add the elements to the index!
        {
            py::gil_scoped_release l;
            int start = 0;
            ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                appr_alg->addPoint((void*)items.data(row), (size_t)id);
                });
            cur_l += rows;
        }
    }

    void build(int scaling) { 

        // build the approximate hsp
        appr_alg->build(100, 10); 

        // initialize a heirarchical partitioning for fast entrypoint
        appr_alg->createHierarchicalPartitioning(scaling);
    }

    // the search function
    py::object knnQuery_return_numpy(py::object input, size_t k = 1, int num_threads = -1) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        hnswlib::labeltype* data_numpy_l;
        dist_t* data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0) num_threads = num_threads_default;
        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            // preparing output
            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];

            // perform the search (in parallel)
            ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn((void*)items.data(row), k);
                if (result.size() != k)
                    throw std::runtime_error(
                        "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                for (int i = k - 1; i >= 0; i--) {
                    auto& result_tuple = result.top();
                    data_numpy_d[row * k + i] = result_tuple.first;
                    data_numpy_l[row * k + i] = result_tuple.second;
                    result.pop();
                }
            });
        }
        py::capsule free_when_done_l(data_numpy_l, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) {
            delete[] f;
            });

        return py::make_tuple(
            py::array_t<hnswlib::labeltype>(
                { rows, k },  // shape
                { k * sizeof(hnswlib::labeltype),
                  sizeof(hnswlib::labeltype) },  // C-style contiguous strides for each index
                data_numpy_l,  // the data pointer
                free_when_done_l),
            py::array_t<dist_t>(
                { rows, k },  // shape
                { k * sizeof(dist_t), sizeof(dist_t) },  // C-style contiguous strides for each index
                data_numpy_d,  // the data pointer
                free_when_done_d));
    }
};


PYBIND11_PLUGIN(GraphHierarchy) {
    py::module m("GraphHierarchy");

    py::class_<Index<float>>(m, "Index")
        // .def(py::init(&Index<float>::createFromParams), py::arg("params"))
        /* WARNING: Index::createFromIndex is not thread-safe with Index::addItems */
        // .def(py::init(&Index<float>::createFromIndex), py::arg("index"))
        .def(py::init<const std::string&, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index", &Index<float>::init_new_index, py::arg("max_elements"), py::arg("max_neighbors") = 32, py::arg("random_seed") = 100)
        .def("add_items", &Index<float>::addItems, py::arg("data"), py::arg("ids") = py::none(),
             py::arg("num_threads") = -1)
        .def("build", &Index<float>::build, py::arg("scaling") = 10)
        .def("set_beam_size", &Index<float>::set_beam_size, py::arg("beam_size") = 10)
        .def("knn_query",
            &Index<float>::knnQuery_return_numpy,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1);
    // .def("get_items", &Index<float>::getData, py::arg("ids") = py::none(), py::arg("return_type") = "numpy")
    // .def("get_ids_list", &Index<float>::getIdsList)
    // .def("get_max_elements", &Index<float>::getMaxElements)
    // .def("get_current_count", &Index<float>::getCurrentCount)
    // .def_readonly("space", &Index<float>::space_name)
    // .def_readonly("dim", &Index<float>::dim)
    // .def_readwrite("num_threads", &Index<float>::num_threads_default)

    return m.ptr();
}