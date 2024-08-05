'''
    Cole Foster
    August 4th, 2024

    SISAP 2024 Indexing Challenge
'''
import argparse
import GraphHierarchy
import h5py
import numpy as np
import os
from pathlib import Path
from urllib.request import urlretrieve
import time

data_directory = "/users/cfoste18/scratch/datasets/LAION"

def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def prepare(kind, size):
    dataset_base_url = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge"
    task = {
        "query": "http://ingeotec.mx/~sadit/sisap2024-data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5",
        "dataset": f"{dataset_base_url}/laion2B-en-{kind}-n={size}.h5",
    }

    for version, url in task.items():
        download(url, os.path.join(data_directory, kind, size, f"{version}.h5"))

def store_results(dst, algo, kind, D, I, buildtime, querytime, params, size):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['data'] = kind
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['size'] = size
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()


def run(size, scaling, partitions):
    kind = "clip768v2"
    key = "emb"
    print(f"Running HNSW on {kind}-{size}")
    index_identifier = f"HNSW-s-{scaling}-p-{partitions}"
    
    #> Download dataset if necessary
    prepare(kind, size)
    D=768

    #> 
    scaling = 10
    max_neighbors = 32

    #> Initialize the HNSW index
    index = GraphHierarchy.Index(space='ip', dim=D) # possible options are l2, cosine or ip

    #> Load the dataset: 
    start_time = time.time()
    with h5py.File(os.path.join(data_directory, kind, size, "dataset.h5"), 'r') as f:
        dataset = f[key]
        N,DD = dataset.shape
        print(f'Datset has N={N} rows and D={DD} columns')
        index.init_index(max_elements=N, scaling=scaling, max_neighbors=max_neighbors, random_seed=10)
        print(" * it init!")
        
        # determine number of rows
        total_rows = dataset.shape[0]
        chunk_size = 100000

        # iterate over the dataset, add each chunk
        for start_index in range(0, total_rows, chunk_size):
            end_index = min(start_index + chunk_size, total_rows)

            # load this chunk into memory
            data_chunk = dataset[start_index:end_index]

            # add it to hnsw index
            index.add_items(data_chunk)
    print(f" * done adding items {time.time() - start_time:.4} (s)")

    # construct
    index.build(partitions)
    build_time = time.time() - start_time
    print(f"Done Constructing Index in {build_time:.4f} (s)")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=str,
        default="300K"
    )
    parser.add_argument(
        "-s",
        "--scaling",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-p",
        "--partitions",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    assert args.size in ["300K", "10M", "100M"]

    print("Running Script With:")
    print(f"  * N={args.size}")
    print(f"  * s={args.scaling}")
    print(f"  * p={args.partitions} ")
    run(args.size, args.scaling, args.partitions)
    print(f"Done! Have a good day!")
