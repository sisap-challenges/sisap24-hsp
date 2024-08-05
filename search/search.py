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


def run(size, M, ef_construction):

    #> Initialize the HNSW index
    index = GraphHierarchy.Index(space='ip', dim=768) # possible options are l2, cosine or ip
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=str,
        default="300K"
    )
    parser.add_argument(
        "-M",
        type=int,
        default=20,
    )
    parser.add_argument(
        "-E",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    assert args.size in ["300K", "10M", "100M"]

    print("Running Script With:")
    print(f"  * N={args.size}")
    print(f"  * M={args.M}                  | HNSW Parameter M")
    print(f"  * EFC={args.E}               | HNSW Parameter ef_construction")
    run(args.size, args.M, args.E)
    print(f"Done! Have a good day!")
