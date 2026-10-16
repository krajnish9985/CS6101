#!/bin/bash
cd "$(dirname "$0")/.."
mkdir -p data/raw/sift
wget -c -O data/raw/sift/sift-128-euclidean.hdf5 http://ann-benchmarks.com/sift-128-euclidean.hdf5
