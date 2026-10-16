import argparse

import numpy as np

from common import PROC, RAW, mib, read_vecs, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--dev-size", type=int, default=100_000, help="size of the SIFT development subset")
args = ap.parse_args()

src = RAW / "sift" / "sift"
if (src / "sift_base.fvecs").exists():
    base = read_vecs(src / "sift_base.fvecs", np.float32)
    query = read_vecs(src / "sift_query.fvecs", np.float32)
    gt = read_vecs(src / "sift_groundtruth.ivecs", np.int32)
    source = "TEXMEX sift.tar.gz (fvecs/ivecs)"
else:
    import h5py
    with h5py.File(RAW / "sift" / "sift-128-euclidean.hdf5", "r") as f:
        base = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        query = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        gt = np.ascontiguousarray(f["neighbors"][:], dtype=np.int32)
    source = "ann-benchmarks sift-128-euclidean.hdf5"

print("SIFT1M base", base.shape, "query", query.shape, "gt", gt.shape)
np.save(PROC / "sift1m_base.npy", base)
np.save(PROC / "sift1m_query.npy", query)
np.save(PROC / "sift1m_gt_supplied.npy", gt)

n = args.dev_size
np.save(PROC / "sift100k_base.npy", base[:n])
np.save(PROC / "sift100k_query.npy", query)

for name, b in (("sift1m", base), ("sift100k", base[:n])):
    save_json({"dataset": name, "source": source, "corpus_size": int(b.shape[0]),
               "num_queries": int(query.shape[0]), "original_dim": 128, "search_dim": int(b.shape[1]),
               "metric": "L2", "raw_float32_bytes": int(b.shape[0]) * int(b.shape[1]) * 4,
               "raw_float32_MiB": mib(int(b.shape[0]) * int(b.shape[1]) * 4),
               "subset_rule": "first %d base vectors of SIFT1M" % n if name == "sift100k" else None},
              PROC / f"{name}_meta.json")
print("done")
