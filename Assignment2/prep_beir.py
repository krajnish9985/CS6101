import argparse
import time

import numpy as np
import pyarrow.parquet as pq
import sklearn
from sklearn.decomposition import PCA

from common import PROC, RAW, mib, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["quora", "nq", "msmarco"])
ap.add_argument("--sample", type=int, default=200_000, help="PCA fitting-sample size (corpus only)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--solver", default="auto", choices=["auto", "randomized", "full", "covariance_eigh"],
                help="sklearn PCA svd_solver (teammate used randomized, seed 42)")
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--query-split", default="auto", help="queries/<split>.parquet; auto = test (dev for msmarco)")
ap.add_argument("--batch", type=int, default=20_000)
args = ap.parse_args()

src = RAW / "beir" / args.dataset
corpus_files = sorted((src / "corpus").glob("*.parquet"))
split = args.query_split if args.query_split != "auto" else ("dev" if args.dataset == "msmarco" else "test")
query_file = src / "queries" / f"{split}.parquet"
assert query_file.exists(), f"missing {query_file}"
assert corpus_files, f"no parquet files in {src/'corpus'} -- run scripts/download_beir.py on the login node"


def emb_batches(files, batch):
    for f in files:
        pf = pq.ParquetFile(f)
        for rb in pf.iter_batches(batch_size=batch, columns=["emb"]):
            col = rb.column(0)
            vals = col.flatten().to_numpy(zero_copy_only=False)
            yield np.asarray(vals, dtype=np.float32).reshape(len(col), -1)


def n_rows(files):
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def project(pca, x):
    y = ((x - pca.mean_) @ pca.components_.T).astype(np.float32)
    y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return y


t0 = time.time()
N = n_rows(corpus_files)
rng = np.random.default_rng(args.seed)
m = min(args.sample, N)
sample_idx = np.sort(rng.choice(N, size=m, replace=False))
print(f"{args.dataset}: corpus N={N:,}; PCA sample={m:,} (seed {args.seed})", flush=True)

Xs, off, ptr, D0 = [], 0, 0, None
for blk in emb_batches(corpus_files, args.batch):
    D0 = blk.shape[1]
    hi = off + blk.shape[0]
    j = np.searchsorted(sample_idx, hi)
    if j > ptr:
        Xs.append(blk[sample_idx[ptr:j] - off])
        ptr = j
    off = hi
Xs = np.concatenate(Xs)
assert Xs.shape[0] == m and off == N, (Xs.shape, off, N)
print(f"pass 1 done ({time.time()-t0:.0f}s), original dim = {D0}", flush=True)

solver = args.solver if args.solver != "auto" else (
    "covariance_eigh" if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 5) else "full")
pca = PCA(n_components=args.dim, svd_solver=solver, random_state=args.seed).fit(Xs)
evr = float(pca.explained_variance_ratio_.sum())
print(f"PCA fitted ({solver}); variance explained by {args.dim} comps = {evr:.4f}", flush=True)
del Xs

out = np.lib.format.open_memmap(PROC / f"{args.dataset}_base.npy", mode="w+", dtype=np.float32,
                                shape=(N, args.dim))
off = 0
for blk in emb_batches(corpus_files, args.batch):
    out[off:off + blk.shape[0]] = project(pca, blk)
    off += blk.shape[0]
out.flush()
del out
print(f"pass 2 done ({time.time()-t0:.0f}s)", flush=True)

Q = np.concatenate(list(emb_batches([query_file], args.batch)))
Qp = project(pca, Q)
np.save(PROC / f"{args.dataset}_query.npy", Qp)
np.savez(PROC / f"{args.dataset}_pca.npz", mean=pca.mean_, components=pca.components_,
         explained_variance_ratio=pca.explained_variance_ratio_)

save_json({"dataset": args.dataset, "source": "CohereLabs/beir-embed-english-v3 (corpus + test queries)",
           "corpus_size": int(N), "num_queries": int(Q.shape[0]), "original_dim": int(D0),
           "search_dim": args.dim, "metric": "L2 on unit-normalised vectors (== cosine ranking)",
           "raw_float32_bytes": int(N) * args.dim * 4, "raw_float32_MiB": mib(int(N) * args.dim * 4),
           "pca_impl": f"sklearn {sklearn.__version__} PCA(svd_solver='{solver}')",
           "pca_fit_on": "uniform random sample of corpus embeddings only (no queries)",
           "pca_sample_size": int(m), "pca_seed": args.seed, "pca_explained_variance": evr,
           "query_split": split, "post_processing": "project -> float32 -> L2-normalise (corpus and queries)",
           "prep_time_s": time.time() - t0}, PROC / f"{args.dataset}_meta.json")
print(f"saved. total {time.time()-t0:.0f}s")
