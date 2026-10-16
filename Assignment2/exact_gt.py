import argparse
import time

import faiss
import numpy as np

from common import (K_GT, PROC, RES, base_path, gt_path, load_base, load_queries, k_recall_at_k, latency_summary,
                    median_run, n_threads_default, peak_rss_mb, recalls, save_json, system_info, timing_subset)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--threads", type=int, default=n_threads_default())
ap.add_argument("--n-timing", type=int, default=1000)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--recompute", action="store_true")
ap.add_argument("--no-timing", action="store_true", help="only compute/verify ground truth")
args = ap.parse_args()
ds = args.dataset

X = np.load(base_path(ds), mmap_mode="r")
Q = load_queries(ds)
N, d = X.shape
print(f"{ds}: base {X.shape} queries {Q.shape}", flush=True)

faiss.omp_set_num_threads(args.threads)
t0 = time.perf_counter()
index = faiss.IndexFlatL2(d)
for s0 in range(0, N, 500_000):
    index.add(np.ascontiguousarray(X[s0:s0 + 500_000], dtype=np.float32))
build_time = time.perf_counter() - t0

if gt_path(ds).exists() and not args.recompute:
    I = np.load(gt_path(ds))
    print("loaded saved exact GT", I.shape)
    gt_time = None
else:
    t0 = time.perf_counter()
    Is = []
    for s in range(0, Q.shape[0], 2000):
        _, Ib = index.search(Q[s:s + 2000], K_GT)
        Is.append(Ib.astype(np.int32))
    I = np.concatenate(Is)
    gt_time = time.perf_counter() - t0
    np.save(gt_path(ds), I)
    print(f"exact GT computed in {gt_time:.1f}s with {args.threads} threads -> {gt_path(ds)}", flush=True)

out = {"dataset": ds, "method": "FAISS IndexFlatL2 (exact)", "system": system_info(),
       "build_threads": args.threads, "build_time_s": build_time, "gt_compute_time_s": gt_time,
       "index_size_bytes": int(index.ntotal) * d * 4,
       "num_queries": int(Q.shape[0])}

if ds == "sift1m":
    sup = np.load(PROC / "sift1m_gt_supplied.npy")
    agree = {"top1_agreement": float((I[:, 0] == sup[:, 0]).mean()),
             "10recall@10_vs_supplied": k_recall_at_k(I, sup, 10),
             "100recall@100_vs_supplied": k_recall_at_k(I, sup, 100)}
    print("agreement with supplied SIFT1M GT:", agree)
    out["supplied_gt_agreement"] = agree
    out.update(recalls(I, sup))
else:
    out.update(recalls(I, I))

if args.no_timing:
    save_json(out, RES / f"exact_{ds}.json")
    raise SystemExit(0)

qidx = timing_subset(Q.shape[0], args.n_timing)
qs = [Q[i:i + 1] for i in qidx]
faiss.omp_set_num_threads(1)


def run():
    for q in qs[:args.warmup]:
        index.search(q, 10)
    lat = np.empty(len(qs))
    for t, q in enumerate(qs):
        t0 = time.perf_counter()
        index.search(q, 10)
        lat[t] = time.perf_counter() - t0
    return lat


lat, rep = median_run(run, args.repeats)
out.update(latency_summary(lat))
out.update({"timing_threads": 1, "query_mode": "one query per search() call", "timing_queries": int(len(qidx)),
            "repeat_mean_latencies_ms": rep, "peak_rss_MiB": peak_rss_mb()})
save_json(out, RES / f"exact_{ds}.json")
print(f"exact {ds}: mean latency {out['mean_latency_ms']:.2f} ms  QPS {out['qps']:.1f}")
