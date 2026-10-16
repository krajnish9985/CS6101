import argparse
import json
import resource
import time

import numpy as np

from cache_tools import fincore
from common import IDX, RES, load_queries, load_reference_gt, peak_rss_mb, recalls, save_json, system_info
from diskann import DiskIndex, proc_read_bytes

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="msmarco")
ap.add_argument("--dir", default=None)
ap.add_argument("--mode", required=True, choices=["cold", "warm"])
ap.add_argument("--Ls", default="100")
ap.add_argument("--W", type=int, default=4)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--nq", type=int, default=0, help="0 = all queries")
ap.add_argument("--warmup-queries", type=int, default=3000, help="warm mode: queries used to exercise the index")
ap.add_argument("--fincore-json", default=None, help="cold mode: residency recorded right before this process")
ap.add_argument("--tag", default="")
args = ap.parse_args()

d = IDX / (args.dir or f"disk_{args.dataset}")
t0 = time.perf_counter()
idx = DiskIndex(d)
t_load = time.perf_counter() - t0
Q = load_queries(args.dataset)
gt = load_reference_gt(args.dataset)
if args.nq:
    Q, gt = Q[:args.nq], gt[:args.nq]
Ls = [int(x) for x in args.Ls.split(",")]
if args.mode == "cold":
    assert len(Ls) == 1, "cold mode: one L per process (evict before each run)"
lim = resource.getrlimit(resource.RLIMIT_AS)[0]
resident_ram = idx.codes.nbytes + idx.centroids.nbytes + idx.meta["N"] * 4

pre = None
if args.mode == "cold" and args.fincore_json:
    pre = json.load(open(args.fincore_json))["files"][0]["after"]
if args.mode == "warm":
    wq = np.random.default_rng(1).permutation(Q.shape[0])
    wq = Q[np.resize(wq, args.warmup_queries)]
    tw = time.perf_counter()
    idx.run(wq, k=args.k, L=max(Ls), W=args.W)
    print(f"warm-up: {args.warmup_queries} queries at L={max(Ls)} in {time.perf_counter()-tw:.1f}s", flush=True)
    pre = fincore(idx.path)

for L in Ls:
    io0 = proc_read_bytes()
    ids, st, lat = idx.run(Q, k=args.k, L=L, W=args.W)
    io1 = proc_read_bytes()
    nq = Q.shape[0]
    s = st.mean(axis=0)
    r = {"dataset": args.dataset, "mode": args.mode, "L_search": L, "beam_width_W": args.W, "num_queries": int(nq),
         **recalls(ids, gt),
         "mean_latency_ms": float(lat.mean() * 1e3), "p50_latency_ms": float(np.percentile(lat, 50) * 1e3),
         "p95_latency_ms": float(np.percentile(lat, 95) * 1e3), "qps": float(1.0 / lat.mean()),
         "mean_latency_first100_ms": float(lat[:100].mean() * 1e3),
         "reads_per_query": float(s[0]), "bytes_read_per_query": float(s[1]), "rounds_per_query": float(s[2]),
         "expansions_per_query": float(s[3]), "pq_dists_per_query": float(s[4]), "full_dists_per_query": float(s[5]),
         "storage_bytes_per_query": float((io1 - io0) / nq) if io0 >= 0 else None,
         "fincore_before_run": pre, "peak_rss_MiB": peak_rss_mb(), "rlimit_as_bytes": None if lim < 0 else int(lim),
         "resident_structures_bytes": int(resident_ram), "index_load_s": t_load,
         "disk_index_bytes": idx.meta["disk_index_bytes"], "system": system_info(), "threads": 1,
         "query_mode": "one query per call"}
    save_json(r, RES / f"ec_{args.mode}_L{L}_W{args.W}{args.tag}.json")
    print(f"[{args.mode}] L={L:4d} W={args.W}  1R@1={r['recall_1_at_1']:.4f} 10R@10={r['recall_10_at_10']:.4f}  "
          f"mean={r['mean_latency_ms']:.3f}ms p95={r['p95_latency_ms']:.3f}ms QPS={r['qps']:.0f}  "
          f"reads/q={r['reads_per_query']:.1f} rounds/q={r['rounds_per_query']:.1f} "
          f"storageKB/q={(r['storage_bytes_per_query'] or 0)/1024:.1f}  peakRSS={r['peak_rss_MiB']:.0f}MiB", flush=True)
idx.close()
