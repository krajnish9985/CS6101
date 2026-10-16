import argparse

import numpy as np

from common import (IDX, RES, latency_summary, load_base, load_json, load_queries, load_reference_gt,
                    median_run, n_threads_default, recalls, save_json, system_info, timing_subset)
from vamana import Vamana

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--Ls", default="10,20,30,50,75,100,150,200")
ap.add_argument("--threads", type=int, default=n_threads_default())
ap.add_argument("--n-timing", type=int, default=1000)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--repeats", type=int, default=3)
args = ap.parse_args()

X = load_base(args.dataset)
Q = load_queries(args.dataset)
gt = load_reference_gt(args.dataset)
idx = Vamana.load(IDX / f"vamana_{args.tag}", X)
build = load_json(RES / f"vamana_build_{args.tag}.json")
qidx = timing_subset(Q.shape[0], args.n_timing)

points = []
for L in [int(v) for v in args.Ls.split(",")]:
    I, ne, nd = idx.search(Q, k=10, L=L, threads=args.threads)
    lat, rep = median_run(lambda: idx.time_queries(Q, qidx, k=10, L=L, warmup=args.warmup), args.repeats)
    p = {"L_search": L, "R": idx.meta["R"], "alpha": idx.meta["alpha"], "L_build": idx.meta["L_build"],
         **recalls(I, gt), **latency_summary(lat), "avg_dist_evals": float(nd.mean()),
         "avg_vertices_expanded": float(ne.mean()), "repeat_mean_latencies_ms": rep}
    points.append(p)
    print(f"L={L:4d}  1R@1={p['recall_1_at_1']:.4f}  10R@10={p['recall_10_at_10']:.4f}  "
          f"lat={p['mean_latency_ms']:.3f}ms  QPS={p['qps']:.0f}  exp={p['avg_vertices_expanded']:.1f}  "
          f"dist={p['avg_dist_evals']:.0f}", flush=True)

save_json({"dataset": args.dataset, "tag": args.tag, "method": "Vamana (ours)", "R": idx.meta["R"],
           "alpha": idx.meta["alpha"], "L_build": idx.meta["L_build"], "build_time_s": build["build_time_s"],
           "graph_bytes": build.get("graph_bytes"), "graph_plus_vectors_bytes": build.get("graph_plus_vectors_bytes"),
           "eval_system": system_info(), "timing_threads": 1, "query_mode": "one query per call",
           "timing_queries": int(len(qidx)), "recall_queries": int(Q.shape[0]), "points": points},
          RES / f"vamana_search_{args.tag}.json")
