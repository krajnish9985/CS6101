import argparse
import time

import faiss
import numpy as np

from common import (IDX, RES, file_size, latency_summary, load_base, load_json, load_queries,
                    load_reference_gt, median_run, n_threads_default, peak_rss_mb, recalls, save_json,
                    system_info, timing_subset)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--M", type=int, default=32)
ap.add_argument("--efc", type=int, default=200)
ap.add_argument("--efs", default="16,32,64,128,256,512")
ap.add_argument("--threads", type=int, default=n_threads_default())
ap.add_argument("--n-timing", type=int, default=1000)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--rebuild", action="store_true")
args = ap.parse_args()
ds = args.dataset

X = load_base(ds)
Q = load_queries(ds)
gt = load_reference_gt(ds)
N, d = X.shape
path = IDX / f"hnsw_{ds}_M{args.M}_efc{args.efc}.faiss"
bpath = path.with_suffix(".build.json")
faiss.omp_set_num_threads(args.threads)

if path.exists() and not args.rebuild:
    t0 = time.perf_counter()
    index = faiss.read_index(str(path))
    print(f"loaded {path} in {time.perf_counter()-t0:.1f}s (load time excluded from latency)")
    build = load_json(bpath)
else:
    index = faiss.IndexHNSWFlat(d, args.M)
    index.hnsw.efConstruction = args.efc
    t0 = time.perf_counter()
    index.add(X)
    bt = time.perf_counter() - t0
    faiss.write_index(index, str(path))
    build = {"build_time_s": bt, "build_threads": args.threads, "peak_rss_MiB_after_build": peak_rss_mb(),
             "system": system_info()}
    save_json(build, bpath)
    print(f"built HNSW M={args.M} efC={args.efc} in {bt:.1f}s with {args.threads} threads", flush=True)
del X

qidx = timing_subset(Q.shape[0], args.n_timing)
qs = [Q[i:i + 1] for i in qidx]
points = []
for ef in [int(e) for e in args.efs.split(",")]:
    index.hnsw.efSearch = ef
    faiss.omp_set_num_threads(args.threads)
    _, I = index.search(Q, 10)
    r = recalls(I, gt)
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
    p = {"efSearch": ef, **r, **latency_summary(lat), "repeat_mean_latencies_ms": rep}
    points.append(p)
    print(f"efSearch={ef:4d}  1R@1={p['recall_1_at_1']:.4f}  10R@10={p['recall_10_at_10']:.4f}  "
          f"lat={p['mean_latency_ms']:.3f}ms  QPS={p['qps']:.0f}", flush=True)

save_json({"dataset": ds, "method": "FAISS IndexHNSWFlat", "M": args.M, "efConstruction": args.efc,
           "index_size_bytes": file_size(path), **build, "eval_system": system_info(), "timing_threads": 1,
           "query_mode": "one query per search() call", "timing_queries": int(len(qidx)),
           "recall_queries": int(Q.shape[0]), "points": points}, RES / f"hnsw_{ds}.json")
