import argparse
import time

from common import IDX, RES, load_base, mib, n_threads_default, peak_rss_mb, save_json, system_info
from vamana import Vamana

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--R", type=int, default=64)
ap.add_argument("--L", type=int, default=100, help="L_build")
ap.add_argument("--alpha", type=float, default=1.2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=n_threads_default())
ap.add_argument("--batch-frac", type=float, default=0.02, help="max parallel insertion batch as a fraction of N")
ap.add_argument("--n-sub", type=int, default=None, help="build on the first n points only (scaling study)")
ap.add_argument("--no-save", action="store_true")
ap.add_argument("--tag", default=None)
args = ap.parse_args()
tag = args.tag or f"{args.dataset}_R{args.R}_L{args.L}_a{args.alpha}"

t0 = time.perf_counter()
X = load_base(args.dataset, args.n_sub)
print(f"loaded {args.dataset} {X.shape} in {time.perf_counter()-t0:.1f}s  (rss {peak_rss_mb():.0f} MiB)", flush=True)
rss_before = peak_rss_mb()

idx = Vamana.build(X, R=args.R, L=args.L, alpha=args.alpha, seed=args.seed, threads=args.threads,
                   max_batch_frac=args.batch_frac)
peak = peak_rss_mb()
st = idx.degree_stats()
assert st["max_degree_le_R"], "some vertex has out-degree > R!"
print(f"build {idx.meta['build_time_s']:.1f}s  avg deg {st['avg_out_degree']:.2f}  max deg {st['max_out_degree']}  "
      f"peak RSS {peak:.0f} MiB", flush=True)

out = {"tag": tag, "dataset": args.dataset, "n_points": int(X.shape[0]), **idx.meta, **st,
       "peak_rss_MiB": peak, "rss_after_loading_vectors_MiB": rss_before, "system": system_info(),
       "graph_representation": "fixed-width int32 adjacency [N,R] (padded -1) + int32 degree[N]",
       "language": "C++17 (OpenMP, -O3 -march=native) called from Python via ctypes"}
if not args.no_save:
    nbytes = idx.save(IDX / f"vamana_{tag}")
    out.update({"graph_bytes": nbytes, "graph_MiB": mib(nbytes),
                "graph_plus_vectors_bytes": nbytes + X.nbytes, "graph_plus_vectors_MiB": mib(nbytes + X.nbytes)})
    print(f"saved graph: {mib(nbytes):.1f} MiB (+ vectors {mib(X.nbytes):.1f} MiB)")
save_json(out, RES / f"vamana_build_{tag}.json")
