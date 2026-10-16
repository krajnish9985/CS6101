import argparse

import numpy as np

from common import IDX, RES, gt_path, load_base, load_queries, save_json
from vamana import Vamana

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="sift100k")
ap.add_argument("--tag", default="sift100k_ref")
ap.add_argument("--qid", type=int, default=0)
ap.add_argument("--L", type=int, default=100)
args = ap.parse_args()

X = load_base(args.dataset)
Q = load_queries(args.dataset)
gt = np.load(gt_path(args.dataset))
idx = Vamana.load(IDX / f"vamana_{args.tag}", X)
I, ne, nd = idx.search(Q[args.qid:args.qid + 1], k=10, L=args.L, threads=1)
exact = gt[args.qid, :10].tolist()
mine = I[0].tolist()
r = len(set(exact) & set(mine)) / 10
res = {"dataset": args.dataset, "tag": args.tag, "query_id": args.qid, "L_search": args.L,
       "exact_top10": exact, "vamana_top10": mine, "recall_10_at_10": r,
       "vertices_expanded": int(ne[0]), "distance_evaluations": int(nd[0])}
for k, v in res.items():
    print(f"{k:22s}: {v}")
save_json(res, RES / f"sanity_{args.dataset}_q{args.qid}.json")
