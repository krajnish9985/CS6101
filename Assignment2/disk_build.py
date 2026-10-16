import argparse
import json
import os
import time

import numpy as np

from common import IDX, RES, base_path, mib, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--tag", required=True, help="Vamana graph tag (indices/vamana_<tag>)")
ap.add_argument("--pq-m", type=int, default=32, help="PQ sub-quantizers (bytes per vector)")
ap.add_argument("--pq-train", type=int, default=262_144)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
args = ap.parse_args()

gdir = IDX / f"vamana_{args.tag}"
out = IDX / (args.out or f"disk_{args.dataset}")
out.mkdir(parents=True, exist_ok=True)
gmeta = json.loads((gdir / "meta.json").read_text())
nbrs = np.load(gdir / "nbrs.npy", mmap_mode="r")
deg = np.load(gdir / "deg.npy")
X = np.load(base_path(args.dataset), mmap_mode="r")
N, D = X.shape
R = nbrs.shape[1]
M = args.pq_m
assert D % M == 0
t0 = time.time()

rng = np.random.default_rng(args.seed)
tr = np.ascontiguousarray(X[np.sort(rng.choice(N, size=min(args.pq_train, N), replace=False))], dtype=np.float32)
try:
    import faiss
    pq = faiss.ProductQuantizer(D, M, 8)
    pq.train(tr)
    cent = faiss.vector_to_array(pq.centroids).reshape(M, 256, D // M).astype(np.float32)
    encode = lambda x: pq.compute_codes(np.ascontiguousarray(x, dtype=np.float32))
    pq_impl = f"FAISS ProductQuantizer(d={D}, M={M}, nbits=8)"
except ImportError:
    from sklearn.cluster import KMeans
    ds = D // M
    cent = np.stack([KMeans(256, n_init=1, max_iter=20, random_state=args.seed).fit(tr[:, m * ds:(m + 1) * ds])
                     .cluster_centers_ for m in range(M)]).astype(np.float32)

    def encode(x):
        x = np.asarray(x, dtype=np.float32)
        c = np.empty((x.shape[0], M), dtype=np.uint8)
        for m in range(M):
            s = x[:, m * ds:(m + 1) * ds]
            d2 = (s ** 2).sum(1)[:, None] - 2 * s @ cent[m].T + (cent[m] ** 2).sum(1)[None]
            c[:, m] = d2.argmin(1)
        return c
    pq_impl = f"sklearn KMeans PQ (M={M}, 256 centroids)"
codes = np.empty((N, M), dtype=np.uint8)
for s in range(0, N, 500_000):
    codes[s:s + 500_000] = encode(X[s:s + 500_000])
np.save(out / "pq_codes.npy", codes)
np.save(out / "pq_centroids.npy", cent)
t_pq = time.time() - t0
print(f"PQ trained+encoded in {t_pq:.0f}s ({pq_impl})", flush=True)

SECTOR = 4096
node_bytes = 4 * D + 4 + 4 * R
nps = SECTOR // node_bytes
assert nps >= 1, "node record larger than a sector"
n_sect = (N + nps - 1) // nps
t1 = time.time()
with open(out / "disk.index", "wb") as f:
    hdr = np.zeros(SECTOR, dtype=np.uint8)
    h = np.array([0x4B534944, N, D, R, gmeta["medoid"], node_bytes, nps, SECTOR], dtype=np.int64)
    hdr[:h.nbytes] = h.view(np.uint8)
    f.write(hdr.tobytes())
    step = nps * 20_000
    for a in range(0, N, step):
        b = min(N, a + step)
        n = b - a
        rec = np.zeros((((n + nps - 1) // nps) * nps, node_bytes), dtype=np.uint8)
        rec[:n, :4 * D] = np.ascontiguousarray(X[a:b], dtype=np.float32).view(np.uint8).reshape(n, 4 * D)
        rec[:n, 4 * D:4 * D + 4] = deg[a:b].astype(np.uint32).view(np.uint8).reshape(n, 4)
        rec[:n, 4 * D + 4:] = np.ascontiguousarray(nbrs[a:b], dtype=np.int32).view(np.uint8).reshape(n, 4 * R)
        sect = np.zeros((rec.shape[0] // nps, SECTOR), dtype=np.uint8)
        sect[:, :nps * node_bytes] = rec.reshape(-1, nps * node_bytes)
        f.write(sect.tobytes())
    f.flush()
    os.fsync(f.fileno())
size = (out / "disk.index").stat().st_size
assert size == (1 + n_sect) * SECTOR, (size, n_sect)
t_layout = time.time() - t1
print(f"disk.index written: {mib(size):.1f} MiB, {nps} nodes/sector, node={node_bytes} B ({t_layout:.0f}s)")

meta = {"dataset": args.dataset, "graph_tag": args.tag, "N": int(N), "D": int(D), "R": int(R), "L_build": gmeta["L_build"],
        "alpha": gmeta["alpha"], "graph_build_time_s": gmeta["build_time_s"], "medoid": int(gmeta["medoid"]),
        "sector_bytes": SECTOR, "node_bytes": node_bytes, "nodes_per_sector": nps, "n_sectors": int(n_sect + 1),
        "disk_index_bytes": size, "pq_impl": pq_impl, "pq_M": M, "pq_bits": 8, "pq_train_size": int(tr.shape[0]),
        "pq_codes_bytes": int(codes.nbytes), "pq_centroids_bytes": int(cent.nbytes),
        "visited_tags_bytes": int(N) * 4, "pq_time_s": t_pq, "layout_time_s": t_layout}
(out / "disk_meta.json").write_text(json.dumps(meta, indent=2))
save_json(meta, RES / f"ec_disk_build_{args.dataset}.json")
print(json.dumps(meta, indent=2))
