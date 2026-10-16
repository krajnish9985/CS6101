import time

import numpy as np

from vamana import Vamana, default_threads

rng = np.random.default_rng(0)
C, W = rng.normal(size=(200, 32)) * 3, rng.normal(size=(32, 128))
make = lambda m: (C[rng.integers(200, size=m)] + rng.normal(size=(m, 32))) @ W
X = (make(20000) + rng.normal(size=(20000, 128))).astype(np.float32)
Q = (make(300) + rng.normal(size=(300, 128))).astype(np.float32)
gt = np.argsort((Q**2).sum(1)[:, None] - 2 * Q @ X.T + (X**2).sum(1)[None], axis=1)[:, :10]
t = time.time()
idx = Vamana.build(X, R=32, L=64, alpha=1.2, threads=default_threads(), verbose=False)
print(f"build {time.time()-t:.1f}s", idx.degree_stats())
for L in (10, 20, 50, 100):
    I, ne, nd = idx.search(Q, 10, L)
    r = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(I, gt)])
    print(f"L={L:3d} 10-recall@10={r:.3f} expanded={ne.mean():.0f} dists={nd.mean():.0f}")
assert r > 0.95, "recall too low -- something is wrong"
print("SMOKE TEST PASSED")
