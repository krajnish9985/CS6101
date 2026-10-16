from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PA2_ROOT", Path(__file__).resolve().parent))
DATA = Path(os.environ.get("PA2_DATA", ROOT / "data"))
RAW = DATA / "raw"
PROC = DATA / "processed"
IDX = ROOT / "indices"
RES = ROOT / "results"
PLOTS = ROOT / "plots"
for _d in (PROC, IDX, RES, PLOTS):
    _d.mkdir(parents=True, exist_ok=True)

DATASETS = ["sift100k", "sift1m", "quora", "nq"]
K_GT = 100


def base_path(ds): return PROC / f"{ds}_base.npy"
def query_path(ds): return PROC / f"{ds}_query.npy"
def gt_path(ds): return PROC / f"{ds}_gt.npy"
def meta_path(ds): return PROC / f"{ds}_meta.json"


def load_base(ds, n=None):
    X = np.load(base_path(ds), mmap_mode="r")
    if n is not None:
        X = X[:n]
    return np.ascontiguousarray(X, dtype=np.float32)


def load_queries(ds):
    return np.ascontiguousarray(np.load(query_path(ds)), dtype=np.float32)


def load_reference_gt(ds):
    if ds == "sift1m" and (PROC / "sift1m_gt_supplied.npy").exists():
        return np.load(PROC / "sift1m_gt_supplied.npy")
    return np.load(gt_path(ds))


def read_vecs(path, dtype):
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    a = a.reshape(-1, d + 1)[:, 1:]
    return np.ascontiguousarray(a.view(dtype))


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_np_default))


def load_json(path):
    return json.loads(Path(path).read_text())


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def k_recall_at_k(I, gt, k):
    A = np.asarray(I)[:, :k]
    G = np.asarray(gt)[:, :k]
    hits = (A[:, :, None] == G[:, None, :]).any(axis=2).sum(axis=1)
    return float(hits.mean() / k)


def recalls(I, gt):
    return {"recall_1_at_1": k_recall_at_k(I, gt, 1), "recall_10_at_10": k_recall_at_k(I, gt, 10)}


def latency_summary(lat_s):
    lat = np.asarray(lat_s)
    m = float(lat.mean())
    return {"mean_latency_ms": m * 1e3, "p50_latency_ms": float(np.percentile(lat, 50)) * 1e3,
            "p95_latency_ms": float(np.percentile(lat, 95)) * 1e3, "qps": 1.0 / m}


def median_run(fn, repeats):
    runs = [fn() for _ in range(max(1, repeats))]
    runs.sort(key=lambda a: float(np.mean(a)))
    return runs[len(runs) // 2], [float(np.mean(r)) * 1e3 for r in runs]


def timing_subset(nq, n=1000, seed=0):
    if nq <= n:
        return np.arange(nq)
    return np.sort(np.random.default_rng(seed).choice(nq, size=n, replace=False))


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024.0


def n_threads_default():
    for k in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        if os.environ.get(k):
            return int(os.environ[k])
    return _ncpu()


def _ncpu():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def system_info():
    model = "unknown"
    if sys.platform == "darwin":
        try:
            model = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                   text=True).stdout.strip()
        except Exception:
            pass
    try:
        out = "" if sys.platform == "darwin" else subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("Model name"):
                model = line.split(":", 1)[1].strip()
    except Exception:
        pass
    return {"hostname": platform.node(), "cpu_model": model,
            "cpus_available": _ncpu(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def file_size(path):
    return os.path.getsize(path)


def mib(nbytes):
    return nbytes / 2**20
