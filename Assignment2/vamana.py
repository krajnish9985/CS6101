from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "vamana.cpp"


def platform_machine() -> str:
    import platform
    return platform.machine() + "_" + platform.processor()


def _cpu_signature() -> str:
    model = "unknown"
    try:
        if sys.platform == "darwin":
            return "mac_" + platform_machine()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") or line.startswith("flags"):
                    model += line
                    if line.startswith("flags"):
                        break
    except OSError:
        pass
    return hashlib.md5(model.encode()).hexdigest()[:10]


def _lib_path() -> Path:
    return _HERE / "build" / f"libvamana_{_cpu_signature()}.so"


def compile_lib(force: bool = False) -> Path:
    out = _lib_path()
    if out.exists() and not force and out.stat().st_mtime >= _SRC.stat().st_mtime:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".so", dir=out.parent)
    os.close(fd)
    if sys.platform == "darwin":
        try:
            omp = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True).stdout.strip()
        except OSError:
            omp = ""
        omp = omp or "/opt/homebrew/opt/libomp"
        cmd = [os.environ.get("CXX", "clang++"), "-O3", "-Xpreprocessor", "-fopenmp", f"-I{omp}/include",
               f"-L{omp}/lib", "-lomp", f"-Wl,-rpath,{omp}/lib", "-fPIC", "-shared", "-std=c++17", str(_SRC),
               "-o", tmp]
    else:
        cmd = [os.environ.get("CXX", "g++"), "-O3", "-march=native", "-fopenmp", "-fPIC", "-shared", "-std=c++17",
               str(_SRC), "-o", tmp]
    print("[vamana] compiling:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    os.replace(tmp, out)
    return out


_lib = ctypes.CDLL(str(compile_lib()))
_vp, _i64, _i32, _f32, _f64, _u64 = (ctypes.c_void_p, ctypes.c_int64, ctypes.c_int,
                                     ctypes.c_float, ctypes.c_double, ctypes.c_uint64)
_lib.vamana_build.argtypes = [_vp, _i64, _i32, _i32, _i32, _f32, _u64, _i32, _f64, _vp, _vp, _vp, _i32]
_lib.vamana_build.restype = _i32
_lib.vamana_search.argtypes = [_vp, _i64, _i32, _vp, _vp, _i32, ctypes.c_int32, _vp, _i64, _i32, _i32, _i32,
                               _vp, _vp, _vp, _vp]
_lib.vamana_search.restype = None
_lib.vamana_medoid.argtypes = [_vp, _i64, _i32, _i32]
_lib.vamana_medoid.restype = ctypes.c_int32


def _f32c(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def default_threads() -> int:
    for k in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        if os.environ.get(k):
            return int(os.environ[k])
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


class Vamana:
    def __init__(self, X: np.ndarray, nbrs: np.ndarray, deg: np.ndarray, medoid: int, meta: dict):
        self.X = _f32c(X)
        self.nbrs = np.ascontiguousarray(nbrs, dtype=np.int32)
        self.deg = np.ascontiguousarray(deg, dtype=np.int32)
        self.medoid = int(medoid)
        self.meta = meta
        self.N, self.D = self.X.shape
        self.R = self.nbrs.shape[1]

    @classmethod
    def build(cls, X, R=64, L=100, alpha=1.2, seed=0, threads=None, max_batch_frac=0.02, verbose=True):
        X = _f32c(X)
        N, D = X.shape
        threads = threads or default_threads()
        nbrs = np.empty((N, R), dtype=np.int32)
        deg = np.empty(N, dtype=np.int32)
        stats = np.zeros(8, dtype=np.float64)
        t0 = time.perf_counter()
        rc = _lib.vamana_build(X.ctypes.data, N, D, R, L, float(alpha), int(seed), int(threads),
                               float(max_batch_frac), nbrs.ctypes.data, deg.ctypes.data, stats.ctypes.data,
                               int(verbose))
        wall = time.perf_counter() - t0
        if rc != 0:
            raise RuntimeError(f"vamana_build failed with code {rc}")
        meta = dict(R=R, L_build=L, alpha=float(alpha), seed=int(seed), threads=int(threads),
                    max_batch_frac=float(max_batch_frac), N=int(N), D=int(D), build_time_s=wall,
                    t_init_s=stats[0], t_pass1_s=stats[1], t_pass2_s=stats[2],
                    batches_pass1=int(stats[3]), batches_pass2=int(stats[4]),
                    build_avg_expanded_per_insert=stats[5], build_avg_dists_per_insert=stats[6],
                    medoid=int(stats[7]),
                    entry_point="database point nearest to the dataset mean (exact scan)")
        return cls(X, nbrs, deg, int(stats[7]), meta)

    def save(self, d) -> int:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "nbrs.npy", self.nbrs)
        np.save(d / "deg.npy", self.deg)
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2))
        return self.graph_bytes(d)

    @staticmethod
    def graph_bytes(d) -> int:
        d = Path(d)
        return sum((d / f).stat().st_size for f in ("nbrs.npy", "deg.npy", "meta.json"))

    @classmethod
    def load(cls, d, X):
        d = Path(d)
        meta = json.loads((d / "meta.json").read_text())
        return cls(X, np.load(d / "nbrs.npy"), np.load(d / "deg.npy"), meta["medoid"], meta)

    def search(self, Q, k=10, L=100, threads=None):
        Q = _f32c(Q)
        nq = Q.shape[0]
        ids = np.empty((nq, k), dtype=np.int32)
        dist = np.empty((nq, k), dtype=np.float32)
        ne = np.empty(nq, dtype=np.int32)
        nd = np.empty(nq, dtype=np.int32)
        _lib.vamana_search(self.X.ctypes.data, self.N, self.D, self.nbrs.ctypes.data, self.deg.ctypes.data,
                           self.R, self.medoid, Q.ctypes.data, nq, k, max(L, k), int(threads or default_threads()),
                           ids.ctypes.data, dist.ctypes.data, ne.ctypes.data, nd.ctypes.data)
        return ids, ne, nd

    def time_queries(self, Q, qidx, k=10, L=100, warmup=50):
        Q = _f32c(Q)
        ids = np.empty(k, dtype=np.int32)
        dist = np.empty(k, dtype=np.float32)
        ne = np.empty(1, dtype=np.int32)
        nd = np.empty(1, dtype=np.int32)
        f = _lib.vamana_search
        args = (self.X.ctypes.data, self.N, self.D, self.nbrs.ctypes.data, self.deg.ctypes.data, self.R,
                self.medoid)
        outs = (ids.ctypes.data, dist.ctypes.data, ne.ctypes.data, nd.ctypes.data)
        base, stride, Lq = Q.ctypes.data, 4 * self.D, max(L, k)
        for i in qidx[:warmup]:
            f(*args, base + int(i) * stride, 1, k, Lq, 1, *outs)
        lat = np.empty(len(qidx), dtype=np.float64)
        pc = time.perf_counter
        for t, i in enumerate(qidx):
            ptr = base + int(i) * stride
            t0 = pc()
            f(*args, ptr, 1, k, Lq, 1, *outs)
            lat[t] = pc() - t0
        return lat

    def degree_stats(self, chunk=1_000_000) -> dict:
        deg, N, R = self.deg, self.N, self.R
        big = np.iinfo(np.int32).max
        self_loops = dups = oor = 0
        indeg = np.zeros(N, dtype=np.int64)
        for s0 in range(0, N, chunk):
            rows = self.nbrs[s0:s0 + chunk]
            valid = np.arange(R)[None, :] < deg[s0:s0 + chunk, None]
            ids = np.arange(s0, s0 + rows.shape[0], dtype=np.int32)[:, None]
            self_loops += int(((rows == ids) & valid).sum())
            srt = np.sort(np.where(valid, rows, big), axis=1)
            dups += int(((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] != big)).sum())
            v = rows[valid]
            oor += int(((v < 0) | (v >= N)).sum())
            v = v[(v >= 0) & (v < N)]
            indeg += np.bincount(v, minlength=N)
        return dict(avg_out_degree=float(deg.mean()), max_out_degree=int(deg.max()), min_out_degree=int(deg.min()),
                    zero_degree_vertices=int((deg == 0).sum()), self_loops=self_loops, duplicate_edges=dups,
                    out_of_range_edges=oor, max_degree_le_R=bool(deg.max() <= R),
                    in_degree_zero_vertices=int((indeg == 0).sum()))
