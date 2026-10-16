from __future__ import annotations

import ctypes
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from vamana import _cpu_signature

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "diskann.cpp"


def _compile() -> Path:
    out = _HERE / "build" / f"libdiskann_{_cpu_signature()}.so"
    if out.exists() and out.stat().st_mtime >= _SRC.stat().st_mtime:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".so", dir=out.parent)
    os.close(fd)
    cmd = [os.environ.get("CXX", "g++"), "-O3", "-march=native", "-fopenmp-simd", "-fPIC", "-shared", "-std=c++17",
           str(_SRC), "-o", tmp]
    print("[diskann] compiling:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    os.replace(tmp, out)
    return out


_lib = ctypes.CDLL(str(_compile()))
_vp, _i64, _i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
_lib.dk_open.argtypes = [ctypes.c_char_p, _i64, _i32, _i32, _i32, _i32, ctypes.c_int32, _vp, _i32, _vp]
_lib.dk_open.restype = _vp
_lib.dk_close.argtypes = [_vp]
_lib.dk_search.argtypes = [_vp, _vp, _i32, _i32, _i32, _vp, _vp, _vp]
_lib.dk_search.restype = _i32

STAT_NAMES = ["reads", "bytes_read", "rounds", "expanded", "pq_dists", "full_dists", "sector_reuse"]


def proc_read_bytes() -> int:
    with open("/proc/self/io") as f:
        for line in f:
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    return -1


class DiskIndex:

    def __init__(self, d):
        d = Path(d)
        self.meta = json.loads((d / "disk_meta.json").read_text())
        m = self.meta
        self.codes = np.ascontiguousarray(np.load(d / "pq_codes.npy"))
        self.centroids = np.ascontiguousarray(np.load(d / "pq_centroids.npy"), dtype=np.float32)
        self.path = str(d / "disk.index")
        self.h = _lib.dk_open(self.path.encode(), m["N"], m["D"], m["R"], m["nodes_per_sector"], m["node_bytes"],
                              m["medoid"], self.codes.ctypes.data, m["pq_M"], self.centroids.ctypes.data)
        if not self.h:
            raise OSError(f"cannot open {self.path}")
        self.D = m["D"]

    def close(self):
        if self.h:
            _lib.dk_close(self.h)
            self.h = None

    def run(self, Q, k=10, L=100, W=4):
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        nq = Q.shape[0]
        ids = np.empty((nq, k), dtype=np.int32)
        dist = np.empty(k, dtype=np.float32)
        st = np.zeros((nq, 7), dtype=np.int64)
        lat = np.empty(nq)
        f, h, base, stride = _lib.dk_search, self.h, Q.ctypes.data, 4 * self.D
        idp, dp, stp = ids.ctypes.data, dist.ctypes.data, st.ctypes.data
        pc = time.perf_counter
        for i in range(nq):
            t0 = pc()
            rc = f(h, base + i * stride, k, L, W, idp + i * 4 * k, dp, stp + i * 56)
            lat[i] = pc() - t0
            if rc != 0:
                raise RuntimeError(f"dk_search failed ({rc}) on query {i}")
        return ids, st, lat
