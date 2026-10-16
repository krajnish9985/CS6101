import argparse
import json
import os
import subprocess
import time


def _mincore(path):
    import ctypes
    import mmap
    size = os.path.getsize(path)
    if size == 0:
        return 0, 0, 0
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    ps = mmap.PAGESIZE
    npages = (size + ps - 1) // ps
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        vec = (ctypes.c_ubyte * npages)()
        libc.mincore(ctypes.c_void_p(addr), size, vec)
        libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)
    pages = sum(b & 1 for b in vec)
    return pages * ps, pages, size


def fincore(path):
    import shutil
    if shutil.which("fincore") is None:
        res, pages, size = _mincore(path)
        return {"file": path, "resident_bytes": res, "resident_pages": pages, "size_bytes": size,
                "resident_pct": 100.0 * min(res, size) / size if size else 0.0, "tool": "mincore(2) fallback"}
    out = subprocess.run(["fincore", "--bytes", "--noheadings", "--output", "RES,PAGES,SIZE", path],
                         capture_output=True, text=True, check=True).stdout.split()
    res, pages, size = int(out[0]), int(out[1]), int(out[2])
    return {"file": path, "resident_bytes": res, "resident_pages": pages, "size_bytes": size,
            "resident_pct": 100.0 * min(res, size) / size if size else 0.0, "tool": "fincore"}


def evict(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "evict"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rec = {"time": time.strftime("%H:%M:%S"), "files": []}
    for p in a.files:
        before = fincore(p)
        if a.cmd == "evict":
            evict(p)
            after = fincore(p)
            print(f"{p}: before {before['resident_pct']:.2f}% ({before['resident_bytes']:,} B)  ->  "
                  f"after fadvise(DONTNEED) {after['resident_pct']:.4f}% ({after['resident_bytes']:,} B)")
            rec["files"].append({"before": before, "after": after})
        else:
            print(f"{p}: resident {before['resident_pct']:.2f}% ({before['resident_bytes']:,} of {before['size_bytes']:,} B)")
            rec["files"].append({"status": before})
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rec, f, indent=2)
