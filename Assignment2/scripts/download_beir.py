import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
dest = ROOT / "data" / "raw" / "beir"
names = sys.argv[1:] or ["quora", "nq"]
patterns = []
for n in names:
    patterns += [f"{n}/corpus/*.parquet", f"{n}/queries/test.parquet"]
print("downloading", patterns, "->", dest)
snapshot_download(repo_id="CohereLabs/beir-embed-english-v3", repo_type="dataset",
                  allow_patterns=patterns, local_dir=str(dest), max_workers=8)
for p in sorted(dest.rglob("*.parquet")):
    print(f"{p.stat().st_size/2**20:10.1f} MiB  {p.relative_to(dest)}")
