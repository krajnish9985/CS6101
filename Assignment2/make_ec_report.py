import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import PLOTS, PROC, RES

J = lambda p: json.loads(Path(p).read_text())
runs = sorted([J(p) for p in glob.glob(str(RES / "ec_cold_L*.json")) + glob.glob(str(RES / "ec_warm_L*.json"))],
              key=lambda r: (r["mode"], r["beam_width_W"], r["L_search"]))
bm = J(RES / "ec_disk_build_msmarco.json") if (RES / "ec_disk_build_msmarco.json").exists() else None
gb = J(RES / "vamana_build_msmarco_main.json") if (RES / "vamana_build_msmarco_main.json").exists() else None
pm = J(PROC / "msmarco_meta.json") if (PROC / "msmarco_meta.json").exists() else None
MiB = lambda b: b / 2**20
md = []
if pm:
    md.append(f"MS MARCO: {pm['corpus_size']:,} passages, {pm['num_queries']:,} {pm.get('query_split','dev')} queries; "
              f"PCA 1024->128 ({pm['pca_impl']}, sample {pm['pca_sample_size']:,}, seed {pm['pca_seed']}, "
              f"explained variance {pm['pca_explained_variance']:.4f}); raw 128-d float32 = {pm['raw_float32_MiB']:.0f} MiB.\n\n")
if bm:
    md.append("| Item | Value |\n|---|---|\n")
    rows = [("Graph builder", f"our Vamana (Part 2), R={bm['R']}, L_build={bm['L_build']}, alpha={bm['alpha']}, "
                              f"build {bm['graph_build_time_s']:.0f}s" + (f", peak build RSS {gb['peak_rss_MiB']:.0f} MiB" if gb else "")),
            ("Disk record", f"[float32 vec {bm['D']}][uint32 deg][int32 nbrs {bm['R']}] = {bm['node_bytes']} B; "
                            f"{bm['nodes_per_sector']} records per 4096-B sector (never straddling sectors)"),
            ("Lookup", "node i -> sector 1 + i / nps, byte offset (i mod nps) * node_bytes (direct, no index)"),
            ("disk.index size", f"{MiB(bm['disk_index_bytes']):,.1f} MiB ({bm['n_sectors']:,} sectors)"),
            ("Compressed vectors (RAM)", f"{bm['pq_impl']}, {bm['pq_M']} B/vector -> {MiB(bm['pq_codes_bytes']):.1f} MiB codes "
                                         f"+ {bm['pq_centroids_bytes']/1024:.0f} KiB centroids"),
            ("Other resident RAM", f"visited tags uint32[N] = {MiB(bm['visited_tags_bytes']):.1f} MiB; candidate list; W sector buffers"),
            ("PQ / layout time", f"{bm['pq_time_s']:.0f}s / {bm['layout_time_s']:.0f}s")]
    md += [f"| {a} | {b} |\n" for a, b in rows]
    md.append("\n")
if runs:
    md.append("| Cache | L | W | 1-recall@1 | 10-recall@10 | mean ms | p50 ms | p95 ms | QPS | reads/q | KiB read/q "
              "| storage KiB/q | rounds/q | expansions/q | peak RSS MiB | index resident before run |\n|" + "---|" * 16 + "\n")
    for r in runs:
        fb = r.get("fincore_before_run") or {}
        md.append(f"| {r['mode']} | {r['L_search']} | {r['beam_width_W']} | {r['recall_1_at_1']:.4f} | {r['recall_10_at_10']:.4f} | "
                  f"{r['mean_latency_ms']:.3f} | {r['p50_latency_ms']:.3f} | {r['p95_latency_ms']:.3f} | {r['qps']:.0f} | "
                  f"{r['reads_per_query']:.1f} | {r['bytes_read_per_query']/1024:.1f} | "
                  f"{(r['storage_bytes_per_query'] or 0)/1024:.1f} | {r['rounds_per_query']:.1f} | {r['expansions_per_query']:.1f} | "
                  f"{r['peak_rss_MiB']:.0f} | {fb.get('resident_pct', float('nan')):.2f}% ({fb.get('resident_bytes', 0):,} B) |\n")
    r0 = runs[0]
    md.append(f"\nRAM limit: address space limited to {r0['rlimit_as_bytes'] / 2**30 if r0['rlimit_as_bytes'] else 'unlimited'} GiB "
              f"(ulimit -v) for every search process; permanently resident structures = {MiB(r0['resident_structures_bytes']):.1f} MiB. "
              f"CPU {r0['system']['cpu_model']}, 1 thread, queries issued one at a time, all {r0['num_queries']:,} queries.\n")
    plt.figure(figsize=(5.4, 4))
    for mode, mk in (("cold", "o-"), ("warm", "s-")):
        rr = [r for r in runs if r["mode"] == mode]
        if rr:
            plt.plot([r["mean_latency_ms"] for r in rr], [r["recall_10_at_10"] for r in rr], mk, label=f"{mode} cache")
            for r in rr:
                plt.annotate(f"L={r['L_search']}", (r["mean_latency_ms"], r["recall_10_at_10"]), fontsize=7,
                             textcoords="offset points", xytext=(3, -9))
    plt.xscale("log"); plt.xlabel("mean latency / query (ms, log)"); plt.ylabel("10-recall@10")
    plt.title(f"MS MARCO disk-resident search (W={r0['beam_width_W']})"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS / "ec_disk_recall_latency.png", dpi=150)
dev = Path(RES / "ec_device.txt")
if dev.exists():
    md.append("\nStorage device:\n```\n" + dev.read_text() + "```\n")
for p in sorted(glob.glob(str(RES / "ec_fincore_cold_L*.json"))):
    f = J(p)["files"][0]
    md.append(f"- {Path(p).stem}: before eviction {f['before']['resident_pct']:.2f}% -> after fadvise(DONTNEED) "
              f"{f['after']['resident_pct']:.4f}% ({f['after']['resident_bytes']:,} B resident) [{f['after'].get('tool','fincore')}]\n")
(RES / "ec_tables.txt").write_text("".join(md))
print("".join(md))
