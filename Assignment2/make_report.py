import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, PROC, RES, ROOT

FULL = ["sift1m", "quora", "nq"]
NICE = {"sift1m": "SIFT1M", "quora": "Quora", "nq": "NQ", "sift100k": "SIFT100K"}


def J(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def f(x, nd=4):
    if x is None:
        return "–"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:.{nd}f}"


def MiB(b):
    return None if b is None else b / 2**20


def table(head, rows):
    s = "| " + " | ".join(head) + " |\n|" + "|".join(["---"] * len(head)) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s + "\n"


S = {k: [] for k in ["DATASETS", "PART1", "PART2", "PART3A", "PART3B"]}
md = S["DATASETS"]
cpus = set()

rows = []
pca_rows = []
for ds in ["sift1m", "sift100k", "quora", "nq"]:
    m = J(PROC / f"{ds}_meta.json")
    if not m:
        continue
    rows.append([NICE[ds], f(m["corpus_size"]), f(m["num_queries"]), m["original_dim"], m["search_dim"],
                 f"{m['raw_float32_MiB']:.1f} MiB"])
    if "pca_impl" in m:
        pca_rows.append([NICE[ds], m["pca_impl"], f(m["pca_sample_size"]), m["pca_seed"],
                         f"{m['pca_explained_variance']:.4f}"])
md.append(table(["Dataset", "Corpus size", "#Queries", "Orig. dim", "Search dim", "Raw float32 (128-d corpus)"], rows))
if pca_rows:
    md.append("PCA (fit on corpus only, applied to corpus + queries, float32, then L2-normalised):\n\n")
    md.append(table(["Dataset", "PCA implementation", "Fit sample size", "Seed", "Variance explained (128 comps)"],
                    pca_rows))
e = J(RES / "exact_sift1m.json")
if e and "supplied_gt_agreement" in e:
    a = e["supplied_gt_agreement"]
    md.append(f"SIFT1M check: FAISS exact vs supplied ground truth — top-1 agreement {a['top1_agreement']:.4f}, "
              f"10-recall@10 {a['10recall@10_vs_supplied']:.4f}, 100-recall@100 {a['100recall@100_vs_supplied']:.4f} "
              "(differences, if any, come from exact distance ties).\n\n")

md = S["PART1"]
for ds in FULL:
    ex, hn = J(RES / f"exact_{ds}.json"), J(RES / f"hnsw_{ds}.json")
    if not (ex or hn):
        continue
    rows = []
    if ex:
        cpus.add(ex["system"]["cpu_model"])
        rows.append(["Exact (IndexFlatL2)", "–", f(ex["recall_1_at_1"]), f(ex["recall_10_at_10"]),
                     f(ex["mean_latency_ms"], 3), f(ex["qps"], 1), f(ex["build_time_s"], 2),
                     f(MiB(ex["index_size_bytes"]), 1)])
    if hn:
        cpus.add(hn["eval_system"]["cpu_model"])
        for p in hn["points"]:
            rows.append([f"HNSW M={hn['M']}, efC={hn['efConstruction']}", f"efSearch={p['efSearch']}",
                         f(p["recall_1_at_1"]), f(p["recall_10_at_10"]), f(p["mean_latency_ms"], 3),
                         f(p["qps"], 1), f(hn["build_time_s"], 1), f(MiB(hn["index_size_bytes"]), 1)])
        plt.figure(figsize=(5.2, 3.8))
        x = [p["mean_latency_ms"] for p in hn["points"]]
        y = [p["recall_10_at_10"] for p in hn["points"]]
        plt.plot(x, y, "o-", label="FAISS HNSW")
        for p in hn["points"]:
            plt.annotate(str(p["efSearch"]), (p["mean_latency_ms"], p["recall_10_at_10"]), fontsize=7,
                         textcoords="offset points", xytext=(3, -8))
        plt.xscale("log"); plt.xlabel("mean latency / query (ms, log)"); plt.ylabel("10-recall@10")
        plt.title(f"{NICE[ds]}: HNSW (labels = efSearch)"); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(PLOTS / f"part1_hnsw_{ds}.png", dpi=150); plt.close()
    md.append(f"### {NICE[ds]}\n\n")
    md.append(table(["Method", "Query param", "1-recall@1", "10-recall@10", "Mean latency (ms)", "QPS",
                     "Build time (s)", "Index size (MiB)"], rows))
    if hn:
        md.append(f"![HNSW {ds}](plots/part1_hnsw_{ds}.png)\n\n")
        md.append(f"HNSW build threads: {hn.get('build_threads')}; recall over all {hn['recall_queries']} queries; "
                  f"latency over {hn['timing_queries']} fixed random queries.\n\n")

md = S["PART2"]
rows = []
for ds in FULL:
    b = J(RES / f"vamana_build_{ds}_main.json")
    if not b:
        continue
    cpus.add(b["system"]["cpu_model"])
    rows.append([NICE[ds], b["R"], b["L_build"], b["alpha"], f(b["build_time_s"], 1), f(b["avg_out_degree"], 2),
                 b["max_out_degree"], f(b.get("graph_MiB"), 1), f(b.get("graph_plus_vectors_MiB"), 1),
                 f(b["peak_rss_MiB"], 0), b["threads"], b["medoid"]])
if rows:
    md.append(table(["Dataset", "R", "L_build", "α", "Build time (s)", "Avg out-deg", "Max out-deg",
                     "Graph file (MiB)", "Graph+vectors (MiB)", "Peak build RSS (MiB)", "Threads", "Entry point s"],
                    rows))
    md.append("Max out-degree ≤ R verified for every graph (assert in vamana_build.py; also 0 self-loops, "
              "0 duplicate edges — see results/vamana_build_*.json).\n\n")

sc = []
for p in glob.glob(str(RES / "vamana_build_scale_*.json")):
    b = J(p)
    sc.append((b["dataset"], b["n_points"], b["build_time_s"], b["peak_rss_MiB"], b["threads"]))
if sc:
    sc.sort()
    md.append("### Build-time scaling (development procedure)\n\n")
    md.append(table(["Dataset", "N", "Build time (s)", "Peak RSS (MiB)", "Threads"],
                    [[NICE.get(d, d), f(n), f(t, 1), f(r, 0), th] for d, n, t, r, th in sc]))
    for d in sorted(set(s[0] for s in sc)):
        pts = [(n, t) for dd, n, t, _, _ in sc if dd == d]
        if len(pts) >= 2:
            ln, lt = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
            bexp, a = np.polyfit(ln, lt, 1)
            full = {"sift1m": 1_000_000, "quora": 522_931, "nq": 2_681_468}.get(d)
            est = f"; extrapolated full-size ({full:,}) build ≈ {np.exp(a) * full ** bexp:.0f}s" if full else ""
            md.append(f"Fit for {NICE.get(d, d)}: time ∝ N^{bexp:.2f}{est}.\n\n")

s = J(RES / "sanity_sift100k_q0.json")
if s:
    md.append("### SIFT100K single-query sanity check\n\n")
    md.append(table(["Field", "Value"], [["query id", s["query_id"]], ["L_search", s["L_search"]],
                                          ["exact top-10 ids", s["exact_top10"]],
                                          ["Vamana top-10 ids", s["vamana_top10"]],
                                          ["10-recall@10", s["recall_10_at_10"]],
                                          ["vertices expanded", s["vertices_expanded"]],
                                          ["distance evaluations", s["distance_evaluations"]]]))

md = S["PART3A"]
dev = [J(p) for p in sorted(glob.glob(str(RES / "vamana_search_sift100k_*.json")))]
if dev:
    rows = []
    fig1, ax1 = plt.subplots(figsize=(5.2, 3.8))
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.8))
    for r in dev:
        lab = f"R={r['R']}, α={r['alpha']}, L_build={r['L_build']}"
        for p in r["points"]:
            rows.append([r["R"], r["alpha"], r["L_build"], p["L_search"], f(p["recall_1_at_1"]),
                         f(p["recall_10_at_10"]), f(p["mean_latency_ms"], 3), f(p["qps"], 0),
                         f(p["avg_dist_evals"], 1), f(p["avg_vertices_expanded"], 1)])
        Ls = [p["L_search"] for p in r["points"]]
        ys = [p["recall_10_at_10"] for p in r["points"]]
        ax1.plot(Ls, ys, "o-", label=lab)
        ax2.plot([p["mean_latency_ms"] for p in r["points"]], ys, "o-", label=lab)
    hn = J(RES / "hnsw_sift100k.json")
    if hn:
        ax2.plot([p["mean_latency_ms"] for p in hn["points"]], [p["recall_10_at_10"] for p in hn["points"]],
                 "s--", color="gray", label="FAISS HNSW")
    ax1.set_xscale("log"); ax1.set_xlabel("L_search (log)"); ax1.set_ylabel("10-recall@10")
    ax1.set_title("SIFT100K: recall vs L_search"); ax1.grid(alpha=.3); ax1.legend(fontsize=7)
    ax2.set_xscale("log"); ax2.set_xlabel("mean latency / query (ms, log)"); ax2.set_ylabel("10-recall@10")
    ax2.set_title("SIFT100K: recall vs latency"); ax2.grid(alpha=.3); ax2.legend(fontsize=7)
    fig1.tight_layout(); fig1.savefig(PLOTS / "part3a_recall_vs_L.png", dpi=150)
    fig2.tight_layout(); fig2.savefig(PLOTS / "part3a_recall_vs_latency.png", dpi=150)
    plt.close("all")
    md.append(table(["R", "α", "L_build", "L_search", "1-recall@1", "10-recall@10", "Mean latency (ms)", "QPS",
                     "Dist evals/query", "Vertices expanded/query"], rows))
    md.append("![recall vs L](plots/part3a_recall_vs_L.png)\n![recall vs latency](plots/part3a_recall_vs_latency.png)\n\n")
    bs = [J(p) for p in sorted(glob.glob(str(RES / "vamana_build_sift100k_*.json")))]
    md.append(table(["Graph", "Build time (s)", "Avg out-deg", "Max out-deg", "Graph (MiB)"],
                    [[b["tag"], f(b["build_time_s"], 1), f(b["avg_out_degree"], 2), b["max_out_degree"],
                      f(b.get("graph_MiB"), 1)] for b in bs if b]))


def fastest_at(points, target):
    ok = [p for p in points if p["recall_10_at_10"] >= target]
    return min(ok, key=lambda p: p["mean_latency_ms"]) if ok else None


md = S["PART3B"]
for ds in FULL:
    ex, hn, va = J(RES / f"exact_{ds}.json"), J(RES / f"hnsw_{ds}.json"), J(RES / f"vamana_search_{ds}_main.json")
    if not (hn or va):
        continue
    plt.figure(figsize=(5.4, 4))
    if hn:
        plt.plot([p["mean_latency_ms"] for p in hn["points"]], [p["recall_10_at_10"] for p in hn["points"]], "s-",
                 label=f"FAISS HNSW (M={hn['M']}, efC={hn['efConstruction']})")
    if va:
        cpus.add(va["eval_system"]["cpu_model"])
        plt.plot([p["mean_latency_ms"] for p in va["points"]], [p["recall_10_at_10"] for p in va["points"]], "o-",
                 label=f"Vamana (R={va['R']}, L_build={va['L_build']}, α={va['alpha']})")
    if ex:
        plt.plot([ex["mean_latency_ms"]], [ex["recall_10_at_10"]], "k*", ms=12, label="FAISS exact")
    plt.xscale("log"); plt.xlabel("mean latency / query (ms, log)"); plt.ylabel("10-recall@10")
    plt.title(f"{NICE[ds]}: Vamana vs HNSW"); plt.grid(alpha=.3); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(PLOTS / f"part3b_compare_{ds}.png", dpi=150); plt.close()

    rows = []
    if ex:
        rows.append(["Exact", "–", f(ex["recall_1_at_1"]), f(ex["recall_10_at_10"]), f(ex["mean_latency_ms"], 3),
                     f(ex["qps"], 1), f(ex["build_time_s"], 2), f(MiB(ex["index_size_bytes"]), 1), "–", "–"])
    seen = set()
    for tgt in (0.90, 0.95, 0.99):
        if hn and (p := fastest_at(hn["points"], tgt)) and ("h", p["efSearch"]) not in seen:
            seen.add(("h", p["efSearch"]))
            rows.append([f"HNSW (≥{tgt})", f"efSearch={p['efSearch']}", f(p["recall_1_at_1"]),
                         f(p["recall_10_at_10"]), f(p["mean_latency_ms"], 3), f(p["qps"], 0),
                         f(hn["build_time_s"], 1), f(MiB(hn["index_size_bytes"]), 1), "–", "–"])
        if va and (p := fastest_at(va["points"], tgt)) and ("v", p["L_search"]) not in seen:
            seen.add(("v", p["L_search"]))
            rows.append([f"Vamana (≥{tgt})", f"L_search={p['L_search']}", f(p["recall_1_at_1"]),
                         f(p["recall_10_at_10"]), f(p["mean_latency_ms"], 3), f(p["qps"], 0),
                         f(va["build_time_s"], 1),
                         f"{f(MiB(va['graph_plus_vectors_bytes']), 1)} ({f(MiB(va['graph_bytes']), 1)} graph)",
                         f(p["avg_dist_evals"], 0), f(p["avg_vertices_expanded"], 1)])
    md.append(f"### {NICE[ds]}\n\n![compare {ds}](plots/part3b_compare_{ds}.png)\n\n")
    md.append(table(["Method", "Param", "1-recall@1", "10-recall@10", "Mean latency (ms)", "QPS", "Build time (s)",
                     "Index size (MiB)", "Dist evals/q", "Expanded/q"], rows))
    if va:
        md.append("<details><summary>All Vamana operating points</summary>\n\n")
        md.append(table(["L_search", "1-recall@1", "10-recall@10", "Mean (ms)", "p95 (ms)", "QPS", "Dist evals/q",
                         "Expanded/q"],
                        [[p["L_search"], f(p["recall_1_at_1"]), f(p["recall_10_at_10"]), f(p["mean_latency_ms"], 3),
                          f(p["p95_latency_ms"], 3), f(p["qps"], 0), f(p["avg_dist_evals"], 0),
                          f(p["avg_vertices_expanded"], 1)] for p in va["points"]]))
        md.append("</details>\n\n")

tables = "".join("".join(v) for v in S.values())
(RES / "tables.txt").write_text(tables)

if len(cpus) > 1:
    print("WARNING: results come from different CPU models:", sorted(cpus))
print("wrote results/tables.txt and plots/*.png")
