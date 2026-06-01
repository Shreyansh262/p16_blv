#!/usr/bin/env python3
"""
analyze_gemma_coverage.py
BLV keyword coverage analysis on Gemma teacher captions.
Outputs:
  ~/p16_blv/results/gemma_coverage.json
  ~/p16_blv/results/gemma_coverage_chart.png
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HOME        = Path.home()
CAPS_FILE   = HOME / "p16_blv/data/generated/all_captions_gemma.json"
RESULTS_DIR = HOME / "p16_blv/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Metric keyword patterns ─────────────────────────────────────────────────
MCF_METRICS = {
    "blv_spatial": [
        r"\bleft\b", r"\bright\b", r"\bahead\b", r"\bbehind\b", r"\bcenter\b",
        r"\bmeters?\b", r"\bsteps?\b", r"\bnear\b", r"\bfar\b",
    ],
    "blv_social": [
        r"\bperson\b", r"\bpeople\b", r"\bman\b", r"\bwoman\b",
        r"\bchild\b", r"\bindividual\b", r"\bsomeone\b",
    ],
    "blv_action": [
        r"\bwalking\b", r"\bstanding\b", r"\bmoving\b", r"\bsitting\b",
        r"\bholding\b", r"\bcarrying\b", r"\bpicking\b", r"\brunning\b",
    ],
    "blv_ambience": [
        r"\bindoor\b", r"\boutdoor\b", r"\broom\b", r"\bhallway\b",
        r"\bkitchen\b", r"\bstreet\b", r"\bcorridor\b", r"\blighting\b",
    ],
}

NAF_METRICS = {
    "nav_obstacles": [
        r"\bobstacle\b", r"\bbox\b", r"\bchair\b", r"\btable\b",
        r"\bdoor\b", r"\bwall\b", r"\bobject\b", r"\bblocking\b", r"\bbarrier\b",
    ],
    "nav_step_changes": [
        r"\bstair", r"\bstep\b", r"\bramp\b", r"\belevation\b",
        r"\bdown\b", r"\bup ahead\b", r"\bcurb\b",
    ],
    "nav_directions": [
        r"\bleft\b", r"\bright\b", r"\bahead\b", r"\bstraight\b",
        r"\bbehind\b", r"\bturn\b",
    ],
    "nav_moving_hazard": [
        r"\bmoving\b", r"\bapproaching\b", r"\bwalking toward\b",
        r"\bvehicle\b", r"\bperson moving\b", r"\bcoming\b",
    ],
    "nav_distances": [
        r"\bmeters?\b", r"\bfoot\b", r"\bfeet\b", r"\bsteps away\b",
        r"\bapproximately\b", r"\bclose\b", r"\bnearby\b",
        r"\bnearby\s+\d", r"\d+\s*(?:meters?|feet|foot)",
    ],
}

ALL_METRICS = {**MCF_METRICS, **NAF_METRICS}


def compile_patterns(metrics):
    return {
        name: [re.compile(p, re.IGNORECASE) for p in pats]
        for name, pats in metrics.items()
    }


def check_metric(text, patterns):
    return any(p.search(text) for p in patterns)


def print_table(rows, headers):
    col_widths = [max(len(str(r[i])) for r in [headers] + rows)
                  for i in range(len(headers))]
    sep = "  ".join("-" * w for w in col_widths)
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))


def make_donut(ax, group_name, metric_names, coverage, colors, n_total, bg):
    sizes   = [max(coverage[m], 0.5) for m in metric_names]
    labels  = [f"{m}\n{coverage[m]:.1f}%" for m in metric_names]
    explode = [0.04] * len(sizes)

    ax.pie(
        sizes,
        labels=None,
        colors=colors,
        startangle=90,
        explode=explode,
        wedgeprops=dict(width=0.55, edgecolor=bg, linewidth=2),
        normalize=True,
    )

    ax.text(0, 0, f"{group_name}\nn={n_total}",
            ha='center', va='center', fontsize=11,
            fontweight='bold', color='white')

    patches = [mpatches.Patch(color=colors[i], label=labels[i])
               for i in range(len(metric_names))]
    ax.legend(handles=patches, loc='lower center',
              bbox_to_anchor=(0.5, -0.24), ncol=2,
              fontsize=9, framealpha=0.15,
              labelcolor='white', facecolor=bg)
    ax.set_title(f"{group_name} Coverage",
                 color='white', fontsize=13, fontweight='bold', pad=14)
    ax.set_facecolor(bg)


def main():
    print(f"Loading {CAPS_FILE} ...")
    with open(CAPS_FILE) as f:
        data = json.load(f)

    data = [d for d in data if len(d.get("blv_description", "").strip()) > 10]
    n = len(data)
    print(f"  {n} valid captions\n")

    # Preview first 2 entries
    print("── First 2 entries " + "─" * 52)
    for i, entry in enumerate(data[:2]):
        desc = entry.get("blv_description", "")
        print(f"\n[{i}] video_id : {entry.get('video_id', '?')}")
        print(f"    dataset  : {entry.get('dataset', '?')}")
        print(f"    blv_desc : {desc[:250]}{'...' if len(desc) > 250 else ''}")
    print("\n" + "─" * 70 + "\n")

    mcf_compiled = compile_patterns(MCF_METRICS)
    naf_compiled = compile_patterns(NAF_METRICS)
    all_compiled = {**mcf_compiled, **naf_compiled}

    counts      = {name: 0 for name in all_compiled}
    per_caption = []

    for entry in data:
        text = entry.get("blv_description", "")
        row  = {"video_id": entry.get("video_id", ""),
                "dataset":  entry.get("dataset", "")}
        for name, patterns in all_compiled.items():
            passed   = check_metric(text, patterns)
            row[name] = int(passed)
            if passed:
                counts[name] += 1
        per_caption.append(row)

    coverage = {name: round(cnt / n * 100, 1) for name, cnt in counts.items()}

    results = {
        "total_captions": n,
        "coverage_pct":   coverage,
        "per_caption":    per_caption,
    }
    out_json = RESULTS_DIR / "gemma_coverage.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out_json}\n")

    # Summary tables
    print("── MCF Metrics " + "─" * 55)
    mcf_rows = [[m, f"{coverage[m]:.1f}%", f"{counts[m]}/{n}"]
                for m in MCF_METRICS]
    print_table(mcf_rows, ["Metric", "Coverage", "Count"])

    print("\n── NAF Metrics " + "─" * 55)
    naf_rows = [[m, f"{coverage[m]:.1f}%", f"{counts[m]}/{n}"]
                for m in NAF_METRICS]
    print_table(naf_rows, ["Metric", "Coverage", "Count"])

    # ── Charts ─────────────────────────────────────────────────────────────
    BG         = '#1a1a2e'
    COLORS_MCF = ['#4cc9f0', '#4361ee', '#7209b7', '#f72585']
    COLORS_NAF = ['#06d6a0', '#118ab2', '#073b4c', '#ffd166', '#ef476f']

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(BG)

    make_donut(axes[0], "MCF", list(MCF_METRICS), coverage, COLORS_MCF, n, BG)
    make_donut(axes[1], "NAF", list(NAF_METRICS), coverage, COLORS_NAF, n, BG)

    fig.suptitle("Gemma Teacher Caption — BLV Keyword Coverage",
                 color='white', fontsize=15, fontweight='bold', y=1.02)

    out_chart = RESULTS_DIR / "gemma_coverage_chart.png"
    plt.tight_layout()
    plt.savefig(out_chart, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\nChart saved → {out_chart}")


if __name__ == "__main__":
    main()
