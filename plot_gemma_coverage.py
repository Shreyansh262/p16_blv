#!/usr/bin/env python3
"""
plot_gemma_coverage.py  —  Presentation-ready BLV coverage chart.
Reads  ~/p16_blv/results/gemma_coverage.json
Writes ~/p16_blv/results/gemma_coverage_chart.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

HOME     = Path.home()
IN_FILE  = HOME / "p16_blv/results/gemma_coverage.json"
OUT_FILE = HOME / "p16_blv/results/gemma_coverage_chart.png"

# ── palette ────────────────────────────────────────────────────────────────
BG        = '#0d1117'
PANEL_BG  = '#161b22'
BAR_TRACK = '#21262d'        # uncovered portion
TEXT      = '#e6edf3'
MUTED     = '#8b949e'
C_HIGH    = '#3fb950'        # ≥ 90 %   green
C_MID     = '#d29922'        # 50–90 %  amber
C_LOW     = '#f85149'        # < 50 %   red-orange

MCF_ORDER = ['blv_spatial', 'blv_social', 'blv_ambience', 'blv_action']
NAF_ORDER = ['nav_directions', 'nav_distances', 'nav_obstacles',
             'nav_moving_hazard', 'nav_step_changes']

LABELS = {
    'blv_spatial':       'Spatial',
    'blv_social':        'Social',
    'blv_action':        'Action',
    'blv_ambience':      'Ambience',
    'nav_obstacles':     'Obstacles',
    'nav_step_changes':  'Step Changes',
    'nav_directions':    'Directions',
    'nav_moving_hazard': 'Moving Hazard',
    'nav_distances':     'Distances',
}

KEYWORDS = {
    'blv_spatial':       'left · right · ahead · meter · near',
    'blv_social':        'person · people · man · woman · child',
    'blv_action':        'walking · standing · moving · sitting',
    'blv_ambience':      'indoor · outdoor · room · hallway · street',
    'nav_obstacles':     'box · chair · table · door · wall · barrier',
    'nav_step_changes':  'stair · step · ramp · elevation · curb',
    'nav_directions':    'left · right · ahead · straight · turn',
    'nav_moving_hazard': 'moving · approaching · vehicle · coming',
    'nav_distances':     'meter · feet · steps away · approximately',
}


def bar_color(pct):
    if pct >= 90:
        return C_HIGH
    elif pct >= 50:
        return C_MID
    return C_LOW


def draw_panel(ax, metrics, coverage, group_title):
    ax.set_facecolor(PANEL_BG)

    n   = len(metrics)
    BAR = 0.52

    for i, m in enumerate(metrics):
        pct   = coverage[m]
        color = bar_color(pct)
        y     = n - 1 - i          # top-to-bottom order

        # track (uncovered)
        ax.barh(y, 100, height=BAR, color=BAR_TRACK,
                left=0, zorder=2, linewidth=0)
        # filled (covered)
        ax.barh(y, pct, height=BAR, color=color,
                left=0, zorder=3, linewidth=0, alpha=0.92)

        # metric label  (left of chart)
        ax.text(-2, y, LABELS[m], va='center', ha='right',
                fontsize=11.5, fontweight='600', color=TEXT)

        # keyword hint  (below metric label, tiny)
        ax.text(-2, y - 0.30, KEYWORDS[m], va='center', ha='right',
                fontsize=7.2, color=MUTED, style='italic')

        # percentage badge (right of track)
        ax.text(102, y, f'{pct:.1f}%', va='center', ha='left',
                fontsize=11, fontweight='bold', color=color)

    # threshold reference lines
    for x, lbl in [(50, '50%'), (90, '90%')]:
        ax.axvline(x, color=MUTED, lw=0.8, ls='--', alpha=0.45, zorder=1)
        ax.text(x, n - 0.05, lbl, ha='center', va='bottom',
                fontsize=8, color=MUTED, alpha=0.7)

    # axes
    ax.set_xlim(-2, 115)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v)}%'))
    ax.tick_params(axis='x', colors=MUTED, labelsize=8.5, length=3, pad=4)
    ax.tick_params(axis='y', left=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color=BAR_TRACK, linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)

    # panel title
    ax.set_title(group_title, color=TEXT, fontsize=13,
                 fontweight='bold', loc='left', pad=10)


def main():
    with open(IN_FILE) as f:
        results = json.load(f)
    coverage = results['coverage_pct']
    n_total  = results['total_captions']

    # ── figure layout ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 8), facecolor=BG)

    # leave room on the left for labels
    ax1 = fig.add_axes([0.14, 0.12, 0.36, 0.72])
    ax2 = fig.add_axes([0.58, 0.12, 0.36, 0.72])

    draw_panel(ax1, MCF_ORDER, coverage, 'MCF  ·  Caption Quality')
    draw_panel(ax2, NAF_ORDER, coverage, 'NAF  ·  Navigation Safety')

    # ── legend ──────────────────────────────────────────────────────────────
    leg_items = [
        mpatches.Patch(fc=C_HIGH,    label='High  ≥ 90%'),
        mpatches.Patch(fc=C_MID,     label='Mid  50–89%'),
        mpatches.Patch(fc=C_LOW,     label='Low  < 50%'),
        mpatches.Patch(fc=BAR_TRACK, label='Not covered'),
    ]
    fig.legend(handles=leg_items, loc='lower center',
               ncol=4, fontsize=10, framealpha=0,
               labelcolor=TEXT, facecolor=BG,
               bbox_to_anchor=(0.5, 0.01),
               handlelength=1.4, handleheight=0.9, handletextpad=0.5,
               columnspacing=1.6)

    # ── title & subtitle ────────────────────────────────────────────────────
    fig.text(0.5, 0.96,
             'Gemma-31B Teacher Caption Coverage — BLV Metrics',
             ha='center', va='top', fontsize=16, fontweight='bold',
             color=TEXT)
    fig.text(0.5, 0.91,
             f'keyword matching  ·  n = {n_total:,} captions',
             ha='center', va='top', fontsize=10, color=MUTED)

    fig.savefig(OUT_FILE, dpi=200, bbox_inches='tight', facecolor=BG)
    print(f'Saved → {OUT_FILE}')


if __name__ == '__main__':
    main()
