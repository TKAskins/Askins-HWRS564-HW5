"""
extract_sensor_freq.py
Counts how many trained trees selected each physical sensor location
(sensor type + depth) at the optimal tree level, across all objectives,
cost thresholds, and crops. Outputs JSON for the visualisation.
"""
import json
import pickle as pkl
from pathlib import Path
import numpy as np
from collections import defaultdict
from sklearn.tree import _tree

RESULTS = Path(__file__).resolve().parent / 'results' / 'Flood'
COSTS = [round(c, 1) for c in np.arange(0.0, 1.1, 0.1)]


def unique_sensors_up_to_level(tree, feat_names, max_level):
    t = tree.tree_
    sensors = set()
    stack = [(0, 1)]
    while stack:
        node, depth = stack.pop()
        if t.children_left[node] == _tree.TREE_LEAF:
            continue
        if depth <= max_level:
            parts = feat_names[t.feature[node]].split(':')
            sensors.add((parts[0], int(parts[1][1:])))
            stack.append((t.children_left[node], depth + 1))
            stack.append((t.children_right[node], depth + 1))
    return sensors


def load_and_count(pkl_dir, fname, targ_col_for_rtinfo):
    try:
        with open(pkl_dir / f'tree_{fname}.pkl', 'rb') as f:
            tree = pkl.load(f)
        with open(pkl_dir / f'rtinfo_{fname}.pkl', 'rb') as f:
            rt = pkl.load(f)
        with open(pkl_dir / f'df_train_{fname}.pkl', 'rb') as f:
            df = pkl.load(f)
    except FileNotFoundError:
        return set()

    opt_level = int(np.argmax(rt.var_df_by_depth['var_exp'])) + 1
    feat_names = [c for c in df.columns if c not in
                  (['flux_200','avg_wc_top_half','flux_60'])]
    return unique_sensors_up_to_level(tree, feat_names, opt_level)


# ── catalogue ────────────────────────────────────────────────────────────────

catalogue = []

for crop, flux_col, flux_depth, flux_dir in [
    ('corn', 'flux_200', '200', '200cm'),
    ('beans', 'flux_60',  '60',  '60cm'),
]:
    # flux — no cost
    catalogue.append((crop, 'flux', RESULTS / flux_dir,
                       f'Flood_{crop}_flux_{flux_depth}cm', flux_col))
    # flux — cost sweep
    for c in COSTS:
        catalogue.append((crop, 'flux', RESULTS / flux_dir,
                          f'Flood_{crop}_flux{flux_depth}_cost{c}', flux_col))
    # wc — no cost
    catalogue.append((crop, 'wc', RESULTS / 'avg_wc',
                      f'Flood_{crop}_avg_wc_top_half', 'avg_wc_top_half'))
    # wc — cost sweep
    for c in COSTS:
        catalogue.append((crop, 'wc', RESULTS / 'avg_wc',
                          f'Flood_{crop}_avg_wc_top_half_cost{c}', 'avg_wc_top_half'))
    # flux+wc — no cost
    catalogue.append((crop, 'flux_wc', RESULTS / flux_dir,
                       f'Flood_{crop}_flux_{flux_depth}cm_wc', flux_col))
    # flux+wc — cost sweep
    for c in COSTS:
        catalogue.append((crop, 'flux_wc', RESULTS / flux_dir,
                          f'Flood_{crop}_flux{flux_depth}_wc_cost{c}', flux_col))


# ── count ────────────────────────────────────────────────────────────────────

# counts[crop][sensor_type][depth] = number of trees that selected it
counts = {
    'corn':  defaultdict(lambda: defaultdict(int)),
    'beans': defaultdict(lambda: defaultdict(int)),
}
total_trees = {'corn': 0, 'beans': 0}

for crop, obj, pkl_dir, fname, targ_col in catalogue:
    sensors = load_and_count(pkl_dir, fname, targ_col)
    total_trees[crop] += 1
    for stype, depth in sensors:
        counts[crop][stype][depth] += 1


# ── serialise ────────────────────────────────────────────────────────────────

out = {}
for crop in ['corn', 'beans']:
    out[crop] = {'total_trees': total_trees[crop], 'sensors': {}}
    for stype in ['wc', 'p']:
        out[crop]['sensors'][stype] = {
            str(d): counts[crop][stype][d]
            for d in sorted(counts[crop][stype].keys())
        }

print(json.dumps(out, indent=2))
