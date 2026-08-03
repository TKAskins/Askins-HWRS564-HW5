"""
extract_sensor_variance.py
For each crop, averages sklearn's tree.feature_importances_ (fraction of total
variance reduction attributable to each feature) by physical sensor location
(type + depth), across all 36 trees: 3 objectives (flux, water content,
flux + water content) x 12 cost configurations (baseline + cost weights
0.0-1.0). Companion to extract_sensor_freq.py -- same tree catalogue, same
crop/sensor keys, so the two outputs can be joined directly by (crop, type,
depth) in the sensor-frequency bar chart.
"""
import json
import pickle as pkl
from pathlib import Path
import numpy as np
from collections import defaultdict

RESULTS = Path(__file__).resolve().parent / 'results' / 'Flood'
COSTS = [round(c, 1) for c in np.arange(0.0, 1.1, 0.1)]


def load_importances(pkl_dir, fname, targ_cols):
    try:
        with open(pkl_dir / f'tree_{fname}.pkl', 'rb') as f:
            tree = pkl.load(f)
        with open(pkl_dir / f'df_train_{fname}.pkl', 'rb') as f:
            df = pkl.load(f)
    except FileNotFoundError:
        return None

    feat_names = [c for c in df.columns if c not in targ_cols]
    imp = tree.feature_importances_

    agg = defaultdict(float)
    for i, name in enumerate(feat_names):
        parts = name.split(':')
        if len(parts) != 3:
            continue
        stype = parts[0]
        depth = int(parts[1][1:])
        agg[(stype, depth)] += imp[i]
    return agg


# ── catalogue (mirrors extract_sensor_freq.py) ────────────────────────────────

catalogue = []

for crop, flux_col, flux_depth, flux_dir in [
    ('corn', 'flux_200', '200', '200cm'),
    ('beans', 'flux_60', '60', '60cm'),
]:
    targ_cols_flux = [flux_col]
    targ_cols_wc = ['avg_wc_top_half']
    targ_cols_fluxwc = [flux_col, 'avg_wc_top_half']

    # flux — no cost
    catalogue.append((crop, RESULTS / flux_dir,
                       f'Flood_{crop}_flux_{flux_depth}cm', targ_cols_flux))
    # flux — cost sweep
    for c in COSTS:
        catalogue.append((crop, RESULTS / flux_dir,
                          f'Flood_{crop}_flux{flux_depth}_cost{c}', targ_cols_flux))
    # wc — no cost
    catalogue.append((crop, RESULTS / 'avg_wc',
                      f'Flood_{crop}_avg_wc_top_half', targ_cols_wc))
    # wc — cost sweep
    for c in COSTS:
        catalogue.append((crop, RESULTS / 'avg_wc',
                          f'Flood_{crop}_avg_wc_top_half_cost{c}', targ_cols_wc))
    # flux+wc — no cost
    catalogue.append((crop, RESULTS / flux_dir,
                       f'Flood_{crop}_flux_{flux_depth}cm_wc', targ_cols_fluxwc))
    # flux+wc — cost sweep
    for c in COSTS:
        catalogue.append((crop, RESULTS / flux_dir,
                          f'Flood_{crop}_flux{flux_depth}_wc_cost{c}', targ_cols_fluxwc))


# ── accumulate ────────────────────────────────────────────────────────────────

# sums[crop][(stype, depth)] = sum of feature_importances_ across all trees
# (0 contribution from trees that don't use that sensor at all)
sums = {'corn': defaultdict(float), 'beans': defaultdict(float)}
n_trees = {'corn': 0, 'beans': 0}

for crop, pkl_dir, fname, targ_cols in catalogue:
    agg = load_importances(pkl_dir, fname, targ_cols)
    if agg is None:
        continue
    n_trees[crop] += 1
    for sensor, imp in agg.items():
        sums[crop][sensor] += imp


# ── serialise (percent of variance explained, averaged over all trees) ───────

out = {}
for crop in ['corn', 'beans']:
    total = n_trees[crop]
    out[crop] = {'total_trees': total, 'sensors': {'wc': {}, 'p': {}}}
    for (stype, depth), s in sums[crop].items():
        pct = round((s / total) * 100, 3) if total else 0.0
        out[crop]['sensors'][stype][str(depth)] = pct
    for stype in ['wc', 'p']:
        out[crop]['sensors'][stype] = dict(
            sorted(out[crop]['sensors'][stype].items(), key=lambda kv: int(kv[0])))

print(json.dumps(out, indent=2))
