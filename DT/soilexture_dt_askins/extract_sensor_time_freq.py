"""
extract_sensor_time_freq.py
For corn decision trees, counts how many unique physical-sensor selections
(sensor type + depth + timestep, deduplicated per tree) fall into early
(day 0-4), middle (day 5-9), and late (day 10-15) timestep bins, broken
down by prediction objective (flux, water content, flux + water content).
"""
import json
import pickle as pkl
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.tree import _tree

RESULTS = Path(__file__).resolve().parent / 'results' / 'Flood'
COSTS = [round(c, 1) for c in np.arange(0.0, 1.1, 0.1)]

CROP = 'corn'
FLUX_COL, FLUX_DEPTH, FLUX_DIR = 'flux_200', '200', '200cm'

BINS = [('early (day 0-4)', 0, 4), ('middle (day 5-9)', 5, 9), ('late (day 10-15)', 10, 15)]


def day_bin(t):
    for label, lo, hi in BINS:
        if lo <= t <= hi:
            return label
    return None


def unique_reads_up_to_level(tree, feat_names, max_level):
    """Set of (sensor_type, depth, time) splits used at DT levels 1..max_level."""
    t = tree.tree_
    reads = set()
    stack = [(0, 1)]
    while stack:
        node, depth = stack.pop()
        if t.children_left[node] == _tree.TREE_LEAF:
            continue
        if depth <= max_level:
            parts = feat_names[t.feature[node]].split(':')
            stype = parts[0]
            sdepth = int(parts[1][1:])
            stime = int(parts[2][1:])
            reads.add((stype, sdepth, stime))
            stack.append((t.children_left[node], depth + 1))
            stack.append((t.children_right[node], depth + 1))
    return reads


def load_and_read(pkl_dir, fname):
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
                  ('flux_200', 'avg_wc_top_half', 'flux_60')]
    return unique_reads_up_to_level(tree, feat_names, opt_level)


# ── catalogue (corn only, 36 trees = 3 objectives x 12 cost configs) ─────────

catalogue = []

catalogue.append(('flux', RESULTS / FLUX_DIR, f'Flood_{CROP}_flux_{FLUX_DEPTH}cm'))
for c in COSTS:
    catalogue.append(('flux', RESULTS / FLUX_DIR, f'Flood_{CROP}_flux{FLUX_DEPTH}_cost{c}'))

catalogue.append(('wc', RESULTS / 'avg_wc', f'Flood_{CROP}_avg_wc_top_half'))
for c in COSTS:
    catalogue.append(('wc', RESULTS / 'avg_wc', f'Flood_{CROP}_avg_wc_top_half_cost{c}'))

catalogue.append(('flux_wc', RESULTS / FLUX_DIR, f'Flood_{CROP}_flux_{FLUX_DEPTH}cm_wc'))
for c in COSTS:
    catalogue.append(('flux_wc', RESULTS / FLUX_DIR, f'Flood_{CROP}_flux{FLUX_DEPTH}_wc_cost{c}'))


# ── count ────────────────────────────────────────────────────────────────────

# counts[objective][bin_label] = number of unique sensor-reads (deduped per tree)
counts = defaultdict(lambda: defaultdict(int))
n_trees = defaultdict(int)
unique_sensors = defaultdict(set)  # union of (type, depth, time) across all trees per objective

for obj, pkl_dir, fname in catalogue:
    reads = load_and_read(pkl_dir, fname)
    n_trees[obj] += 1
    for stype, sdepth, stime in reads:
        counts[obj][day_bin(stime)] += 1
        unique_sensors[obj].add((stype, sdepth, stime))


# ── build table ──────────────────────────────────────────────────────────────

objectives = ['flux', 'wc', 'flux_wc']
bin_labels = [b[0] for b in BINS]

rows = []
for obj in objectives:
    row = {'objective': obj, 'n_trees': n_trees[obj], 'n_unique_sensors': len(unique_sensors[obj])}
    total = 0
    for label in bin_labels:
        row[label] = counts[obj][label]
        total += counts[obj][label]
    row['total'] = total
    rows.append(row)

total_unique = set()
for obj in objectives:
    total_unique |= unique_sensors[obj]

total_row = {'objective': 'TOTAL', 'n_trees': sum(n_trees[o] for o in objectives),
             'n_unique_sensors': len(total_unique)}
grand_total = 0
for label in bin_labels:
    s = sum(counts[o][label] for o in objectives)
    total_row[label] = s
    grand_total += s
total_row['total'] = grand_total
rows.append(total_row)

# ── print ────────────────────────────────────────────────────────────────────

header = (f"{'objective':<10} {'n_trees':>8} {'n_unique_sensors':>17} " +
          ' '.join(f'{l:>18}' for l in bin_labels) + f" {'total':>8}")
print(header)
print('-' * len(header))
for row in rows:
    print(f"{row['objective']:<10} {row['n_trees']:>8} {row['n_unique_sensors']:>17} " +
          ' '.join(f"{row[l]:>18}" for l in bin_labels) +
          f" {row['total']:>8}")

out_dir = Path(__file__).resolve().parent / 'report_tables'
out_dir.mkdir(exist_ok=True)

df_out = pd.DataFrame(rows)[['objective', 'n_trees', 'n_unique_sensors'] + bin_labels + ['total']]
csv_path = out_dir / 'table_corn_sensor_time_bins.csv'
df_out.to_csv(csv_path, index=False)
print(f'\n[saved -> report_tables/{csv_path.name}]')

json_path = out_dir / 'table_corn_sensor_time_bins.json'
with open(json_path, 'w') as f:
    json.dump({'corn': rows}, f, indent=2)
print(f'[saved -> report_tables/{json_path.name}]')
