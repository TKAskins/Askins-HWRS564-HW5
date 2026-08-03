# inspect_trees.py
# Prints which sensor types and soil depths appear at each decision tree level (1-5)
# for the no-cost baselines and selected cost-threshold variants (0.0, 0.3, 0.5, 0.7, 1.0).
# Also prints the root-split feature across all cost levels for each objective.
# Covers three objectives: flux (200cm), water content (avg_wc_top_half), and flux+wc combined.

import pickle as pkl
from pathlib import Path
import numpy as np
from collections import Counter

results = Path(__file__).resolve().parent / 'results' / 'Flood'


def load_tree_and_rt(pkl_dir, fname):
    with open(pkl_dir / f'tree_{fname}.pkl', 'rb') as f:
        tree = pkl.load(f)
    try:
        with open(pkl_dir / f'rtinfo_{fname}.pkl', 'rb') as f:
            rt = pkl.load(f)
    except Exception:
        rt = None
    return tree, rt


def get_splits_by_level(tree_obj, feature_names):
    """Return list of (depth, feature_name) for every internal node."""
    t = tree_obj.tree_
    splits = []
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        if t.children_left[node] != -1:  # internal node
            feat = feature_names[t.feature[node]]
            splits.append((depth + 1, feat))
            stack.append((t.children_left[node], depth + 1))
            stack.append((t.children_right[node], depth + 1))
    return splits


def parse_feat(name):
    """Parse 'wc:d20:t3' -> (sensor_type, depth_cm, time)."""
    parts = name.split(':')
    if len(parts) != 3:
        return name, None, None
    sensor = parts[0]
    depth = int(parts[1][1:])
    time = int(parts[2][1:])
    return sensor, depth, time


def summarise_splits(splits, max_level=5):
    """For each tree level 1..max_level, count sensor types and depth bins."""
    by_level = {}
    for level, feat in splits:
        if level > max_level:
            continue
        sensor, depth, _ = parse_feat(feat)
        by_level.setdefault(level, []).append((sensor, depth))
    return by_level


def print_level_summary(label, splits, max_level=5):
    by_level = summarise_splits(splits, max_level)
    print(f'\n=== {label} ===')
    for lvl in range(1, max_level + 1):
        entries = by_level.get(lvl, [])
        if not entries:
            continue
        sensor_counts = Counter(s for s, d in entries)
        depth_vals = sorted(set(d for s, d in entries))
        depth_range = f'{min(depth_vals)}-{max(depth_vals)} cm'
        print(f'  Level {lvl}: {dict(sensor_counts)}  depths={depth_range}  (n={len(entries)} nodes)')


# ── 1. No-cost baselines ──────────────────────────────────────────────────────
print('\n' + '='*70)
print('NO-COST BASELINES')
print('='*70)

for label, pkl_dir, fname in [
    ('flux (no cost)',    results / '200cm',  'Flood_corn_flux_200cm'),
    ('wc   (no cost)',    results / 'avg_wc', 'Flood_corn_avg_wc_top_half'),
    ('flux+wc (no cost)', results / '200cm',  'Flood_corn_flux_200cm_wc'),
]:
    tree, rt = load_tree_and_rt(pkl_dir, fname)
    feat_names = [c for c in rt.df.columns if c != rt.target_name] if rt else []
    # fallback: get from df_train pickle
    if not feat_names:
        with open(pkl_dir / f'df_train_{fname}.pkl', 'rb') as f:
            df = pkl.load(f)
        feat_names = list(df.columns)
    splits = get_splits_by_level(tree, feat_names)
    print_level_summary(label, splits, max_level=5)

# ── 2. Cost sweep: flux ──────────────────────────────────────────────────────
print('\n' + '='*70)
print('COST SWEEP — flux target (depth200_cost)')
print('='*70)

pkl_dir = results / '200cm'
with open(pkl_dir / 'df_train_Flood_depth200_cost0.0.pkl', 'rb') as f:
    df_ref = pkl.load(f)
feat_names_flux = list(df_ref.columns)

for cost in [0.0, 0.3, 0.5, 0.7, 1.0]:
    fname = f'Flood_depth200_cost{cost}'
    tree, _ = load_tree_and_rt(pkl_dir, fname)
    splits = get_splits_by_level(tree, feat_names_flux)
    print_level_summary(f'flux cost={cost}', splits, max_level=5)

# ── 3. Cost sweep: wc ───────────────────────────────────────────────────────
print('\n' + '='*70)
print('COST SWEEP — wc target (avg_wc_top_half_cost)')
print('='*70)

pkl_dir = results / 'avg_wc'
with open(pkl_dir / 'df_train_Flood_corn_avg_wc_top_half_cost0.0.pkl', 'rb') as f:
    df_ref = pkl.load(f)
feat_names_wc = list(df_ref.columns)

for cost in [0.0, 0.3, 0.5, 0.7, 1.0]:
    fname = f'Flood_corn_avg_wc_top_half_cost{cost}'
    tree, _ = load_tree_and_rt(pkl_dir, fname)
    splits = get_splits_by_level(tree, feat_names_wc)
    print_level_summary(f'wc cost={cost}', splits, max_level=5)

# ── 4. Cost sweep: flux+wc ──────────────────────────────────────────────────
print('\n' + '='*70)
print('COST SWEEP — flux+wc target (flux200_wc_cost)')
print('='*70)

pkl_dir = results / '200cm'
with open(pkl_dir / 'df_train_Flood_corn_flux200_wc_cost0.0.pkl', 'rb') as f:
    df_ref = pkl.load(f)
feat_names_fwc = list(df_ref.columns)

for cost in [0.0, 0.3, 0.5, 0.7, 1.0]:
    fname = f'Flood_corn_flux200_wc_cost{cost}'
    tree, _ = load_tree_and_rt(pkl_dir, fname)
    splits = get_splits_by_level(tree, feat_names_fwc)
    print_level_summary(f'flux+wc cost={cost}', splits, max_level=5)

# ── 5. Top-split feature across all cost levels ─────────────────────────────
print('\n' + '='*70)
print('ROOT SPLIT FEATURE across all cost levels')
print('='*70)

groups = [
    ('flux',    results / '200cm',  'Flood_depth200_cost{}',              feat_names_flux),
    ('wc',      results / 'avg_wc', 'Flood_corn_avg_wc_top_half_cost{}',  feat_names_wc),
    ('flux+wc', results / '200cm',  'Flood_corn_flux200_wc_cost{}',       feat_names_fwc),
]

costs = [round(c, 1) for c in np.arange(0.0, 1.1, 0.1)]

for group_label, pkl_dir, fname_tmpl, feat_names in groups:
    print(f'\n{group_label}:')
    for cost in costs:
        fname = fname_tmpl.format(cost)
        tree, _ = load_tree_and_rt(pkl_dir, fname)
        root_feat = feat_names[tree.tree_.feature[0]]
        sensor, depth, time = parse_feat(root_feat)
        print(f'  cost={cost:.1f}  root split: {sensor} @ {depth}cm  t={time}')
