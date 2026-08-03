# compare_stability.py
# Compares the original DT run (results/Flood, split_random_state=0) against
# the two stability-check re-runs (results/Flood_iter2/3, different train/test
# split seeds) for the no-cost baselines (flux, wc, flux+wc):
#   - root-split feature, and level 1-2 splits
#   - root-split feature across the full cost-threshold sweep
#   - train/test R2 and RMSE of the full (unpruned) tree
# All three iterations share the same source CSV; only the 80/20 trial split
# used to fit each tree differs, so agreement here means the tree structure
# and accuracy are not artifacts of one particular split.

import pickle as pkl
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

RESULTS_ROOT = Path(__file__).resolve().parent / 'results'
ITERATIONS = ['Flood', 'Flood_iter2', 'Flood_iter3']
PLANT_TYPE = 'beans'
DEPTH = 60
COSTS = [round(c, 1) for c in np.arange(0.0, 1.1, 0.1)]


def load_pkl(path):
    with open(path, 'rb') as f:
        return pkl.load(f)


def parse_feat(name):
    parts = name.split(':')
    if len(parts) != 3:
        return name, None, None
    return parts[0], int(parts[1][1:]), int(parts[2][1:])


def get_splits_by_level(tree_obj):
    t = tree_obj.tree_
    splits = []
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        if t.children_left[node] != -1:
            splits.append((depth + 1, t.feature[node]))
            stack.append((t.children_left[node], depth + 1))
            stack.append((t.children_right[node], depth + 1))
    return splits


def level_summary(tree_obj, feat_names, max_level=2):
    splits = get_splits_by_level(tree_obj)
    by_level = {}
    for level, fidx in splits:
        if level > max_level:
            continue
        sensor, depth, _ = parse_feat(feat_names[fidx])
        by_level.setdefault(level, []).append((sensor, depth))
    return by_level


def print_level_summary(by_level, max_level=2, indent='    '):
    for lvl in range(1, max_level + 1):
        entries = by_level.get(lvl, [])
        if not entries:
            continue
        sensor_counts = Counter(s for s, d in entries)
        depth_vals = sorted(set(d for s, d in entries))
        depth_range = f'{min(depth_vals)}-{max(depth_vals)} cm'
        print(f'{indent}Level {lvl}: {dict(sensor_counts)}  depths={depth_range}  (n={len(entries)} nodes)')


def score_tree(pkl_dir, fname, targ_names):
    tree = load_pkl(pkl_dir / f'tree_{fname}.pkl')
    df_train = load_pkl(pkl_dir / f'df_train_{fname}.pkl')
    df_test = load_pkl(pkl_dir / f'df_test_{fname}.pkl')
    feat_names = [c for c in df_train.columns if c not in targ_names]

    X_train = df_train[feat_names].to_numpy()
    y_train = df_train[targ_names].to_numpy()
    X_test = df_test[feat_names].to_numpy()
    y_test = df_test[targ_names].to_numpy()

    pred_train = tree.predict(X_train)
    pred_test = tree.predict(X_test)

    scores = {}
    for i, targ in enumerate(targ_names):
        yt_tr = y_train[:, i] if y_train.ndim > 1 else y_train
        yt_te = y_test[:, i] if y_test.ndim > 1 else y_test
        pt_tr = pred_train[:, i] if pred_train.ndim > 1 else pred_train
        pt_te = pred_test[:, i] if pred_test.ndim > 1 else pred_test
        scores[targ] = dict(
            r2_train=r2_score(yt_tr, pt_tr),
            r2_test=r2_score(yt_te, pt_te),
            rmse_train=np.sqrt(mean_squared_error(yt_tr, pt_tr)),
            rmse_test=np.sqrt(mean_squared_error(yt_te, pt_te)),
        )
    return tree, feat_names, scores


# ── 1. No-cost baselines: level 1-2 splits + accuracy ──────────────────────
print('=' * 78)
print('NO-COST BASELINES — level 1-2 splits and full-tree accuracy')
print('=' * 78)

baselines = [
    ('flux', f'{DEPTH}cm', f'Flood_{PLANT_TYPE}_flux_{DEPTH}cm', [f'flux_{DEPTH}']),
    ('wc', 'avg_wc', f'Flood_{PLANT_TYPE}_avg_wc_top_half', ['avg_wc_top_half']),
    ('flux+wc', f'{DEPTH}cm', f'Flood_{PLANT_TYPE}_flux_{DEPTH}cm_wc', [f'flux_{DEPTH}', 'avg_wc_top_half']),
]

for label, subdir, fname, targ_names in baselines:
    print(f'\n--- {label} ---')
    for it in ITERATIONS:
        pkl_dir = RESULTS_ROOT / it / subdir
        try:
            tree, feat_names, scores = score_tree(pkl_dir, fname, targ_names)
        except FileNotFoundError as e:
            print(f'  [{it}] MISSING: {e}')
            continue
        root_feat = feat_names[tree.tree_.feature[0]]
        sensor, depth, time = parse_feat(root_feat)
        print(f'  [{it}] root split: {sensor} @ {depth}cm t={time}  (n_leaves={tree.get_n_leaves()}, depth={tree.get_depth()})')
        for targ in targ_names:
            s = scores[targ]
            print(f'      {targ}: R2 train={s["r2_train"]:.4f} test={s["r2_test"]:.4f}  '
                  f'RMSE train={s["rmse_train"]:.4f} test={s["rmse_test"]:.4f}')
        by_level = level_summary(tree, feat_names, max_level=2)
        print_level_summary(by_level, max_level=2, indent='      ')

# ── 2. Root split across cost sweep ─────────────────────────────────────────
print('\n' + '=' * 78)
print('ROOT SPLIT FEATURE across cost sweep, per iteration')
print('=' * 78)

sweeps = [
    ('flux', f'{DEPTH}cm', f'Flood_{PLANT_TYPE}_flux{DEPTH}_cost{{}}'),
    ('wc', 'avg_wc', f'Flood_{PLANT_TYPE}_avg_wc_top_half_cost{{}}'),
    ('flux+wc', f'{DEPTH}cm', f'Flood_{PLANT_TYPE}_flux{DEPTH}_wc_cost{{}}'),
]

for label, subdir, fname_tmpl in sweeps:
    print(f'\n--- {label} ---')
    for it in ITERATIONS:
        pkl_dir = RESULTS_ROOT / it / subdir
        row = []
        for cost in COSTS:
            fname = fname_tmpl.format(cost)
            try:
                tree = load_pkl(pkl_dir / f'tree_{fname}.pkl')
                df_train = load_pkl(pkl_dir / f'df_train_{fname}.pkl')
            except FileNotFoundError:
                row.append('MISSING')
                continue
            targ_guess = [c for c in df_train.columns if c.startswith('flux_') or c == 'avg_wc_top_half']
            feat_names = [c for c in df_train.columns if c not in targ_guess]
            root_feat = feat_names[tree.tree_.feature[0]]
            sensor, depth, _ = parse_feat(root_feat)
            row.append(f'{sensor}@{depth}')
        print(f'  [{it}] ' + '  '.join(f'c{c:.1f}={v}' for c, v in zip(COSTS, row)))

print('\nDone.')
