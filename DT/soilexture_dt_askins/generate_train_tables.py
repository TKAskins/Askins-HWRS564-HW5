"""
generate_train_tables.py
Training-set counterpart to the report_tables/table_<plant>_<objective>.csv
tables (which report *test* R²/RMSE, produced via compare_trees.py). This
script evaluates each tree on its own training data instead, at the same
optimal level (argmax of training var_exp), and writes matching tables
named table_<plant>_<objective>_train.csv with training rather than test
R²/RMSE.

Also recomputes test R²/RMSE for a sanity check against the existing
table_<plant>_<objective>.csv values (printed, not written) before trusting
the newly computed training numbers -- the tree/level-selection/prediction
logic here is a copy of compare_trees.py's, generalised to run over both
crops in one pass instead of one crop per PLANT_TYPE edit.
"""

import pickle as pkl
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import _tree

RESULTS = Path(__file__).resolve().parent / 'results' / 'Flood'
OUT_DIR = Path(__file__).resolve().parent / 'report_tables'
OUT_DIR.mkdir(exist_ok=True)

COSTS = [round(c, 1) for c in np.arange(0.0, 1.1, 0.1)]


# ── helpers (mirrors compare_trees.py) ─────────────────────────────────────

def load(pkl_dir, fname):
    def _load(name):
        with open(pkl_dir / f'{name}_{fname}.pkl', 'rb') as f:
            return pkl.load(f)
    return _load('tree'), _load('rtinfo'), _load('df_train'), _load('df_test')


def apply_truncated(X, tree_, max_depth):
    n = X.shape[0]
    nodes = np.zeros(n, dtype=int)
    for i in range(n):
        node, depth = 0, 0
        while tree_.children_left[node] != _tree.TREE_LEAF and depth < max_depth:
            depth += 1
            node = (tree_.children_left[node]
                    if X[i, tree_.feature[node]] <= tree_.threshold[node]
                    else tree_.children_right[node])
        nodes[i] = node
    return nodes


def predict_at_level(df_feats, tree, level):
    X = df_feats.to_numpy()
    nodes = apply_truncated(X, tree.tree_, level)
    pred = tree.tree_.value[nodes].squeeze(-1)
    if pred.shape[1] == 1:
        pred = pred.squeeze(-1)
    return pred


def inverse_signed_log(y):
    return np.sign(y) * np.expm1(np.abs(y))


def parse_feat(name):
    parts = name.split(':')
    return parts[0], int(parts[1][1:])


def unique_sensors_up_to_level(tree, feat_names, max_level):
    t = tree.tree_
    sensors = set()
    stack = [(0, 1)]
    while stack:
        node, depth = stack.pop()
        if t.children_left[node] == _tree.TREE_LEAF:
            continue
        if depth <= max_level:
            sensors.add(parse_feat(feat_names[t.feature[node]]))
            stack.append((t.children_left[node], depth + 1))
            stack.append((t.children_right[node], depth + 1))
    return sensors


def sensor_summary_str(sensors):
    wc_depths = sorted(d for s, d in sensors if s == 'wc')
    p_depths = sorted(d for s, d in sensors if s == 'p')
    wc_str = (f'{len(wc_depths)}wc@{wc_depths[0]}cm' if len(wc_depths) == 1
              else f'{len(wc_depths)}wc@{wc_depths[0]}-{wc_depths[-1]}cm') if wc_depths else ''
    p_str = (f'{len(p_depths)}p@{p_depths[0]}cm' if len(p_depths) == 1
             else f'{len(p_depths)}p@{p_depths[0]}-{p_depths[-1]}cm') if p_depths else ''
    sensor_str = ', '.join(filter(None, [wc_str, p_str]))
    return f'{len(sensors)} ({sensor_str})', len(wc_depths), len(p_depths)


def score(df, tree, feat_names, targ_cols, level):
    """R²/RMSE (overall + per-target, log-space and physical-units for flux)."""
    pred = predict_at_level(df[feat_names], tree, level)
    y = df[targ_cols].to_numpy()
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    r2_per_target = r2_score(y, pred, multioutput='raw_values')
    r2 = float(np.mean(r2_per_target))
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    r2_by_target = {col: round(float(r2_per_target[i]), 4) for i, col in enumerate(targ_cols)}
    rmse_per_target = np.sqrt(np.mean((y - pred) ** 2, axis=0))
    rmse_by_target = {col: round(float(rmse_per_target[i]), 6) for i, col in enumerate(targ_cols)}

    rmse_physical_by_target = {}
    for i, col in enumerate(targ_cols):
        if col.startswith('flux'):
            y_phys = inverse_signed_log(y[:, i])
            pred_phys = inverse_signed_log(pred[:, i])
            rmse_physical_by_target[col] = round(
                float(np.sqrt(np.mean((y_phys - pred_phys) ** 2))), 6)

    return dict(r2=round(r2, 4), rmse=round(rmse, 6),
                r2_by_target=r2_by_target, rmse_by_target=rmse_by_target,
                rmse_physical_by_target=rmse_physical_by_target)


def evaluate(tree, rt, df_tr, df_te, targ_cols):
    opt_level = int(np.argmax(rt.var_df_by_depth['var_exp'])) + 1
    total_cost = float(rt.var_df_by_depth.loc[opt_level, 'total_cost'])

    feat_names = [c for c in df_te.columns if c not in targ_cols]
    train = score(df_tr, tree, feat_names, targ_cols, opt_level)
    test = score(df_te, tree, feat_names, targ_cols, opt_level)

    sensors = unique_sensors_up_to_level(tree, feat_names, opt_level)
    sensor_str, wc_n, p_n = sensor_summary_str(sensors)

    return dict(opt_level=opt_level, total_cost=round(total_cost, 1),
                sensors=sensor_str, n_sensors=len(sensors), wc_sensors=wc_n, p_sensors=p_n,
                train=train, test=test)


# ── catalogue (generalised over both crops) ─────────────────────────────────

def build_catalogue(plant):
    flux_depth = 200 if plant == 'corn' else 60
    flux_dir = f'{flux_depth}cm'
    flux_col = f'flux_{flux_depth}'

    catalogue = []

    catalogue.append(dict(objective='flux', cost_thresh=None,
        pkl_dir=RESULTS / flux_dir, fname=f'Flood_{plant}_flux_{flux_depth}cm',
        targ_cols=[flux_col]))
    for c in COSTS:
        catalogue.append(dict(objective='flux', cost_thresh=c,
            pkl_dir=RESULTS / flux_dir, fname=f'Flood_{plant}_flux{flux_depth}_cost{c}',
            targ_cols=[flux_col]))

    catalogue.append(dict(objective='wc', cost_thresh=None,
        pkl_dir=RESULTS / 'avg_wc', fname=f'Flood_{plant}_avg_wc_top_half',
        targ_cols=['avg_wc_top_half']))
    for c in COSTS:
        catalogue.append(dict(objective='wc', cost_thresh=c,
            pkl_dir=RESULTS / 'avg_wc', fname=f'Flood_{plant}_avg_wc_top_half_cost{c}',
            targ_cols=['avg_wc_top_half']))

    catalogue.append(dict(objective='flux+wc', cost_thresh=None,
        pkl_dir=RESULTS / flux_dir, fname=f'Flood_{plant}_flux_{flux_depth}cm_wc',
        targ_cols=[flux_col, 'avg_wc_top_half']))
    for c in COSTS:
        catalogue.append(dict(objective='flux+wc', cost_thresh=c,
            pkl_dir=RESULTS / flux_dir, fname=f'Flood_{plant}_flux{flux_depth}_wc_cost{c}',
            targ_cols=[flux_col, 'avg_wc_top_half']))

    return catalogue, flux_col


# ── run for both crops ───────────────────────────────────────────────────────

PLANT_LABEL = {'corn': 'Corn', 'beans': 'Beans'}
OBJ_LABEL = {'flux': 'Flux Only', 'wc': 'WC Only', 'flux+wc': 'Flux+WC Content'}

for plant in ['corn', 'beans']:
    catalogue, flux_col = build_catalogue(plant)

    rows = []
    for entry in catalogue:
        try:
            tree, rt, df_tr, df_te = load(entry['pkl_dir'], entry['fname'])
        except FileNotFoundError:
            continue
        metrics = evaluate(tree, rt, df_tr, df_te, entry['targ_cols'])
        rows.append({'objective': entry['objective'],
                      'cost_thresh': 'none' if entry['cost_thresh'] is None else entry['cost_thresh'],
                      **metrics})

    df = pd.DataFrame(rows)

    for obj in ['flux', 'wc', 'flux+wc']:
        sub = df[df['objective'] == obj].reset_index(drop=True)
        if sub.empty:
            continue

        # sanity check: compare recomputed test R²/RMSE against the existing
        # table_<plant>_<obj>.csv (or table_<plant>_flux_wc.csv, raw format)
        print(f'\n[{plant} / {obj}] recomputed test R² (sanity check vs report_tables/table_{plant}_{obj.replace("+","_")}.csv):')
        print(sub[['cost_thresh']].assign(
            r2_test=sub['test'].apply(lambda d: d['r2']),
            rmse_test=sub['test'].apply(lambda d: d['rmse']),
        ).to_string(index=False))

        title = f'{PLANT_LABEL[plant]} {OBJ_LABEL[obj]} (Training)'
        out_rows = []
        for _, r in sub.iterrows():
            out_rows.append([r['cost_thresh'], r['opt_level'],
                              r['train']['r2'], r['train']['rmse'],
                              r['total_cost'], r['sensors']])

        fname = f"table_{plant}_{obj.replace('+', '_')}_train.csv"
        with open(OUT_DIR / fname, 'w', newline='') as f:
            f.write(f'{title},,,,,\n')
            f.write('Cost Threshold,Levels,R2,RMSE,Cost,Sensors\n')
            for row in out_rows:
                sensors_field = row[5]
                if ',' in sensors_field:
                    sensors_field = f'"{sensors_field}"'
                f.write(f'{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{sensors_field}\n')
        print(f'  [saved -> report_tables/{fname}]')

        if obj == 'flux+wc':
            bd_rows = []
            for _, r in sub.iterrows():
                tr = r['train']
                r2_flux = tr['r2_by_target'].get(flux_col, float('nan'))
                r2_wc = tr['r2_by_target'].get('avg_wc_top_half', float('nan'))
                rmse_flux = tr['rmse_by_target'].get(flux_col, float('nan'))
                rmse_wc = tr['rmse_by_target'].get('avg_wc_top_half', float('nan'))
                bd_rows.append([r['cost_thresh'], tr['r2'], round(r2_flux, 4), round(r2_wc, 4),
                                 round(r2_flux - r2_wc, 4), rmse_flux, rmse_wc,
                                 r['total_cost'], r['sensors']])

            bd_fname = f'table_{plant}_flux_wc_breakdown_train.csv'
            with open(OUT_DIR / bd_fname, 'w', newline='') as f:
                f.write(f'{PLANT_LABEL[plant]} Flux+WC Content (Training),,,,,,,,\n')
                f.write('Cost Threshold,Combined R2,Flux R2,WC R2,R2 Gap (flux-wc),Flux RMSE,WC RMSE,Cost,Sensors\n')
                for row in bd_rows:
                    sensors_field = row[8]
                    if ',' in sensors_field:
                        sensors_field = f'"{sensors_field}"'
                    f.write(','.join(str(v) for v in row[:8]) + f',{sensors_field}\n')
            print(f'  [saved -> report_tables/{bd_fname}]')

print('\nDone.')
