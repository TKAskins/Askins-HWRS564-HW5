#compare_trees.py
#Compares cost, unique sensors, and test accuracy across all tree variants
#at each tree's optimal DT level (level that maximises variance explained).

import pickle as pkl
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import _tree


RESULTS = Path(__file__).resolve().parent / 'results' / 'Flood'
PLANT_TYPE = 'beans'


# ── helpers ──────────────────────────────────────────────────────────────────

def load(pkl_dir, fname):
    def _load(name):
        with open(pkl_dir / f'{name}_{fname}.pkl', 'rb') as f:
            return pkl.load(f)
    tree   = _load('tree')
    rt     = _load('rtinfo')
    df_tr  = _load('df_train')
    df_te  = _load('df_test')
    return tree, rt, df_tr, df_te


def apply_truncated(X, tree_, max_depth):
    """Walk every sample to the leaf, stopping at max_depth."""
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
    # value shape: (n_nodes, n_outputs, max_n_classes)
    # index along axis 0 to get (n_samples, n_outputs, 1), then squeeze
    pred = tree.tree_.value[nodes].squeeze(-1)   # (n_samples, n_outputs)
    if pred.shape[1] == 1:
        pred = pred.squeeze(-1)                  # (n_samples,) for single-output
    return pred


def inverse_signed_log(y):
    """Inverse of sign(x)*log1p(|x|): recovers x in original (physical) units."""
    return np.sign(y) * np.expm1(np.abs(y))


def parse_feat(name):
    """'wc:d20:t3' -> (sensor_type, soil_depth_cm)."""
    parts = name.split(':')
    return parts[0], int(parts[1][1:])


def unique_sensors_up_to_level(tree, feat_names, max_level):
    """Return set of (sensor_type, soil_depth) used at DT levels 1..max_level."""
    t = tree.tree_
    sensors = set()
    stack = [(0, 1)]
    while stack:
        node, depth = stack.pop()
        if t.children_left[node] == _tree.TREE_LEAF:
            continue
        if depth <= max_level:
            sensors.add(parse_feat(feat_names[t.feature[node]]))
            stack.append((t.children_left[node],  depth + 1))
            stack.append((t.children_right[node], depth + 1))
    return sensors


def evaluate(tree, rt, df_tr, df_te, targ_cols, feat_cols):
    """Return dict with metrics at optimal DT level."""
    # optimal level = level that maximises var_exp on training data
    opt_level = int(np.argmax(rt.var_df_by_depth['var_exp'])) + 1
    total_cost = float(rt.var_df_by_depth.loc[opt_level, 'total_cost'])
    var_exp_train = float(rt.var_df_by_depth.loc[opt_level, 'var_exp'])

    # test R² — handle single or multi-target
    feat_names = [c for c in df_te.columns if c not in targ_cols]
    pred_te = predict_at_level(df_te[feat_names], tree, opt_level)
    y_te    = df_te[targ_cols].to_numpy()

    # pred_te is already (n_samples,) for single-output or (n_samples, n_outputs) for multi
    if pred_te.ndim == 1:
        pred_te = pred_te.reshape(-1, 1)
    if y_te.ndim == 1:
        y_te = y_te.reshape(-1, 1)

    # per-target R² (raw_values gives one R² per output column)
    r2_per_target = r2_score(y_te, pred_te, multioutput='raw_values')
    r2_te   = float(np.mean(r2_per_target))
    rmse_te = float(np.sqrt(mean_squared_error(y_te, pred_te)))
    # map per-target R² back to named columns
    r2_by_target = {col: round(float(r2_per_target[i]), 4) for i, col in enumerate(targ_cols)}

    # per-target RMSE (on the scale the model was actually trained/evaluated on)
    rmse_per_target = np.sqrt(np.mean((y_te - pred_te) ** 2, axis=0))
    rmse_by_target = {col: round(float(rmse_per_target[i]), 6) for i, col in enumerate(targ_cols)}

    # for flux columns specifically: invert the sign-preserving log transform
    # back to physical units (cm/day) before computing RMSE, since flux was
    # log-transformed prior to training and rmse_by_target above is in log-space
    rmse_physical_by_target = {}
    for i, col in enumerate(targ_cols):
        if col.startswith('flux'):
            y_phys    = inverse_signed_log(y_te[:, i])
            pred_phys = inverse_signed_log(pred_te[:, i])
            rmse_physical_by_target[col] = round(
                float(np.sqrt(np.mean((y_phys - pred_phys) ** 2))), 6)

    # unique physical sensors at optimal level
    sensors = unique_sensors_up_to_level(tree, feat_names, opt_level)
    n_sensors = len(sensors)

    wc_depths = sorted(d for s, d in sensors if s == 'wc')
    p_depths  = sorted(d for s, d in sensors if s == 'p')

    def fmt_sensor(count, depths):
        if count == 0:
            return ''
        if count == 1:
            return f'{count}wc@{depths[0]}cm' if count == len(wc_depths) else f'{count}p@{depths[0]}cm'
        return f'{count}@{depths[0]}-{depths[-1]}cm'

    wc_str = (f'{len(wc_depths)}wc@{wc_depths[0]}cm' if len(wc_depths) == 1
              else f'{len(wc_depths)}wc@{wc_depths[0]}-{wc_depths[-1]}cm') if wc_depths else ''
    p_str  = (f'{len(p_depths)}p@{p_depths[0]}cm' if len(p_depths) == 1
              else f'{len(p_depths)}p@{p_depths[0]}-{p_depths[-1]}cm') if p_depths else ''
    sensor_str = ', '.join(filter(None, [wc_str, p_str]))
    sensor_summary = f'{n_sensors} ({sensor_str})'

    return dict(
        opt_level    = opt_level,
        var_exp_train= round(var_exp_train, 4),
        r2_test      = round(r2_te, 4),
        r2_by_target = r2_by_target,
        rmse_test    = round(rmse_te, 6),
        rmse_by_target= rmse_by_target,
        rmse_physical_by_target = rmse_physical_by_target,
        total_cost   = round(total_cost, 1),
        sensors      = sensor_summary,
        n_sensors    = n_sensors,
        wc_sensors   = len(wc_depths),
        p_sensors    = len(p_depths),
    )


# ── tree catalogue ───────────────────────────────────────────────────────────

COSTS = [round(c, 1) for c in np.arange(0.0, 1.1, 0.1)]

catalogue = []

# flux — no-cost baseline
catalogue.append(dict(
    objective='flux', cost_thresh=None,
    pkl_dir=RESULTS/'60cm', fname='Flood_beans_flux_60cm',
    targ_cols=['flux_60'],
))

# flux — cost sweep
for c in COSTS:
    catalogue.append(dict(
        objective='flux', cost_thresh=c,
        pkl_dir=RESULTS/'60cm', fname=f'Flood_beans_flux60_cost{c}',
        targ_cols=['flux_60'],
    ))

# wc — no-cost baseline
catalogue.append(dict(
    objective='wc', cost_thresh=None,
    pkl_dir=RESULTS/'avg_wc', fname='Flood_beans_avg_wc_top_half',
    targ_cols=['avg_wc_top_half'],
))

# wc — cost sweep
for c in COSTS:
    catalogue.append(dict(
        objective='wc', cost_thresh=c,
        pkl_dir=RESULTS/'avg_wc', fname=f'Flood_beans_avg_wc_top_half_cost{c}',
        targ_cols=['avg_wc_top_half'],
    ))

# flux+wc — no-cost baseline
catalogue.append(dict(
    objective='flux+wc', cost_thresh=None,
    pkl_dir=RESULTS/'60cm', fname='Flood_beans_flux_60cm_wc',
    targ_cols=['flux_60', 'avg_wc_top_half'],
))

# flux+wc — cost sweep
for c in COSTS:
    catalogue.append(dict(
        objective='flux+wc', cost_thresh=c,
        pkl_dir=RESULTS/'60cm', fname=f'Flood_beans_flux60_wc_cost{c}',
        targ_cols=['flux_60', 'avg_wc_top_half'],
    ))


# ── run ──────────────────────────────────────────────────────────────────────

rows = []
for entry in catalogue:
    tree, rt, df_tr, df_te = load(entry['pkl_dir'], entry['fname'])
    targ_cols = entry['targ_cols']
    feat_cols = [c for c in df_te.columns if c not in targ_cols]
    metrics = evaluate(tree, rt, df_tr, df_te, targ_cols, feat_cols)
    rows.append({
        'objective':   entry['objective'],
        'cost_thresh': 'none' if entry['cost_thresh'] is None else entry['cost_thresh'],
        **metrics,
    })

df = pd.DataFrame(rows)


# ── print tables ─────────────────────────────────────────────────────────────

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 60)
pd.set_option('display.float_format', '{:.4f}'.format)

OUT_DIR = Path(__file__).resolve().parent / 'report_tables'
OUT_DIR.mkdir(exist_ok=True)

for obj in ['flux', 'wc', 'flux+wc']:
    sub = df[df['objective'] == obj].copy()
    sub = sub.reset_index(drop=True)
    print(f'\n{"="*100}')
    print(f'  OBJECTIVE: {obj}')
    print(f'{"="*100}')
    table_cols = ['cost_thresh','opt_level','var_exp_train','r2_test','rmse_test',
                  'total_cost','sensors']

    if obj == 'flux':
        # add RMSE back-transformed into physical units (cm/day), since flux
        # was log-transformed before training and rmse_test above is in log-space
        sub['rmse_test_cm_per_day'] = sub['rmse_physical_by_target'].apply(
            lambda d: d.get('flux_60', float('nan')))
        table_cols = table_cols + ['rmse_test_cm_per_day']

    print(sub[table_cols].to_string(index=False))

    csv_name = f"table_{PLANT_TYPE}_{obj.replace('+', '_')}.csv"
    sub[table_cols].to_csv(OUT_DIR / csv_name, index=False)
    print(f'  [saved -> report_tables/{csv_name}]')

    if obj == 'flux+wc':
        # expand per-target R² and RMSE into separate columns
        sub_split = sub.copy()
        sub_split['r2_flux']   = sub_split['r2_by_target'].apply(lambda d: d.get('flux_60', float('nan')))
        sub_split['r2_wc']     = sub_split['r2_by_target'].apply(lambda d: d.get('avg_wc_top_half', float('nan')))
        sub_split['rmse_flux'] = sub_split['rmse_by_target'].apply(lambda d: d.get('flux_60', float('nan')))
        sub_split['rmse_wc']   = sub_split['rmse_by_target'].apply(lambda d: d.get('avg_wc_top_half', float('nan')))
        sub_split['rmse_flux_cm_per_day'] = sub_split['rmse_physical_by_target'].apply(
            lambda d: d.get('flux_60', float('nan')))
        sub_split['r2_gap (flux-wc)'] = sub_split['r2_flux'] - sub_split['r2_wc']
        breakdown_cols = ['cost_thresh','r2_flux','r2_wc','r2_gap (flux-wc)','rmse_flux',
                           'rmse_flux_cm_per_day','rmse_wc','total_cost','sensors']
        print(f'\n  -- Per-target R² / RMSE breakdown --')
        print(sub_split[breakdown_cols].to_string(index=False))

        sub_split[breakdown_cols].to_csv(OUT_DIR / f'table_{PLANT_TYPE}_flux_wc_breakdown.csv', index=False)
        print(f'  [saved -> report_tables/table_{PLANT_TYPE}_flux_wc_breakdown.csv]')

# ── cross-objective summary at optimal no-cost level ────────────────────────
print(f'\n{"="*90}')
print('  CROSS-OBJECTIVE SUMMARY  (no-cost baseline vs best cost-penalised variant)')
print(f'{"="*90}')

summary_rows = []
for obj in ['flux', 'wc', 'flux+wc']:
    sub = df[df['objective'] == obj]
    baseline = sub[sub['cost_thresh'] == 'none'].iloc[0]

    # "best" cost variant = highest r2_test (excluding baseline)
    cost_sub = sub[sub['cost_thresh'] != 'none']
    best = cost_sub.loc[cost_sub['r2_test'].idxmax()]

    print(f'\n{obj}:')
    print(f"  Baseline (no cost) :  R²={baseline['r2_test']:.4f}  RMSE={baseline['rmse_test']:.6f}  cost={baseline['total_cost']:.1f}  "
          f"sensors={baseline['n_sensors']} (wc={baseline['wc_sensors']} p={baseline['p_sensors']})")
    print(f"  Best cost-penalised:  R²={best['r2_test']:.4f}  RMSE={best['rmse_test']:.6f}  cost={best['total_cost']:.1f}  "
          f"sensors={best['n_sensors']} (wc={best['wc_sensors']} p={best['p_sensors']})  "
          f"[cost_thresh={best['cost_thresh']}]")
    dr2   = best['r2_test']   - baseline['r2_test']
    drmse = best['rmse_test'] - baseline['rmse_test']
    dcost = best['total_cost'] - baseline['total_cost']
    print(f"  Delta               :  ΔR²={dr2:+.4f}  ΔRMSE={drmse:+.6f}  Δcost={dcost:+.1f}  "
          f"Δsensors={best['n_sensors']-baseline['n_sensors']:+d}")

    summary_rows.append({
        'objective': obj, 'variant': 'baseline (no cost)',
        'cost_thresh': 'none', 'r2_test': baseline['r2_test'], 'rmse_test': baseline['rmse_test'],
        'total_cost': baseline['total_cost'], 'n_sensors': baseline['n_sensors'],
        'wc_sensors': baseline['wc_sensors'], 'p_sensors': baseline['p_sensors'],
    })
    summary_rows.append({
        'objective': obj, 'variant': 'best cost-penalised',
        'cost_thresh': best['cost_thresh'], 'r2_test': best['r2_test'], 'rmse_test': best['rmse_test'],
        'total_cost': best['total_cost'], 'n_sensors': best['n_sensors'],
        'wc_sensors': best['wc_sensors'], 'p_sensors': best['p_sensors'],
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / f'table_{PLANT_TYPE}_cross_objective_summary.csv', index=False)
print(f'\n  [saved -> report_tables/table_{PLANT_TYPE}_cross_objective_summary.csv]')
