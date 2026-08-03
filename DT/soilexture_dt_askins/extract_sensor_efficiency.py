"""
extract_sensor_efficiency.py
For each corn objective, aggregates feature importance by physical sensor location
(type + depth) across the full cost-threshold sweep, then computes standalone
sensor cost so both can be plotted together.
"""
import json, pickle as pkl, math
from pathlib import Path
import numpy as np
from collections import defaultdict

RESULTS = Path(__file__).resolve().parent / 'results' / 'Flood'
COSTS   = [round(c, 1) for c in np.arange(0.0, 1.1, 0.1)]

# ── cost model (matching training) ───────────────────────────────────────────
INITIAL_COST     = {'wc': 4.0, 'p': 10.0}
SENSOR_COST      = {'wc': 2.0, 'p': 1.0}
MEASUREMENT_COST = {'wc': 0.1, 'p': 0.1}

def depth_cost(d):
    return 0.2 * math.ceil(d / 5)

def standalone_cost(stype, depth):
    """Cost of adding a single sensor to an otherwise empty network."""
    return (INITIAL_COST[stype] + SENSOR_COST[stype]
            + depth_cost(depth) + MEASUREMENT_COST[stype])

# ── helpers ───────────────────────────────────────────────────────────────────
def load(pkl_dir, fname):
    def _load(tag):
        with open(pkl_dir / f'{tag}_{fname}.pkl', 'rb') as f:
            return pkl.load(f)
    return _load('tree'), _load('rtinfo'), _load('df_train')

def sensor_importances(tree, feat_names):
    """Return dict {(stype, depth): total_importance} from sklearn feature_importances_."""
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

def opt_r2(rt, df_te, tree, targ_cols):
    """Return test R² at optimal level (uses training var_exp to pick level)."""
    from sklearn.metrics import r2_score
    from sklearn.tree import _tree
    opt_level = int(np.argmax(rt.var_df_by_depth['var_exp'])) + 1
    feat_names = [c for c in df_te.columns if c not in targ_cols]
    X = df_te[feat_names].to_numpy()
    n = X.shape[0]
    nodes = np.zeros(n, dtype=int)
    t = tree.tree_
    for i in range(n):
        node, depth = 0, 0
        while t.children_left[node] != _tree.TREE_LEAF and depth < opt_level:
            depth += 1
            node = (t.children_left[node] if X[i, t.feature[node]] <= t.threshold[node]
                    else t.children_right[node])
        nodes[i] = node
    pred = t.value[nodes].squeeze(-1)
    if pred.ndim == 2: pred = pred[:, 0]
    y = df_te[targ_cols[0]].to_numpy()
    return float(r2_score(y, pred))

# ── define trees ──────────────────────────────────────────────────────────────
objectives = {
    'flux': {
        'dir': RESULTS / '200cm',
        'baseline': 'Flood_corn_flux_200cm',
        'sweep':    ['Flood_corn_flux200_cost{c}'.format(c=c) for c in COSTS],
        'targ':     'flux_200',
    },
    'wc': {
        'dir': RESULTS / 'avg_wc',
        'baseline': 'Flood_corn_avg_wc_top_half',
        'sweep':    ['Flood_corn_avg_wc_top_half_cost{c}'.format(c=c) for c in COSTS],
        'targ':     'avg_wc_top_half',
    },
    'flux_wc': {
        'dir': RESULTS / '200cm',
        'baseline': 'Flood_corn_flux_200cm_wc',
        'sweep':    ['Flood_corn_flux200_wc_cost{c}'.format(c=c) for c in COSTS],
        'targ':     'flux_200',   # rtinfo is keyed on flux
    },
}

# ── aggregate importances over all 12 trees per objective ─────────────────────
output = {}

for obj_key, cfg in objectives.items():
    fnames = [cfg['baseline']] + cfg['sweep']
    targ   = cfg['targ']
    pkl_dir = cfg['dir']

    # accumulate weighted importance for each sensor location
    weighted_imp  = defaultdict(float)   # (stype, depth) -> sum(r2 * importance)
    weight_total  = 0.0
    sensor_appears = defaultdict(int)    # how many trees used this sensor

    for fname in fnames:
        try:
            tree, rt, df_tr = load(pkl_dir, fname)
        except FileNotFoundError:
            continue

        feat_names = [c for c in df_tr.columns
                      if c not in ('flux_200','avg_wc_top_half','flux_60')]

        # use feature_importances_ from the FULL tree (not truncated)
        imp_map = sensor_importances(tree, feat_names)

        # weight by training var_exp at optimal level (proxy for model quality)
        opt_level = int(np.argmax(rt.var_df_by_depth['var_exp'])) + 1
        quality   = float(rt.var_df_by_depth.loc[opt_level, 'var_exp'])
        if quality <= 0:
            continue

        for sensor, imp in imp_map.items():
            if imp > 0:
                weighted_imp[sensor]   += quality * imp
                sensor_appears[sensor] += 1
        weight_total += quality

    if weight_total == 0:
        continue

    # normalise
    sensors = []
    for (stype, depth), wimp in weighted_imp.items():
        avg_imp = wimp / weight_total          # weighted-average importance
        cost    = standalone_cost(stype, depth)
        efficiency = avg_imp / cost if cost > 0 else 0
        sensors.append({
            'label':      f'{stype}@{depth}cm',
            'type':       stype,
            'depth':      depth,
            'importance': round(avg_imp * 100, 3),   # as %
            'cost':       round(cost, 2),
            'efficiency': round(efficiency * 100, 4),
            'n_trees':    sensor_appears[(stype, depth)],
        })

    # top 10 by efficiency
    top10 = sorted(sensors, key=lambda x: x['efficiency'], reverse=True)[:10]
    # also sort by efficiency for display
    output[obj_key] = sorted(top10, key=lambda x: x['efficiency'], reverse=True)

print(json.dumps(output, indent=2))
