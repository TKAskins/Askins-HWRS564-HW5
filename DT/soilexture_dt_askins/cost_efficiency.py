# cost_efficiency.py
# Marginal cost per unit of variance explained, by tree depth level, for the
# beans no-cost baselines and the full cost-threshold sweep. Uses the
# total_cost / var_exp columns already computed in each tree's rtinfo pickle
# (RegressionTreeInfo.var_df_by_depth) -- no refitting needed.
#
# marginal_cost      = total_cost[level] - total_cost[level-1]
# marginal_var_exp   = var_exp[level] - var_exp[level-1]
# cost_per_var_exp   = marginal_cost / marginal_var_exp   (NA if marginal_var_exp <= 0:
#                       that split added cost without explaining more variance)

import pickle as pkl
from pathlib import Path
import numpy as np

RESULTS = Path(__file__).resolve().parent / 'results' / 'Flood'
PLANT_TYPE = 'beans'
DEPTH = 60
COSTS = [round(c, 1) for c in np.arange(0.0, 1.1, 0.1)]


def load_pkl(p):
    with open(p, 'rb') as f:
        return pkl.load(f)


def marginal_table(rtinfo):
    vdbd = rtinfo.var_df_by_depth
    levels = vdbd.index.values
    var_exp = vdbd['var_exp'].values.astype(float)
    total_cost = vdbd['total_cost'].values.astype(float)

    var_exp_prev = np.concatenate([[0.0], var_exp[:-1]])
    cost_prev = np.concatenate([[0.0], total_cost[:-1]])

    d_var = var_exp - var_exp_prev
    d_cost = total_cost - cost_prev
    ratio = np.where(d_var > 0, d_cost / np.where(d_var > 0, d_var, 1), np.nan)

    return levels, var_exp, total_cost, d_var, d_cost, ratio


def print_full_table(label, rtinfo):
    levels, var_exp, total_cost, d_var, d_cost, ratio = marginal_table(rtinfo)
    print(f'\n--- {label} ---')
    print('  lvl   var_exp   Δvar_exp   total_cost   Δcost   Δcost/Δvar_exp')
    for lvl, ve, tc, dv, dc, r in zip(levels, var_exp, total_cost, d_var, d_cost, ratio):
        r_str = f'{r:9.2f}' if not np.isnan(r) else '     -   '
        print(f'  {lvl:3d}   {ve:7.4f}   {dv:8.4f}   {tc:10.3f}   {dc:6.3f}   {r_str}')


def summarize_sweep(label, subdir, fname_tmpl):
    print(f'\n--- {label}: cost sweep summary (at each tree\'s var_exp-peak level) ---')
    print('  cost   peak_lvl   var_exp   total_cost   last_step_Δcost/Δvar_exp')
    for cost in COSTS:
        fname = fname_tmpl.format(cost)
        try:
            rtinfo = load_pkl(RESULTS / subdir / f'rtinfo_{fname}.pkl')
        except FileNotFoundError:
            print(f'  {cost:.1f}   MISSING')
            continue
        levels, var_exp, total_cost, d_var, d_cost, ratio = marginal_table(rtinfo)
        peak_i = int(np.argmax(var_exp))
        r_str = f'{ratio[peak_i]:9.2f}' if not np.isnan(ratio[peak_i]) else '     -   '
        print(f'  {cost:.1f}   {levels[peak_i]:6d}   {var_exp[peak_i]:7.4f}   '
              f'{total_cost[peak_i]:10.3f}   {r_str}')


baselines = [
    ('flux (no cost)', f'{DEPTH}cm', f'Flood_{PLANT_TYPE}_flux_{DEPTH}cm'),
    ('wc (no cost)', 'avg_wc', f'Flood_{PLANT_TYPE}_avg_wc_top_half'),
    ('flux+wc (no cost)', f'{DEPTH}cm', f'Flood_{PLANT_TYPE}_flux_{DEPTH}cm_wc'),
]

print('=' * 78)
print('BASELINE TREES — marginal cost per unit variance explained, by level')
print('=' * 78)
for label, subdir, fname in baselines:
    rtinfo = load_pkl(RESULTS / subdir / f'rtinfo_{fname}.pkl')
    print_full_table(label, rtinfo)

print('\n' + '=' * 78)
print('COST-SWEEP TREES — marginal cost per unit variance explained')
print('=' * 78)

sweeps = [
    ('flux', f'{DEPTH}cm', f'Flood_{PLANT_TYPE}_flux{DEPTH}_cost{{}}'),
    ('wc', 'avg_wc', f'Flood_{PLANT_TYPE}_avg_wc_top_half_cost{{}}'),
    ('flux+wc', f'{DEPTH}cm', f'Flood_{PLANT_TYPE}_flux{DEPTH}_wc_cost{{}}'),
]

for label, subdir, fname_tmpl in sweeps:
    summarize_sweep(label, subdir, fname_tmpl)

# full per-level detail for the two ends of the sweep, for reference
print('\n' + '=' * 78)
print('COST-SWEEP TREES — full per-level detail at cost=0.0 and cost=1.0')
print('=' * 78)
for label, subdir, fname_tmpl in sweeps:
    for cost in [0.0, 1.0]:
        fname = fname_tmpl.format(cost)
        try:
            rtinfo = load_pkl(RESULTS / subdir / f'rtinfo_{fname}.pkl')
        except FileNotFoundError:
            continue
        print_full_table(f'{label} cost={cost}', rtinfo)
