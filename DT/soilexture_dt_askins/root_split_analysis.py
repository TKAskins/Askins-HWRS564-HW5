"""
root_split_analysis.py
Ad-hoc analysis: for each crop x objective, look at every tree's root split
(the single split that explains the most variance) and check whether the
same physical sensor location dominates across the 12 cost configurations,
and how much of the total variance that first split alone explains.
"""
import pickle as pkl
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

RESULTS = Path(__file__).resolve().parent / 'results' / 'Flood'
COSTS = [round(c, 1) for c in np.arange(0.0, 1.1, 0.1)]

EXCLUDE = ('flux_200', 'avg_wc_top_half', 'flux_60')


def load(pkl_dir, fname):
    with open(pkl_dir / f'tree_{fname}.pkl', 'rb') as f:
        tree = pkl.load(f)
    with open(pkl_dir / f'rtinfo_{fname}.pkl', 'rb') as f:
        rt = pkl.load(f)
    with open(pkl_dir / f'df_train_{fname}.pkl', 'rb') as f:
        df = pkl.load(f)
    return tree, rt, df


def root_info(pkl_dir, fname):
    try:
        tree, rt, df = load(pkl_dir, fname)
    except FileNotFoundError:
        return None
    feat_names = [c for c in df.columns if c not in EXCLUDE]
    root_feat_idx = tree.tree_.feature[0]
    root_feat = feat_names[root_feat_idx]
    var_exp_root = float(rt.var_df_by_depth.loc[1, 'var_exp'])
    opt_level = int(np.argmax(rt.var_df_by_depth['var_exp'])) + 1
    var_exp_opt = float(rt.var_df_by_depth.loc[opt_level, 'var_exp'])
    return root_feat, var_exp_root, var_exp_opt, opt_level


def catalogue_for(crop, flux_depth, flux_dir):
    cat = []
    cat.append(('flux', RESULTS / flux_dir, f'Flood_{crop}_flux_{flux_depth}cm'))
    for c in COSTS:
        cat.append(('flux', RESULTS / flux_dir, f'Flood_{crop}_flux{flux_depth}_cost{c}'))
    cat.append(('wc', RESULTS / 'avg_wc', f'Flood_{crop}_avg_wc_top_half'))
    for c in COSTS:
        cat.append(('wc', RESULTS / 'avg_wc', f'Flood_{crop}_avg_wc_top_half_cost{c}'))
    cat.append(('flux_wc', RESULTS / flux_dir, f'Flood_{crop}_flux_{flux_depth}cm_wc'))
    for c in COSTS:
        cat.append(('flux_wc', RESULTS / flux_dir, f'Flood_{crop}_flux{flux_depth}_wc_cost{c}'))
    return cat


summary_rows = []
detail_rows = []

for crop, flux_depth, flux_dir in [('corn', '200', '200cm'), ('beans', '60', '60cm')]:
    print(f'\n{"="*90}\nCROP: {crop}\n{"="*90}')
    cat = catalogue_for(crop, flux_depth, flux_dir)

    by_obj = defaultdict(list)
    for obj, pkl_dir, fname in cat:
        info = root_info(pkl_dir, fname)
        if info is None:
            continue
        by_obj[obj].append((fname, *info))

    for obj in ['flux', 'wc', 'flux_wc']:
        entries = by_obj[obj]
        root_counter = Counter(e[1] for e in entries)
        avg_var_exp_root = np.mean([e[2] for e in entries])
        avg_var_exp_opt = np.mean([e[3] for e in entries])
        top_sensor, top_count = root_counter.most_common(1)[0]
        print(f'\n  Objective: {obj}  ({len(entries)} trees)')
        print(f'  Root-split sensor frequency: {root_counter.most_common()}')
        print(f'  Mean var_exp from root split alone: {avg_var_exp_root:.4f}')
        print(f'  Mean var_exp at optimal depth      : {avg_var_exp_opt:.4f}')
        print(f'  Root split as fraction of optimal-depth var_exp: {avg_var_exp_root/avg_var_exp_opt:.4f}')
        for fname, root_feat, ver, veo, lvl in entries:
            print(f'    {fname:<45} root={root_feat:<14} var_exp_root={ver:.4f}  var_exp_opt={veo:.4f} (lvl {lvl})')
            detail_rows.append(dict(
                crop=crop, objective=obj, config=fname, root_sensor=root_feat,
                var_exp_root=round(ver, 4), var_exp_opt=round(veo, 4), opt_level=lvl,
            ))

        summary_rows.append(dict(
            crop=crop, objective=obj, n_trees=len(entries),
            dominant_root_sensor=top_sensor, dominant_count=top_count,
            mean_var_exp_root=round(avg_var_exp_root, 4),
            mean_var_exp_opt=round(avg_var_exp_opt, 4),
            root_share_of_opt=round(avg_var_exp_root / avg_var_exp_opt, 4),
        ))

# ── save to Excel ────────────────────────────────────────────────────────────

out_dir = Path(__file__).resolve().parent / 'report_tables'
out_dir.mkdir(exist_ok=True)
xlsx_path = out_dir / 'table_root_split_analysis.xlsx'

summary_df = pd.DataFrame(summary_rows)
detail_df = pd.DataFrame(detail_rows)

with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    summary_df.to_excel(writer, sheet_name='summary', index=False)
    detail_df.to_excel(writer, sheet_name='detail', index=False)

print(f'\n[saved -> report_tables/{xlsx_path.name}]')
