import pickle as pkl
from pathlib import Path
from sklearn.tree import _tree
import numpy as np

results = Path(__file__).resolve().parent / 'results' / 'Flood' / 'avg_wc'

for cost in [0.0, 0.1, 0.2, 0.3, 0.4]:
    fname = f'Flood_corn_avg_wc_top_half_cost{cost}'
    with open(results / f'tree_{fname}.pkl', 'rb') as f:
        tree = pkl.load(f)
    with open(results / f'rtinfo_{fname}.pkl', 'rb') as f:
        rt = pkl.load(f)
    with open(results / f'df_train_{fname}.pkl', 'rb') as f:
        df_tr = pkl.load(f)

    feat_names = [c for c in df_tr.columns if c != 'avg_wc_top_half']
    t = tree.tree_
    print(f'--- cost={cost} ---')
    print('var_exp by level:  ', rt.var_df_by_depth['var_exp'].round(4).to_dict())
    print('total_cost by level:', rt.var_df_by_depth['total_cost'].round(2).to_dict())

    def walk(node, depth, max_depth=5):
        if t.children_left[node] == _tree.TREE_LEAF or depth > max_depth:
            return
        feat = feat_names[t.feature[node]]
        thresh = t.threshold[node]
        print('  ' * depth + f'L{depth+1} node{node}: {feat} <= {thresh:.4f}')
        walk(t.children_left[node], depth + 1, max_depth)
        walk(t.children_right[node], depth + 1, max_depth)

    walk(0, 0)
    print()
