import pickle as pkl
from pathlib import Path

base = Path(__file__).resolve().parent / 'results' / 'Flood'

print('=== BASELINE TREES (standard sklearn) ===')
baseline = {
    'corn flux':     base / '200cm'  / 'tree_Flood_corn_flux_200cm.pkl',
    'corn wc':       base / 'avg_wc' / 'tree_Flood_corn_avg_wc_top_half.pkl',
    'beans flux':    base / '60cm'   / 'tree_Flood_beans_flux_60cm.pkl',
    'beans wc':      base / 'avg_wc' / 'tree_Flood_beans_avg_wc_top_half.pkl',
}
print(f'{"Tree":<18} {"Root MSE":>12} {"n":>6} {"Total SSE":>12} {"n_nodes":>8}')
print('-' * 62)
for label, path in baseline.items():
    with open(path, 'rb') as f:
        tree = pkl.load(f)
    t = tree.tree_
    mse = t.impurity[0]
    n   = t.n_node_samples[0]
    sse = mse * n
    print(f'{label:<18} {mse:>12.8f} {n:>6} {sse:>12.6f} {t.node_count:>8}')

print()
print('=== COST TREES (modified sklearn) ===')
cost = {
    'corn flux c=0.5':  base / '200cm'  / 'tree_Flood_corn_flux200_cost0.5.pkl',
    'corn flux c=1.0':  base / '200cm'  / 'tree_Flood_corn_flux200_cost1.0.pkl',
    'corn wc c=0.5':    base / 'avg_wc' / 'tree_Flood_corn_avg_wc_top_half_cost0.5.pkl',
    'beans flux c=0.5': base / '60cm'   / 'tree_Flood_beans_flux60_cost0.5.pkl',
    'beans wc c=0.5':   base / 'avg_wc' / 'tree_Flood_beans_avg_wc_top_half_cost0.5.pkl',
}
print(f'{"Tree":<22} {"Root MSE":>14} {"n":>6} {"Total SSE":>14} {"n_nodes":>8}')
print('-' * 68)
for label, path in cost.items():
    with open(path, 'rb') as f:
        tree = pkl.load(f)
    t = tree.tree_
    mse = t.impurity[0]
    n   = t.n_node_samples[0]
    sse = mse * n
    print(f'{label:<22} {mse:>14.4f} {n:>6} {sse:>14.2f} {t.node_count:>8}')
    print(f'  {"2500 / SSE":>20} = {2500/sse*100:.2f}% of total SSE')
