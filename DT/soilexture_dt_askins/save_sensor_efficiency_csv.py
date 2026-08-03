"""Saves sensor efficiency tables as CSVs in report_tables/."""
import csv, json, subprocess, sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / 'report_tables'
OUT_DIR.mkdir(exist_ok=True)

# re-run the extractor to get the data
result = subprocess.run(
    [sys.executable, str(Path(__file__).resolve().parent / 'extract_sensor_efficiency.py')],
    capture_output=True, text=True
)
data = json.loads(result.stdout)

labels = {
    'flux':   'Flux (200 cm)',
    'wc':     'Water Content (avg root zone)',
    'fluxwc': 'Flux + Water Content (combined)',
}

FIELDS = ['rank', 'sensor', 'type', 'importance_pct', 'standalone_cost', 'efficiency', 'n_trees_of_12']

for key, sensors in data.items():
    fname = OUT_DIR / f'table_corn_sensor_efficiency_{key}.csv'
    with open(fname, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, s in enumerate(sensors, 1):
            w.writerow({
                'rank':             i,
                'sensor':           s['label'],
                'type':             'Water content' if s['type'] == 'wc' else 'Pressure',
                'importance_pct':   round(s['importance'], 3),
                'standalone_cost':  s['cost'],
                'efficiency':       round(s['importance'] / s['cost'], 4),
                'n_trees_of_12':    s['n_trees'],
            })
    print(f'Saved: {fname.name}')
