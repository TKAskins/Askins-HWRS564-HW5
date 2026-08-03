# run_stability_iterations.py
# Re-runs the full DT pipeline (all six run_dt_flood_* stages) with different
# train/test split seeds, to check that results/Flood/ is stable across
# independent draws of the same source data. Writes to sibling folders
# results/Flood_iter2/ and results/Flood_iter3/ so the original run under
# results/Flood/ (split_random_state=0) is never touched.

from dt_cost_by_level_main import run_all

if __name__ == '__main__':
    run_all(results_dirname='Flood_iter2', split_random_state=1)
    run_all(results_dirname='Flood_iter3', split_random_state=2)
    print('Stability iterations done.')
