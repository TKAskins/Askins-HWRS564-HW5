# dt_cost_by_level_main.py
# Derek Groenendyk
# Written: 2019/06/07
# Modifed: 2026/02/05 by Trevor Askins
# decision tree training and analysis for cost using class object


from collections import OrderedDict
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle as pkl
import sklearn as sklearn
from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from dt_cost_by_level_class import RegressionTreeInfo

print(sklearn.__file__)
print(sklearn.__version__)

# Set to 'corn', 'beans', or 'multi' to select which CSV to train on.
# 'multi' combines all available Flood CSVs; single plant types load only their own CSV.
PLANT_TYPE = 'beans'


def apply(X, tree, max_depth=None):
    """Traverse tree to leaf node, stopping at max_depth if specified."""
    if max_depth is None:
        max_depth = tree.max_depth
    n_samples = X.shape[0]
    nodes = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        node = 0
        depth = 0
        while tree.children_left[node] != _tree.TREE_LEAF and depth <= max_depth:
            depth += 1
            if X[i, tree.feature[node]] <= tree.threshold[node]:
                node = tree.children_left[node]
            else:
                node = tree.children_right[node]
        nodes[i] = node
    return nodes


def predict(X, tree, max_depth):
    """Predict using tree truncated at max_depth."""
    if max_depth is None:
        max_depth = tree.max_depth
    nodes = apply(X, tree, max_depth)
    return tree.value.take(nodes)


def signed_log(x):
    """Sign-preserving log transform; finite for negative, zero, and positive x."""
    return np.sign(x) * np.log1p(np.abs(x))


def get_rt_pred(directory, fname, level=None, auto_lvl=True, return_pred=False):

    if True:
        pkl_file = open(str(directory / ('df_train_' + fname + '.pkl')), 'rb')
        df_train = pkl.load(pkl_file)
        pkl_file.close()

    if True:
        pkl_file = open(str(directory / ('df_test_' + fname + '.pkl')), 'rb')
        df_test = pkl.load(pkl_file)
        pkl_file.close()

    if True:
        pkl_file = open(str(directory / ('tree_' + fname + '.pkl')), 'rb')
        reg_tree = pkl.load(pkl_file)
        pkl_file.close()

    col_names = df_test.columns.values
    feat_names = [cname for cname in col_names if 'flux' not in cname]
    targ_names = [cname for cname in col_names if 'flux' in cname]  

    num_targets = len(targ_names)

    if True and num_targets == 1:
        target_name = targ_names[0] 

        pkl_file = open(str(directory / ('rtinfo_' + fname + '.pkl')), 'rb')
        rtinfo = pkl.load(pkl_file)
        pkl_file.close()



    # print(col_names)          

    df_train_feats = df_train[feat_names]
    df_train_targs = df_train[targ_names]
    df_test_feats = df_test[feat_names]
    df_test_targs = df_test[targ_names] 


    # print(df_train)
    # print(df_test)


    if auto_lvl:
        level = np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1
        # print(fname, level)

    pred_train = predict(df_train_feats.to_numpy(), reg_tree.tree_, max_depth=level)
    pred_test = predict(df_test_feats.to_numpy(), reg_tree.tree_, max_depth=level)

    r2_train = r2_score(df_train_targs.to_numpy(), pred_train)          
    r2_test = r2_score(df_test_targs.to_numpy(), pred_test)     

    rmse_train = np.sqrt(mean_squared_error(df_train_targs.to_numpy(), pred_train))
    rmse_test = np.sqrt(mean_squared_error(df_test_targs.to_numpy(), pred_test))

    if not return_pred:
        return r2_train, r2_test, rmse_train, rmse_test
    else:
        return pred_train, pred_test, df_train_targs, df_test_targs


def collapse_feat_midx(df, time_offset = 0):

    idx = pd.IndexSlice

    df_new = df.stack(0,future_stack=True).T
    df_new = df_new.reorder_levels([2, 1, 0], axis=1)
    df_new.columns = df_new.columns.to_flat_index()

    cols = []
    for col in df_new.columns.values:
        cols.append(':'.join([col[0], 'd' + str(col[1]), 't' + str(col[2] - time_offset)]))
    df_new.columns = cols   

    return df_new


def collapse_targ_midx(df):

    idx = pd.IndexSlice

    df_new = df.stack(0).T
    # df_new = df_new.reorder_levels([2, 1, 0], axis=1)
    df_new.columns = df_new.columns.to_flat_index()

    cols = [':'.join([col[0], 'd' + str(col[1]), 't' + str(col[2])]) for \
        col in df_new.columns.values]
    df_new.columns = cols   

    return df_new   


def create_input_df(df_train, df_test, times, depths, params):

    idx = pd.IndexSlice

    df_train_new = collapse_feat_midx(df_train.loc[idx[times, depths], (params)])
    df_test_new = collapse_feat_midx(df_test.loc[idx[times, depths], (params)])

    # print(df_train_new.columns)

    # df_train_new = pd.concat([df_train_new, df_train_targets], axis=1)
    # df_test_new = pd.concat([df_test_new, df_test_targets], axis=1)

    df_train_new.sort_index(inplace=True)
    df_test_new.sort_index(inplace=True)

    return df_train_new, df_test_new   


def analyze_dt(results_dir, results_fname, trial_name, target_name):

    '''
    if you have already created a decision tree and want to analyze it,
    you only need the following inputs. Where:
    reg_tree is the existing regression tree
    df is the DataFrame containing the features and the target.
    targ_name is the name of the column containing the target values
    trial_name can be defined freely.
    ''' 

    print_flag = False

    pkl_file = open(str(results_dir / ('tree_' + results_fname + '.pkl')), 'rb')
    reg_tree = pkl.load(pkl_file)
    pkl_file.close()


    pkl_file = open(str(results_dir / ('df_train_' + results_fname + '.pkl')), 'rb')
    df_train = pkl.load(pkl_file)
    pkl_file.close()

    # target_name = 'flux_' + str(depth)


    initial_cost = OrderedDict(zip(['wc','p'],[4.0,10.0]))
    sensor_cost = OrderedDict(zip(['wc','p'],[2.0,1.0]))
    depth_cost = OrderedDict(zip([str(depth) for depth in range(5,55,5)],[0.2 * np.ceil(depth / 5) for depth in range(5,55,5)]))
    measurement_cost = OrderedDict(zip(['wc','p'],[0.1,0.1]))

    rtinfo = RegressionTreeInfo(reg_tree, df_train, target_name,
        trial_name, print_flag=print_flag,
        initial_cost=initial_cost,
        sensor_cost=sensor_cost,
        depth_cost=depth_cost,
        measurement_cost=measurement_cost)

    return rtinfo



def obs_split(df, labels, random_state=0):

    ltrain, ltest = train_test_split(labels, test_size=0.2, random_state=random_state)

    idx = pd.IndexSlice

    df_train = df.loc[idx[:, :], idx[:, ltrain]]
    df_test = df.loc[idx[:, :], idx[:, ltest]]

    return df_train, df_test


def fit_sklearn(df, target_name,max_depth=None,cost_flag=False,cost_threshold=0.0,
    imp_threshold=0.0,
    initial_cost=[0.0],
    sensor_cost=[0.0],
    depth_cost=[0.0],
    measurement_cost=[0.0]
    ):

    feature_dict = df.drop(target_name, axis=1)

    if cost_flag:
        reg_tree = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=0,
            initial_cost=initial_cost,
            sensor_cost=sensor_cost,
            depth_cost=depth_cost,
            measurement_cost=measurement_cost,
            cost_threshold=cost_threshold,
            imp_threshold=imp_threshold,
            new_version_flag=True
        )

    else:
        reg_tree = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=0,
        )


    print('tree instantiated')

    # tree_final = reg_tree.fit(feature_dict, df[target_name])

    tree_final = reg_tree.fit(feature_dict, df[target_name])

    print('tree fit')

    return tree_final



def run_dt_flood_flux(results_dirname='Flood', split_random_state=0):

    repo_dir     = Path(__file__).resolve().parent.parent.parent
    hydrus_dir   = repo_dir / 'Hydrus' / 'python_hydrus_code'

    # auto-detect CSVs: Flood_<planttype>_<unit>.csv
    csv_files = sorted(hydrus_dir.glob(f'Flood_{PLANT_TYPE}_*.csv')) if PLANT_TYPE != 'multi' else sorted(hydrus_dir.glob('Flood_*_*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No Flood CSV found in {hydrus_dir}')

    if len(csv_files) > 1:
        # multiple plant types â€” combine all CSVs and use full depth list
        # depths 50/150 cm are not on the 4 cm grid; nearest available are 48/148
        df = pd.concat(
            [pd.read_csv(str(f), header=[0, 1], index_col=[0, 1]) for f in csv_files],
            axis=1)
        plant_type = 'multi'
        depth_list = [48, 100, 148, 200]
    else:
        df = pd.read_csv(str(csv_files[0]), header=[0, 1], index_col=[0, 1])
        plant_type = csv_files[0].stem.split('_')[1]
        depth_list = [200] if plant_type == 'corn' else [60]

    results_base = Path(__file__).resolve().parent / 'results' / results_dirname

    print_flag = False
    log_target = True
    max_depth  = 10

    initial_cost     = OrderedDict(zip(['wc','p'], [4.0, 10.0]))
    sensor_cost      = OrderedDict(zip(['wc','p'], [2.0, 1.0]))
    depth_cost       = OrderedDict(zip([str(d) for d in range(4, 204, 4)],
                                       [0.2 * np.ceil(d / 5) for d in range(4, 204, 4)]))
    measurement_cost = OrderedDict(zip(['wc','p'], [0.1, 0.1]))

    labels = df.columns.get_level_values('trial').unique().tolist()
    df_train_full, df_test_full = obs_split(df, labels, random_state=split_random_state)

    idx           = pd.IndexSlice
    depth_list    = [200] if plant_type == 'corn' else [60]
    obs_depths    = range(4, 204, 4)
    feature_types = ['wc', 'p']
    trial_name    = 'Flood'

    for adepth in depth_list:

        targ_depths = [adepth]
        targ_names  = ['flux_' + str(adepth)]
        targ_folder = str(adepth) + 'cm'

        df_train_targets = df_train_full.loc[idx[:, targ_depths], ('f')].T
        target_names = ['flux_' + str(col[1]) for col in df_train_targets.columns.values]
        df_train_targets.columns = target_names
        df_train_targets = df_train_targets.sum(axis=1)

        df_test_targets = df_test_full.loc[idx[:, targ_depths], ('f')].T
        target_names_test = ['flux_' + str(col[1]) for col in df_test_targets.columns.values]
        df_test_targets.columns = target_names_test
        df_test_targets = df_test_targets.sum(axis=1)

        df_train_targets.name = targ_names[0]
        df_test_targets.name  = targ_names[0]

        if log_target:
            df_train_targets = df_train_targets.apply(signed_log)
            df_test_targets  = df_test_targets.apply(signed_log)

        df_train = collapse_feat_midx(df_train_full.loc[idx[:, obs_depths], feature_types])
        df_test  = collapse_feat_midx(df_test_full.loc[idx[:, obs_depths], feature_types])

        df_train = pd.concat([df_train, df_train_targets], axis=1)
        df_test  = pd.concat([df_test,  df_test_targets],  axis=1)

        nobs_dir = results_base / targ_folder
        nobs_dir.mkdir(parents=True, exist_ok=True)

        results_fname = f'{trial_name}_{plant_type}_flux_{adepth}cm'

        reg_tree = fit_sklearn(df_train, targ_names, max_depth=max_depth)

        target_name = targ_names[0]
        rtinfo = RegressionTreeInfo(reg_tree, df_train, target_name, trial_name,
            print_flag=print_flag,
            initial_cost=initial_cost,
            sensor_cost=sensor_cost,
            depth_cost=depth_cost,
            measurement_cost=measurement_cost)

        with open(str(nobs_dir / ('rtinfo_'   + results_fname + '.pkl')), 'wb') as f:
            pkl.dump(rtinfo,    f)
        with open(str(nobs_dir / ('df_train_' + results_fname + '.pkl')), 'wb') as f:
            pkl.dump(df_train,  f)
        with open(str(nobs_dir / ('df_test_'  + results_fname + '.pkl')), 'wb') as f:
            pkl.dump(df_test,   f)
        with open(str(nobs_dir / ('tree_'     + results_fname + '.pkl')), 'wb') as f:
            pkl.dump(reg_tree,  f)

        print(f'Depth {adepth} cm done.')

    print('Done...')


def run_dt_flood_flux_cost(results_dirname='Flood', split_random_state=0):

    repo_dir   = Path(__file__).resolve().parent.parent.parent
    hydrus_dir = repo_dir / 'Hydrus' / 'python_hydrus_code'

    csv_files = sorted(hydrus_dir.glob(f'Flood_{PLANT_TYPE}_*.csv')) if PLANT_TYPE != 'multi' else sorted(hydrus_dir.glob('Flood_*_*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No Flood CSV found in {hydrus_dir}')

    if len(csv_files) > 1:
        df = pd.concat(
            [pd.read_csv(str(f), header=[0, 1], index_col=[0, 1]) for f in csv_files],
            axis=1)
        plant_type = 'multi'
        depth_list = [48, 100, 148, 200]
    else:
        df = pd.read_csv(str(csv_files[0]), header=[0, 1], index_col=[0, 1])
        plant_type = csv_files[0].stem.split('_')[1]
        depth_list = [200] if plant_type == 'corn' else [60]

    results_base = Path(__file__).resolve().parent / 'results' / results_dirname

    print_flag        = False
    log_target        = True
    max_depth         = 10
    imp_threshold     = 2500.0
    cost_threshold_list = [round(ac, 1) for ac in np.arange(0.0, 1.1, 0.1)]

    initial_cost     = OrderedDict(zip(['wc', 'p'], [4.0, 10.0]))
    sensor_cost      = OrderedDict(zip(['wc', 'p'], [2.0, 1.0]))
    depth_cost       = OrderedDict(zip([str(d) for d in range(4, 204, 4)],
                                       [0.2 * np.ceil(d / 5) for d in range(4, 204, 4)]))
    measurement_cost = OrderedDict(zip(['wc', 'p'], [0.1, 0.1]))

    labels = df.columns.get_level_values('trial').unique().tolist()
    df_train_full, df_test_full = obs_split(df, labels, random_state=split_random_state)

    idx           = pd.IndexSlice
    obs_depths    = range(4, 204, 4)
    feature_types = ['wc', 'p']
    trial_name    = 'Flood'

    for adepth in depth_list:

        targ_depths = [adepth]
        targ_names  = ['flux_' + str(adepth)]
        targ_folder = str(adepth) + 'cm'

        df_train_targets = df_train_full.loc[idx[:, targ_depths], ('f')].T
        target_names = ['flux_' + str(col[1]) for col in df_train_targets.columns.values]
        df_train_targets.columns = target_names
        df_train_targets = df_train_targets.sum(axis=1)

        df_test_targets = df_test_full.loc[idx[:, targ_depths], ('f')].T
        target_names_test = ['flux_' + str(col[1]) for col in df_test_targets.columns.values]
        df_test_targets.columns = target_names_test
        df_test_targets = df_test_targets.sum(axis=1)

        df_train_targets.name = targ_names[0]
        df_test_targets.name  = targ_names[0]

        if log_target:
            df_train_targets = df_train_targets.apply(signed_log)
            df_test_targets  = df_test_targets.apply(signed_log)

        df_train = collapse_feat_midx(df_train_full.loc[idx[:, obs_depths], feature_types])
        df_test  = collapse_feat_midx(df_test_full.loc[idx[:, obs_depths], feature_types])

        df_train = pd.concat([df_train, df_train_targets], axis=1)
        df_test  = pd.concat([df_test,  df_test_targets],  axis=1)

        nobs_dir = results_base / targ_folder
        nobs_dir.mkdir(parents=True, exist_ok=True)

        for acost in cost_threshold_list:

            results_fname = f'{trial_name}_{plant_type}_flux{adepth}_cost{acost}'

            reg_tree = fit_sklearn(df_train, targ_names, max_depth=max_depth,
                cost_flag=True,
                cost_threshold=acost,
                imp_threshold=imp_threshold,
                initial_cost=list(initial_cost.values()),
                sensor_cost=list(sensor_cost.values()),
                depth_cost=list(depth_cost.values()),
                measurement_cost=list(measurement_cost.values()))

            target_name = targ_names[0]
            rtinfo = RegressionTreeInfo(reg_tree, df_train, target_name, trial_name,
                print_flag=print_flag,
                initial_cost=initial_cost,
                sensor_cost=sensor_cost,
                depth_cost=depth_cost,
                measurement_cost=measurement_cost)

            with open(str(nobs_dir / ('rtinfo_'   + results_fname + '.pkl')), 'wb') as f:
                pkl.dump(rtinfo,   f)
            with open(str(nobs_dir / ('df_train_' + results_fname + '.pkl')), 'wb') as f:
                pkl.dump(df_train, f)
            with open(str(nobs_dir / ('df_test_'  + results_fname + '.pkl')), 'wb') as f:
                pkl.dump(df_test,  f)
            with open(str(nobs_dir / ('tree_'     + results_fname + '.pkl')), 'wb') as f:
                pkl.dump(reg_tree, f)

            print(f'Depth {adepth} cm, cost {acost} done.')

    print('Done...')


def run_dt_flood_wc(results_dirname='Flood', split_random_state=0):

    repo_dir   = Path(__file__).resolve().parent.parent.parent
    hydrus_dir = repo_dir / 'Hydrus' / 'python_hydrus_code'

    csv_files = sorted(hydrus_dir.glob(f'Flood_{PLANT_TYPE}_*.csv')) if PLANT_TYPE != 'multi' else sorted(hydrus_dir.glob('Flood_*_*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No Flood CSV found in {hydrus_dir}')

    if len(csv_files) > 1:
        df = pd.concat(
            [pd.read_csv(str(f), header=[0, 1], index_col=[0, 1]) for f in csv_files],
            axis=1)
        plant_type = 'multi'
    else:
        df = pd.read_csv(str(csv_files[0]), header=[0, 1], index_col=[0, 1])
        plant_type = csv_files[0].stem.split('_')[1]

    results_base = Path(__file__).resolve().parent / 'results' / results_dirname

    print_flag    = False
    max_depth     = 10
    target_col    = 'avg_wc_top_half'

    initial_cost     = OrderedDict(zip(['wc', 'p'], [4.0, 10.0]))
    sensor_cost      = OrderedDict(zip(['wc', 'p'], [2.0, 1.0]))
    depth_cost       = OrderedDict(zip([str(d) for d in range(4, 204, 4)],
                                       [0.2 * np.ceil(d / 5) for d in range(4, 204, 4)]))
    measurement_cost = OrderedDict(zip(['wc', 'p'], [0.1, 0.1]))

    labels = df.columns.get_level_values('trial').unique().tolist()
    df_train_full, df_test_full = obs_split(df, labels, random_state=split_random_state)

    all_depths    = sorted(df.index.get_level_values('d').unique())
    idx_sel       = pd.IndexSlice
    obs_depths    = range(4, 204, 4)
    feature_types = ['wc', 'p']
    trial_name    = 'Flood'

    def avg_wc_top_half(df_full, trials):
        """Average wc across all times and depths in the top half of each trial's root zone."""
        targets = {}
        for trial in trials:
            rd_part       = next(p for p in trial.split('.') if p.startswith('rd'))
            root_depth    = int(rd_part[2:])
            top_half      = root_depth // 2
            depths_in_range = [d for d in all_depths if d <= top_half] or [all_depths[0]]
            wc_vals       = df_full.loc[idx_sel[:, depths_in_range], ('wc', trial)]
            targets[trial] = wc_vals.mean()
        return pd.Series(targets, name=target_col)

    train_trials = df_train_full.columns.get_level_values('trial').unique().tolist()
    test_trials  = df_test_full.columns.get_level_values('trial').unique().tolist()

    df_train_targets = avg_wc_top_half(df_train_full, train_trials)
    df_test_targets  = avg_wc_top_half(df_test_full,  test_trials)

    df_train = collapse_feat_midx(df_train_full.loc[idx_sel[:, obs_depths], feature_types])
    df_test  = collapse_feat_midx(df_test_full.loc[idx_sel[:, obs_depths], feature_types])

    df_train = pd.concat([df_train, df_train_targets], axis=1)
    df_test  = pd.concat([df_test,  df_test_targets],  axis=1)

    nobs_dir = results_base / 'avg_wc'
    nobs_dir.mkdir(parents=True, exist_ok=True)

    results_fname = f'{trial_name}_{plant_type}_avg_wc_top_half'

    reg_tree = fit_sklearn(df_train, [target_col], max_depth=max_depth)

    rtinfo = RegressionTreeInfo(reg_tree, df_train, target_col, trial_name,
        print_flag=print_flag,
        initial_cost=initial_cost,
        sensor_cost=sensor_cost,
        depth_cost=depth_cost,
        measurement_cost=measurement_cost)

    with open(str(nobs_dir / ('rtinfo_'   + results_fname + '.pkl')), 'wb') as f:
        pkl.dump(rtinfo,   f)
    with open(str(nobs_dir / ('df_train_' + results_fname + '.pkl')), 'wb') as f:
        pkl.dump(df_train, f)
    with open(str(nobs_dir / ('df_test_'  + results_fname + '.pkl')), 'wb') as f:
        pkl.dump(df_test,  f)
    with open(str(nobs_dir / ('tree_'     + results_fname + '.pkl')), 'wb') as f:
        pkl.dump(reg_tree, f)

    print('Done...')


def run_dt_flood_wc_cost(results_dirname='Flood', split_random_state=0):

    repo_dir   = Path(__file__).resolve().parent.parent.parent
    hydrus_dir = repo_dir / 'Hydrus' / 'python_hydrus_code'

    csv_files = sorted(hydrus_dir.glob(f'Flood_{PLANT_TYPE}_*.csv')) if PLANT_TYPE != 'multi' else sorted(hydrus_dir.glob('Flood_*_*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No Flood CSV found in {hydrus_dir}')

    if len(csv_files) > 1:
        df = pd.concat(
            [pd.read_csv(str(f), header=[0, 1], index_col=[0, 1]) for f in csv_files],
            axis=1)
        plant_type = 'multi'
    else:
        df = pd.read_csv(str(csv_files[0]), header=[0, 1], index_col=[0, 1])
        plant_type = csv_files[0].stem.split('_')[1]

    results_base = Path(__file__).resolve().parent / 'results' / results_dirname

    print_flag          = False
    max_depth           = 10
    imp_threshold       = 2500.0
    cost_threshold_list = [round(ac, 1) for ac in np.arange(0.0, 1.1, 0.1)]
    target_col          = 'avg_wc_top_half'

    initial_cost     = OrderedDict(zip(['wc', 'p'], [4.0, 10.0]))
    sensor_cost      = OrderedDict(zip(['wc', 'p'], [2.0, 1.0]))
    depth_cost       = OrderedDict(zip([str(d) for d in range(4, 204, 4)],
                                       [0.2 * np.ceil(d / 5) for d in range(4, 204, 4)]))
    measurement_cost = OrderedDict(zip(['wc', 'p'], [0.1, 0.1]))

    labels = df.columns.get_level_values('trial').unique().tolist()
    df_train_full, df_test_full = obs_split(df, labels, random_state=split_random_state)

    all_depths    = sorted(df.index.get_level_values('d').unique())
    idx_sel       = pd.IndexSlice
    obs_depths    = range(4, 204, 4)
    feature_types = ['wc', 'p']
    trial_name    = 'Flood'

    def avg_wc_top_half(df_full, trials):
        """Average wc across all times and depths in the top half of each trial's root zone."""
        targets = {}
        for trial in trials:
            rd_part         = next(p for p in trial.split('.') if p.startswith('rd'))
            root_depth      = int(rd_part[2:])
            top_half        = root_depth // 2
            depths_in_range = [d for d in all_depths if d <= top_half] or [all_depths[0]]
            wc_vals         = df_full.loc[idx_sel[:, depths_in_range], ('wc', trial)]
            targets[trial]  = wc_vals.mean()
        return pd.Series(targets, name=target_col)

    train_trials = df_train_full.columns.get_level_values('trial').unique().tolist()
    test_trials  = df_test_full.columns.get_level_values('trial').unique().tolist()

    df_train_targets = avg_wc_top_half(df_train_full, train_trials)
    df_test_targets  = avg_wc_top_half(df_test_full,  test_trials)

    df_train = collapse_feat_midx(df_train_full.loc[idx_sel[:, obs_depths], feature_types])
    df_test  = collapse_feat_midx(df_test_full.loc[idx_sel[:, obs_depths], feature_types])

    df_train = pd.concat([df_train, df_train_targets], axis=1)
    df_test  = pd.concat([df_test,  df_test_targets],  axis=1)

    nobs_dir = results_base / 'avg_wc'
    nobs_dir.mkdir(parents=True, exist_ok=True)

    for acost in cost_threshold_list:

        results_fname = f'{trial_name}_{plant_type}_avg_wc_top_half_cost{acost}'

        reg_tree = fit_sklearn(df_train, [target_col], max_depth=max_depth,
            cost_flag=True,
            cost_threshold=acost,
            imp_threshold=imp_threshold,
            initial_cost=list(initial_cost.values()),
            sensor_cost=list(sensor_cost.values()),
            depth_cost=list(depth_cost.values()),
            measurement_cost=list(measurement_cost.values()))

        rtinfo = RegressionTreeInfo(reg_tree, df_train, target_col, trial_name,
            print_flag=print_flag,
            initial_cost=initial_cost,
            sensor_cost=sensor_cost,
            depth_cost=depth_cost,
            measurement_cost=measurement_cost)

        with open(str(nobs_dir / ('rtinfo_'   + results_fname + '.pkl')), 'wb') as f:
            pkl.dump(rtinfo,   f)
        with open(str(nobs_dir / ('df_train_' + results_fname + '.pkl')), 'wb') as f:
            pkl.dump(df_train, f)
        with open(str(nobs_dir / ('df_test_'  + results_fname + '.pkl')), 'wb') as f:
            pkl.dump(df_test,  f)
        with open(str(nobs_dir / ('tree_'     + results_fname + '.pkl')), 'wb') as f:
            pkl.dump(reg_tree, f)

        print(f'Cost {acost} done.')

    print('Done...')


def run_dt_flood_flux_wc_cost(results_dirname='Flood', split_random_state=0):

    repo_dir   = Path(__file__).resolve().parent.parent.parent
    hydrus_dir = repo_dir / 'Hydrus' / 'python_hydrus_code'

    csv_files = sorted(hydrus_dir.glob(f'Flood_{PLANT_TYPE}_*.csv')) if PLANT_TYPE != 'multi' else sorted(hydrus_dir.glob('Flood_*_*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No Flood CSV found in {hydrus_dir}')

    if len(csv_files) > 1:
        df = pd.concat(
            [pd.read_csv(str(f), header=[0, 1], index_col=[0, 1]) for f in csv_files],
            axis=1)
        plant_type = 'multi'
        depth_list = [48, 100, 148, 200]
    else:
        df = pd.read_csv(str(csv_files[0]), header=[0, 1], index_col=[0, 1])
        plant_type = csv_files[0].stem.split('_')[1]
        depth_list = [200] if plant_type == 'corn' else [60]

    results_base = Path(__file__).resolve().parent / 'results' / results_dirname

    print_flag          = False
    max_depth           = 10
    imp_threshold       = 2500.0
    cost_threshold_list = [round(ac, 1) for ac in np.arange(0.0, 1.1, 0.1)]
    wc_col              = 'avg_wc_top_half'

    initial_cost     = OrderedDict(zip(['wc', 'p'], [4.0, 10.0]))
    sensor_cost      = OrderedDict(zip(['wc', 'p'], [2.0, 1.0]))
    depth_cost       = OrderedDict(zip([str(d) for d in range(4, 204, 4)],
                                       [0.2 * np.ceil(d / 5) for d in range(4, 204, 4)]))
    measurement_cost = OrderedDict(zip(['wc', 'p'], [0.1, 0.1]))

    labels = df.columns.get_level_values('trial').unique().tolist()
    df_train_full, df_test_full = obs_split(df, labels, random_state=split_random_state)

    all_depths    = sorted(df.index.get_level_values('d').unique())
    idx_sel       = pd.IndexSlice
    obs_depths    = range(4, 204, 4)
    feature_types = ['wc', 'p']
    trial_name    = 'Flood'

    train_trials = df_train_full.columns.get_level_values('trial').unique().tolist()
    test_trials  = df_test_full.columns.get_level_values('trial').unique().tolist()

    def avg_wc_top_half(df_full, trials):
        """Average wc across all times and depths in the top half of each trial's root zone."""
        targets = {}
        for trial in trials:
            rd_part         = next(p for p in trial.split('.') if p.startswith('rd'))
            root_depth      = int(rd_part[2:])
            top_half        = root_depth // 2
            depths_in_range = [d for d in all_depths if d <= top_half] or [all_depths[0]]
            wc_vals         = df_full.loc[idx_sel[:, depths_in_range], ('wc', trial)]
            targets[trial]  = wc_vals.mean()
        return pd.Series(targets, name=wc_col)

    # wc target is independent of flux depth â€” compute once
    df_train_wc = avg_wc_top_half(df_train_full, train_trials)
    df_test_wc  = avg_wc_top_half(df_test_full,  test_trials)

    # feature matrices are also depth-independent â€” compute once
    df_train_feats = collapse_feat_midx(df_train_full.loc[idx_sel[:, obs_depths], feature_types])
    df_test_feats  = collapse_feat_midx(df_test_full.loc[idx_sel[:, obs_depths], feature_types])

    for adepth in depth_list:

        targ_depths  = [adepth]
        flux_col     = 'flux_' + str(adepth)
        targ_folder  = str(adepth) + 'cm'

        df_train_flux = df_train_full.loc[idx_sel[:, targ_depths], ('f')].T
        df_train_flux.columns = ['flux_' + str(col[1]) for col in df_train_flux.columns.values]
        df_train_flux = df_train_flux.sum(axis=1)
        df_train_flux.name = flux_col

        df_test_flux = df_test_full.loc[idx_sel[:, targ_depths], ('f')].T
        df_test_flux.columns = ['flux_' + str(col[1]) for col in df_test_flux.columns.values]
        df_test_flux = df_test_flux.sum(axis=1)
        df_test_flux.name = flux_col

        # sign-preserving log-transform flux (handles flux >= 0)
        df_train_flux = df_train_flux.apply(signed_log)
        df_test_flux  = df_test_flux.apply(signed_log)

        targ_cols = [flux_col, wc_col]

        df_train = pd.concat([df_train_feats, df_train_flux, df_train_wc], axis=1)
        df_test  = pd.concat([df_test_feats,  df_test_flux,  df_test_wc],  axis=1)

        nobs_dir = results_base / targ_folder
        nobs_dir.mkdir(parents=True, exist_ok=True)

        for acost in cost_threshold_list:

            results_fname = f'{trial_name}_{plant_type}_flux{adepth}_wc_cost{acost}'

            reg_tree = fit_sklearn(df_train, targ_cols, max_depth=max_depth,
                cost_flag=True,
                cost_threshold=acost,
                imp_threshold=imp_threshold,
                initial_cost=list(initial_cost.values()),
                sensor_cost=list(sensor_cost.values()),
                depth_cost=list(depth_cost.values()),
                measurement_cost=list(measurement_cost.values()))

            # rtinfo uses the flux target for split/variance analysis
            rtinfo = RegressionTreeInfo(reg_tree, df_train, flux_col, trial_name,
                print_flag=print_flag,
                initial_cost=initial_cost,
                sensor_cost=sensor_cost,
                depth_cost=depth_cost,
                measurement_cost=measurement_cost)

            with open(str(nobs_dir / ('rtinfo_'   + results_fname + '.pkl')), 'wb') as f:
                pkl.dump(rtinfo,   f)
            with open(str(nobs_dir / ('df_train_' + results_fname + '.pkl')), 'wb') as f:
                pkl.dump(df_train, f)
            with open(str(nobs_dir / ('df_test_'  + results_fname + '.pkl')), 'wb') as f:
                pkl.dump(df_test,  f)
            with open(str(nobs_dir / ('tree_'     + results_fname + '.pkl')), 'wb') as f:
                pkl.dump(reg_tree, f)

            print(f'Depth {adepth} cm, cost {acost} done.')

    print('Done...')


def run_dt_flood_flux_wc(results_dirname='Flood', split_random_state=0):

    repo_dir   = Path(__file__).resolve().parent.parent.parent
    hydrus_dir = repo_dir / 'Hydrus' / 'python_hydrus_code'

    csv_files = sorted(hydrus_dir.glob(f'Flood_{PLANT_TYPE}_*.csv')) if PLANT_TYPE != 'multi' else sorted(hydrus_dir.glob('Flood_*_*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No Flood CSV found in {hydrus_dir}')

    if len(csv_files) > 1:
        df = pd.concat(
            [pd.read_csv(str(f), header=[0, 1], index_col=[0, 1]) for f in csv_files],
            axis=1)
        plant_type = 'multi'
        depth_list = [48, 100, 148, 200]
    else:
        df = pd.read_csv(str(csv_files[0]), header=[0, 1], index_col=[0, 1])
        plant_type = csv_files[0].stem.split('_')[1]
        depth_list = [200] if plant_type == 'corn' else [60]

    results_base = Path(__file__).resolve().parent / 'results' / results_dirname

    print_flag = False
    log_target = True
    max_depth  = 10
    wc_col     = 'avg_wc_top_half'

    initial_cost     = OrderedDict(zip(['wc', 'p'], [4.0, 10.0]))
    sensor_cost      = OrderedDict(zip(['wc', 'p'], [2.0, 1.0]))
    depth_cost       = OrderedDict(zip([str(d) for d in range(4, 204, 4)],
                                       [0.2 * np.ceil(d / 5) for d in range(4, 204, 4)]))
    measurement_cost = OrderedDict(zip(['wc', 'p'], [0.1, 0.1]))

    labels = df.columns.get_level_values('trial').unique().tolist()
    df_train_full, df_test_full = obs_split(df, labels, random_state=split_random_state)

    all_depths    = sorted(df.index.get_level_values('d').unique())
    idx_sel       = pd.IndexSlice
    obs_depths    = range(4, 204, 4)
    feature_types = ['wc', 'p']
    trial_name    = 'Flood'

    train_trials = df_train_full.columns.get_level_values('trial').unique().tolist()
    test_trials  = df_test_full.columns.get_level_values('trial').unique().tolist()

    def avg_wc_top_half(df_full, trials):
        targets = {}
        for trial in trials:
            rd_part         = next(p for p in trial.split('.') if p.startswith('rd'))
            root_depth      = int(rd_part[2:])
            top_half        = root_depth // 2
            depths_in_range = [d for d in all_depths if d <= top_half] or [all_depths[0]]
            wc_vals         = df_full.loc[idx_sel[:, depths_in_range], ('wc', trial)]
            targets[trial]  = wc_vals.mean()
        return pd.Series(targets, name=wc_col)

    # wc target is depth-independent — compute once
    df_train_wc = avg_wc_top_half(df_train_full, train_trials)
    df_test_wc  = avg_wc_top_half(df_test_full,  test_trials)

    # feature matrices are also depth-independent — compute once
    df_train_feats = collapse_feat_midx(df_train_full.loc[idx_sel[:, obs_depths], feature_types])
    df_test_feats  = collapse_feat_midx(df_test_full.loc[idx_sel[:, obs_depths], feature_types])

    for adepth in depth_list:

        targ_depths = [adepth]
        flux_col    = 'flux_' + str(adepth)
        targ_folder = str(adepth) + 'cm'

        df_train_flux = df_train_full.loc[idx_sel[:, targ_depths], ('f')].T
        df_train_flux.columns = ['flux_' + str(col[1]) for col in df_train_flux.columns.values]
        df_train_flux = df_train_flux.sum(axis=1)
        df_train_flux.name = flux_col

        df_test_flux = df_test_full.loc[idx_sel[:, targ_depths], ('f')].T
        df_test_flux.columns = ['flux_' + str(col[1]) for col in df_test_flux.columns.values]
        df_test_flux = df_test_flux.sum(axis=1)
        df_test_flux.name = flux_col

        if log_target:
            df_train_flux = df_train_flux.apply(signed_log)
            df_test_flux  = df_test_flux.apply(signed_log)

        targ_cols = [flux_col, wc_col]

        df_train = pd.concat([df_train_feats, df_train_flux, df_train_wc], axis=1)
        df_test  = pd.concat([df_test_feats,  df_test_flux,  df_test_wc],  axis=1)

        nobs_dir = results_base / targ_folder
        nobs_dir.mkdir(parents=True, exist_ok=True)

        results_fname = f'{trial_name}_{plant_type}_flux_{adepth}cm_wc'

        reg_tree = fit_sklearn(df_train, targ_cols, max_depth=max_depth)

        rtinfo = RegressionTreeInfo(reg_tree, df_train, flux_col, trial_name,
            print_flag=print_flag,
            initial_cost=initial_cost,
            sensor_cost=sensor_cost,
            depth_cost=depth_cost,
            measurement_cost=measurement_cost)

        with open(str(nobs_dir / ('rtinfo_'   + results_fname + '.pkl')), 'wb') as f:
            pkl.dump(rtinfo,   f)
        with open(str(nobs_dir / ('df_train_' + results_fname + '.pkl')), 'wb') as f:
            pkl.dump(df_train, f)
        with open(str(nobs_dir / ('df_test_'  + results_fname + '.pkl')), 'wb') as f:
            pkl.dump(df_test,  f)
        with open(str(nobs_dir / ('tree_'     + results_fname + '.pkl')), 'wb') as f:
            pkl.dump(reg_tree, f)

        print(f'Depth {adepth} cm done.')

    print('Done...')


def run_all(results_dirname='Flood', split_random_state=0):
    """Run all six DT pipelines, writing under results/<results_dirname>/.

    split_random_state controls the train/test trial split (obs_split);
    used to fit independent iterations against the same source data for
    stability checks without touching a previous run's output.
    """

    run_dt_flood_flux(results_dirname, split_random_state)

    run_dt_flood_flux_cost(results_dirname, split_random_state)

    run_dt_flood_wc(results_dirname, split_random_state)

    run_dt_flood_wc_cost(results_dirname, split_random_state)

    run_dt_flood_flux_wc(results_dirname, split_random_state)

    run_dt_flood_flux_wc_cost(results_dirname, split_random_state)


def main():

    ### functions to create and analyze decision trees ###

    run_all()

    ### analyze existing decision tree ##

    # analyze_dt()

    # get_rt_pred()
    pass

if __name__=='__main__':
    main()

