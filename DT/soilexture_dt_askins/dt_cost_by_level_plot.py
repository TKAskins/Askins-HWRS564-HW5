# dt_cost_by_level_plot.py
# Derek Groenendyk
# Written: 2019/06/07
# Modifed: 2026/02/05
# decision analysis for cost using class object


from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from pathlib import Path
import pickle as pkl
import sklearn as sklearn
from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from dt_cost_by_level_class import RegressionTreeInfo

# ── Results config ────────────────────────────────────────────────────────────
_RESULTS_DIR = Path(__file__).resolve().parent / 'results' / 'Flood'
_TRIAL_NAME  = 'Flood'
_PLANT_TYPE  = 'corn'    # 'corn' | 'beans' | 'multi'

# ── Tree selection ─────────────────────────────────────────────────────────────
# Set _OBJECTIVE to match the run_dt_* function whose results you want to plot:
#   'flux'          → run_dt_flood_flux()
#   'flux_cost'     → run_dt_flood_flux_cost()      (also set _COST_THRESHOLD)
#   'wc'            → run_dt_flood_wc()
#   'wc_cost'       → run_dt_flood_wc_cost()        (also set _COST_THRESHOLD)
#   'flux_wc'       → run_dt_flood_flux_wc()
#   'flux_wc_cost'  → run_dt_flood_flux_wc_cost()   (also set _COST_THRESHOLD)
_OBJECTIVE      = 'flux_wc_cost'
_COST_THRESHOLD = 0.5    # used when 'cost' in _OBJECTIVE; range 0.0–1.0
#0.0 — pure variance minimization, no cost penalty. Picks the best-predicting sensors regardless of cost.
#0.5 — balanced.
#1.0 — pure cost minimization, no variance penalty. Picks the cheapest sensors regardless of predictive power.
# ──────────────────────────────────────────────────────────────────────────────

# _DEPTH_LIST and _TARGET_LABEL are derived automatically — do not edit manually
if _OBJECTIVE in ('wc', 'wc_cost'):
    _DEPTH_LIST   = [None]
    _TARGET_LABEL = 'Avg. WC in top half of root zone'
    _TARGET_UNIT  = '[-]'
elif _OBJECTIVE in ('flux_wc', 'flux_wc_cost'):
    _TARGET_LABEL = 'Flux + Avg. WC'
    _TARGET_UNIT  = '[ln(cm) / -]'
    if _PLANT_TYPE == 'corn':
        _DEPTH_LIST = [200]
    elif _PLANT_TYPE == 'beans':
        _DEPTH_LIST = [100]
    else:
        _DEPTH_LIST = [48, 100, 148, 200]
elif _PLANT_TYPE == 'corn':
    _DEPTH_LIST   = [200]
    _TARGET_LABEL = 'Flux'
    _TARGET_UNIT  = '[ln(cm)]'
elif _PLANT_TYPE == 'beans':
    _DEPTH_LIST   = [100]
    _TARGET_LABEL = 'Flux'
    _TARGET_UNIT  = '[ln(cm)]'
else:                             # multi
    _DEPTH_LIST   = [48, 100, 148, 200]
    _TARGET_LABEL = 'Flux'
    _TARGET_UNIT  = '[ln(cm)]'

# figures are grouped per-DT: figs/<objective>[_<cost_threshold>]/
_FIGS_DIR_TAG = f'{_OBJECTIVE}_{_COST_THRESHOLD}' if 'cost' in _OBJECTIVE else _OBJECTIVE
_FIGS_DIR = Path(__file__).resolve().parent / 'figs' / _FIGS_DIR_TAG


def _depth_tag(depth):
    """Human-readable depth label for titles and filenames."""
    return 'avg_wc' if depth is None else f'{depth}cm'


def _get_fname(depth):
    """Results filename stem for the current objective and depth."""
    if _OBJECTIVE == 'flux':
        return f'{_TRIAL_NAME}_{_PLANT_TYPE}_flux_{depth}cm'
    elif _OBJECTIVE == 'flux_cost':
        return f'{_TRIAL_NAME}_depth{depth}_cost{_COST_THRESHOLD}'
    elif _OBJECTIVE == 'wc':
        return f'{_TRIAL_NAME}_{_PLANT_TYPE}_avg_wc_top_half'
    elif _OBJECTIVE == 'wc_cost':
        return f'{_TRIAL_NAME}_{_PLANT_TYPE}_avg_wc_top_half_cost{_COST_THRESHOLD}'
    elif _OBJECTIVE == 'flux_wc':
        return f'{_TRIAL_NAME}_{_PLANT_TYPE}_flux_{depth}cm_wc'
    elif _OBJECTIVE == 'flux_wc_cost':
        return f'{_TRIAL_NAME}_{_PLANT_TYPE}_flux{depth}_wc_cost{_COST_THRESHOLD}'
    raise ValueError(f'Unknown _OBJECTIVE: {_OBJECTIVE!r}')


def _get_results_dir(depth):
    """Results directory for the current objective and depth."""
    if _OBJECTIVE in ('wc', 'wc_cost'):
        return _RESULTS_DIR / 'avg_wc'
    return _RESULTS_DIR / f'{depth}cm'  # flux, flux_cost, flux_wc, flux_wc_cost


def load_results(depth):
    """Load rtinfo, df_train, df_test, tree for the selected objective and depth."""
    if _OBJECTIVE == 'flux':
        results_dir = _RESULTS_DIR / f'{depth}cm'
        fname = f'{_TRIAL_NAME}_{_PLANT_TYPE}_flux_{depth}cm'
    elif _OBJECTIVE == 'flux_cost':
        results_dir = _RESULTS_DIR / f'{depth}cm'
        fname = f'{_TRIAL_NAME}_depth{depth}_cost{_COST_THRESHOLD}'
    elif _OBJECTIVE == 'wc':
        results_dir = _RESULTS_DIR / 'avg_wc'
        fname = f'{_TRIAL_NAME}_{_PLANT_TYPE}_avg_wc_top_half'
    elif _OBJECTIVE == 'wc_cost':
        results_dir = _RESULTS_DIR / 'avg_wc'
        fname = f'{_TRIAL_NAME}_{_PLANT_TYPE}_avg_wc_top_half_cost{_COST_THRESHOLD}'
    elif _OBJECTIVE == 'flux_wc':
        results_dir = _RESULTS_DIR / f'{depth}cm'
        fname = f'{_TRIAL_NAME}_{_PLANT_TYPE}_flux_{depth}cm_wc'
    elif _OBJECTIVE == 'flux_wc_cost':
        results_dir = _RESULTS_DIR / f'{depth}cm'
        fname = f'{_TRIAL_NAME}_{_PLANT_TYPE}_flux{depth}_wc_cost{_COST_THRESHOLD}'
    else:
        raise ValueError(f'Unknown _OBJECTIVE: {_OBJECTIVE!r}')

    with open(str(results_dir / ('rtinfo_'   + fname + '.pkl')), 'rb') as f:
        rtinfo   = pkl.load(f)
    with open(str(results_dir / ('df_train_' + fname + '.pkl')), 'rb') as f:
        df_train = pkl.load(f)
    with open(str(results_dir / ('df_test_'  + fname + '.pkl')), 'rb') as f:
        df_test  = pkl.load(f)
    with open(str(results_dir / ('tree_'     + fname + '.pkl')), 'rb') as f:
        tree     = pkl.load(f)
    return rtinfo, df_train, df_test, tree


def get_rt_pred(directory, fname, level=None, auto_lvl=True, return_pred=False):
    """Evaluate tree predictions at a given depth level."""
    with open(str(directory / ('df_train_' + fname + '.pkl')), 'rb') as f:
        df_train = pkl.load(f)
    with open(str(directory / ('df_test_'  + fname + '.pkl')), 'rb') as f:
        df_test  = pkl.load(f)
    with open(str(directory / ('tree_'     + fname + '.pkl')), 'rb') as f:
        reg_tree = pkl.load(f)

    col_names  = df_test.columns.values
    # feature columns follow the pattern type:dDEPTH:tDAY; target columns do not
    feat_names = [c for c in col_names if ':d' in str(c) and ':t' in str(c)]
    targ_names = [c for c in col_names if c not in feat_names]

    if len(targ_names) == 1 or auto_lvl:
        with open(str(directory / ('rtinfo_' + fname + '.pkl')), 'rb') as f:
            rtinfo = pkl.load(f)

    df_train_feats = df_train[feat_names]
    df_train_targs = df_train[targ_names]
    df_test_feats  = df_test[feat_names]
    df_test_targs  = df_test[targ_names]

    if auto_lvl:
        level = np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1

    pred_train = predict(df_train_feats.to_numpy(), reg_tree.tree_, max_depth=level)
    pred_test  = predict(df_test_feats.to_numpy(),  reg_tree.tree_, max_depth=level)

    r2_train   = r2_score(df_train_targs.to_numpy(), pred_train)
    r2_test    = r2_score(df_test_targs.to_numpy(),  pred_test)
    rmse_train = np.sqrt(mean_squared_error(df_train_targs.to_numpy(), pred_train))
    rmse_test  = np.sqrt(mean_squared_error(df_test_targs.to_numpy(),  pred_test))

    if not return_pred:
        return r2_train, r2_test, rmse_train, rmse_test
    else:
        return pred_train, pred_test, df_train_targs, df_test_targs


def plot_tree(tree):

    sklearn.tree.plot_tree(tree)
    # plt.show()    

    if False:
        # sklearn.tree.export_graphviz(tree, out_file='./sklearn_results/sklearn_mpg_dot.data')
        dot_data = sklearn.tree.export_graphviz(tree)#, out_file='sklearn_dot')
        graph = graphviz.Source(dot_data)
        graph.format = 'png'
        graph.render("./sklearn_results/sklearn_mpg")     

    # output_pdf(tree, name='temp_sklearn')    


def process_rtinfo(rtinfo):

    leaves = np.nonzero(rtinfo.var_df['is_leaf'].values)[0]
    # print(rtinfo.var_df.iloc[0])
    # print(leaves[:10])
    # print(len(leaves))

    depth = 15
    nodes = rtinfo.nodes_by_depth[depth - 1]
    # print(nodes)
    total_sse = rtinfo.var_df.reindex(nodes)[['sse_left', 'sse_right']].sum().sum()
    # print(rtinfo.var_df.reindex(nodes)[['sse_left', 'sse_right']])
    # print(total_sse)

    # print(rtinfo.var_df_by_depth['sse'])

    # print(key_list)
    
    temp_df = rtinfo.var_df[:][rtinfo.var_df['depth'] < 10]
    
    feat_df = rtinfo.var_df['feature']
    # print(feat_df)

    count_df = feat_df.value_counts()

    print(count_df)

    # print(count_df.iloc[:20])

    for i in range(len(count_df)):

        feature = count_df.index[i]

        depths = rtinfo.var_df['depth'][rtinfo.var_df['feature'] == feature]
        nodes = rtinfo.var_df.index[rtinfo.var_df['feature'] == feature]

        print(feature, count_df.iloc[i])
        # print(depths)

        # temp_df = rtinfo.var_df[:][rtinfo.var_df['feature'] == feature]
        temp_df = rtinfo.var_df.reindex(nodes)
        temp_df.sort_values(by=['depth'], ascending=True, inplace=True)

        # print(temp_df)
        # print(temp_df[['depth','cost','feature']])

    # prev_obs_dict = populate_obs_dict(rtinfo.var_df['feature'].values)
    # print(prev_obs_dict['wc'])
    # print(len(prev_obs_dict['wc']['d']))
    # print(len(prev_obs_dict['p']['d']))

    # print(np.unique(rtinfo.var_df['feature']))

    # raise

    print(rtinfo.var_df_by_depth['var_exp'])


def regdt_level_2panel_plot():

    params = {
        'axes.labelsize':6,
        'xtick.labelsize':6,
        'ytick.labelsize':6,
    }

    plt.rcParams.update(params)

    depth_num = _DEPTH_LIST[0]  # first target depth
    rtinfo, _, _, _ = load_results(depth_num)
    level = int(np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1)
    save_dir = _FIGS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    n_levels = len(rtinfo.var_df_by_depth)
    levels = range(1, n_levels + 1)

    fig = plt.figure(figsize=(8,3))
    axes = fig.subplots(1,2)

    ax = axes[0]
    var_data = rtinfo.var_df_by_depth['var_exp']
    ax.plot(levels, var_data)

    ax.plot([level, level], [0,1], lw=0.5)
    ax.set_ylim([0,1])
    
    ax.set_xlabel('Level [-]')
    ax.set_ylabel('Explained Variance [-]')

    if False:
        max_varexp = round(np.amax(rtinfo.var_df_by_depth['var_exp']),2)
        max_varexp_lvl = np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1
        # ax.text(0.5, 0.75, 'Max Var. Exp. = ' + str(max_varexp), transform=ax.transAxes)
        ax.text(0.1, 0.25, 'Maximum Explained Variance Level = ' + str(max_varexp_lvl), 
            transform=ax.transAxes,
            fontsize=6)

    ax = axes[1]
    var_data = rtinfo.var_df_by_depth['sse']
    ax.plot(levels, var_data)

    sse_max = float(rtinfo.var_df_by_depth['sse'].max()) * 1.1
    ax.plot([level, level], [0, sse_max], lw=0.5)
    ax.set_ylim([0, sse_max])

    ax.set_xlabel('Level [-]')
    ax.set_ylabel('SSE [ln(cm)$^2$]')

    fname = 'regdt_level_panel'
    
    fig.savefig(str(save_dir / (fname + '.png')), format='png', dpi=300)  


def regdt_level_plots(rtinfo, trial_name, save_dir, tag=''):

    ### by depth ###

    num_levels = 10


    if False:
        ## variance
        fig = plt.figure()
        ax = rtinfo.var_df_by_depth['ssq_condition'].plot()
        ax.set_xlabel('Splits [-]')
        ax.set_ylabel('ssq_cond')
        fname = trial_name + '_' + 'sse_cond'
        if tag !='':
            fname += '_' + tag
        ax.set_title(fname)
        fig.savefig('./figs/' + fname + '.png', format='png')       

    fig = plt.figure()
    ax = rtinfo.var_df_by_depth['var_exp'].plot()
    ax.set_xlabel('Splits [-]')
    ax.set_ylabel('Explained Variance [-]')
    fname = trial_name + '_' + 'var_exp'
    if tag !='':
        fname += '_' + tag
    ax.set_title(fname)

    if True:
        max_varexp = round(np.amax(rtinfo.var_df_by_depth['var_exp']),2)
        max_varexp_lvl = np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1
        ax.text(0.5, 0.75, 'Max Var. Exp. = ' + str(max_varexp), transform=ax.transAxes)
        ax.text(0.5, 0.70, 'Max Var. Exp. Level = ' + str(max_varexp_lvl), transform=ax.transAxes)


    fig.savefig(str(save_dir / (fname + '.png')), format='png')               

    fig = plt.figure()
    ax = rtinfo.var_df_by_depth['sse'].plot()
    ax.set_xlabel('Splits [-]')
    ax.set_ylabel('SSE [ln(cm)$^2$]')
    fname = trial_name + '_' + 'sse'
    if tag !='':
        fname += '_' + tag
    ax.set_title(fname)
    fig.savefig(str(save_dir / (fname + '.png')), format='png')               

    fig = plt.figure()  
    ax = rtinfo.var_df_by_depth['sse_red'].plot()
    ax.set_xlabel('Splits [-]')
    ax.set_ylabel('Reduction of SSE [ln(cm)$^2$]')   
    fname = trial_name + '_' + 'sse_red'
    if tag !='':
        fname += '_' + tag
    ax.set_title(fname)
    fig.savefig(str(save_dir / (fname + '.png')), format='png')           


    ## cost
    fig = plt.figure()  
    ax = rtinfo.var_df_by_depth['total_cost'].plot()
    ax.set_xlabel('Splits [-]')
    ax.set_ylabel('Total Cost [$]') 
    fname = trial_name + '_' + 'total_cost'
    if tag !='':
        fname += '_' + tag
    ax.set_title(fname)
    fig.savefig(str(save_dir / (fname + '.png')), format='png')           

    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    actual_levels = min(num_levels, len(rtinfo.nodes_by_depth))
    for i in range(actual_levels):
        print(i)
        nodes = rtinfo.nodes_by_depth[i]
        interior = [n for n in nodes if n in rtinfo.var_df.index]
        cost = rtinfo.var_df.loc[interior, 'cost'].sum() if interior else 0.0
        ax.scatter(i+1, cost)
    # ax = rtinfo.var_df_by_depth['cost'].plot()
    ax.set_xlabel('Splits [-]')
    ax.set_ylabel('cost for split [$]')
    fname = trial_name + '_' + 'cost'
    if tag !='':
        fname += '_' + tag
    ax.set_title(fname)
    fig.savefig(str(save_dir / (fname + '.png')), format='png')           


    if False:
        ## cost and variance
        ## needs fixin'
        fig = plt.figure()
        ax = rtinfo.var_df_by_depth.plot(x='total_cost', y='sse_red', legend=True)
        ax.set_xlabel('Total Cost [$]')
        ax.set_ylabel('Reduction of SSE [ln(cm)$^2$]')
        fname = trial_name + '_' + 'ssered_v_totalcost'
        if tag !='':
            fname += '_' + tag
        ax.set_title(fname)
        fig.savefig(str(save_dir / (fname + '.png')), format='png')   


    if False:
        ## targets
        fig = plt.figure()  
        ax = rtinfo.var_df_by_depth['num_targets'].plot()
        ax.set_xlabel('Splits [-]')
        ax.set_ylabel('number of targets')
        fname = trial_name + '_' + 'targs'
        if tag !='':
            fname += '_' + tag
        ax.set_title(fname)
        fig.savefig(str(save_dir / (fname + '.png')), format='png')   


def regdt_bar_plots(rtinfo, df, trial_name, tag=''):

    depth = 4

    nodes = rtinfo.nodes_by_depth[depth - 1]
    leaf_nodes = rtinfo.var_df.loc[(rtinfo.var_df['depth'] < depth) & rtinfo.var_df['is_leaf']].index

    nodes_at_depth = nodes.copy()
    nodes_at_depth.extend(leaf_nodes)   

    depth_df = rtinfo.var_df.reindex(nodes_at_depth)

    # impurity
    ax = depth_df['impurity'].plot(kind='bar')
    ax.set_xlabel('node')
    ax.set_ylabel('impurity')
    

    fname = trial_name + '_' + 'impurity' + '_bar'
    if tag !='':
        fname += '_' + tag      

    ax.set_title(fname + '\nImpurity at Depth = ' + str(depth))

    plt.gcf().savefig('./figs/' + fname + '.png', format='png') 

    ax = depth_df['sse'].plot(kind='bar')
    ax.set_xlabel('node')
    ax.set_ylabel('SSE [ln(cm)$^2$]')

    fname = trial_name + '_' + 'sse' + '_bar'
    if tag !='':
        fname += '_' + tag      

    ax.set_title(fname + '\nSSE at Depth = ' + str(depth))

    plt.gcf().savefig('./figs/' + fname + '.png', format='png')     

    ax = depth_df[['var_exp', 'var_exp_left', 'var_exp_right']].plot(kind='bar')
    ax.set_xlabel('node')
    # ax.set_ylabel('SSE [ln(cm)$^2$]')

    fname = trial_name + '_' + 'var_exp' + '_bar'
    if tag !='':
        fname += '_' + tag      

    ax.set_title(fname + '\nVar. Exp. at Depth = ' + str(depth))

    plt.gcf().savefig('./figs/' + fname + '.png', format='png') 


    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
              ax.get_xticklabels() + ax.get_yticklabels()): \
    item.set_fontsize(20)

    plt.show()


def regdt_scatter_plots_v1(rtinfo, df, df_full):

    ### df_full contains only features

    val_list = np.setdiff1d(np.arange(1326), df.index.values)
    val_list = np.arange(1326)

    plot_info_list = [
        # ['var_exp', 100.0, 'Variance Explained'],
        # ['impurity', 50.0, 'Impurity'],
        # ['cost', 1.0, 'Cost'],
        ['r2', 0.0, '$R^2$'],
        ['total_cost', 0.0, 'Level Cost'],
        ['r2_cost', 0.0, '$R^2$']
    ]

    max_level = 4

    for plot_info in plot_info_list:

        fig = plt.figure()
        ax = fig.add_subplot(111)

        obs_num = 0
        obs_dict = OrderedDict()
        x_list = []
        y_list = []
        s_list = []

        plot_type = plot_info[0]
        marker_size = plot_info[1]
        title = plot_info[2]

        level = 0
        while level < max_level:
            level += 1

            if plot_type in ['r2', 'total_cost', 'r2_cost']:

                if plot_type == 'r2':
                    pred = predict(full_df.loc[val_list].to_numpy(), rtinfo.tree_, max_depth=level)
                    # print(np.amax(pred))
                    # print(df.loc[val_list, target_name])
                    r2 = r2_score(rtinfo.full_targ_vals.loc[val_list].to_numpy(), pred)
                    print(r2)
                    y_list.append(r2)
                    x_list.append(level)
                elif plot_type == 'total_cost':
                    y_list.append(rtinfo.var_df_by_depth.loc[level, 'cost'])
                    x_list.append(level)
                else:
                    pred = predict(full_df.loc[val_list].to_numpy(), rtinfo.tree_, max_depth=level)
                    r2 = r2_score(rtinfo.full_targ_vals.loc[val_list].to_numpy(), pred)
                    y_list.append(r2)
                    x_list.append(rtinfo.var_df_by_depth.loc[level, 'cost'])
                

            elif plot_type in ['var_exp', 'impurity', 'cost']:

                for i,an in enumerate(rtinfo.nodes_by_depth[level - 1]):

                    feat = rtinfo.tree_.feature[an]

                    try:
                        yloc = obs_dict[feat]
                    except KeyError:
                        obs_num += 1
                        obs_dict[feat] = obs_num
                        yloc = obs_num

                    x_list.append(level)
                    y_list.append(yloc)

                    if plot_type == 'var_exp':
                        s = rtinfo.var_df.loc[an, 'var_exp'] * marker_size
                    elif plot_type == 'impurity':
                        s = rtinfo.var_df.loc[an, 'impurity'] * marker_size
                    elif plot_type == 'cost':
                        s = rtinfo.var_df.loc[an, 'cost'] * marker_size + 1.0
                    s_list.append(s)

        print(s_list)

        if len(s_list) == 0:
            ax.scatter(x_list, y_list)
        else:
            ax.scatter(x_list, y_list, s=s_list)

        print(obs_dict)

        if plot_type in ['var_exp', 'impurity', 'cost']:
            ax.set_yticks(list(obs_dict.values()))
            ax.set_yticklabels(rtinfo.feature_names[list(obs_dict.keys())])
            ax.set_title(title)

        if plot_type in ['r2', 'total_cost', 'r2_cost']:
            ax.set_ylabel(title)
            ax.set_title(trial_name)



        if plot_type == 'r2_cost':
            ax.set_xlabel('Total Cost [$]')
            # print(x_list)
            # print(y_list)
        else:
            ax.set_xticks(range(1, max_level + 1))
            ax.set_xlabel('Splits [-]')          


        # fname = 'temp_' + plot_type + '_lvl' + str(max_level)

        # trial_name = 'Inf'
        # trial_name = 'Drain'
        fname = trial_name + '_' + plot_type + '_lvl' + str(max_level)

        # fig = ax.figure
        fig.savefig('./figs/' + fname + '.png', format='png')



    # same as above, just a different code structure
    plot_info_list = [
        ['var_exp', 100.0, 'Variance Explained'],
        ['impurity', 50.0, 'Impurity'],
        ['cost', 1.0, 'Cost'],
    ]

    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    offset = -0.15

    for p,plot_info in enumerate(plot_info_list):
        offset += 0.15

        obs_num = 0
        obs_dict = OrderedDict()
        x_list = []
        y_list = []
        s_list = []

        plot_type = plot_info[0]
        marker_size = plot_info[1]
        title = plot_info[2]

        level = 0
        while level < max_level:
            level += 1              

            for i,an in enumerate(rtinfo.nodes_by_depth[level - 1]):
                
                feat = rtinfo.tree_.feature[an]

                if feat == _tree.TREE_UNDEFINED:
                    print('undefined: ',_tree.TREE_UNDEFINED)
                    continue

                try:
                    yloc = obs_dict[feat]
                except KeyError:
                    obs_num += 1
                    obs_dict[feat] = obs_num
                    yloc = obs_num
                x_list.append(level + offset)
                y_list.append(yloc)

                if plot_type == 'var_exp':
                    s = rtinfo.var_df.loc[an, 'var_exp'] * marker_size
                elif plot_type == 'impurity':
                    s = rtinfo.var_df.loc[an, 'impurity'] * marker_size
                elif plot_type == 'cost':
                    s = rtinfo.var_df.loc[an, 'cost'] * marker_size + 1.0
                s_list.append(s)

        print(s_list)

        ax.scatter(x_list, y_list, s=s_list, label=title)

        print(obs_dict)

        if p == 0:
            ax.set_yticks(list(obs_dict.values()))
            ax.set_yticklabels(rtinfo.feature_names[list(obs_dict.keys())])
            ax.set_title(trial_name)

        ax.set_xticks(range(1, max_level + 1))
        ax.set_xlabel('Splits [-]')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fname = trial_name + '_lvl' + str(max_level)

    fig.savefig('./figs/' + fname + '.png', format='png')


def regdt_scatter_plots_v2(rtinfo, df, df_test, tag=''):

    ### df_test contains features and targets

    val_list = np.setdiff1d(np.arange(1326), df.index.values)
    val_list = np.arange(1326)


    # print(df)
    # print(df_test)
    # raise

    plot_info_list = [
        # ['var_exp', 100.0, 'Variance Explained'],
        # ['impurity', 50.0, 'Impurity'],
        # ['cost', 1.0, 'Cost'],
        ['r2', 0.0, '$R^2$'],
        ['total_cost', 0.0, 'Level Cost'],
        ['r2_cost', 0.0, '$R^2$']
    ]

    max_level = 4

    for plot_info in plot_info_list:

        fig = plt.figure()
        ax = fig.add_subplot(111)

        obs_num = 0
        obs_dict = OrderedDict()
        x_list = []
        y_list = []
        s_list = []

        plot_type = plot_info[0]
        marker_size = plot_info[1]
        title = plot_info[2]

        col_names = df_test.columns.values
        feat_names = [cname for cname in col_names if 'flux' not in cname]
        targ_names = [cname for cname in col_names if 'flux' in cname]

        df_test_feats = df[feat_names]
        df_test_targs = df[targ_names]

        level = 0
        while level < max_level:
            level += 1

            if plot_type in ['r2', 'total_cost', 'r2_cost']:

                if plot_type == 'r2':
                    pred = predict(df_test_feats.to_numpy(), rtinfo.tree_, max_depth=level)
                    # print(np.amax(pred))
                    # print(df.loc[val_list, target_name])
                    r2 = r2_score(df_test_targs.to_numpy(), pred)
                    print(r2)
                    y_list.append(r2)
                    x_list.append(level)
                elif plot_type == 'total_cost':
                    y_list.append(rtinfo.var_df_by_depth.loc[level, 'cost'])
                    x_list.append(level)
                else:
                    pred = predict(df_test_feats.to_numpy(), rtinfo.tree_, max_depth=level)
                    r2 = r2_score(df_test_targs.to_numpy(), pred)
                    y_list.append(r2)
                    x_list.append(rtinfo.var_df_by_depth.loc[level, 'cost'])
                

            elif plot_type in ['var_exp', 'impurity', 'cost']:

                for i,an in enumerate(rtinfo.nodes_by_depth[level - 1]):
                    feat = rtinfo.tree_.feature[an]
                    try:
                        yloc = obs_dict[feat]
                    except KeyError:
                        obs_num += 1
                        obs_dict[feat] = obs_num
                        yloc = obs_num
                    x_list.append(level)
                    y_list.append(yloc)

                    if plot_type == 'var_exp':
                        s = rtinfo.var_df.loc[an, 'var_exp'] * marker_size
                    elif plot_type == 'impurity':
                        s = rtinfo.var_df.loc[an, 'impurity'] * marker_size
                    elif plot_type == 'cost':
                        s = rtinfo.var_df.loc[an, 'cost'] * marker_size + 1.0
                    s_list.append(s)

        print(s_list)

        if len(s_list) == 0:
            ax.scatter(x_list, y_list)
        else:
            ax.scatter(x_list, y_list, s=s_list)

        print(obs_dict)

        if plot_type in ['var_exp', 'impurity', 'cost']:
            ax.set_yticks(list(obs_dict.values()))
            ax.set_yticklabels(rtinfo.feature_names[list(obs_dict.keys())])
            ax.set_title(title)

        if plot_type in ['r2', 'total_cost', 'r2_cost']:
            ax.set_ylabel(title)
            ax.set_title(rtinfo.trial_name)



        if plot_type == 'r2_cost':
            ax.set_xlabel('Total Cost [$]')
            # print(x_list)
            # print(y_list)
        else:
            ax.set_xticks(range(1, max_level + 1))
            ax.set_xlabel('Splits [-]')          


        # fname = 'temp_' + plot_type + '_lvl' + str(max_level)

        # trial_name = 'Inf'
        # trial_name = 'Drain'
        fname = rtinfo.trial_name + '_' + plot_type + '_lvl' + str(max_level)

        if tag !='':
            fname += '_' + tag

        ax.set_title(fname)     

        print(fname)

        # fig = ax.figure
        fig.savefig('./figs/' + fname + '.png', format='png')



    # same as above, just a different code structure
    plot_info_list = [
        ['var_exp', 100.0, 'Variance Explained'],
        ['impurity', 50.0, 'Impurity'],
        ['cost', 1.0, 'Cost'],
    ]

    
    fig = plt.figure()
    fig.subplots_adjust(left=0.20)
    ax = fig.add_subplot(111)

    offset = -0.15

    for p,plot_info in enumerate(plot_info_list):
        offset += 0.15

        obs_num = 0
        obs_dict = OrderedDict()
        x_list = []
        y_list = []
        s_list = []

        plot_type = plot_info[0]
        marker_size = plot_info[1]
        title = plot_info[2]

        level = 0
        while level < max_level:
            level += 1              

            for i,an in enumerate(rtinfo.nodes_by_depth[level - 1]):

                feat = rtinfo.tree_.feature[an]

                if feat == _tree.TREE_UNDEFINED:
                    print('undefined: ',_tree.TREE_UNDEFINED)
                    continue

                try:
                    yloc = obs_dict[feat]
                except KeyError:
                    obs_num += 1
                    obs_dict[feat] = obs_num
                    yloc = obs_num
                x_list.append(level + offset)
                y_list.append(yloc)

                if plot_type == 'var_exp':
                    s = rtinfo.var_df.loc[an, 'var_exp'] * marker_size
                elif plot_type == 'impurity':
                    s = rtinfo.var_df.loc[an, 'impurity'] * marker_size
                elif plot_type == 'cost':
                    s = rtinfo.var_df.loc[an, 'cost'] * marker_size + 1.0
                s_list.append(s)
                # print(an, s)

        ax.scatter(x_list, y_list, s=s_list, label=title)

        # print(x_list)
        # print(y_list)
        # print(s_list)

        # print(obs_dict)

        if p == 0:
            ax.set_yticks(list(obs_dict.values()))
            ax.set_yticklabels(rtinfo.feature_names[list(obs_dict.keys())])
            ax.set_title(rtinfo.trial_name)

        ax.set_xticks(range(1, max_level + 1))
        ax.set_xlabel('Splits [-]')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

    fname = rtinfo.trial_name + '_lvl' + str(max_level)

    if tag !='':
        fname += '_' + tag

    ax.set_title(fname)     

    fig.savefig('./figs/' + fname + '.png', format='png')   


def regdt_scatter_plots_v3():

    params = {
        'axes.titlesize' : 6,
        'axes.labelsize':6,
        'xtick.labelsize':5,
        'ytick.labelsize':5,
    }

    plt.rcParams.update(params)

    # Plot cumulative improvement and cost by tree level for each target depth.
    # Flood has a single result per depth â€” no cost-weighting comparison.

    color_list = ['r', 'g', 'b', 'y']

    _FIGS_DIR.mkdir(parents=True, exist_ok=True)

    for idepth, depth_num in enumerate(_DEPTH_LIST):

        rtinfo, _, _, _ = load_results(depth_num)

        max_level = int(np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1)
        max_level = min(max_level, 4)

        columns = ['name', 'level', 'cost', 'improvement', 'var_exp', 'impurity']

        # Collect node-level data
        rows = []
        for ilvl in range(max_level):
            nodes = rtinfo.nodes_by_depth[ilvl]
            for anode in nodes:
                name = rtinfo.node_names[anode]
                if name != 'undefined!' and anode in rtinfo.var_df.index:
                    rows.append({
                        'name'       : name,
                        'level'      : ilvl + 1,
                        'cost'       : rtinfo.var_df.loc[anode, 'cost'],
                        'improvement': rtinfo.var_df.loc[anode, 'var_exp'],
                        'var_exp'    : rtinfo.var_df.loc[anode, 'var_exp'],
                        'impurity'   : rtinfo.var_df.loc[anode, 'impurity'],
                    })

        if not rows:
            continue

        measurement_df = pd.DataFrame(rows, columns=columns)

        # Cumulative bar chart: improvement and cost by level
        fig_bar = plt.figure(figsize=(8, 4))
        axes_bar = fig_bar.subplots(1, 2)
        fig_bar.subplots_adjust(left=0.2)

        for iplt, var_type in enumerate(['improvement', 'cost']):

            ax = axes_bar[iplt]

            for ilvl in range(max_level):
                x = ilvl + 1
                idx = measurement_df['level'] <= (ilvl + 1)
                var_val = measurement_df.loc[idx, var_type].sum()

                if ilvl == 0:
                    ax.bar(x, var_val, 0.6,
                           facecolor=color_list[idepth % len(color_list)],
                           label=_depth_tag(depth_num))
                else:
                    ax.bar(x, var_val, 0.6,
                           facecolor=color_list[idepth % len(color_list)])

            ax.set_ylabel('Improvement [ln(cm)$^2$]' if var_type == 'improvement' else 'Cost [$]')
            ax.set_xlabel('Level [-]')
            ax.set_xticks(range(1, max_level + 1))

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper left',
                      bbox_transform=ax.transAxes, fontsize=6)

        fname = f'{_TRIAL_NAME}_{_depth_tag(depth_num)}_cum_by_depth'
        fig_bar.savefig(str(_FIGS_DIR / (fname + '.png')), format='png', dpi=300)
        print(fname)


def plot_bar_trialcompare():
    """Cumulative improvement bar chart by tree level for each Flood target depth."""

    params = {
        'axes.titlesize' : 6,
        'axes.labelsize':6,
        'xtick.labelsize':5,
        'ytick.labelsize':5,
    }

    plt.rcParams.update(params)

    color_list = ['r', 'g', 'b', 'y']
    var_type    = 'improvement'

    _FIGS_DIR.mkdir(parents=True, exist_ok=True)

    num_depths  = len(_DEPTH_LIST)
    ncols       = min(num_depths, 4)
    nrows       = (num_depths + ncols - 1) // ncols

    fig_bar, axes_bar = plt.subplots(nrows, ncols,
                                     figsize=(4 * ncols, 4 * nrows),
                                     squeeze=False)
    fig_bar.subplots_adjust(left=0.12, wspace=0.35)

    for idepth, depth_num in enumerate(_DEPTH_LIST):

        ax = axes_bar[idepth // ncols][idepth % ncols]

        rtinfo, _, _, _ = load_results(depth_num)

        max_level = int(np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1)

        # Build node-level dataframe
        rows = []
        for ilvl in range(max_level):
            nodes = rtinfo.nodes_by_depth[ilvl]
            for anode in nodes:
                name = rtinfo.node_names[anode]
                if name != 'undefined!' and anode in rtinfo.var_df.index:
                    rows.append({
                        'level'      : ilvl + 1,
                        'improvement': rtinfo.var_df.loc[anode, 'var_exp'],
                        'cost'       : rtinfo.var_df.loc[anode, 'cost'],
                    })

        if not rows:
            continue

        measurement_df = pd.DataFrame(rows)

        ylabel = 'Variance Reduction [ln(cm)$^2$]' if var_type == 'improvement' else 'Cost [$]'

        for ilvl in range(max_level):
            x       = ilvl + 1
            idx     = measurement_df['level'] <= (ilvl + 1)
            var_val = measurement_df.loc[idx, var_type].sum()

            if ilvl == 0:
                ax.bar(x, var_val, 0.6,
                       facecolor=color_list[idepth % len(color_list)],
                       label=_depth_tag(depth_num))
            else:
                ax.bar(x, var_val, 0.6,
                       facecolor=color_list[idepth % len(color_list)])

        ax.set_ylabel(ylabel)
        ax.set_xlabel('Level [-]')
        ax.set_xticks(range(1, max_level + 1))
        ax.set_title(f'{_TRIAL_NAME} {_depth_tag(depth_num)}')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper left',
                  bbox_transform=ax.transAxes, fontsize=6)

    # Hide any unused subplots
    for j in range(num_depths, nrows * ncols):
        axes_bar[j // ncols][j % ncols].set_visible(False)

    fname = var_type + '_cum_by_depth_' + _TRIAL_NAME

    fig_bar.savefig(str(_FIGS_DIR / (fname + '.png')), format='png', dpi=300)
    print(fname)


def apply(X, tree, max_depth=None):
    """Finds the terminal region (=leaf node) for each sample in X."""

    if max_depth == None:
        max_depth = tree.max_depth

    n_samples = X.shape[0]
    nodes = np.zeros(n_samples, dtype=int)

    # print(n_samples)

    # print(max_depth)

    for i in range(n_samples):

        node = 0
        depth = 0

        # feature = tree.feature[node]

        # While node not a leaf
        while tree.children_left[node] != _tree.TREE_LEAF and depth <= max_depth:
            depth += 1

            if X[i, tree.feature[node]] <= tree.threshold[node]:
                node = tree.children_left[node]
            else:
                node = tree.children_right[node]

        nodes[i] = node

        # print('Depth: ', depth, node)

    # print(nodes)

    return nodes


def predict(X, tree, max_depth):

    if max_depth == None:
        max_depth = tree.max_depth  

    # print(X)
    # print(tree)
    # print(max_depth)

    nodes = apply(X, tree, max_depth)

    # tree.value shape: (n_nodes, n_outputs, 1) — index by node then squeeze class dim
    prediction = np.squeeze(tree.value[nodes], axis=-1)

    return prediction



def plot_imp_cost():

    # Plot total_cost at each tree level across all target depths
    # Uses rtinfo loaded directly via load_results(depth)

    plot_levels = [3, 5, 7, 10]   # tree levels to highlight (1-indexed)
    dtype = 'cost'

    colors = ['r', 'g', 'b', 'y']

    save_dir = _FIGS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.80)

    for idepth, depth_num in enumerate(_DEPTH_LIST):

        rtinfo, _, _, _ = load_results(depth_num)

        regdt_level_plots(rtinfo, f'{_TRIAL_NAME}_{_depth_tag(depth_num)}', save_dir, tag='')

        for i, alevel in enumerate(plot_levels):

            if alevel > len(rtinfo.var_df_by_depth):
                continue

            if dtype == 'var_exp':
                var_data = rtinfo.var_df_by_depth['var_exp'][alevel - 1]

            elif dtype == 'sse':
                nodes = rtinfo.nodes_by_depth[alevel - 1]
                var_data = rtinfo.var_df.reindex(nodes)[['sse_left', 'sse_right']].sum().sum()

            elif dtype == 'cost':
                var_data = rtinfo.var_df_by_depth['total_cost'][alevel - 1]

            if idepth == 0:
                ax.scatter(depth_num, var_data, c=colors[i], label='level ' + str(alevel))
            else:
                ax.scatter(depth_num, var_data, c=colors[i])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=[1.01, 1.0], bbox_transform=ax.transAxes)

    if None not in _DEPTH_LIST:
        ax.set_xticks(_DEPTH_LIST)
    ax.set_xlabel('Target Depth [cm]')
    ax.set_ylabel({'var_exp': 'Explained Variance [-]', 'sse': 'SSE [ln(cm)$^2$]', 'cost': 'Total Cost [$]'}.get(dtype, dtype))

    fname = 'imp_cost_' + _TRIAL_NAME + '_' + dtype
    ax.set_title(fname)

    fig.savefig(str(save_dir / (fname + '.png')), format='png')


def plot_imp_cost_panel():
    """Plot total cost and SSE at the auto-selected level for each Flood target depth."""

    params = {
        'axes.labelsize': 8,
        'figure.titlesize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    }
    plt.rcParams.update(params)

    dtypes = ['cost', 'sse']
    colors = ['r', 'g']

    num_depths = len(_DEPTH_LIST)
    fig, axes = plt.subplots(num_depths, len(dtypes),
                             figsize=(4 * len(dtypes), 3 * num_depths))
    if num_depths == 1:
        axes = [axes]  # ensure 2-D indexing works

    fig.subplots_adjust(bottom=0.1, hspace=0.4)

    max_xaxis = [-99, -99]

    for idepth, depth_num in enumerate(_DEPTH_LIST):

        rtinfo, _, _, _ = load_results(depth_num)
        level = int(np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1)

        for idtype, adtype in enumerate(dtypes):

            ax = axes[idepth][idtype]

            if adtype == 'sse':
                nodes    = rtinfo.nodes_by_depth[level - 1]
                var_data = rtinfo.var_df.reindex(nodes)[['sse_left', 'sse_right']].sum().sum()
            elif adtype == 'cost':
                var_data = rtinfo.var_df_by_depth['total_cost'][level - 1]

            ax.scatter(depth_num, var_data, c=colors[idtype], s=15)

            if var_data > max_xaxis[idtype]:
                max_xaxis[idtype] = var_data

            if idtype == 0:
                ax.set_ylabel(f'{_depth_tag(depth_num)}')
            if idepth == 0:
                ax.set_title('Total Cost' if adtype == 'cost'
                             else 'Sum of Squared Error, $cm^2$')
            if idepth == num_depths - 1:
                ax.set_xlabel('Target Depth [cm]')

    plt.tight_layout()

    _FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fname = 'imp_cost_panel_' + _TRIAL_NAME
    fig.savefig(str(_FIGS_DIR / (fname + '.png')), format='png', dpi=300)


def plot_imp_cost_combined():
    """Plot var_exp, SSE, and total cost at the auto-level for each Flood target depth."""

    dtypes = ['var_exp', 'sse', 'cost']
    colors = ['r', 'g', 'b', 'y']

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    fig.subplots_adjust(right=0.80)

    for idepth, depth_num in enumerate(_DEPTH_LIST):

        rtinfo, _, _, _ = load_results(depth_num)
        level = int(np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1)

        for idtype, dtype in enumerate(dtypes):

            if dtype == 'var_exp':
                var_data = rtinfo.var_df_by_depth['var_exp'][level - 1]
            elif dtype == 'sse':
                nodes    = rtinfo.nodes_by_depth[level - 1]
                var_data = rtinfo.var_df.reindex(nodes)[['sse_left', 'sse_right']].sum().sum()
            elif dtype == 'cost':
                var_data = rtinfo.var_df_by_depth['total_cost'][level - 1]

            label = dtype if idepth == 0 else None
            ax.scatter(depth_num, var_data,
                       c=colors[idtype % len(colors)],
                       label=label, s=20)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=[1.01, 1.0], bbox_transform=ax.transAxes)

    if None not in _DEPTH_LIST:
        ax.set_xticks(_DEPTH_LIST)
    ax.set_xlabel('Target Depth [cm]')
    ax.set_ylabel('Value')

    fname = 'imp_cost_combined_' + _TRIAL_NAME
    ax.set_title(fname)

    _FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(_FIGS_DIR / (fname + '.png')), format='png')


def pred_plot_v2():

    depth = _DEPTH_LIST[0]  # first target depth
    trial_name = _TRIAL_NAME

    rtinfo, _, _, _ = load_results(depth)
    auto_lvl  = True
    max_level = int(np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1)

    results_dir   = _get_results_dir(depth)
    results_fname = _get_fname(depth)

    title  = f'{trial_name} DT Predictions'
    xlabel = f'Simulated {_TARGET_LABEL} {_TARGET_UNIT}'
    ylabel = f'Predicted {_TARGET_LABEL} {_TARGET_UNIT}'

    param = 'time'

    # param_list = times
    # cmap = rand_cmap.rand_cmap(len(times), type='bright',
    #     first_color_black=False, last_color_black=False)
    # clist = []
    # for i in range(len(param_list)):
    #     rgba = cmap(i/(len(param_list) - 1))
    #     clist.append(rgba)
    # print(clist)
    # raise

    clist = ['g']


    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.80)

    g_plist = [[1e9, 1e9], [-1e9, -1e9]]


    pred_list = get_rt_pred(results_dir, results_fname,
        level=max_level, auto_lvl=auto_lvl, return_pred=True)      

    # print(param, ':', aobs)

    pred_train, pred_test, true_train, true_test = pred_list 


    r_list = get_rt_pred(results_dir, results_fname,
        level=max_level, auto_lvl=auto_lvl, return_pred=False)

    r2_train, r2_test, rmse_train, rmse_test = r_list

    text = f'$R^2=${round(r2_test,2)}'

    ax.text(0.75, 0.25, text,transform=ax.transAxes)

    # print(true_test)
    # true_test.hist()
    # plt.show()
    # raise


    label = param[0]

    # ax.scatter(true_test, pred_test, c=[clist[iobs]], label=label)
    ax.scatter(true_test, pred_test, label=label)

    # print(true_test)

    # plist = [[1e9, 1e9], [-1e9, -1e9]]

    # plist[0][0] = np.amin(true_test.values)
    # plist[0][1] = np.amin(pred_test)

    # plist[1][0] = np.amax(true_test.values)
    # plist[1][1] = np.amax(pred_test)

    # for i in range(2):
    #     if plist[0][i] < g_plist[0][i]:
    #         g_plist[0][i] = plist[0][i]

    #     if plist[1][i] > g_plist[1][i]:
    #         g_plist[1][i] = plist[1][i]

    # pmin = np.amin(g_plist[0])
    # pmax = np.amax(g_plist[1])

    # ax.plot([pmin, pmax], [pmin, pmax], 'k--')

    # print(g_plist)


    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels, bbox_to_anchor=[1.01, 1.0], bbox_transform=ax.transAxes)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fname = 'pred_' + f'{_TRIAL_NAME}_{_PLANT_TYPE}_flux_{depth}cm'

    ax.set_title(title)

    _FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(_FIGS_DIR / (fname + '.png')), format='png')           


def pred_plot_2panel():
    """Predicted vs simulated scatter for each Flood target depth (single panel per depth)."""

    params = {
        'axes.titlesize' : 6,
        'axes.labelsize' : 6,
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
    }
    plt.rcParams.update(params)

    num_depths = len(_DEPTH_LIST)
    fig, axes = plt.subplots(1, num_depths,
                             figsize=(4 * num_depths, 3),
                             squeeze=False)
    axes = axes[0]  # shape (num_depths,)

    fig.subplots_adjust(bottom=0.2)

    for i, depth in enumerate(_DEPTH_LIST):

        ax = axes[i]

        results_dir   = _get_results_dir(depth)
        results_fname = _get_fname(depth)

        rtinfo, _, _, _ = load_results(depth)
        auto_lvl  = True
        max_level = int(np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1)

        pred_list = get_rt_pred(results_dir, results_fname,
            level=max_level, auto_lvl=auto_lvl, return_pred=True)
        pred_train, pred_test, true_train, true_test = pred_list

        r_list = get_rt_pred(results_dir, results_fname,
            level=max_level, auto_lvl=auto_lvl, return_pred=False)
        r2_train, r2_test, rmse_train, rmse_test = r_list

        text = f'$R^2$={round(r2_test, 2)}'
        ax.text(0.05, 0.90, text, transform=ax.transAxes, fontsize=6)

        ax.scatter(true_test, pred_test, s=3)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot(xlim, ylim, 'k--', lw=0.5, zorder=99)

        ax.set_xlabel(f'Simulated {_TARGET_LABEL} {_TARGET_UNIT}')
        ax.set_title(f'{_TRIAL_NAME} {_depth_tag(depth)}')

        if i == 0:
            ax.set_ylabel(f'Predicted {_TARGET_LABEL} {_TARGET_UNIT}')

    fname = 'pred_2panel_' + _TRIAL_NAME

    _FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(_FIGS_DIR / (fname + '.png')), format='png', dpi=300)


def r2_plot_by_depth():

    depth_list = _DEPTH_LIST
    trial_name = _TRIAL_NAME

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(right=0.80)

    clist = ['r', 'g', 'b']

    x_vals = list(range(len(depth_list))) if None in depth_list else depth_list
    x_labels = [_depth_tag(d) for d in depth_list]

    for idepth, adepth in enumerate(depth_list):

        results_dir   = _get_results_dir(adepth)
        results_fname = _get_fname(adepth)

        r2_train, r2_test, rmse_train, rmse_test = get_rt_pred(
            results_dir, results_fname, auto_lvl=True, return_pred=False)

        xval = x_vals[idepth]
        if idepth == 0:
            ax.scatter(xval, r2_test,  c=clist[0], label='test')
            ax.scatter(xval, r2_train, c=clist[1], label='train')
        else:
            ax.scatter(xval, r2_test,  c=clist[0])
            ax.scatter(xval, r2_train, c=clist[1])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=[1.01, 1.0], bbox_transform=ax.transAxes)

    ax.set_xlabel('Target' if None in depth_list else 'Depth [cm]')
    ax.set_ylabel('$R^2$ [-]')
    ax.set_ylim([0.0, 1.0])
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels)

    fname = 'r2_' + trial_name + '_by_depth'
    ax.set_title(fname)

    _FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(_FIGS_DIR / (fname + '.png')), format='png')


def get_unique_feature_names(rtinfo, level=None):

    all_features = []
    all_nodes = []

    if not level:
        level = np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1
        print(level)

    for alvl in range(0,level):
        nodes = rtinfo.nodes_by_depth[alvl]
        # print(nodes)
        all_nodes.extend(nodes)
        all_features.extend(rtinfo.node_names[nodes])

    # print(all_features)
    # print(all_nodes)
    all_features = set(all_features)

    feature_depths = []
    feature_times = []

    for aname in all_features:
        if aname != 'undefined!':
            res = [i for i in range(len(aname)) if aname.startswith(":", i)]
            ind = res[-1]
            feature_depths.append(aname[:ind])
            feature_times.append(aname[ind+1:])

    unique_features = set(feature_depths)

    # print(feature_depths)
    # print(feature_times)

    return unique_features, all_features


def numobs_plot_by_depth():

    depth_list = _DEPTH_LIST
    fname_tag  = _TRIAL_NAME

    plot_type = 'numobs'

    clist = ['r', 'g', 'b']
    width  = 1
    offset = width / 2

    fig = plt.figure(figsize=(8, 3))
    ax  = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(bottom=0.20)

    max_num_features = -99

    global_features  = []
    global_allfeatures = []

    for idepth, adepth in enumerate(depth_list):

        rtinfo, _, _, _ = load_results(adepth)

        level = int(np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1)

        features, all_features = get_unique_feature_names(rtinfo, level)
        num_features = len(features)

        if num_features > max_num_features:
            max_num_features = num_features

        global_features.extend(features)
        global_allfeatures.extend(all_features)

        xval  = idepth if adepth is None else adepth
        label = _depth_tag(adepth)

        if idepth == 0:
            ax.bar(xval, num_features, color=clist[0], width=width, label=label)
        else:
            ax.bar(xval, num_features, color=clist[0], width=width)

    unique_gfeatures   = set(global_features)
    unique_allfeatures = set(global_allfeatures)

    print('num_unique_features:', len(unique_gfeatures))
    print('num_all_features:', len(unique_allfeatures))

    ax.set_xlabel('Target' if None in depth_list else 'Target Depth [cm]')
    ax.set_ylabel('Number of Unique Measurements [-]')
    ax.set_ylim([0, max_num_features + 2])

    loc = 'upper right'
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=loc, bbox_transform=ax.transAxes)

    if None not in depth_list:
        ax.set_xticks(depth_list)
    ax.set_title(_TRIAL_NAME)

    plt.tight_layout()

    fname = plot_type + '_by_depth_' + fname_tag

    _FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(_FIGS_DIR / (fname + '.png')), format='png')


def numobs_plot_by_cost():

    params = {
        'axes.labelsize':6,
        'xtick.labelsize':6,
        'ytick.labelsize':6,
    }

    plt.rcParams.update(params)

    depth_list = _DEPTH_LIST
    fname_tag  = _TRIAL_NAME

    cost_threshold_list = np.round(np.arange(0.0, 1.1, 0.1), 1)

    clist = ['r', 'g', 'b']
    width  = 0.025
    offset = width / 2

    fig = plt.figure(figsize=(6, 3))
    ax  = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()
    ax2.invert_yaxis()
    fig.subplots_adjust(bottom=0.20)

    max_num_features = -99
    max_sse = -99

    # For each cost threshold, accumulate features/SSE across all target depths
    for icost, acost in enumerate(cost_threshold_list):

        global_features_cost   = []
        global_allfeatures_cost = []
        global_sse_cost        = []

        for adepth in depth_list:

            rtinfo, _, _, _ = load_results(adepth)

            level = int(np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1)

            features, all_features = get_unique_feature_names(rtinfo, level)

            global_features_cost.extend(features)
            global_allfeatures_cost.extend(all_features)

            nodes    = rtinfo.nodes_by_depth[level - 1]
            var_data = rtinfo.var_df.reindex(nodes)[['sse_left', 'sse_right']].sum().sum()
            global_sse_cost.append(var_data)

        unique_gfeatures_cost = set(global_features_cost)
        num_cost_features     = len(unique_gfeatures_cost)
        sse                   = sum(global_sse_cost)

        if num_cost_features > max_num_features:
            max_num_features = num_cost_features
        if sse > max_sse:
            max_sse = sse

        ax.bar(acost + offset,  num_cost_features, color=clist[1], width=width)
        ax2.bar(acost + offset, sse,               color=clist[0], width=width)

    ax.set_xlabel('Cost Weight [-]')
    ax.set_ylabel('Number of Unique Measurements [-]')
    ax2.set_ylabel('SSE [ln(cm)$^2$]')

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0, max_num_features + 2])
    ax.set_yticks(range(max_num_features + 2 + 1))
    ax.set_yticklabels([str(int(t)) for t in ax.get_yticks()])
    ax2.set_ylim([max_sse + 200, 0])

    ax.set_xticks(cost_threshold_list)
    ax.tick_params(axis='x', rotation=45)

    ax.set_title(_TRIAL_NAME)

    plt.tight_layout()

    fname = 'numobs_by_cost_' + fname_tag

    _FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(_FIGS_DIR / (fname + '.png')), format='png', dpi=300)


def r2_ratio_plot_v2():

    clist = ['r', 'g', 'b', 'y']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.80)

    for idepth, depth in enumerate(_DEPTH_LIST):

        results_dir   = _get_results_dir(depth)
        results_fname = _get_fname(depth)

        rtinfo, _, _, _ = load_results(depth)
        auto_lvl  = True
        max_level = int(np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1)

        r_list = get_rt_pred(results_dir, results_fname, level=max_level,
            auto_lvl=auto_lvl, return_pred=False)

        r2_train, r2_test, rmse_train, rmse_test = r_list

        print('depth:', depth)
        print('r2_train: ', r2_train)
        print('r2_test:', r2_test)
        print('RMSE_train:', rmse_train)
        print('RMSE_test', rmse_test)
        print('r2_ratio:', r2_train / r2_test)

        ax.scatter(depth, r2_train / r2_test, c=[clist[idepth % len(clist)]],
                   label='depth=' + str(depth))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=[1.01, 1.0], bbox_transform=ax.transAxes)

    ax.set_xlabel('Target Depth [cm]')
    ax.set_ylabel('$R^2$ Train / $R^2$ Test [-]')

    fname = 'r2_rat_' + _TRIAL_NAME

    ax.set_title(fname)

    _FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(_FIGS_DIR / (fname + '.png')), format='png')           


def rmse_plot():

    clist = ['r', 'g', 'b', 'y']

    fig = plt.figure()
    fig.subplots_adjust(right=0.8)
    ax = fig.add_subplot(111)

    for idepth, depth in enumerate(_DEPTH_LIST):

        results_dir   = _get_results_dir(depth)
        results_fname = _get_fname(depth)

        r_list = get_rt_pred(results_dir, results_fname,
            auto_lvl=True, return_pred=False)

        r2_train, r2_test, rmse_train, rmse_test = r_list

        xval  = idepth if depth is None else depth
        label = _depth_tag(depth)

        ax.scatter(xval, rmse_test, c=[clist[idepth % len(clist)]], label=label)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=[1.01, 1.0], bbox_transform=ax.transAxes)

    ax.set_xlabel('Target' if None in _DEPTH_LIST else 'Target Depth [cm]')
    ax.set_ylabel('RMSE [ln(cm)]')

    fname = 'rmse_' + _TRIAL_NAME

    ax.set_title(fname)

    _FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(_FIGS_DIR / (fname + '.png')), format='png')           


def obs_matrix_plot():
    """Observation frequency / var-exp matrix across target depths for Flood data."""

    trial_name = _TRIAL_NAME
    obs_depths = list(range(4, 204, 4))   # available feature depths for Flood

    sens_type = 'both'
    var_type  = 'varexp_wght'

    for depth_num in _DEPTH_LIST:

        rtinfo, df_train, df_test, _ = load_results(depth_num)

        col_names  = df_test.columns.values
        feat_names_list = [c for c in col_names if 'flux' not in c]

        max_level = int(np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1)

        freq_data   = np.zeros((len(obs_depths), 1))
        varexp_data = np.zeros((len(obs_depths), 1))

        level = 0
        while level < max_level:
            level += 1

            nodes    = rtinfo.nodes_by_depth[level - 1]
            depth_df = rtinfo.var_df.reindex(nodes).set_index('feature')

            for an in nodes:

                feat = rtinfo.tree_.feature[an]

                if feat == _tree.TREE_UNDEFINED:
                    continue

                feat_name   = rtinfo.feature_names[feat]
                fname_split = feat_name.split(':')

                if sens_type != 'both':
                    if fname_split[0] != sens_type:
                        continue

                var_exp = depth_df.loc[feat_name, 'var_exp']
                if isinstance(var_exp, pd.Series):
                    var_exp = np.sum(var_exp.to_numpy())

                # Feature name format: type:dDEPTH  (no time component for Flood)
                try:
                    fdepth = int(fname_split[1][1:])
                    fdepth_idx = obs_depths.index(fdepth)
                except (IndexError, ValueError):
                    continue

                freq_data[fdepth_idx, 0]   += 1

                if varexp_data[fdepth_idx, 0] == 0:
                    varexp_data[fdepth_idx, 0] = var_exp
                else:
                    varexp_data[fdepth_idx, 0] = (varexp_data[fdepth_idx, 0] + var_exp) / 2.

        fig = plt.figure()
        fig.subplots_adjust(right=0.8)
        ax = fig.add_subplot(111)

        for i in range(len(obs_depths)):
            j = 0
            if var_type == 'freq':
                data = freq_data[i, j]
            elif var_type == 'var_exp':
                data = varexp_data[i, j]
            elif var_type == 'varexp_wght':
                data = varexp_data[i, j] * freq_data[i, j]

            if var_type == 'var_exp':
                text_data = '{:0.3f}'.format(data)
            else:
                text_data = '{:0.0f}'.format(data)

            x = 1
            y = len(obs_depths) - i

            if var_type == 'var_exp':
                data *= 100.0

            ax.scatter(x, y, s=data * 5 + 1, c='r')
            ax.text(x - 0.1, y + 0.015, text_data, fontsize=6)

        yticks  = list(range(1, len(obs_depths) + 1))
        ylabels = [str(item) for item in obs_depths[::-1]]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=5)
        ax.set_ylim(yticks[0] - 0.5, yticks[-1] + 0.5)
        ax.set_xticks([1])
        ax.set_xticklabels(['features'])
        ax.set_ylabel('Feature Depth [cm]')

        if var_type == 'freq':
            title_str = 'Observation Frequency'
        elif var_type == 'var_exp':
            title_str = 'Explained Variance'
        elif var_type == 'varexp_wght':
            title_str = 'Weighted Frequency'

        ax.set_title(f'{title_str}: {trial_name} {_depth_tag(depth_num)}, {sens_type}')

        fname_out = f'obsmatrix_{trial_name}_{_depth_tag(depth_num)}_{sens_type}_{var_type}'
        _FIGS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(_FIGS_DIR / (fname_out + '.png')), format='png')


def obs_freq_plot():
    """Feature-depth frequency / var-exp plot for each Flood target depth."""

    trial_name = _TRIAL_NAME
    obs_depths  = list(range(4, 204, 4))  # available feature depths for Flood

    sens_type = 'wc'
    var_type  = 'var_exp'

    for depth_num in _DEPTH_LIST:

        rtinfo, _, _, _ = load_results(depth_num)

        max_level = int(np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1)

        freq_data = np.zeros(len(obs_depths))

        level = 0
        while level < max_level:
            level += 1

            nodes    = rtinfo.nodes_by_depth[level - 1]
            depth_df = rtinfo.var_df.reindex(nodes).set_index('feature')

            for an in nodes:

                feat = rtinfo.tree_.feature[an]

                if feat == _tree.TREE_UNDEFINED:
                    continue

                feat_name   = rtinfo.feature_names[feat]
                fname_split = feat_name.split(':')

                if fname_split[0] != sens_type:
                    continue

                var_exp = depth_df.loc[feat_name, 'var_exp']
                if isinstance(var_exp, pd.Series):
                    var_exp = np.sum(var_exp.to_numpy())

                try:
                    fdepth     = int(fname_split[1][1:])
                    fdepth_idx = obs_depths.index(fdepth)
                except (IndexError, ValueError):
                    continue

                if var_type == 'freq':
                    freq_data[fdepth_idx] += 1
                elif var_type == 'var_exp':
                    if freq_data[fdepth_idx] == 0:
                        freq_data[fdepth_idx] = var_exp
                    else:
                        freq_data[fdepth_idx] = (freq_data[fdepth_idx] + var_exp) / 2.

        idx_max = int(np.argmax(freq_data))

        fig = plt.figure()
        fig.subplots_adjust(right=0.8)
        ax  = fig.add_subplot(111)

        for i, fdepth in enumerate(obs_depths):
            if freq_data[i] > 0:
                ax.scatter(1, len(obs_depths) - i, s=freq_data[i] * 5 + 1, c='r')

        ax.scatter(1, len(obs_depths) - idx_max, marker='*', c='y', s=200)

        yticks  = list(range(1, len(obs_depths) + 1))
        ylabels = [str(d) for d in obs_depths[::-1]]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=5)
        ax.set_ylim(0.5, len(obs_depths) + 0.5)
        ax.set_xticks([1])
        ax.set_xticklabels([sens_type])
        ax.set_ylabel('Feature Depth [cm]')

        ax.set_title(
            f'Observation Frequency: {trial_name} {_depth_tag(depth_num)}, {sens_type}')

        fname_out = f'obsfreq_{trial_name}_{_depth_tag(depth_num)}_{sens_type}_{var_type}'
        _FIGS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(_FIGS_DIR / (fname_out + '.png')), format='png')

    return


def plot_top10_sensors():
    """Grouped horizontal bar chart of the 10 sensors (type + depth) with greatest
    cumulative variance contribution.  Variance on the bottom x-axis (blue) and
    cumulative sensor cost on the top x-axis (orange)."""

    params = {
        'axes.labelsize':  8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 8,
        'axes.titlesize':  8,
    }
    plt.rcParams.update(params)

    _FIGS_DIR.mkdir(parents=True, exist_ok=True)

    for depth_num in _DEPTH_LIST:

        rtinfo, _, _, _ = load_results(depth_num)

        max_level = int(np.argmax(rtinfo.var_df_by_depth['var_exp']) + 1)

        # accumulate var_exp and cost per (type, depth) sensor
        sensor_varexp = {}
        sensor_cost   = {}

        for ilvl in range(max_level):
            for anode in rtinfo.nodes_by_depth[ilvl]:
                if anode not in rtinfo.var_df.index:
                    continue
                feat = rtinfo.tree_.feature[anode]
                if feat == _tree.TREE_UNDEFINED:
                    continue
                feat_name = rtinfo.feature_names[feat]
                parts = feat_name.split(':')
                if len(parts) < 2:
                    continue
                sensor_key = f'{parts[0]}:{parts[1]}'  # e.g. 'wc:d20'

                var_exp = rtinfo.var_df.loc[anode, 'var_exp']
                cost    = rtinfo.var_df.loc[anode, 'cost']
                if isinstance(var_exp, pd.Series):
                    var_exp = float(var_exp.sum())
                if isinstance(cost, pd.Series):
                    cost = float(cost.sum())

                sensor_varexp[sensor_key] = sensor_varexp.get(sensor_key, 0.0) + float(var_exp)
                sensor_cost[sensor_key]   = sensor_cost.get(sensor_key, 0.0)   + float(cost)

        if not sensor_varexp:
            continue

        # rank by variance, take top 10
        sorted_sensors = sorted(sensor_varexp.items(), key=lambda x: x[1], reverse=True)[:10]
        labels         = [s[0] for s in sorted_sensors]
        values_varexp  = [s[1] for s in sorted_sensors]
        values_cost    = [sensor_cost.get(lbl, 0.0) for lbl in labels]

        y      = np.arange(len(labels))
        height = 0.35

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twiny()

        ax1.barh(y + height / 2, values_varexp, height, color='steelblue',  label='Explained Variance')
        ax2.barh(y - height / 2, values_cost,   height, color='darkorange', label='Cost')

        ax1.set_yticks(y)
        ax1.set_yticklabels(labels)
        ax1.invert_yaxis()

        ax1.set_xlabel('Cumulative Explained Variance [ln(cm)$^2$]', color='steelblue')
        ax1.tick_params(axis='x', colors='steelblue')
        ax2.set_xlabel('Cumulative Cost [$]', color='darkorange')
        ax2.tick_params(axis='x', colors='darkorange')
        ax1.set_ylabel('Sensor [type:depth]')
        ax1.set_title(f'{_TRIAL_NAME} {_depth_tag(depth_num)} — Top 10 Sensors by Variance Contribution')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=7)

        plt.tight_layout()

        fname = f'top10_sensors_{_TRIAL_NAME}_{_depth_tag(depth_num)}'
        fig.savefig(str(_FIGS_DIR / (fname + '.png')), format='png', dpi=300)
        print(fname)


def main():

    regdt_level_2panel_plot()     # var_exp and SSE vs tree depth level
    r2_plot_by_depth()            # RÂ² train/test at each target depth
    pred_plot_v2()                # predicted vs simulated flux scatter (single)
    r2_ratio_plot_v2()            # RÂ² train/test ratio by depth
    rmse_plot()                   # RMSE by target depth
    pred_plot_2panel()            # predicted vs simulated, one panel per depth
    numobs_plot_by_depth()        # number of unique feature measurements by depth
    numobs_plot_by_cost()         # number of features vs cost weight
    plot_imp_cost()               # cost / sse / var_exp at levels, per depth
    plot_imp_cost_panel()         # cost & sse panel plot, per depth
    plot_imp_cost_combined()      # all dtypes combined, per depth
    plot_bar_trialcompare()       # cumulative improvement bar chart by level
    regdt_scatter_plots_v3()      # improvement / cost bar chart per level
    obs_matrix_plot()             # feature-depth frequency matrix, per depth
    obs_freq_plot()               # feature-depth var-exp frequency, per depth
    plot_top10_sensors()          # top 10 sensors by variance contribution

if __name__ == "__main__":
    main()
