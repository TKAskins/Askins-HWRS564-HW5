# dt_cost_level_class.py
# Derek Groenendyk
# Written: 2019/06/07
# Modifed: 2019/06/25
# deceision analysis in class form (referenced, not ran)


import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import _tree


class RegressionTreeInfo:
    def __init__(self, tree, df, target_name, trial_name, 
        initial_cost, sensor_cost, depth_cost, measurement_cost,
        print_flag=False,):


        self.tree = tree
        self.target_name = target_name
        self.df = df

        self.trial_name = trial_name

        self.print_flag = print_flag

        self.target = self.df[self.target_name]

        self.initial_cost = initial_cost
        self.sensor_cost = sensor_cost
        self.depth_cost = depth_cost
        self.measurement_cost = measurement_cost

        self.analyze_tree()


    def analyze_tree(self):

        # feature_names = ['displacement','horsepower','weight']
        self.feature_names = self.df.drop(columns=[self.target_name]).columns

        self.tree_ = self.tree.tree_
        # afhelp(tree_))
        # raise

        neg_idx = np.nonzero(self.tree_.feature < 0.0)[0]

        node_names = [
            self.feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in self.tree_.feature
        ]
        self.node_names = np.array(node_names)

        self.num_nodes = self.tree_.node_count


        if self.print_flag:
            print('num_nodes', self.num_nodes) 
            print('num targets', len(self.df.index))   

        self.parent_dict = {0:0}
        self.children_dict = {}

        for node in range(self.tree_.node_count):
            # parent_dict[node] = [self.tree_.children_left[node], self.tree_.children_right[node]]
            self.parent_dict[self.tree_.children_left[node]] = node
            self.parent_dict[self.tree_.children_right[node]] = node
            self.children_dict[node] = [self.tree_.children_left[node],
                self.tree_.children_right[node]]

        self.nodes_by_depth = []
        self.cdict = {}

        self.recurse_nodes(0, 1, [])

        # print(len(cdict.keys()))
        self.key_list = sorted(self.cdict.keys())

        self.get_variance()

        self.var_df['feature'] = self.node_names[self.var_df.index]
        self.var_df['cost'] = np.zeros(len(self.var_df.index))
        self.var_df['cost_savings'] = np.zeros(len(self.var_df.index))
        # self.var_df['p_cost'] = np.zeros(len(self.var_df.index))
        
        self.var_df_by_depth['cost'] = \
            np.zeros(len(self.var_df_by_depth.index))
        self.var_df_by_depth['sse'] = \
            np.zeros(len(self.var_df_by_depth.index))
        self.var_df_by_depth['sse_red'] = \
            np.zeros(len(self.var_df_by_depth.index))

        self.calc_cost()

        global_sse = np.sum((self.target.values -
            np.mean(self.target.values)) ** 2)
        # print(global_sse)

        for i,nodes in enumerate(self.nodes_by_depth):
            depth = i + 1
            nodes_at_depth = nodes.copy()

            # print(nodes)
            # print(var_df.index)

            # sum cost of only nodes at this depth
            self.var_df_by_depth.loc[depth, 'cost'] = \
                self.var_df.reindex(nodes_at_depth)['cost'].sum()

            leaf_nodes = self.var_df.loc[(self.var_df['depth'] < depth) & \
                self.var_df['is_leaf']].index

            # nodes at depth including previous leaf nodes
            total_sse = self.var_df.reindex(nodes_at_depth)[['sse_left',
                'sse_right']].sum().sum()
            # print(depth, total_sse)
            self.var_df_by_depth.loc[depth, 'sse'] = total_sse
            self.var_df_by_depth.loc[depth, 'sse_red'] = global_sse - total_sse

        self.var_df_by_depth['total_cost'] = \
            self.var_df_by_depth['cost'].cumsum()
        self.var_df_by_depth['var_exp_imp'] = 1.0 - \
            self.var_df_by_depth['var_exp']


    def recurse_nodes(self, node, depth, cnodes, exclude_undef=False):

        indent = "  " * depth

        # print(depth, node)

        try:
            ndepth = self.nodes_by_depth[depth - 1]
        except:
            self.nodes_by_depth.append([])
            ndepth = self.nodes_by_depth[depth - 1]

        if exclude_undef:
            if self.tree_.children_left[node] == _tree.TREE_LEAF:
            # if self.tree_.children_left[node] == _tree.TREE_UNDEFINED:
                return cnodes
            else:
                ndepth.append(node)
        else:
            ndepth.append(node)
        # if depth == 4:
        #   print(depth, node, nodes_by_depth[depth -1])

        if self.tree_.feature[node] != _tree.TREE_UNDEFINED:

            left_cnodes = []
            left_cnodes = self.recurse_nodes(self.tree_.children_left[node],
                depth + 1, left_cnodes)

            # print('L',left_cnodes)

            right_cnodes = []
            right_cnodes = self.recurse_nodes(self.tree_.children_right[node],
                depth + 1, right_cnodes)

            # print(left_cnodes, right_cnodes)
            if len(left_cnodes) == 0:
                cnodes = [[], sorted(right_cnodes)]
                # cnodes = sorted(right_cnodes)
            elif len(right_cnodes) == 0:
                cnodes = [sorted(left_cnodes), []]
                # cnodes = sorted(left_cnodes)
            else:
                cnodes = [sorted(left_cnodes) , sorted(right_cnodes)]
                # cnodes = sorted(left_cnodes + right_cnodes)

            # cdict[node] = sorted(cnodes)

            # list of all nodes below current level for
            # left and right nodes
            self.cdict[node] = cnodes

            cnodes = list(itertools.chain.from_iterable(cnodes))   

            cnodes.append(node)     

            return cnodes            

        else:
            # cnodes.append(node)
            # pass
            return cnodes


    def get_variance(self, global_var_exp=True):

        # print(tree_.impurity[0])
        # print(np.var(self.target))
        global_mean = np.mean((self.target))
        global_mse = np.var((self.target))
        global_ssq = np.sum((self.target.values - global_mean) ** 2)
        global_num_nodes = len(self.target.values)

        if self.print_flag:
            print(global_num_nodes)

        # global_num_nodes = 10

        depths = np.arange(len(self.nodes_by_depth)) + 1

        self.var_df = pd.DataFrame(
            # np.zeros((len(cdict.keys()), 11)), 
            index=sorted(self.cdict.keys()), 
            columns=['var_total', 'var_left', 
            'var_right', 'impurity', 'impurity_calc', 'var_exp', 'var_exp_part',
             'sse_left', 'sse_right', 'sse', 'depth', 'is_leaf', 'var_exp_left',
             'var_exp_right','cost','cost_savings'])

        self.var_df_by_depth = pd.DataFrame(
            # np.zeros((len(depths), 5)), 
            index=depths,
            columns=['var_total', 'var_exp', 'var_exp_part', 'ssq_condition', 
            'num_targets','cost', 'sse','sse_red','total_cost','var_exp_imp'])

        targets_by_depth = []
        
        # print(var_df)

        num_undef = 0.0

        var_sides = ['var_left', 'var_right']

        for i,nodes in enumerate(self.nodes_by_depth):

            all_targets = []

            depth = i + 1

            ssq_by_node = []
            ssq_pct_by_node = []
            var_by_node = []
            mse_pct_decrease_by_node = []
            numnodes_by_node = []
            par_nodes = []

            global_ssq_condition = 0.0
            num_global_groups = 0
            total_targets = 0
            # ssq_condition = {}
            # ssq_errors = {}     

            if self.print_flag:
                print('Depth', depth)


            for anode in nodes:
                if False:
                    par_node = self.parent_dict[anode]
                    par_nodes.append(par_node)              

                    anode_targ_values, anode_targ_idx = \
                        self.get_target_values(anode, side_ind)   

                    pnode_targ_values, pnode_targ_idx = \
                        self.get_target_values(par_node, side_ind)

                if self.node_names[anode] == 'undefined!':
                    num_undef += 1
                    continue

                leaf_flag = False


                if np.any(np.isin(np.array(self.children_dict[anode]), _tree.TREE_LEAF)):
                    leaf_flag = True
                self.var_df.loc[anode, 'is_leaf'] = leaf_flag

                if anode == 0:
                    par_node = _tree.TREE_LEAF
                    side_ind = _tree.TREE_LEAF
                else:
                    par_node = self.parent_dict[anode]
                    side_ind = self.children_dict[par_node].index(anode)

                anode_targ_values, anode_targ_idx = \
                    self.get_target_values(par_node, side_ind)   

                # left_node,right_node = children_dict[anode]
                var_by_node.append(np.var(anode_targ_values)) 
                numnodes_by_node.append(len(anode_targ_idx))                

                anode_mean = np.mean(anode_targ_values)
                anode_mse = np.var(anode_targ_values)

                temp_df_values = self.target.values[anode_targ_idx - 1]
                anode_ssq = np.sum((temp_df_values - anode_mean) ** 2)                  
                
                self.var_df.loc[anode, 'depth'] = depth
                self.var_df.loc[anode, 'impurity'] = self.tree_.impurity[anode]
                self.var_df.loc[anode, 'var_total'] = np.var(anode_targ_values)
                self.var_df.loc[anode, 'sse'] = \
                    np.sum((anode_targ_values - anode_mean) ** 2)
            
                if not leaf_flag:

                    left_targ_values, left_targ_idx = \
                        self.get_target_values(anode, 0)   

                    right_targ_values, right_targ_idx = \
                        self.get_target_values(anode, 1)  

                    all_targets.extend(left_targ_idx) 
                    all_targets.extend(right_targ_idx)

                    var_left = np.var(left_targ_values)
                    var_right = np.var(right_targ_values)

                    # var_df.loc[anode, 'var_total'] = np.var(anode_targ_values)
                    self.var_df.loc[anode, 'var_left'] = var_left
                    self.var_df.loc[anode, 'var_right'] = var_right             

                    temp_numnodes = len(left_targ_values) + \
                        len(right_targ_values)

                    if temp_numnodes == 0 and False:
                        print(anode, temp_numnodes, var_left, var_right)

                    self.var_df.loc[anode, 'impurity_calc'] = \
                        (len(left_targ_values) / temp_numnodes) * var_left + \
                        (len(right_targ_values) / temp_numnodes) * var_right

                    if anode != 0 and False:
                        # print(var_df.loc[par_node])
                        print('var_total == par_side_var: ', \
                            self.var_df.loc[anode]['var_total'] == \
                            self.var_df.loc[par_node][var_sides[side_ind]])

                    if False:
                        pnode_targ_mean = np.mean(pnode_targ_values)
                        pnode_ssq = np.sum((pnode_targ_values - \
                            pnode_targ_mean) ** 2)

                        temp_targvals = self.target.values[anode_targ_idx - 1]
                        ssq_alt = np.sum((temp_targvals - \
                            pnode_targ_mean) ** 2)

                        ssq_by_node.append(ssq_alt)
                        ssq_pct_by_node.append(ssq_alt / pnode_ssq)
                    
                    num_groups = 2
                    ssq_condition = 0.0
                    temp_global_ssq = 0.0

                    for j,targ_values in enumerate([left_targ_values,
                        right_targ_values]):

                        targ_mean = np.mean(targ_values)
                        targ_num = len(targ_values)

                        total_targets += targ_num

                        side_global_ssq = targ_num *  (targ_mean - \
                            global_mean) ** 2

                        temp_global_ssq += side_global_ssq
                        num_global_groups += 1

                        local_ssq = targ_num *  (targ_mean - anode_mean) ** 2

                        ssq_condition += local_ssq

                        sse = np.sum((targ_values - targ_mean) ** 2)

                        # print(global_mse)
                        # print(side_global_ssq)
                        # print(global_ssq)
                        # print(anode)

                        if j == 0:
                            self.var_df.loc[anode, 'sse_left'] = sse
                            if global_var_exp:
                                self.var_df.loc[anode, 'var_exp_left'] = \
                                    (side_global_ssq - (1 - 1) * global_mse) / \
                                    (global_ssq + global_mse)                   
                            else:
                                self.var_df.loc[anode, 'var_exp_left'] = \
                                    (local_ssq - (1 - 1) * anode_mse) / \
                                    (anode_ssq + anode_mse)                                 
                        else:
                            self.var_df.loc[anode, 'sse_right'] = sse
                            if global_var_exp:
                                self.var_df.loc[anode, 'var_exp_right'] = \
                                    (side_global_ssq - (1 - 1) * global_mse) / \
                                    (global_ssq + global_mse)                   
                            else:
                                self.var_df.loc[anode, 'var_exp_right'] = \
                                    (local_ssq - (1 - 1) * anode_mse) / \
                                    (anode_ssq + anode_mse)

                else:
                    self.var_df.loc[anode, 'var_left'] = 0.0
                    self.var_df.loc[anode, 'var_right'] = 0.0
                    self.var_df.loc[anode, 'sse_left'] = 0.0
                    self.var_df.loc[anode, 'sse_right'] = 0.0
                    self.var_df.loc[anode, 'var_exp_left'] = 0.0                 
                    self.var_df.loc[anode, 'var_exp_right'] = 0.0
                    self.var_df.loc[anode, 'impurity_calc'] =  anode_mse

                    temp_global_ssq = 0.0
                    ssq_condition = 0.0

                    num_groups = 1

                    targ_num = len(anode_targ_values)

                    total_targets += targ_num

                    temp_global_ssq = targ_num *  (anode_mean - \
                        global_mean) ** 2

                    num_global_groups += 1
                                        

                if global_var_exp:

                    var_exp = (temp_global_ssq - (num_groups) * global_mse) / \
                        (global_ssq + global_mse)    

                    var_exp_partial = (temp_global_ssq - (num_groups) *
                        global_mse) / (temp_global_ssq + (global_num_nodes - \
                        (num_groups - 1)) * global_mse)

                else:

                    var_exp = (ssq_condition - (num_groups - 1) * anode_mse) / \
                        (anode_ssq + anode_mse)

                    var_exp_partial = (ssq_condition - (num_groups - 1) * \
                        anode_mse) / (ssq_condition + (len(anode_targ_values) - \
                        (num_groups - 1)) * anode_mse)

                global_ssq_condition += temp_global_ssq             

                # print(len(anode_targ_values))

                self.var_df.loc[anode, 'var_exp'] = var_exp
                self.var_df.loc[anode, 'var_exp_part'] = var_exp_partial

            leaf_df = self.var_df.loc[(self.var_df['depth'] < depth) & \
                self.var_df['is_leaf']]
            leaf_df = leaf_df.sort_values(by=['depth'], ascending=False)
            leaf_nodes = leaf_df.index
            # print('number of leaves: ', len(leaf_nodes))

            for j,anode in enumerate(leaf_nodes):

                if anode in nodes:
                    # skip leaf nodes at current depth
                    continue

                if False:
                    left_targ_values, left_targ_idx = \
                        self.get_target_values(anode, 0)   

                    right_targ_values, right_targ_idx = \
                        self.get_target_values(anode, 1) 

                    idx_list = [left_targ_idx, right_targ_idx]
                
                    for k,targ_values in enumerate([left_targ_values,
                        right_targ_values]): 

                        idx_keep = np.setdiff1d(idx_list[k], all_targets)
                        idx_mask = np.nonzero(np.isin(idx_list[k],
                            idx_keep))[0]
                        # print('overlap: ', len(np.intersect1d(all_targets, idx_keep)))

                        all_targets.extend(idx_keep) 
                        targ_values = targ_values[idx_mask]
                        targ_mean = np.mean(targ_values)
                        targ_num = len(targ_values)

                        if targ_num == 0:
                            continue

                        total_targets += targ_num
                        temp_ssq_cond = targ_num *  (targ_mean - \
                            global_mean) ** 2
                        # print(anode,k,temp_ssq_cond,targ_num, targ_mean)
                        global_ssq_condition += temp_ssq_cond
                        num_global_groups += 1

                par_node = self.parent_dict[anode]
                side_ind = self.children_dict[par_node].index(anode)
                anode_targ_values, anode_targ_idx = \
                    self.get_target_values(par_node, side_ind)

                targ_num = len(anode_targ_values)

                idx_keep = np.setdiff1d(anode_targ_idx, all_targets)
                idx_mask = np.nonzero(np.isin(anode_targ_idx, idx_keep))[0]         
                all_targets.extend(idx_keep)

                total_targets += targ_num

                global_ssq_condition += targ_num * \
                    (np.mean(anode_targ_values) - global_mean) ** 2
                num_global_groups += 1                  

            targets_by_depth.append(all_targets)
            # print(len(all_targets), num_global_groups, global_ssq_condition)              


            var_exp = (global_ssq_condition - (num_global_groups - 1) * \
                global_mse) / (global_ssq + global_mse)

            var_exp_partial = (global_ssq_condition - (num_global_groups - 1) * \
                global_mse) / (global_ssq_condition + (global_num_nodes - \
                (num_global_groups - 1)) * global_mse)

            self.var_df_by_depth.loc[depth, 'var_total'] = global_mse
            self.var_df_by_depth.loc[depth, 'var_exp'] = var_exp
            self.var_df_by_depth.loc[depth, 'var_exp_part'] = var_exp_partial
            self.var_df_by_depth.loc[depth, 'ssq_condition'] = \
                global_ssq_condition

            self.var_df_by_depth.loc[depth, 'num_targets'] = total_targets

        if self.print_flag:
            print('num_undefined: ', num_undef)


    def get_target_values(self, anode, side_ind):

        if side_ind == _tree.TREE_LEAF:
            return self.target.values, np.arange(len(self.df.index)) + 1 

        temp_node = anode

        idx = np.arange(len(self.df.index))

        while 1:
            name = self.node_names[temp_node]
            threshold = self.tree_.threshold[temp_node]

            if side_ind == 0:
                temp_idx = np.nonzero(self.df[name].values[idx] <= threshold)
            else:
                temp_idx = np.nonzero(self.df[name].values[idx] > threshold)

            idx = idx[temp_idx]

            prev_temp_node = temp_node
            temp_node = self.parent_dict[temp_node]
            if prev_temp_node == 0:
                break

            side_ind = self.children_dict[temp_node].index(prev_temp_node)            

        anode_target_values = self.target.values[idx]

        return anode_target_values, idx


    def calc_cost(self):

        include_all_nodes = True

        all_obs = np.array([])

        for i,nodes in enumerate(self.nodes_by_depth):

            for j,node in enumerate(nodes):
                cur_obs = self.node_names[node]

                if cur_obs == 'undefined!':
                    continue

                if np.any(np.isin(all_obs, cur_obs)):
                    cost = 0.0
                    # cost_savings = 0.0
                else:
                    cost = self.calc_soil_obs_cost_v2(all_obs, cur_obs)

                cost_savings = self.calc_soil_obs_cost_savings(all_obs, cur_obs)

                self.var_df.loc[node, 'cost'] = cost
                self.var_df.loc[node, 'cost_savings'] = cost_savings

            if not include_all_nodes:
                nodes = np.array(nodes, dtype=int)
                new_obs = self.node_names[nodes]
                all_obs = np.unique(np.append(all_obs, new_obs))

                undef_idx = np.nonzero(all_obs == 'undefined!')[0]
                if len(undef_idx) > 0:
                    all_obs = np.delete(all_obs, undef_idx)

            else:
                all_obs = np.unique(np.append(all_obs, self.node_names[nodes]))

                undef_idx = np.nonzero(all_obs == 'undefined!')[0]
                if len(undef_idx) > 0:
                    all_obs = np.delete(all_obs, undef_idx)    

        return all_obs


    def calc_soil_obs_cost_v1(self, prev_obs, cur_obs):
        # rules assign cost based on existing features

        prev_obs_dict = self.populate_obs_dict(prev_obs)

        sp_obs = cur_obs.split(':')
        obs_type = sp_obs[0]
        depth = sp_obs[1][1:]
        time = sp_obs[2][1:]

        cost = 0.0
        
        if obs_type == 'wc':
            
            if (not np.any(np.isin(prev_obs_dict[obs_type]['d'], depth))) & \
                (not np.any(np.isin(prev_obs_dict['p']['d'], depth))):
                cost += 5 + 1 # depth & time

            else:
                if np.any(np.isin(prev_obs_dict[obs_type]['d'], depth)):

                    idx = np.nonzero(prev_obs_dict[obs_type]['d'] == depth)[0]

                    if not np.any(np.isin(prev_obs_dict[obs_type]['t'][idx],
                        time)):
                        cost += 1 # new time

                if np.any(np.isin(prev_obs_dict['p']['d'], depth)):
                    cost += 2.5
                    
                    idx = np.nonzero(prev_obs_dict['p']['d'] == depth)[0]
                    if not np.any(np.isin(prev_obs_dict['p']['t'][idx], time)):                 
                        cost += 1
            
        if obs_type == 'p':
            
            if (not np.any(np.isin(prev_obs_dict[obs_type]['d'], depth))) & \
                (not np.any(np.isin(prev_obs_dict['wc']['d'], depth))):
                cost += 12 + 2 # new depth and time

            else:
                if np.any(np.isin(prev_obs_dict[obs_type]['d'], depth)):

                    idx = np.nonzero(prev_obs_dict[obs_type]['d'] == depth)[0]
                    
                    if not np.any(np.isin(prev_obs_dict[obs_type]['t'][idx],
                        time)):
                        cost += 1 # new time

                if np.any(np.isin(prev_obs_dict['wc']['d'], depth)):
                    cost += 6

                    idx = np.nonzero(prev_obs_dict['wc']['d'] == depth)[0]
                    if not np.any(np.isin(prev_obs_dict['wc']['t'][idx], time)):                    
                        cost += 1

        return cost


    def calc_soil_obs_cost_v2(self, prev_obs, cur_obs):
        # rules assign cost based on existing features

        prev_obs_dict = self.populate_obs_dict(prev_obs)

        sp_obs = cur_obs.split(':')
        obs_type = sp_obs[0]
        depth = sp_obs[1][1:]
        time = sp_obs[2][1:]

        cost = 0.0

        # initial setup cost
        if (len(prev_obs_dict[obs_type]['d']) == 0):
            cost += self.initial_cost[obs_type]

        # depth cost - waived if a sensor of any type already exists at
        # this depth or deeper (the borehole already reaches this depth)
        existing_depths = np.concatenate(
            [prev_obs_dict['wc']['d'], prev_obs_dict['p']['d']]).astype(float)
        if not np.any(existing_depths >= float(depth)):
            cost += self.depth_cost[depth]

        # sensor and measurement cost
        if not np.any(np.isin(prev_obs_dict[obs_type]['d'], depth)):
            cost += self.sensor_cost[obs_type]
            cost += self.measurement_cost[obs_type]

        else:
            idx = np.nonzero(prev_obs_dict[obs_type]['d'] == depth)[0]

            if not np.any(np.isin(prev_obs_dict[obs_type]['t'][idx],
                time)):
                cost += self.measurement_cost[obs_type]

        return cost


    def calc_soil_obs_cost_savings(self, prev_obs, cur_obs):
        # rules assign cost based on existing features

        prev_obs_dict = self.populate_obs_dict(prev_obs)

        # print(cur_obs)
        # print(prev_obs)

        sp_obs = cur_obs.split(':')
        obs_type = sp_obs[0]
        depth = sp_obs[1][1:]
        time = sp_obs[2][1:]

        cost = 0.0
        cost -= self.initial_cost[obs_type]
        cost -= self.sensor_cost[obs_type]
        cost -= self.depth_cost[depth]
        cost -= self.measurement_cost[obs_type]

        # print(cost)

        # initial setup cost
        if (len(prev_obs_dict[obs_type]['d']) > 0):
            cost += self.initial_cost[obs_type]

        # depth cost
        if (np.any(np.isin(prev_obs_dict['wc']['d'], depth))) | \
            (np.any(np.isin(prev_obs_dict['p']['d'], depth))):

            num = np.count_nonzero(np.isin(prev_obs_dict['wc']['d'], depth))
            num += np.count_nonzero(np.isin(prev_obs_dict['p']['d'], depth))

            # print(cur_obs, cost, num, self.depth_cost[depth])

            cost += self.depth_cost[depth] * num

            # print('depth',num)

            # print(cur_obs, cost, num, self.depth_cost[depth])

        # sensor and measurement cost
        if np.any(np.isin(prev_obs_dict[obs_type]['d'], depth)):

            num = np.count_nonzero(np.isin(prev_obs_dict[obs_type]['d'], depth))

            cost += self.sensor_cost[obs_type] * num

            # print('sensor',num)

            idx = np.nonzero(prev_obs_dict[obs_type]['d'] == depth)[0]

            if np.any(np.isin(prev_obs_dict[obs_type]['t'][idx],
                time)):

                num = np.count_nonzero(np.isin(prev_obs_dict[obs_type]['t'][idx], time))

                # print('time',num)

                cost += self.measurement_cost[obs_type] * num

        # print(cur_obs, cost)

        # print(cost)

        return cost


    def populate_obs_dict(self, prev_obs):

        prev_obs_dict = {
            'wc':{
                'd':np.array([]),
                't':np.array([])  
            },
            'p':{
                'd':np.array([]),
                't':np.array([])  
            }
        }

        for a_prv_obs in prev_obs:
            sp_prv_obs = a_prv_obs.split(':')
            obs_type = sp_prv_obs[0]
            depth = sp_prv_obs[1][1:]
            time = sp_prv_obs[2][1:]

            prev_obs_dict[obs_type]['d'] = \
                np.append(prev_obs_dict[obs_type]['d'], depth)
            prev_obs_dict[obs_type]['t'] = \
                np.append(prev_obs_dict[obs_type]['t'], time)

        return prev_obs_dict

