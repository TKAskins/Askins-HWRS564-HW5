

import _splitter_cython as splitter
import _criterion_cython as splitter
# import _utils_cython as utils
# import _classes as c

print(df.shape())

n_outputs = 1
n_samples = shape[0] 


max_features = None
min_samples_leaf = 1
min_weight_leaf = 0.0
random_state = 0
monotonic_cst = None
sensor_cost = [2.0,3.0]
sensor_cost = np.array(sensor_cost).astype(np.float64)
time_cost = 1.0
depth_cost = 5.0
cost_threshold = 0.2

criterion = _criterion_cython.MSE(n_outputs, n_samples)

splitter = _splitter_cython.BestSplitter(
    criterion,
    max_features,
    min_samples_leaf,
    min_weight_leaf,
    random_state,
    monotonic_cst,
    sensor_cost,
    time_cost,
    depth_cost,
    cost_threshold,
)

X = 
y = 

sample_weight = None
missing_values_in_feature_mask = False

splitter.init(X, y, sample_weight, missing_values_in_feature_mask)




# start = stack_record.start
# end = stack_record.end
# depth = stack_record.depth
# parent = stack_record.parent
# is_left = stack_record.is_left
impurity = stack_record.impurity
# n_constant_features = stack_record.n_constant_features
lower_bound = stack_record.lower_bound
upper_bound = stack_record.upper_bound

n_node_samples = end - start
splitter.node_reset(start, end, &weighted_n_node_samples)

is_leaf = (depth >= max_depth or
           n_node_samples < min_samples_split or
           n_node_samples < 2 * min_samples_leaf or
           weighted_n_node_samples < 2 * min_weight_leaf)

if first:
    impurity = splitter.node_impurity()
    first = 0

splitter.node_split(
    impurity,
    &split,
    &n_constant_features,
    lower_bound,
    upper_bound,
    tree,
    sensor_types,
    depth_types,
    time_types,
)


impurity =
split =
n_constant_features = 0
lower_bound =
upper_bound = 
tree = 
sensor_types = 
depth_types = 
time_types = 

splitter.node_split()
