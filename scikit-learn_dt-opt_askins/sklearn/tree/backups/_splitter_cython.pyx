# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#
# License: BSD 3 clause

cimport numpy as cnp

from _criterion_cython cimport Criterion
from _tree_cython cimport Tree, Node

from libc.stdlib cimport qsort
from libc.string cimport memcpy
from libc.math cimport isnan
from cython cimport final

import numpy as np

from scipy.sparse import issparse

from _utils_cython cimport log
from _utils_cython cimport rand_int
from _utils_cython cimport rand_uniform
from _utils_cython cimport RAND_R_MAX


from libcpp.vector cimport vector
from libc.stdio cimport printf,sprintf
from libc.math cimport fabs

# from libc.stdint cimport cnp.int32_t

cnp.import_array()

cdef cnp.float64_t INFINITY = np.inf

# Mitigate precision differences between 32 bit and 64 bit
cdef cnp.float32_t FEATURE_THRESHOLD = 1e-7

# Constant to switch between algorithm non zero value extract algorithm
# in SparsePartitioner
cdef cnp.float32_t EXTRACT_NNZ_SWITCH = 0.1

# cdef extern from "stdio.h":
#     FILE *fopen(const char *filename, const char *mode)
#     int fprintf(FILE *stream, const char *format, ...)
#     int fclose(FILE *stream)

# cdef FILE *file = fopen("log.txt", "w")
# with nogil:
#     fprintf(file, "Logging from nogil\n")
# fclose(file)




cdef cnp.float64_t abs_float(cnp.float64_t x) noexcept nogil:
    if x < 0:
        return -x
    else:
        return x

cdef inline void _add_to_split_list(
    SplitRecord rec,
    vector[SplitRecord]& split_list) noexcept nogil:
    """Adds record `rec` to the priority queue `split_list`."""
    split_list.push_back(rec)

cdef inline int find_max_in_2d_array(vector[vector[cnp.float64_t]]  arr) noexcept nogil:
    cdef int rows = arr.size()
    cdef int col = 0
    cdef int max_ind = 0
    cdef cnp.float64_t max_val = arr[0][0]
    cdef int i, j

    # Iterate through the array to find the maximum value
    for i in range(rows):
        if arr[i][col] > max_val:
            max_val = arr[i][col]
            max_ind = i

    return max_ind

cdef inline int find_min_in_2d_array(vector[vector[cnp.float64_t]]  arr) noexcept nogil:
    cdef int rows = arr.size()
    cdef int col = 0
    cdef int min_ind = 0
    cdef cnp.float64_t min_val = arr[0][0]
    cdef int i, j

    # Iterate through the array to find the maximum value
    for i in range(rows):
        if arr[i][col] < min_val:
            min_val = arr[i][col]
            min_ind = i

    return min_ind

cdef inline void inplace_sort_2d_array_by_two_columns(vector[vector[cnp.float64_t]] arr, \
    int col1, int col2) noexcept nogil:
    cdef int i, j, k
    cdef int rows = arr.size()
    cdef int cols = arr[0].size()
    cdef cnp.float64_t tmp

    for i in range(rows - 1):
        for j in range(rows - i - 1):
            if (arr[j][col1] < arr[j + 1][col1]) or \
               ((arr[j][col1] == arr[j + 1][col1]) and (arr[j][col2] < arr[j + 1][col2])):
                # Swap rows
                for k in range(cols):
                    tmp = arr[j][k]
                    arr[j][k] = arr[j + 1][k]
                    arr[j + 1][k] = tmp

cdef inline void set_2d_array_to_zero(vector[vector[cnp.float64_t]] arr) noexcept nogil:
    cdef int i, j
    cdef int rows = arr.size()
    cdef int cols = arr[0].size()

    for i in range(rows):
        for j in range(cols):
            arr[i][j] = 0.0

cdef inline void _init_split(SplitRecord* self, cnp.intp_t start_pos) noexcept nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY
    self.missing_go_to_left = False
    self.n_missing = 0

cdef class Splitter:
    """Abstract splitter class.

    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """

    def __cinit__(
        self,
        Criterion criterion,
        cnp.intp_t max_features,
        cnp.intp_t min_samples_leaf,
        cnp.float64_t min_weight_leaf,
        object random_state,
        const cnp.int8_t[:] monotonic_cst,
        cnp.float64_t[:] sensor_cost,
        cnp.float64_t time_cost,
        cnp.float64_t depth_cost,
        cnp.float64_t cost_threshold,        
    ):

        # Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)

        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features : cnp.intp_t
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf : cnp.intp_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : cnp.float64_t
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : object
            The user inputted random state to be used for pseudo-randomness

        monotonic_cst : const cnp.int8_t[:]
            Monotonicity constraints

        """

        self.criterion = criterion

        self.n_samples = 0
        self.n_features = 0

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.monotonic_cst = monotonic_cst
        self.with_monotonic_cst = monotonic_cst is not None

        self.sensor_cost = sensor_cost
        self.time_cost = time_cost
        self.depth_cost = depth_cost
        self.cost_threshold = cost_threshold

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def __reduce__(self):
        return (type(self), (self.criterion,
                             self.max_features,
                             self.min_samples_leaf,
                             self.min_weight_leaf,
                             self.random_state,
                             self.monotonic_cst,
                             self.sensor_cost,
                             self.time_cost,
                             self.depth_cost,
                             self.cost_threshold), self.__getstate__())

    cdef int init(
        self,
        object X,
        const cnp.float64_t[:, ::1] y,
        const cnp.float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1:
        """Initialize the splitter.

        Take in the input data X, the target Y, and optional sample weights.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.

        y : ndarray, dtype=cnp.float64_t
            This is the vector of targets, or true labels, for the samples represented
            as a Cython memoryview.

        sample_weight : ndarray, dtype=cnp.float64_t
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight. This is represented
            as a Cython memoryview.

        has_missing : bool
            At least one missing values is in X.
        """

        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef cnp.intp_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        self.samples = np.empty(n_samples, dtype=np.intp_t)
        cdef cnp.intp_t[:] samples = self.samples

        cdef cnp.intp_t i, j
        cdef cnp.float64_t weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight is None or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight is not None:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef cnp.intp_t n_features = X.shape[1]
        self.features = np.arange(n_features, dtype=np.intp)
        self.n_features = n_features

        self.feature_values = np.empty(n_samples, dtype=np.float32)
        self.constant_features = np.empty(n_features, dtype=np.intp)

        self.y = y

        self.sample_weight = sample_weight
        if missing_values_in_feature_mask is not None:
            self.criterion.init_sum_missing()
        return 0

    cdef int node_reset(
        self,
        cnp.intp_t start,
        cnp.intp_t end,
        cnp.float64_t* weighted_n_node_samples
    ) except -1 nogil:
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : cnp.intp_t
            The index of the first sample to consider
        end : cnp.intp_t
            The index of the last sample to consider
        weighted_n_node_samples : ndarray, dtype=cnp.float64_t pointer
            The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.init(
            self.y,
            self.sample_weight,
            self.weighted_n_samples,
            self.samples,
            start,
            end
        )

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    cdef int node_split(
        self,
        cnp.float64_t impurity,
        SplitRecord* split,
        cnp.intp_t* n_constant_features,
        cnp.float64_t lower_bound,
        cnp.float64_t upper_bound,
        Tree tree,
        cnp.int32_t[:] sensor_types,
        cnp.int32_t[:] depth_types,
        cnp.int32_t[:] time_types,        
    ) except -1 nogil:

        """Find the best split on node samples[start:end].

        This is a placeholder method. The majority of computation will be done
        here.

        It should return -1 upon errors.
        """

        pass

    cdef void node_value(self, cnp.float64_t* dest) noexcept nogil:
        """Copy the value of node samples[start:end] into dest."""

        self.criterion.node_value(dest)

    cdef inline void clip_node_value(self, cnp.float64_t* dest, cnp.float64_t lower_bound, cnp.float64_t upper_bound) noexcept nogil:
        """Clip the value in dest between lower_bound and upper_bound for monotonic constraints."""

        self.criterion.clip_node_value(dest, lower_bound, upper_bound)

    cdef cnp.float64_t node_impurity(self) noexcept nogil:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()


cdef inline int node_split_best(
    Splitter splitter,
    DensePartitioner partitioner,
    Criterion criterion,
    cnp.float64_t impurity,
    SplitRecord* split,
    cnp.intp_t* n_constant_features,
    bint with_monotonic_cst,
    const cnp.int8_t[:] monotonic_cst,
    cnp.float64_t lower_bound,
    cnp.float64_t upper_bound,
    Tree tree,
    cnp.int32_t[:] sensor_types,
    cnp.int32_t[:] depth_types,
    cnp.int32_t[:] time_types,    
) except -1 nogil:
    """Find the best split on node samples[start:end]

    Returns -1 in case of failure to allocate memory (and raise MemoryError)
    or 0 otherwise.
    """
    # Find the best split
    cdef cnp.intp_t start = splitter.start
    cdef cnp.intp_t end = splitter.end
    cdef cnp.intp_t end_non_missing
    cdef cnp.intp_t n_missing = 0
    cdef bint has_missing = 0
    cdef cnp.intp_t n_searches
    cdef cnp.intp_t n_left, n_right
    cdef bint missing_go_to_left

    cdef cnp.intp_t[:] samples = splitter.samples
    cdef cnp.intp_t[::1] features = splitter.features
    cdef cnp.intp_t[::1] constant_features = splitter.constant_features
    cdef cnp.intp_t n_features = splitter.n_features

    cdef cnp.float32_t[::1] feature_values = splitter.feature_values
    cdef cnp.intp_t max_features = splitter.max_features
    cdef cnp.intp_t min_samples_leaf = splitter.min_samples_leaf
    cdef cnp.float64_t min_weight_leaf = splitter.min_weight_leaf
    cdef cnp.uint32_t* random_state = &splitter.rand_r_state

    cdef SplitRecord best_split, current_split
    cdef cnp.float64_t current_proxy_improvement = -INFINITY
    cdef cnp.float64_t best_proxy_improvement = -INFINITY

    cdef cnp.intp_t f_i = n_features
    cdef cnp.intp_t f_j
    cdef cnp.intp_t p
    cdef cnp.intp_t p_prev

    cdef cnp.intp_t n_visited_features = 0
    # Number of features discovered to be constant during the split search
    cdef cnp.intp_t n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    cdef cnp.intp_t n_drawn_constants = 0
    cdef cnp.intp_t n_known_constants = n_constant_features[0]
    # n_total_constants = n_known_constants + n_found_constants
    cdef cnp.intp_t n_total_constants = n_known_constants

    cdef Node* node
    cdef Node* nodes = tree.nodes
    cdef cnp.float64_t[:] sensor_cost = splitter.sensor_cost
    cdef cnp.float64_t time_cost = splitter.time_cost
    cdef cnp.float64_t depth_cost = splitter.depth_cost
    cdef cnp.float64_t current_cost
    cdef cnp.float64_t max_id
    cdef cnp.float64_t diff
    cdef cnp.float64_t cost_threshold = splitter.cost_threshold    
    cdef cnp.intp_t node_count = tree.node_count
    cdef cnp.intp_t feature
    cdef cnp.int32_t sensor_n
    cdef cnp.int32_t depth_n
    cdef cnp.int32_t time_n
    cdef cnp.int32_t sensor_f
    cdef cnp.int32_t depth_f
    cdef cnp.int32_t time_f
    cdef int max_ind
    cdef int min_ind
    cdef int feature_ind
    cdef int n_ind
    cdef int ind
    cdef int best_ind
    cdef int r_ind
    cdef int c_ind
    # cdef boolean best_flag
    cdef vector[vector[cnp.float64_t]] obj_arr
    cdef vector[SplitRecord] split_list
    cdef char buffer[100]

    # Resize the array
    obj_arr.resize(n_features)
    for i in range(n_features):
        obj_arr[i].resize(3)

    _init_split(&best_split, end)

    partitioner.init_node_split(start, end)

    while (f_i > n_total_constants and  # Stop early if remaining features
                                        # are constant
            (n_visited_features < max_features or
             # At least one drawn features must be non constant
             n_visited_features <= n_found_constants + n_drawn_constants)):

        n_visited_features += 1

        # Draw a feature at random
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                       random_state)

        if f_j < n_known_constants:
            # f_j in the interval [n_drawn_constants, n_known_constants[
            features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]

            n_drawn_constants += 1
            continue

        # f_j in the interval [n_known_constants, f_i - n_found_constants[
        f_j += n_found_constants
        # f_j in the interval [n_total_constants, f_i[
        current_split.feature = features[f_j]
        partitioner.sort_samples_and_feature_values(current_split.feature)
        n_missing = partitioner.n_missing
        end_non_missing = end - n_missing

        if (
            # All values for this feature are missing, or
            end_non_missing == start or
            # This feature is considered constant (max - min <= FEATURE_THRESHOLD)
            feature_values[end_non_missing - 1] <= feature_values[start] + FEATURE_THRESHOLD
        ):
            # We consider this feature constant in this case.
            # Since finding a split among constant feature is not valuable,
            # we do not consider this feature for splitting.
            features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]

            n_found_constants += 1
            n_total_constants += 1
            continue

        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]
        has_missing = n_missing != 0
        criterion.init_missing(n_missing)  # initialize even when n_missing == 0

        # Evaluate all splits

        # If there are missing values, then we search twice for the most optimal split.
        # The first search will have all the missing values going to the right node.
        # The second search will have all the missing values going to the left node.
        # If there are no missing values, then we search only once for the most
        # optimal split.
        n_searches = 2 if has_missing else 1

        for i in range(n_searches):
            missing_go_to_left = i == 1
            criterion.missing_go_to_left = missing_go_to_left
            criterion.reset()

            p = start

            while p < end_non_missing:
                partitioner.next_p(&p_prev, &p)

                if p >= end_non_missing:
                    continue

                if missing_go_to_left:
                    n_left = p - start + n_missing
                    n_right = end_non_missing - p
                else:
                    n_left = p - start
                    n_right = end_non_missing - p + n_missing

                # Reject if min_samples_leaf is not guaranteed
                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue

                current_split.pos = p
                criterion.update(current_split.pos)

                # Reject if monotonicity constraints are not satisfied
                if (
                    with_monotonic_cst and
                    monotonic_cst[current_split.feature] != 0 and
                    not criterion.check_monotonicity(
                        monotonic_cst[current_split.feature],
                        lower_bound,
                        upper_bound,
                    )
                ):
                    continue

                # Reject if min_weight_leaf is not satisfied
                if ((criterion.weighted_n_left < min_weight_leaf) or
                        (criterion.weighted_n_right < min_weight_leaf)):
                    continue

                current_proxy_improvement = criterion.proxy_impurity_improvement()

                if current_proxy_improvement > best_proxy_improvement:
                    best_proxy_improvement = current_proxy_improvement
                    # sum of halves is used to avoid infinite value
                    current_split.threshold = (
                        feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                    )

                    if (
                        current_split.threshold == feature_values[p] or
                        current_split.threshold == INFINITY or
                        current_split.threshold == -INFINITY
                    ):
                        current_split.threshold = feature_values[p_prev]

                    current_split.n_missing = n_missing
                    if n_missing == 0:
                        current_split.missing_go_to_left = n_left > n_right
                    else:
                        current_split.missing_go_to_left = missing_go_to_left

                    current_split.improvement = best_proxy_improvement

                    best_split = current_split  # copy

        # Evaluate when there are missing values and all missing values goes
        # to the right node and non-missing values goes to the left node.
        if has_missing:
            n_left, n_right = end - start - n_missing, n_missing
            p = end - n_missing
            missing_go_to_left = 0

            if not (n_left < min_samples_leaf or n_right < min_samples_leaf):
                criterion.missing_go_to_left = missing_go_to_left
                criterion.update(p)

                if not ((criterion.weighted_n_left < min_weight_leaf) or
                        (criterion.weighted_n_right < min_weight_leaf)):
                    current_proxy_improvement = criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        current_split.threshold = INFINITY
                        current_split.missing_go_to_left = missing_go_to_left
                        current_split.n_missing = n_missing
                        current_split.pos = p
                        best_split = current_split

        #### Cost Calculation
        current_cost = 0.0

        sensor_f = sensor_types[best_split.feature]
        depth_f = depth_types[best_split.feature]
        time_f = time_types[best_split.feature]

        
        time_flag = False
        depth_flag = False
        sensor_flag = False
        best_flag = False

        for node_id in range(node_count):
            node = &nodes[node_id]
            feature = node.feature

            sensor_n = sensor_types[node_id]
            depth_n = depth_types[node_id]
            time_n = time_types[node_id]

            # turning the flags off favors sensors of the same kind and type
            if (best_split.feature == feature) and not best_flag:
                current_cost += sensor_cost[sensor_f] \
                    + time_cost + depth_cost
                # best_flag = True

            else:
                if (sensor_f == sensor_n) and not sensor_flag:
                    # sensor_flag = True
                    current_cost += sensor_cost[sensor_f]

                if (depth_f == depth_n) and (sensor_f == sensor_n) and not depth_flag:
                    # depth_flag = True
                    current_cost += depth_cost

        best_split.cost = current_cost

        ind = <int>(n_visited_features - 1)

        # sprintf(buffer, "%zd",ind)
        # printf("ind: %s\n", buffer)

        # sprintf(buffer, "%f",best_split.improvement)
        # printf("imp: %s\n", buffer)

        # sprintf(buffer, "%f",current_cost)
        # printf("cost: %s\n", buffer)

        obj_arr[ind][0] = best_split.improvement
        obj_arr[ind][1] = best_split.cost
        obj_arr[ind][2] = <float>ind

        _add_to_split_list(best_split,split_list)

    # sprintf(buffer, "%zd",n_features)
    # printf("n_features: %s\n", buffer)

    ### Cost Objective Function
    inplace_sort_2d_array_by_two_columns(obj_arr, 1, 0)

    max_ind = find_max_in_2d_array(obj_arr)
    min_ind = find_min_in_2d_array(obj_arr)

    best_ind = 0



    n_ind = -1

    sprintf(buffer, "%zd",n_ind)
    printf("n_ind: %s\n", buffer)

    sprintf(buffer, "%zd",n_features)
    printf("n_feat: %s\n", buffer)

    while n_ind < <int>n_features - 1:
        n_ind += 1

        diff = fabs(obj_arr[max_ind][0] - obj_arr[n_ind][0]) / \
            (obj_arr[max_ind][0] - obj_arr[min_ind][0])

        sprintf(buffer, "%f",obj_arr[n_ind][0])
        printf("imp: %s\n", buffer)
        # fflush(stdout)

        sprintf(buffer, "%f",diff)
        printf("diff: %s\n", buffer)
        # fflush(stdout)

        sprintf(buffer, "%f",cost_threshold)
        printf("cost_threshold: %s\n", buffer)
        # fflush(stdout)

        if diff <= cost_threshold:
            best_ind = n_ind
            break

    feature_ind = <int>obj_arr[best_ind][2]
    best_split = split_list[feature_ind]

    sprintf(buffer, "%zd",n_ind)
    printf("n_ind: %s\n", buffer)    

    # best_split.cost = obj_arr[best_ind][1]

    # for r_ind in [best_ind, max_ind]:
    for c_ind in range(3):
        if c_ind < 2:
            sprintf(buffer, "%f",obj_arr[max_ind][c_ind])
            printf("%s\n", buffer)
        if c_ind == 2:
            feature_ind = <int>obj_arr[max_ind][c_ind]
            current_split = split_list[feature_ind]
            sprintf(buffer, "%zd",current_split.feature)
            printf("%s\n", buffer)                        

    for c_ind in range(3):
        if c_ind < 2:
            sprintf(buffer, "%f",obj_arr[best_ind][c_ind])
            printf("%s\n", buffer)
        if c_ind == 2:
            feature_ind = <int>obj_arr[best_ind][c_ind]
            current_split = split_list[feature_ind]
            sprintf(buffer, "%zd",current_split.feature)
            printf("%s\n", buffer)                        
    printf("\n") 

    # set_2d_array_to_zero(obj_arr)   
   
    # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
    if best_split.pos < end:
        partitioner.partition_samples_final(
            best_split.pos,
            best_split.threshold,
            best_split.feature,
            best_split.n_missing
        )
        criterion.init_missing(best_split.n_missing)
        criterion.missing_go_to_left = best_split.missing_go_to_left

        criterion.reset()
        criterion.update(best_split.pos)
        criterion.children_impurity(
            &best_split.impurity_left, &best_split.impurity_right
        )
        best_split.improvement = criterion.impurity_improvement(
            impurity,
            best_split.impurity_left,
            best_split.impurity_right
        )

        # shift_missing_values_to_left_if_required(&best_split, samples, end)

    # Respect invariant for constant features: the original order of
    # element in features[:n_known_constants] must be preserved for sibling
    # and child nodes
    memcpy(&features[0], &constant_features[0], sizeof(cnp.intp_t) * n_known_constants)

    # Copy newly found constant features
    memcpy(&constant_features[n_known_constants],
           &features[n_known_constants],
           sizeof(cnp.intp_t) * n_found_constants)

    # Return values
    split[0] = best_split
    n_constant_features[0] = n_total_constants
    return 0


@final
cdef class DensePartitioner:
    """Partitioner specialized for dense data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    cdef:
        const cnp.float32_t[:, :] X
        cdef cnp.intp_t[:] samples
        cdef cnp.float32_t[::1] feature_values
        cdef cnp.intp_t start
        cdef cnp.intp_t end
        cdef cnp.intp_t n_missing
        cdef const unsigned char[::1] missing_values_in_feature_mask

    def __init__(
        self,
        const cnp.float32_t[:, :] X,
        cnp.intp_t[:] samples,
        cnp.float32_t[::1] feature_values,
        const unsigned char[::1] missing_values_in_feature_mask,
    ):
        self.X = X
        self.samples = samples
        self.feature_values = feature_values
        self.missing_values_in_feature_mask = missing_values_in_feature_mask

    cdef inline void init_node_split(self, cnp.intp_t start, cnp.intp_t end) noexcept nogil:
        """Initialize splitter at the beginning of node_split."""
        self.start = start
        self.end = end
        self.n_missing = 0

    cdef inline void sort_samples_and_feature_values(
        self, cnp.intp_t current_feature
    ) noexcept nogil:
        """Simultaneously sort based on the feature_values.

        Missing values are stored at the end of feature_values.
        The number of missing values observed in feature_values is stored
        in self.n_missing.
        """
        cdef:
            cnp.intp_t i, current_end
            cnp.float32_t[::1] feature_values = self.feature_values
            const cnp.float32_t[:, :] X = self.X
            cnp.intp_t[:] samples = self.samples
            cnp.intp_t n_missing = 0
            const unsigned char[::1] missing_values_in_feature_mask = self.missing_values_in_feature_mask

        # Sort samples along that feature; by
        # copying the values into an array and
        # sorting the array in a manner which utilizes the cache more
        # effectively.
        if missing_values_in_feature_mask is not None and missing_values_in_feature_mask[current_feature]:
            i, current_end = self.start, self.end - 1
            # Missing values are placed at the end and do not participate in the sorting.
            while i <= current_end:
                # Finds the right-most value that is not missing so that
                # it can be swapped with missing values at its left.
                if isnan(X[samples[current_end], current_feature]):
                    n_missing += 1
                    current_end -= 1
                    continue

                # X[samples[current_end], current_feature] is a non-missing value
                if isnan(X[samples[i], current_feature]):
                    samples[i], samples[current_end] = samples[current_end], samples[i]
                    n_missing += 1
                    current_end -= 1

                feature_values[i] = X[samples[i], current_feature]
                i += 1
        else:
            # When there are no missing values, we only need to copy the data into
            # feature_values
            for i in range(self.start, self.end):
                feature_values[i] = X[samples[i], current_feature]

        sort(&feature_values[self.start], &samples[self.start], self.end - self.start - n_missing)
        self.n_missing = n_missing

    cdef inline void next_p(self, cnp.intp_t* p_prev, cnp.intp_t* p) noexcept nogil:
        """Compute the next p_prev and p for iteratiing over feature values.

        The missing values are not included when iterating through the feature values.
        """
        cdef:
            cnp.float32_t[::1] feature_values = self.feature_values
            cnp.intp_t end_non_missing = self.end - self.n_missing

        while (
            p[0] + 1 < end_non_missing and
            feature_values[p[0] + 1] <= feature_values[p[0]] + FEATURE_THRESHOLD
        ):
            p[0] += 1

        p_prev[0] = p[0]

        # By adding 1, we have
        # (feature_values[p] >= end) or (feature_values[p] > feature_values[p - 1])
        p[0] += 1



    cdef inline void partition_samples_final(
            self,
            cnp.intp_t best_pos,
            cnp.float64_t best_threshold,
            cnp.intp_t best_feature,
            cnp.intp_t best_n_missing,
        ) noexcept nogil:
            """Partition samples for X at the best_threshold and best_feature.

            If missing values are present, this method partitions `samples`
            so that the `best_n_missing` missing values' indices are in the
            right-most end of `samples`, that is `samples[end_non_missing:end]`.
            """
            cdef:
                # Local invariance: start <= p <= partition_end <= end
                cnp.intp_t start = self.start
                cnp.intp_t p = start
                cnp.intp_t end = self.end - 1
                cnp.intp_t partition_end = end - best_n_missing
                cnp.intp_t[:] samples = self.samples
                const cnp.float32_t[:, :] X = self.X
                cnp.float32_t current_value

            if best_n_missing != 0:
                # Move samples with missing values to the end while partitioning the
                # non-missing samples
                while p < partition_end:
                    # Keep samples with missing values at the end
                    if isnan(X[samples[end], best_feature]):
                        end -= 1
                        continue

                    # Swap sample with missing values with the sample at the end
                    current_value = X[samples[p], best_feature]
                    if isnan(current_value):
                        samples[p], samples[end] = samples[end], samples[p]
                        end -= 1

                        # The swapped sample at the end is always a non-missing value, so
                        # we can continue the algorithm without checking for missingness.
                        current_value = X[samples[p], best_feature]

                    # Partition the non-missing samples
                    if current_value <= best_threshold:
                        p += 1
                    else:
                        samples[p], samples[partition_end] = samples[partition_end], samples[p]
                        partition_end -= 1
            else:
                # Partitioning routine when there are no missing values
                while p < partition_end:
                    if X[samples[p], best_feature] <= best_threshold:
                        p += 1
                    else:
                        samples[p], samples[partition_end] = samples[partition_end], samples[p]
                        partition_end -= 1


cdef inline void sort(cnp.float32_t* feature_values, cnp.intp_t* samples, cnp.intp_t n) noexcept nogil:
    if n == 0:
        return
    cdef cnp.intp_t maxd = 2 * <cnp.intp_t>log(n)
    introsort(feature_values, samples, n, maxd)



cdef void introsort(cnp.float32_t* feature_values, cnp.intp_t *samples,
                    cnp.intp_t n, cnp.intp_t maxd) noexcept nogil:
    cdef cnp.float32_t pivot
    cdef cnp.intp_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(feature_values, samples, n)
            return
        maxd -= 1

        pivot = median3(feature_values, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if feature_values[i] < pivot:
                swap(feature_values, samples, i, l)
                i += 1
                l += 1
            elif feature_values[i] > pivot:
                r -= 1
                swap(feature_values, samples, i, r)
            else:
                i += 1

        introsort(feature_values, samples, l, maxd)
        feature_values += r
        samples += r
        n -= r

cdef void heapsort(cnp.float32_t* feature_values, cnp.intp_t* samples, cnp.intp_t n) noexcept nogil:
    cdef cnp.intp_t start, end

    # heapify
    start = <cnp.intp_t>((n - 2) / 2)
    end = n
    while True:
        sift_down(feature_values, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(feature_values, samples, 0, end)
        sift_down(feature_values, samples, 0, end)
        end = end - 1


cdef inline void swap(cnp.float32_t* feature_values, cnp.intp_t* samples,
                      cnp.intp_t i, cnp.intp_t j) noexcept nogil:
    # Helper for sort
    feature_values[i], feature_values[j] = feature_values[j], feature_values[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline cnp.float32_t median3(cnp.float32_t* feature_values, cnp.intp_t n) noexcept nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef cnp.float32_t a = feature_values[0], b = feature_values[<cnp.intp_t>(n / 2)], c = feature_values[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b

cdef inline void sift_down(cnp.float32_t* feature_values, cnp.intp_t* samples,
                           cnp.intp_t start, cnp.intp_t end) noexcept nogil:
    # Restore heap order in feature_values[start:end] by moving the max element to start.
    cdef cnp.intp_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and feature_values[maxind] < feature_values[child]:
            maxind = child
        if child + 1 < end and feature_values[maxind] < feature_values[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(feature_values, samples, root, maxind)
            root = maxind



cdef class BestSplitter(Splitter):
    """Splitter for finding the best split on dense data."""
    cdef DensePartitioner partitioner
    cdef int init(
        self,
        object X,
        const cnp.float64_t[:, ::1] y,
        const cnp.float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )

    cdef int node_split(
            self,
            cnp.float64_t impurity,
            SplitRecord* split,
            cnp.intp_t* n_constant_features,
            cnp.float64_t lower_bound,
            cnp.float64_t upper_bound,
            Tree tree,
            cnp.int32_t[:] sensor_types,
            cnp.int32_t[:] depth_types,
            cnp.int32_t[:] time_types            
    ) except -1 nogil:
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            impurity,
            split,
            n_constant_features,
            self.with_monotonic_cst,
            self.monotonic_cst,
            lower_bound,
            upper_bound,
            tree,
            sensor_types,
            depth_types,
            time_types,            
        )
