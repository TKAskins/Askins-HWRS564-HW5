# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#
# License: BSD 3 clause

# See _splitter.pyx for details.
cimport numpy as cnp

from _criterion_cython cimport Criterion

# from ..utils._typedefs cimport cnp.float32_t, cnp.float64_t, cnp.intp_t, cnp.int32_t, ucnp.int32_t

from _tree_cython cimport Tree, Node


cdef struct SplitRecord:
    # Data to track sample split
    cnp.intp_t feature         # Which feature to split on.
    cnp.intp_t pos             # Split samples array at the given position,
    #                      # i.e. count of samples below threshold for feature.
    #                      # pos is >= end if the node is a leaf.
    cnp.float64_t threshold       # Threshold to split at.
    cnp.float64_t improvement     # Impurity improvement given parent node.
    cnp.float64_t impurity_left   # Impurity of the left split.
    cnp.float64_t impurity_right  # Impurity of the right split.
    cnp.float64_t lower_bound     # Lower bound on value of both children for monotonicity
    cnp.float64_t upper_bound     # Upper bound on value of both children for monotonicity
    unsigned char missing_go_to_left  # Controls if missing values go to the left node.
    cnp.intp_t n_missing       # Number of missing values for the feature being split on
    cnp.float64_t cost # measurement cost
    cnp.float64_t cost_objective # measurement cost + improvement    

cdef class Splitter:
    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.

    # Internal structures
    cdef public Criterion criterion      # Impurity criterion
    cdef public cnp.intp_t max_features      # Number of features to test
    cdef public cnp.intp_t min_samples_leaf  # Min samples in a leaf
    cdef public cnp.float64_t min_weight_leaf   # Minimum weight in a leaf

    cdef object random_state             # Random state
    cdef cnp.uint32_t rand_r_state           # sklearn_rand_r random number state

    cdef cnp.intp_t[:] samples             # Sample indices in X, y
    cdef cnp.intp_t n_samples                # X.shape[0]
    cdef cnp.float64_t weighted_n_samples       # Weighted number of samples
    cdef cnp.intp_t[::1] features            # Feature indices in X
    cdef cnp.intp_t[::1] constant_features   # Constant features indices
    cdef cnp.intp_t n_features               # X.shape[1]
    cdef cnp.float32_t[::1] feature_values   # temp. array holding feature values

    cdef cnp.intp_t start                    # Start position for the current node
    cdef cnp.intp_t end                      # End position for the current node

    cdef const cnp.float64_t[:, ::1] y
    # Monotonicity constraints for each feature.
    # The encoding is as follows:
    #   -1: monotonic decrease
    #    0: no constraint
    #   +1: monotonic increase
    cdef const cnp.int8_t[:] monotonic_cst
    cdef bint with_monotonic_cst
    cdef const cnp.float64_t[:] sample_weight

    cdef Node* node
    cdef Node* nodes
    cdef cnp.float64_t[:] sensor_cost 
    cdef cnp.float64_t time_cost
    cdef cnp.float64_t depth_cost
    cdef cnp.float64_t cost_threshold
    cdef cnp.float64_t current_cost
    cdef cnp.float64_t max_id
    cdef cnp.float64_t diff    
    cdef cnp.intp_t node_count
    cdef cnp.intp_t feature
    cdef cnp.int32_t sensor_n
    cdef cnp.int32_t depth_n
    cdef cnp.int32_t time_n
    cdef cnp.int32_t sensor_f
    cdef cnp.int32_t depth_f
    cdef cnp.int32_t time_f
    cdef int max_ind
    cdef int feature_ind
    cdef int n_ind
    cdef int ind
    cdef int best_ind


    # The samples vector `samples` is maintained by the Splitter object such
    # that the samples contained in a node are contiguous. With this setting,
    # `node_split` reorganizes the node samples `samples[start:end]` in two
    # subsets `samples[start:pos]` and `samples[pos:end]`.

    # The 1-d  `features` array of size n_features contains the features
    # indices and allows fast sampling without replacement of features.

    # The 1-d `constant_features` array of size n_features holds in
    # `constant_features[:n_constant_features]` the feature ids with
    # constant values for all the samples that reached a specific node.
    # The value `n_constant_features` is given by the parent node to its
    # child nodes.  The content of the range `[n_constant_features:]` is left
    # undefined, but preallocated for performance reasons
    # This allows optimization with depth-based tree building.

    # Methods
    cdef int init(
        self,
        object X,
        const cnp.float64_t[:, ::1] y,
        const cnp.float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1

    cdef int node_reset(
        self,
        cnp.intp_t start,
        cnp.intp_t end,
        cnp.float64_t* weighted_n_node_samples
    ) except -1 nogil

    cdef int node_split(
        self,
        cnp.float64_t impurity,   # Impurity of the node
        SplitRecord* split,
        cnp.intp_t* n_constant_features,
        cnp.float64_t lower_bound,
        cnp.float64_t upper_bound,
        Tree tree,
        cnp.int32_t[:] sensor_types,
        cnp.int32_t[:] depth_types,
        cnp.int32_t[:] time_types,        
    ) except -1 nogil

    cdef void node_value(self, cnp.float64_t* dest) noexcept nogil

    cdef void clip_node_value(self, cnp.float64_t* dest, cnp.float64_t lower_bound, cnp.float64_t upper_bound) noexcept nogil

    cdef cnp.float64_t node_impurity(self) noexcept nogil
