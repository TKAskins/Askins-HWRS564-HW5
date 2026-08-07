# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

# See _tree.pyx for details.

import numpy as np
cimport numpy as cnp

# from ..utils._typedefs cimport float32_t, cnp.float64_t, cnp.intp_t, cnp.int32_t, ucnp.int32_t

from _splitter_cython cimport Splitter, SplitRecord

cdef struct Node:
    # Base storage structure for the nodes in a Tree object

    cnp.intp_t left_child                    # id of the left child of the node
    cnp.intp_t right_child                   # id of the right child of the node
    cnp.intp_t feature                       # Feature used for splitting the node
    cnp.float64_t threshold                  # Threshold value at the node
    cnp.float64_t impurity                   # Impurity of the node (i.e., the value of the criterion)
    cnp.intp_t n_node_samples                # Number of samples at the node
    cnp.float64_t weighted_n_node_samples    # Weighted number of samples at the node
    unsigned char missing_go_to_left     # Whether features have missing values
    cnp.float64_t cost
    cnp.float64_t cost_objective
    cnp.float64_t improvement


cdef class Tree:
    # The Tree object is a binary tree structure constructed by the
    # TreeBuilder. The tree structure is used for predictions and
    # feature importances.

    # Input/Output layout
    cdef public cnp.intp_t n_features        # Number of features in X
    cdef cnp.intp_t* n_classes               # Number of classes in y[:, k]
    cdef public cnp.intp_t n_outputs         # Number of outputs in y
    cdef public cnp.intp_t max_n_classes     # max(n_classes)

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public cnp.intp_t max_depth         # Max depth of the tree
    cdef public cnp.intp_t node_count        # Counter for node IDs
    cdef public cnp.intp_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes
    cdef cnp.float64_t* value                   # (capacity, n_outputs, max_n_classes) array of values
    cdef cnp.intp_t value_stride             # = n_outputs * max_n_classes
    
    cdef public char* derek             # test variable

    # Methods
    cdef cnp.intp_t _add_node(self, cnp.intp_t parent, bint is_left, bint is_leaf,
                          cnp.intp_t feature, cnp.float64_t threshold, cnp.float64_t impurity,
                          cnp.intp_t n_node_samples,
                          cnp.float64_t weighted_n_node_samples,
                          unsigned char missing_go_to_left,
                          cnp.float64_t cost,
                          cnp.float64_t cost_objective,
                          cnp.float64_t improvement
                          ) except -1 nogil
    cdef int _resize(self, cnp.intp_t capacity) except -1 nogil
    cdef int _resize_c(self, cnp.intp_t capacity=*) except -1 nogil

    cdef cnp.ndarray _get_value_ndarray(self)
    cdef cnp.ndarray _get_node_ndarray(self)

    cpdef cnp.ndarray predict(self, object X)

    cpdef cnp.ndarray apply(self, object X)
    cdef cnp.ndarray _apply_dense(self, object X)
    cdef cnp.ndarray _apply_sparse_csr(self, object X)

    cpdef object decision_path(self, object X)
    cdef object _decision_path_dense(self, object X)
    cdef object _decision_path_sparse_csr(self, object X)

    cpdef compute_node_depths(self)
    cpdef compute_feature_importances(self, normalize=*)


# =============================================================================
# Tree builder
# =============================================================================

cdef class TreeBuilder:
    # The TreeBuilder recursively builds a Tree object from training samples,
    # using a Splitter object for splitting internal nodes and assigning
    # values to leaves.
    #
    # This class controls the various stopping criteria and the node splitting
    # evaluation order, e.g. depth-first or best-first.

    cdef Splitter splitter              # Splitting algorithm

    cdef cnp.intp_t min_samples_split       # Minimum number of samples in an internal node
    cdef cnp.intp_t min_samples_leaf        # Minimum number of samples in a leaf
    cdef cnp.float64_t min_weight_leaf         # Minimum weight in a leaf
    cdef cnp.intp_t max_depth               # Maximal tree depth
    cdef cnp.float64_t min_impurity_decrease   # Impurity threshold for early stopping
    # const cnp.int32_t[:] feature_types

    cpdef build(
        self,
        Tree tree,
        object X,
        const cnp.float64_t[:, ::1] y,
        const cnp.float64_t[:] sample_weight=*,
        const unsigned char[::1] missing_values_in_feature_mask=*,
        cnp.int32_t[:] sensor_types=*,
        cnp.int32_t[:] depth_types=*,
        cnp.int32_t[:] time_types=*,
    )

    cdef _check_input(
        self,
        object X,
        const cnp.float64_t[:, ::1] y,
        const cnp.float64_t[:] sample_weight,
    )
