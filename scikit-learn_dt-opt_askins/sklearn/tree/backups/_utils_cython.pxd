# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

# See _utils.pyx for details.

cimport numpy as cnp

from _tree_cython cimport Node

# from ..neighbors._quad_tree cimport Cell
# from ..utils._typedefs cimport cnp.float32_t, cnp.float64_t, cnp.intp_t, int32_t, cnp.uint32_t

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    # It corresponds to the maximum representable value for
    # 32-bit signed integers (i.e. 2^31 - 1).
    RAND_R_MAX = 2147483647


# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef cnp.float32_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (cnp.float32_t*)
    (cnp.intp_t*)
    (unsigned char*)
    (WeightedPQueueRecord*)
    (cnp.float64_t*)
    (cnp.float64_t**)
    (Node*)
    # (Cell*)
    (Node**)

cdef int safe_realloc(realloc_ptr* p, size_t nelems) except -1 nogil


cdef cnp.ndarray sizet_ptr_to_ndarray(cnp.intp_t* data, cnp.intp_t size)


cdef cnp.intp_t rand_int(cnp.intp_t low, cnp.intp_t high,
                     cnp.uint32_t* random_state) noexcept nogil


cdef cnp.float64_t rand_uniform(cnp.float64_t low, cnp.float64_t high,
                            cnp.uint32_t* random_state) noexcept nogil


cdef cnp.float64_t log(cnp.float64_t x) noexcept nogil

# =============================================================================
# WeightedPQueue data structure
# =============================================================================

# A record stored in the WeightedPQueue
cdef struct WeightedPQueueRecord:
    cnp.float64_t data
    cnp.float64_t weight

cdef class WeightedPQueue:
    cdef cnp.intp_t capacity
    cdef cnp.intp_t array_ptr
    cdef WeightedPQueueRecord* array_

    cdef bint is_empty(self) noexcept nogil
    cdef int reset(self) except -1 nogil
    cdef cnp.intp_t size(self) noexcept nogil
    cdef int push(self, cnp.float64_t data, cnp.float64_t weight) except -1 nogil
    cdef int remove(self, cnp.float64_t data, cnp.float64_t weight) noexcept nogil
    cdef int pop(self, cnp.float64_t* data, cnp.float64_t* weight) noexcept nogil
    cdef int peek(self, cnp.float64_t* data, cnp.float64_t* weight) noexcept nogil
    cdef cnp.float64_t get_weight_from_index(self, cnp.intp_t index) noexcept nogil
    cdef cnp.float64_t get_value_from_index(self, cnp.intp_t index) noexcept nogil


# =============================================================================
# WeightedMedianCalculator data structure
# =============================================================================

cdef class WeightedMedianCalculator:
    cdef cnp.intp_t initial_capacity
    cdef WeightedPQueue samples
    cdef cnp.float64_t total_weight
    cdef cnp.intp_t k
    cdef cnp.float64_t sum_w_0_k  # represents sum(weights[0:k]) = w[0] + w[1] + ... + w[k-1]
    cdef cnp.intp_t size(self) noexcept nogil
    cdef int push(self, cnp.float64_t data, cnp.float64_t weight) except -1 nogil
    cdef int reset(self) except -1 nogil
    cdef int update_median_parameters_post_push(
        self, cnp.float64_t data, cnp.float64_t weight,
        cnp.float64_t original_median) noexcept nogil
    cdef int remove(self, cnp.float64_t data, cnp.float64_t weight) noexcept nogil
    cdef int pop(self, cnp.float64_t* data, cnp.float64_t* weight) noexcept nogil
    cdef int update_median_parameters_post_remove(
        self, cnp.float64_t data, cnp.float64_t weight,
        cnp.float64_t original_median) noexcept nogil
    cdef cnp.float64_t get_median(self) noexcept nogil
