
from ...model cimport Node, Vector, Matrix44

from libc.stdint cimport int64_t


cdef double DefaultSnapAngle
cdef int64_t DefaultSegments


cdef class Model:
    cdef readonly str name
    cdef readonly object trimesh_model
    cdef readonly bint created
    cdef readonly bint valid
    cdef Vector bounds
    cdef set dependents
    cdef list buffer_caches

    cpdef bint is_manifold(self)
    cpdef void check_for_changes(self)
    cpdef object build_trimesh(self)

    cpdef void add_dependent(self, Model model)
    cpdef void invalidate(self)
    cpdef object get_trimesh(self)
    cpdef Vector get_bounds(self)
    cdef tuple get_buffers(self, object glctx, dict objects)

    cpdef Model manifold(self)
    cpdef Model flatten(self)
    cpdef Model invert(self)
    cpdef Model repair(self)
    cdef Model _snap_edges(self, double snap_angle, double minimum_area)
    cdef Model _transform(self, Matrix44 transform_matrix)
    cdef Model _uv_remap(self, str mapping)
    cdef Model _slice(self, Vector position, Vector normal)

    @staticmethod
    cdef Model _intersect(list models)

    @staticmethod
    cdef Model _union(list models)

    @staticmethod
    cdef Model _difference(list models)

    @staticmethod
    cdef Model _box()

    @staticmethod
    cdef Model _sphere(int64_t segments)

    @staticmethod
    cdef Model _cylinder(int64_t segments)

    @staticmethod
    cdef Model _cone(int64_t segments)

    @staticmethod
    cdef Model _external(str filename)
