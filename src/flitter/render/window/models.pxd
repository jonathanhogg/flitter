
from ...model cimport Node, Vector, Matrix44

from libc.stdint cimport int64_t


cdef double DefaultSnapAngle
cdef int64_t DefaultSegments


cdef class Model:
    cdef readonly str name
    cdef double touch_timestamp
    cdef double cache_timestamp
    cdef readonly dict cache
    cdef set dependents
    cdef list buffer_caches

    cpdef void unload(self)
    cpdef void check_for_changes(self)
    cpdef bint is_smooth(self)
    cpdef object build_trimesh(self)
    cpdef object build_manifold(self)

    cpdef void add_dependent(self, Model model)
    cpdef void remove_dependent(self, Model model)
    cpdef void invalidate(self)
    cpdef Vector get_bounds(self)
    cpdef object get_trimesh(self)
    cpdef object get_manifold(self)
    cdef tuple get_buffers(self, object glctx, dict objects)

    cpdef Model flatten(self)
    cpdef Model invert(self)
    cpdef Model repair(self)
    cdef Model _snap_edges(self, double snap_angle, double minimum_area)
    cdef Model _transform(self, Matrix44 transform_matrix)
    cdef Model _uv_remap(self, str mapping)
    cdef Model _trim(self, Vector position, Vector normal)

    @staticmethod
    cdef Model _intersect(list models)

    @staticmethod
    cdef Model _union(list models)

    @staticmethod
    cdef Model _difference(list models)

    @staticmethod
    cdef Model _box(str uv_map)

    @staticmethod
    cdef Model _sphere(int64_t segments)

    @staticmethod
    cdef Model _cylinder(int64_t segments)

    @staticmethod
    cdef Model _cone(int64_t segments)

    @staticmethod
    cdef Model _external(str filename)
