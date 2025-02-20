
from ...model cimport Node, Vector, Matrix44
from ...language.vm cimport Function

from libc.stdint cimport int64_t, uint64_t


cdef double DefaultSnapAngle
cdef int64_t DefaultSegments


cpdef tuple build_arrays_from_trimesh(trimesh_model)
cpdef tuple build_arrays_from_manifold(manifold)
cpdef void fill_in_normals(vertices_array, faces_array)


cdef class Model:
    cdef readonly uint64_t id
    cdef readonly double touch_timestamp
    cdef readonly dict cache
    cdef readonly set dependents
    cdef readonly list buffer_caches

    cpdef bint uncache(self, bint buffers)
    cpdef void unload(self)
    cpdef void check_for_changes(self)
    cpdef bint is_manifold(self)
    cpdef double signed_distance(self, double x, double y, double z) noexcept
    cpdef tuple build_arrays(self)
    cpdef object build_trimesh(self)
    cpdef object build_manifold(self)

    cpdef void add_dependent(self, Model model)
    cpdef void remove_dependent(self, Model model)
    cpdef void invalidate(self)
    cpdef Vector get_bounds(self)
    cpdef tuple get_arrays(self)
    cpdef object get_trimesh(self)
    cpdef object get_manifold(self)
    cpdef tuple get_buffers(self, object glctx, dict objects)

    cpdef Model flatten(self)
    cpdef Model invert(self)
    cpdef Model repair(self)
    cdef Model _snap_edges(self, double snap_angle)
    cdef Model _transform(self, Matrix44 transform_matrix)
    cdef Model _uv_remap(self, str mapping)
    cdef Model _trim(self, Vector origin, Vector normal, double smooth, double fillet, double chamfer)

    @staticmethod
    cdef Model _boolean(str operation, list models, double smooth, double fillet, double chamfer)

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

    @staticmethod
    cdef Model _vector(Vector vertices, Vector faces)

    @staticmethod
    cdef Model _sdf(Function function, Model original, Vector minimum, Vector maximum, double resolution)

    @staticmethod
    cdef Model _mix(list models, Vector weights)
