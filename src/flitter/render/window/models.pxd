# cython: language_level=3, profile=True

from ...model cimport Node, Vector, Matrix44


cdef double DefaultSnapAngle


cdef class Model:
    cdef str name
    cdef object trimesh_model
    cdef bint valid

    cdef bint is_constructed(self)
    cdef bint check_valid(self)
    cdef void build_trimesh_model(self)
    cdef tuple get_buffers(self, object glctx, dict objects)
    cdef tuple instance(self, Matrix44 model_matrix)

    cdef Model flatten(self)
    cdef Model invert(self)
    cdef Model snap_edges(self, double snap_angle, double minimum_area)
    cdef Model transform(self, Matrix44 transform_matrix)
    cdef Model slice(self, Vector position, Vector normal)

    @staticmethod
    cdef Model intersect(list models)

    @staticmethod
    cdef Model union(list models)

    @staticmethod
    cdef Model difference(list models)

    @staticmethod
    cdef Model get_box(Node node)

    @staticmethod
    cdef Model get_sphere(Node node)

    @staticmethod
    cdef Model get_cylinder(Node node)

    @staticmethod
    cdef Model get_cone(Node node)

    @staticmethod
    cdef Model get_external(Node node)
