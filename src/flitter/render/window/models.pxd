# cython: language_level=3, profile=True

from ...model cimport Node, Vector, Matrix44


cdef double DefaultSmooth


cdef class Model:
    cdef str name
    cdef bint flat
    cdef bint invert
    cdef object trimesh_model
    cdef bint valid

    cdef bint is_constructed(self)
    cdef bint check_valid(self)
    cdef void build_trimesh_model(self)
    cdef tuple get_buffers(self, object glctx, dict objects)

    cdef Model smooth_shade(self, double smooth, double minimum_area)
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
