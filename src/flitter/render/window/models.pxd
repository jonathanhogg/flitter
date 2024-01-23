# cython: language_level=3, profile=True

from ...model cimport Node, Matrix44


cdef class Model:
    cdef str name
    cdef bint flat
    cdef bint invert
    cdef object trimesh_model

    cdef bint trimesh_model_unchanged(self)
    cdef object build_trimesh_model(self)
    cdef object get_render_trimesh_model(self)
    cdef tuple get_buffers(self, object glctx, dict objects)

    cdef Model transform(self, Matrix44 transform_matrix)

    @staticmethod
    cdef Model intersect(Node node, list models)
    @staticmethod
    cdef Model union(Node node, list models)
    @staticmethod
    cdef Model difference(Node node, list models)
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
