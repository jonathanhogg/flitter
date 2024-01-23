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
    cdef Model intersect(self, Node node, Model model)
    cdef Model union(self, Node node, Model model)
    cdef Model difference(self, Node node, Model model)

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
