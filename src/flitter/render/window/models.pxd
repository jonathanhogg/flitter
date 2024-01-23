# cython: language_level=3, profile=True

from ...model cimport Node, Matrix44


cdef class Model:
    cdef str name
    cdef bint flat
    cdef bint invert
    cdef object trimesh_model

    cdef object get_trimesh_model(self)
    cdef tuple get_buffers(self, object glctx, dict objects)
    cdef Model transform(self, Matrix44 transform_matrix)
    cdef Model intersect(self, Model model)
    cdef Model union(self, Model model)
    cdef Model difference(self, Model model)


cdef class Box(Model):
    @staticmethod
    cdef Box get(Node node)


cdef class Sphere(Model):
    cdef int segments

    @staticmethod
    cdef Sphere get(Node node)


cdef class Cylinder(Model):
    cdef int segments

    @staticmethod
    cdef Cylinder get(Node node)


cdef class Cone(Model):
    cdef int segments

    @staticmethod
    cdef Cone get(Node node)


cdef class ExternalModel(Model):
    cdef str filename

    @staticmethod
    cdef ExternalModel get(Node node)
