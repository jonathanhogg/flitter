# cython: language_level=3, profile=True

from ...model cimport Node


cdef class Model:
    cdef str name

    cdef tuple get_buffers(self, object glctx, dict objects)


cdef class TrimeshModel(Model):
    cdef bint flat
    cdef bint invert
    cdef object trimesh_model

    cdef object get_trimesh_model(self)


cdef class Box(TrimeshModel):
    @staticmethod
    cdef Box get(Node node)


cdef class Sphere(TrimeshModel):
    cdef int subdivisions

    @staticmethod
    cdef Sphere get(Node node)


cdef class Cylinder(TrimeshModel):
    cdef int segments

    @staticmethod
    cdef Cylinder get(Node node)


cdef class Cone(TrimeshModel):
    cdef int segments

    @staticmethod
    cdef Cone get(Node node)


cdef class ExternalModel(TrimeshModel):
    cdef str filename

    @staticmethod
    cdef ExternalModel get(Node node)
