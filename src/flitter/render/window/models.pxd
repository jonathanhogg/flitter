# cython: language_level=3, profile=True


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
    cdef Box get(bint flat, bint invert)


cdef class Sphere(TrimeshModel):
    cdef int subdivisions

    @staticmethod
    cdef Sphere get(bint flat, bint invert, int subdivisions)


cdef class Cylinder(TrimeshModel):
    cdef int segments

    @staticmethod
    cdef Cylinder get(bint flat, bint invert, int segments)


cdef class LoadedModel(TrimeshModel):
    cdef str filename

    @staticmethod
    cdef LoadedModel get(bint flat, bint invert, str filename)
