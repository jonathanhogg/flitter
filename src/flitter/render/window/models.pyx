# cython: language_level=3, profile=True

import cython
from loguru import logger
import numpy as np
import trimesh

from libc.math cimport cos, sin, sqrt

from ... import name_patch
from ...cache import SharedCache


logger = name_patch(logger, __name__)

cdef dict ModelCache = {}
cdef double Tau = 6.283185307179586
cdef double RootHalf = sqrt(0.5)


cdef class Model:
    def __hash__(self):
        return <Py_hash_t>(<void*>self)

    def __eq__(self, other):
        return self is other

    cdef tuple get_buffers(self, object glctx, dict objects):
        raise NotImplementedError()


cdef class TrimeshModel(Model):
    cdef object get_trimesh_model(self):
        raise NotImplementedError()

    cdef tuple get_buffers(self, object glctx, dict objects):
        cdef str name = self.name
        trimesh_model = self.get_trimesh_model()
        if trimesh_model is self.trimesh_model and name in objects:
            return objects[name]
        self.trimesh_model = trimesh_model
        if trimesh_model is None:
            if name in objects:
                del objects[name]
            return None, None
        logger.debug("Preparing model {}", name)
        cdef tuple buffers
        faces = trimesh_model.faces[:,::-1] if self.invert else trimesh_model.faces
        cdef bint has_uv = trimesh_model.visual is not None and isinstance(trimesh_model.visual, trimesh.visual.texture.TextureVisuals)
        vertex_uvs = trimesh_model.visual.uv if has_uv else np.zeros((len(trimesh_model.vertices), 2))
        if self.flat:
            face_normals = -trimesh_model.face_normals if self.invert else trimesh_model.face_normals
            vertex_data = np.empty((len(faces), 3, 8), dtype='f4')
            vertex_data[:,:,0:3] = trimesh_model.vertices[faces]
            vertex_data[:,:,3:6] = face_normals[:,None,:]
            vertex_data[:,:,6:8] = vertex_uvs[faces]
            buffers = (glctx.buffer(vertex_data), None)
        else:
            vertex_normals = -trimesh_model.vertex_normals if self.invert else trimesh_model.vertex_normals
            vertex_data = np.hstack((trimesh_model.vertices, vertex_normals, vertex_uvs)).astype('f4')
            index_data = faces.astype('i4')
            buffers = (glctx.buffer(vertex_data), glctx.buffer(index_data))
        objects[name] = buffers
        return buffers


cdef class Box(TrimeshModel):
    @staticmethod
    cdef Box get(Node node):
        cdef bint flat = node.get_bool('flat', False)
        cdef bint invert = node.get_bool('invert', False)
        cdef str name = '!box/flat' if flat else '!box'
        if invert:
            name += '/invert'
        cdef Box model = ModelCache.get(name)
        if model is None:
            model = Box.__new__(Box)
            model.name = name
            model.flat = flat
            model.invert = invert
            model.trimesh_model = None
            ModelCache[name] = model
        return model

    cdef object get_trimesh_model(self):
        return trimesh.primitives.Box() if self.trimesh_model is None else self.trimesh_model


cdef class Sphere(TrimeshModel):
    @staticmethod
    cdef Sphere get(Node node):
        cdef bint flat = node.get_bool('flat', False)
        cdef bint invert = node.get_bool('invert', False)
        cdef int subdivisions = node.get_int('subdivisions', 2)
        cdef str name = f'!sphere/{subdivisions}'
        if flat:
            name += '/flat'
        if invert:
            name += '/invert'
        cdef Sphere model = ModelCache.get(name)
        if model is None:
            model = Sphere.__new__(Sphere)
            model.name = name
            model.flat = flat
            model.invert = invert
            model.subdivisions = subdivisions
            model.trimesh_model = None
            ModelCache[name] = model
        return model

    cdef object get_trimesh_model(self):
        return trimesh.primitives.Sphere(subdivisions=self.subdivisions) if self.trimesh_model is None else self.trimesh_model


cdef class Cylinder(TrimeshModel):
    @staticmethod
    cdef Cylinder get(Node node):
        cdef bint flat = node.get_bool('flat', False)
        cdef bint invert = node.get_bool('invert', False)
        cdef int segments = node.get_int('segments', 32)
        cdef str name = f'!cylinder/{segments}'
        if flat:
            name += '/flat'
        if invert:
            name += '/invert'
        cdef Cylinder model = ModelCache.get(name)
        if model is None:
            model = Cylinder.__new__(Cylinder)
            model.name = name
            model.flat = flat
            model.invert = invert
            model.segments = segments
            model.trimesh_model = None
            ModelCache[name] = model
        return model

    cdef object get_trimesh_model(self):
        return trimesh.primitives.Cylinder(sections=self.segments) if self.trimesh_model is None else self.trimesh_model


cdef class Cone(TrimeshModel):
    @staticmethod
    cdef Cone get(Node node):
        cdef bint flat = node.get_bool('flat', False)
        cdef bint invert = node.get_bool('invert', False)
        cdef int segments = node.get_int('segments', 32)
        cdef str name = f'!cone/{segments}'
        if flat:
            name += '/flat'
        if invert:
            name += '/invert'
        cdef Cone model = ModelCache.get(name)
        if model is None:
            model = Cone.__new__(Cone)
            model.name = name
            model.flat = flat
            model.invert = invert
            model.segments = segments
            model.trimesh_model = None
            ModelCache[name] = model
        return model

    @cython.cdivision(True)
    @cython.boundscheck(False)
    cdef object get_trimesh_model(self):
        if self.trimesh_model is not None:
            return self.trimesh_model
        cdef int i, j, k, n = self.segments
        cdef object vertices_array = np.empty((n*3+1, 3), dtype='float64')
        cdef double[:,:] vertices = vertices_array
        cdef object vertex_normals_array = np.empty((n*3+1, 3), dtype='float64')
        cdef double[:,:] vertex_normals = vertex_normals_array
        cdef object faces_array = np.empty((n*2, 3), dtype='int64')
        cdef long[:,:] faces = faces_array
        cdef double x, y, th
        vertices[0, 0], vertices[0, 1], vertices[0, 2] = 0, 0, -0.5
        vertex_normals[0, 0], vertex_normals[0, 1], vertex_normals[0, 2] = 0, 0, -1
        for i in range(n):
            j = i * 3
            k = (i-1)*3 if i else (n-1)*3
            th = Tau * i/n
            x = cos(th)
            y = sin(th)
            vertices[j+1, 0], vertices[j+1, 1], vertices[j+1, 2] = x, y, -0.5
            vertex_normals[j+1, 0], vertex_normals[j+1, 1], vertex_normals[j+1, 2] = 0, 0, -1
            vertices[j+2, 0], vertices[j+2, 1], vertices[j+2, 2] = x, y, -0.5
            vertex_normals[j+2, 0], vertex_normals[j+2, 1], vertex_normals[j+2, 2] = x*RootHalf, y*RootHalf, RootHalf
            vertices[j+3, 0], vertices[j+3, 1], vertices[j+3, 2] = 0, 0, 0.5
            vertex_normals[j+3, 0], vertex_normals[j+3, 1], vertex_normals[j+3, 2] = x*RootHalf, y*RootHalf, RootHalf
            faces[i*2, 0], faces[i*2, 1], faces[i*2, 2] = 0, k+1, j+1
            faces[i*2+1, 0], faces[i*2+1, 1], faces[i*2+1, 2] = j+3, k+2, j+2
        self.trimesh_model = trimesh.base.Trimesh(vertices=vertices_array, vertex_normals=vertex_normals_array, faces=faces_array)
        return self.trimesh_model


cdef class ExternalModel(TrimeshModel):
    @staticmethod
    cdef ExternalModel get(Node node):
        cdef str filename = node.get_str('filename', None)
        if not filename:
            return None
        cdef bint flat = node.get_bool('flat', False)
        cdef bint invert = node.get_bool('invert', False)
        cdef str name = filename
        if flat:
            name += '/flat'
        if invert:
            name += '/invert'
        cdef ExternalModel model = ModelCache.get(name)
        if model is None:
            model = ExternalModel.__new__(ExternalModel)
            model.name = name
            model.flat = flat
            model.invert = invert
            model.filename = filename
            model.trimesh_model = None
            ModelCache[name] = model
        return model

    cdef object get_trimesh_model(self):
        return SharedCache[self.filename].read_trimesh_model()
