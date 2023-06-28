# cython: language_level=3, profile=True

import cython
from loguru import logger
import numpy as np
import trimesh

from ... import name_patch
from ...cache import SharedCache


logger = name_patch(logger, __name__)

cdef dict ModelCache = {}


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
    cdef Box get(bint flat, bint invert):
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
    cdef Sphere get(bint flat, bint invert, int subdivisions):
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
    cdef Cylinder get(bint flat, bint invert, int segments):
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
        return trimesh.primitives.Cylinder(segments=self.segments) if self.trimesh_model is None else self.trimesh_model


cdef class LoadedModel(TrimeshModel):
    @staticmethod
    cdef LoadedModel get(bint flat, bint invert, str filename):
        cdef str name = filename
        if flat:
            name += '/flat'
        if invert:
            name += '/invert'
        cdef LoadedModel model = ModelCache.get(name)
        if model is None:
            model = LoadedModel.__new__(LoadedModel)
            model.name = name
            model.flat = flat
            model.invert = invert
            model.filename = filename
            model.trimesh_model = None
            ModelCache[name] = model
        return model

    cdef object get_trimesh_model(self):
        return SharedCache[self.filename].read_trimesh_model()
