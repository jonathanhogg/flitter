
import cython
from loguru import logger
import manifold3d
import numpy as np
import trimesh

from libc.math cimport cos, sin, sqrt, atan2, ceil
from libc.stdint cimport int32_t, int64_t

from ... import name_patch
from ...cache import SharedCache
from ...model cimport true_


logger = name_patch(logger, __name__)

cdef dict ModelCache = {}
cdef double Tau = 6.283185307179586231995926937088370323181152343750231995926937088370323181152343750
cdef double RootHalf = sqrt(0.5)
cdef double DefaultSnapAngle = 0.05
cdef int64_t DefaultSegments = 64
cdef Matrix44 IdentityTransform = Matrix44._identity()


cdef class Model:
    @staticmethod
    def by_name(str name):
        return ModelCache.get(name)

    @staticmethod
    def from_node(Node node):
        raise NotImplementedError()

    def __init__(self, name=None):
        if name is None:
            raise ValueError("Name cannot be None")
        name = str(name)
        if name in ModelCache:
            raise ValueError(f"Model already exists with name: '{name}'")
        self.name = name
        ModelCache[name] = self

    def __hash__(self):
        return <Py_hash_t>(<void*>self)

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<Model: {self.name}>'

    cpdef bint is_manifold(self):
        raise NotImplementedError()

    cpdef void check_for_changes(self):
        raise NotImplementedError()

    cpdef object build_trimesh(self):
        raise NotImplementedError()

    cpdef void add_dependent(self, Model model):
        if self.dependents is None:
            self.dependents = set()
        self.dependents.add(model)

    cpdef void invalidate(self):
        cdef Model model
        cdef dict cache
        if self.valid:
            self.trimesh_model = None
            self.bounds = None
            self.valid = False
            if self.dependents is not None:
                for model in self.dependents:
                    model.invalidate()
            if self.buffer_caches is not None:
                for cache in self.buffer_caches:
                    if self.name in cache:
                        del cache[self.name]

    cpdef object get_trimesh(self):
        if not self.valid:
            self.trimesh_model = self.build_trimesh()
            self.valid = True
        return self.trimesh_model

    cpdef Vector get_bounds(self):
        if not self.valid:
            self.trimesh_model = self.build_trimesh()
            self.valid = True
        elif self.bounds is not None:
            return self.bounds
        self.bounds = Vector.__new__(Vector)
        cdef const double[:, :] bounds
        cdef int64_t i, j
        if self.trimesh_model is not None:
            bounds = self.trimesh_model.bounds
            self.bounds.allocate_numbers(6)
            for i in range(2):
                for j in range(3):
                    self.bounds.numbers[i*3 + j] = bounds[i, j]
        return self.bounds

    cdef tuple get_buffers(self, object glctx, dict objects):
        cdef str name = self.name
        if not self.valid:
            self.trimesh_model = self.build_trimesh()
            self.valid = True
        elif name in objects:
            return objects[name]
        cdef tuple buffers
        if self.buffer_caches is None:
            self.buffer_caches = []
        self.buffer_caches.append(objects)
        if self.trimesh_model is None:
            buffers = None, None
            objects[name] = buffers
            return buffers
        if (visual := self.trimesh_model.visual) is not None and isinstance(visual, trimesh.visual.texture.TextureVisuals) \
                and visual.uv is not None and len(visual.uv) == len(self.trimesh_model.vertices):
            vertex_uvs = visual.uv
        else:
            vertex_uvs = np.zeros((len(self.trimesh_model.vertices), 2))
        vertex_data = np.hstack((self.trimesh_model.vertices, self.trimesh_model.vertex_normals, vertex_uvs)).astype('f4')
        index_data = self.trimesh_model.faces.astype('i4')
        buffers = (glctx.buffer(vertex_data), glctx.buffer(index_data))
        if not self.created:
            logger.debug("Constructed model {} with {} vertices and {} faces", name, len(vertex_data), len(index_data))
            self.created = True
        else:
            logger.trace("Re-constructed model {} with {} vertices and {} faces", name, len(vertex_data), len(index_data))
        objects[name] = buffers
        return buffers

    cpdef Model manifold(self):
        return Manifold._get(self)

    cpdef Model flatten(self):
        return Flatten._get(self)

    cpdef Model invert(self):
        return Invert._get(self)

    cpdef Model repair(self):
        return Repair._get(self)

    cdef Model _snap_edges(self, double snap_angle, double minimum_area):
        if snap_angle <= 0:
            return Flatten._get(self)
        return SnapEdges._get(self, snap_angle, minimum_area)

    def snap_edges(self, snap_angle=DefaultSnapAngle, minimum_area=0):
        return self._snap_edges(float(snap_angle), float(minimum_area))

    cdef Model _transform(self, Matrix44 transform_matrix):
        return Transform._get(self, transform_matrix)

    def transform(self, transform_matrix):
        return self._transform(Matrix44._coerce(transform_matrix))

    cdef Model _uv_remap(self, str mapping):
        return UVRemap._get(self, mapping)

    def uv_remap(self, mapping):
        return self._uv_remap(str(mapping))

    cdef Model _slice(self, Vector origin, Vector normal):
        return Slice._get(self, origin, normal)

    def slice(self, origin, normal):
        return self._slice(Vector._coerce(origin), Vector._coerce(normal))

    @staticmethod
    cdef Model _intersect(list models):
        return BooleanOperation._get('intersection', models)

    @staticmethod
    def intersect(*models):
        return Model._intersect(list(models))

    @staticmethod
    cdef Model _union(list models):
        return BooleanOperation._get('union', models)

    @staticmethod
    def union(*models):
        return Model._union(list(models))

    @staticmethod
    cdef Model _difference(list models):
        return BooleanOperation._get('difference', models)

    @staticmethod
    def difference(*models):
        return Model._difference(list(models))

    @staticmethod
    cdef Model _box(str uv_map):
        return Box._get(uv_map)

    @staticmethod
    def box(uv_map='standard'):
        return Box._get(str(uv_map))

    @staticmethod
    cdef Model _sphere(int64_t segments):
        return Sphere._get(segments)

    @staticmethod
    def sphere(segments=DefaultSegments):
        return Sphere._get(int(segments))

    @staticmethod
    cdef Model _cylinder(int64_t segments):
        return Cylinder._get(segments)

    @staticmethod
    def cylinder(segments=DefaultSegments):
        return Cylinder._get(int(segments))

    @staticmethod
    cdef Model _cone(int64_t segments):
        return Cone._get(segments)

    @staticmethod
    def cone(segments=DefaultSegments):
        return Cone._get(int(segments))

    @staticmethod
    cdef Model _external(str filename):
        return ExternalModel._get(filename)

    @staticmethod
    def external(filename):
        return ExternalModel._get(str(filename))


cdef class UnaryOperation(Model):
    cdef Model original

    cpdef bint is_manifold(self):
        return self.original.is_manifold()

    cpdef void check_for_changes(self):
        if self.valid:
            self.original.check_for_changes()


cdef class Manifold(UnaryOperation):
    @staticmethod
    cdef Manifold _get(Model original):
        assert not isinstance(original, Manifold)
        cdef str name = f'manifold({original.name})'
        cdef Manifold model = <Manifold>ModelCache.get(name, None)
        if model is None:
            model = Manifold.__new__(Manifold)
            model.name = name
            model.original = original
            model.original.add_dependent(model)
            ModelCache[name] = model
        return model

    cpdef bint is_manifold(self):
        return True

    cpdef Model manifold(self):
        return self

    cpdef Model flatten(self):
        return self.original.flatten()

    cpdef Model repair(self):
        return self

    cdef Model _snap_edges(self, double snap_angle, double minimum_area):
        return self.original._snap_edges(snap_angle, minimum_area)

    cpdef object build_trimesh(self):
        cdef bint merged=False, filled=False, hull=False
        trimesh_model = self.original.get_trimesh()
        if trimesh_model is not None:
            if not trimesh_model.is_watertight:
                trimesh_model = trimesh_model.copy()
                trimesh_model.merge_vertices(merge_tex=True, merge_norm=True)
                merged = True
                if not trimesh_model.is_watertight:
                    if trimesh_model.fill_holes():
                        filled = True
                    else:
                        trimesh_model = trimesh_model.convex_hull
                        hull = True
            if not trimesh_model.is_volume:
                logger.error("Mesh is not a volume: {}", self.original.name)
                trimesh_model = None
            elif hull:
                logger.warning("Computed convex hull of non-manifold mesh: {}", self.original.name)
            elif filled:
                logger.debug("Filled holes in non-manifold mesh: {}", self.original.name)
            elif merged:
                logger.trace("Merged vertices of non-manifold mesh: {}", self.original.name)
        return trimesh_model


cdef class Flatten(UnaryOperation):
    @staticmethod
    cdef Flatten _get(Model original):
        cdef str name = f'flatten({original.name})'
        cdef Flatten model = <Flatten>ModelCache.get(name, None)
        if model is None:
            model = Flatten.__new__(Flatten)
            model.name = name
            model.original = original
            model.original.add_dependent(model)
            ModelCache[name] = model
        return model

    cpdef bint is_manifold(self):
        return False

    cpdef Model manifold(self):
        return self.original.manifold()

    cpdef Model flatten(self):
        return self

    cdef Model _snap_edges(self, double snap_angle, double minimum_area):
        return self

    cpdef object build_trimesh(self):
        trimesh_model = self.original.get_trimesh()
        if trimesh_model is not None:
            trimesh_model = trimesh_model.copy()
            trimesh_model.unmerge_vertices()
        return trimesh_model


cdef class Invert(UnaryOperation):
    @staticmethod
    cdef Invert _get(Model original):
        cdef str name = f'invert({original.name})'
        cdef Invert model = <Invert>ModelCache.get(name, None)
        if model is None:
            model = Invert.__new__(Invert)
            model.name = name
            model.original = original
            model.original.add_dependent(model)
            ModelCache[name] = model
        return model

    cpdef Model manifold(self):
        return self.original.manifold().invert()

    cpdef Model invert(self):
        return self.original

    cpdef Model repair(self):
        return self.original.repair().invert()

    cdef Model _snap_edges(self, double snap_angle, double minimum_area):
        return self.original._snap_edges(snap_angle, minimum_area).invert()

    cdef Model _transform(self, Matrix44 transform_matrix):
        return self.original._transform(transform_matrix).invert()

    cpdef object build_trimesh(self):
        trimesh_model = self.original.get_trimesh()
        if trimesh_model is not None:
            trimesh_model = trimesh.base.Trimesh(vertices=trimesh_model.vertices,
                                                 vertex_normals=-trimesh_model.vertex_normals,
                                                 faces=trimesh_model.faces[:, ::-1],
                                                 visual=trimesh_model.visual)
        return trimesh_model


cdef class Repair(UnaryOperation):
    @staticmethod
    cdef Repair _get(Model original):
        cdef str name = f'repair({original.name})'
        cdef Repair model = <Repair>ModelCache.get(name, None)
        if model is None:
            model = Repair.__new__(Repair)
            model.name = name
            model.original = original
            model.original.add_dependent(model)
            ModelCache[name] = model
        return model

    cpdef Model manifold(self):
        return self.original.manifold()

    cpdef Model repair(self):
        return self

    cpdef object build_trimesh(self):
        trimesh_model = self.original.get_trimesh()
        if trimesh_model is not None:
            trimesh_model = trimesh_model.copy()
            trimesh_model.process(validate=True, merge_tex=True, merge_norm=True)
            trimesh_model.remove_unreferenced_vertices()
            trimesh_model.fill_holes()
            trimesh_model.fix_normals()
        return trimesh_model


cdef class SnapEdges(UnaryOperation):
    cdef double snap_angle
    cdef double minimum_area

    @staticmethod
    cdef SnapEdges _get(Model original, double snap_angle, double minimum_area):
        snap_angle = min(max(0, snap_angle), 0.5)
        minimum_area = min(max(0, minimum_area), 1)
        cdef str name = 'snap_edges(' + original.name
        if snap_angle != DefaultSnapAngle:
            name += f', {snap_angle:g}'
        if minimum_area:
            name += f', {minimum_area:g}'
        name += ')'
        cdef SnapEdges model = <SnapEdges>ModelCache.get(name, None)
        if model is None:
            model = SnapEdges.__new__(SnapEdges)
            model.name = name
            model.original = original
            model.original.add_dependent(model)
            model.snap_angle = snap_angle
            model.minimum_area = minimum_area
            ModelCache[name] = model
        return model

    cpdef Model manifold(self):
        return self.original.manifold()

    cpdef Model flatten(self):
        return self.original.flatten()

    cpdef Model repair(self):
        return self.original.repair()._snap_edges(self.snap_angle, self.minimum_area)

    cdef Model _snap_edges(self, double snap_angle, double minimum_area):
        return self.original._snap_edges(snap_angle, minimum_area)

    cpdef object build_trimesh(self):
        trimesh_model = self.original.get_trimesh()
        if trimesh_model is not None:
            trimesh_model = trimesh.graph.smooth_shade(trimesh_model, angle=self.snap_angle*Tau,
                                                       facet_minarea=1/self.minimum_area if self.minimum_area else None)
        return trimesh_model


cdef class Transform(UnaryOperation):
    cdef Matrix44 transform_matrix

    @staticmethod
    cdef Model _get(Model original, Matrix44 transform_matrix):
        if transform_matrix.eq(IdentityTransform) is true_:
            return original
        cdef str name = f'{original.name}@{hex(transform_matrix.hash(False))[3:]}'
        cdef Transform model = <Transform>ModelCache.get(name, None)
        if model is None:
            model = Transform.__new__(Transform)
            model.name = name
            model.original = original
            model.original.add_dependent(model)
            model.transform_matrix = transform_matrix
            ModelCache[name] = model
        return model

    cpdef Model manifold(self):
        return self.original.manifold()._transform(self.transform_matrix)

    cpdef Model repair(self):
        return self.original.repair()._transform(self.transform_matrix)

    cdef Model _transform(self, Matrix44 transform_matrix):
        return self.original._transform(transform_matrix.mmul(self.transform_matrix))

    cpdef object build_trimesh(self):
        trimesh_model = self.original.get_trimesh()
        if trimesh_model is not None:
            transform_array = np.array(self.transform_matrix, dtype='float64').reshape((4, 4)).transpose()
            trimesh_model = trimesh_model.copy().apply_transform(transform_array)
            if self.original.is_manifold() and not trimesh_model.is_volume:
                logger.warning("Result of transform is no longer a volume: {}", self.name)
                return None
        return trimesh_model


cdef class UVRemap(UnaryOperation):
    cdef str mapping

    @staticmethod
    cdef UVRemap _get(Model original, str mapping):
        cdef str name = f'uvremap({original.name}, {mapping})'
        cdef UVRemap model = <UVRemap>ModelCache.get(name, None)
        if model is None:
            model = UVRemap.__new__(UVRemap)
            model.name = name
            model.original = original
            model.original.add_dependent(model)
            model.mapping = mapping
            ModelCache[name] = model
        return model

    cdef object remap_sphere(self, trimesh_model):
        cdef const double[:, :] vertices
        cdef object vertex_uv_array
        cdef double[:, :] vertex_uv
        cdef int64_t i, n
        cdef double x, y, z
        n = len(trimesh_model.vertices)
        vertices = trimesh_model.vertices.astype('float64')
        vertex_uv_array = np.zeros((n, 2), dtype='float64')
        vertex_uv = vertex_uv_array
        for i in range(n):
            x, y, z = vertices[i][0], vertices[i][1], vertices[i][2]
            vertex_uv[i][0] = atan2(y, x) / Tau % 1
            vertex_uv[i][1] = atan2(z, sqrt(x*x + y*y)) / Tau * 2 + 0.5
        visual = trimesh.visual.texture.TextureVisuals(uv=vertex_uv_array)
        return trimesh.base.Trimesh(vertices=trimesh_model.vertices, vertex_normals=trimesh_model.vertex_normals, faces=trimesh_model.faces, visual=visual)

    cpdef object build_trimesh(self):
        trimesh_model = self.original.get_trimesh()
        if trimesh_model is not None:
            if self.mapping is 'sphere':
                trimesh_model = self.remap_sphere(trimesh_model)
        return trimesh_model


cdef class Slice(UnaryOperation):
    cdef Vector origin
    cdef Vector normal

    @staticmethod
    cdef Slice _get(Model original, Vector origin, Vector normal):
        if origin.numbers == NULL or origin.length != 3 or normal.numbers == NULL or normal.length != 3:
            return None
        original = original.manifold()
        cdef str name = f'slice({original.name}, {hex(origin.hash(False) ^ normal.hash(False))[3:]})'
        cdef Slice model = <Slice>ModelCache.get(name, None)
        if model is None:
            model = Slice.__new__(Slice)
            model.name = name
            model.original = original
            model.original.add_dependent(model)
            model.origin = origin
            model.normal = normal.normalize()
            ModelCache[name] = model
        return model

    cpdef bint is_manifold(self):
        return True

    cpdef Model manifold(self):
        return self

    cpdef Model repair(self):
        return self

    cdef Model _transform(self, Matrix44 transform_matrix):
        cdef Vector origin = transform_matrix.vmul(self.origin)
        cdef Vector normal = transform_matrix.inverse_transpose_matrix33().vmul(self.normal).normalize()
        return self.original._transform(transform_matrix)._slice(origin, normal)

    cpdef object build_trimesh(self):
        trimesh_model = self.original.get_trimesh()
        if trimesh_model is not None:
            manifold = manifold3d.Manifold(mesh=manifold3d.Mesh(vert_properties=np.array(trimesh_model.vertices, dtype=np.float32),
                                                                tri_verts=np.array(trimesh_model.faces, dtype=np.uint32)))
            normal = self.normal.neg()
            mesh = manifold.trim_by_plane(normal=tuple(normal), origin_offset=self.origin.dot(normal)).to_mesh()
            if len(mesh.vert_properties) and len(mesh.tri_verts):
                trimesh_model = trimesh.base.Trimesh(vertices=mesh.vert_properties, faces=mesh.tri_verts)
            else:
                trimesh_model = None
                logger.warning("Result of slice was empty mesh: {}", self.name)
        return trimesh_model


cdef class BooleanOperation(Model):
    cdef str operation
    cdef list models

    @staticmethod
    cdef Model _get(str operation, list models):
        cdef Model child_model, first=None
        cdef set existing = set()
        models = list(models)
        models.reverse()
        cdef list collected_models = []
        cdef str name = operation + '('
        while models:
            child_model = models.pop()
            if child_model is None:
                continue
            if operation is 'union' and isinstance(child_model, BooleanOperation) and (<BooleanOperation>child_model).operation is 'union':
                models.extend(reversed((<BooleanOperation>child_model).models))
                continue
            child_model = child_model.manifold()
            if first is None:
                first = child_model
            elif operation is 'difference' and child_model is first:
                return None
            elif child_model in existing:
                continue
            else:
                name += ', '
            name += child_model.name
            existing.add(child_model)
            collected_models.append(child_model)
        if len(collected_models) == 0:
            return None
        if len(collected_models) == 1:
            return collected_models[0]
        name += ')'
        cdef BooleanOperation model = <BooleanOperation>ModelCache.get(name, None)
        if model is None:
            model = BooleanOperation.__new__(BooleanOperation)
            model.name = name
            model.operation = operation
            model.models = collected_models
            for child_model in collected_models:
                child_model.add_dependent(model)
            ModelCache[name] = model
        return model

    cpdef bint is_manifold(self):
        return True

    cpdef void check_for_changes(self):
        cdef Model model
        if self.valid:
            for model in self.models:
                model.check_for_changes()

    cpdef Model manifold(self):
        return self

    cpdef Model repair(self):
        return self

    cdef Model _transform(self, Matrix44 transform_matrix):
        cdef Model model
        cdef list models = []
        for model in self.models:
            models.append(model._transform(transform_matrix))
        return BooleanOperation._get(self.operation, models)

    cdef Model _slice(self, Vector origin, Vector normal):
        cdef Model model
        cdef list models = []
        cdef int64_t i
        for i, model in enumerate(self.models):
            if i == 0 or self.operation is 'union':
                models.append(model._slice(origin, normal))
            else:
                models.append(model)
        return BooleanOperation._get(self.operation, models)

    cpdef object build_trimesh(self):
        cdef list trimesh_models = []
        cdef Model model
        for model in self.models:
            trimesh_model = model.get_trimesh()
            if trimesh_model is not None:
                trimesh_models.append(trimesh_model)
        if not trimesh_models:
            return None
        if len(trimesh_models) == 1:
            return trimesh_models[0]
        if self.operation is 'difference' and len(trimesh_models) > 2:
            unioned_models = trimesh.boolean.boolean_manifold(trimesh_models[1:], 'union')
            trimesh_model = trimesh.boolean.boolean_manifold([trimesh_models[0], unioned_models], 'difference')
        else:
            trimesh_model = trimesh.boolean.boolean_manifold(trimesh_models, self.operation)
        if len(trimesh_model.vertices) and len(trimesh_model.faces):
            return trimesh_model
        logger.warning("Result of {} was empty mesh: {}", self.operation, self.name)
        return None


cdef class PrimitiveModel(Model):
    cpdef void check_for_changes(self):
        pass

    cpdef bint is_manifold(self):
        return False


cdef class Box(PrimitiveModel):
    cdef str uv_map

    Vertices = np.array([
        (+.5, +.5, -.5), (+.5, +.5, +.5), (+.5, -.5, +.5), (+.5, -.5, -.5),  # +X
        (-.5, +.5, +.5), (-.5, +.5, -.5), (-.5, -.5, -.5), (-.5, -.5, +.5),  # -X
        (+.5, +.5, -.5), (-.5, +.5, -.5), (-.5, +.5, +.5), (+.5, +.5, +.5),  # +Y
        (+.5, -.5, +.5), (-.5, -.5, +.5), (-.5, -.5, -.5), (+.5, -.5, -.5),  # -Y
        (+.5, +.5, +.5), (-.5, +.5, +.5), (-.5, -.5, +.5), (+.5, -.5, +.5),  # +Z
        (-.5, +.5, -.5), (+.5, +.5, -.5), (+.5, -.5, -.5), (-.5, -.5, -.5),  # -Z
    ], dtype='f4')
    VertexNormals = np.array([
        (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0),
        (-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0),
        (0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1),
        (0, 0, -1), (0, 0, -1), (0, 0, -1), (0, 0, -1),
    ], dtype='f4')
    VertexUV = {
        'standard': np.array([
            (1/6, 1), (0/6, 1), (0/6, 0), (1/6, 0),
            (2/6, 1), (1/6, 1), (1/6, 0), (2/6, 0),
            (3/6, 1), (2/6, 1), (2/6, 0), (3/6, 0),
            (4/6, 1), (3/6, 1), (3/6, 0), (4/6, 0),
            (5/6, 1), (4/6, 1), (4/6, 0), (5/6, 0),
            (6/6, 1), (5/6, 1), (5/6, 0), (6/6, 0),
        ], dtype='f4'),
        'repeat': np.array([
            (1, 1), (0, 1), (0, 0), (1, 0),
            (1, 1), (0, 1), (0, 0), (1, 0),
            (1, 1), (0, 1), (0, 0), (1, 0),
            (1, 1), (0, 1), (0, 0), (1, 0),
            (1, 1), (0, 1), (0, 0), (1, 0),
            (1, 1), (0, 1), (0, 0), (1, 0),
        ], dtype='f4'),
    }
    Faces = np.array([
        (0, 1, 2), (2, 3, 0),
        (4, 5, 6), (6, 7, 4),
        (8, 9, 10), (10, 11, 8),
        (12, 13, 14), (14, 15, 12),
        (16, 17, 18), (18, 19, 16),
        (20, 21, 22), (22, 23, 20),
    ], dtype='i4')

    @staticmethod
    cdef Box _get(str uv_map):
        uv_map = uv_map if uv_map in Box.VertexUV else 'standard'
        cdef str name = '!box' if uv_map == 'standard' else f'!box({uv_map})'
        cdef Box model = <Box>ModelCache.get(name, None)
        if model is None:
            model = Box.__new__(Box)
            model.name = name
            model.uv_map = uv_map
            model.trimesh_model = None
            ModelCache[name] = model
        return model

    cpdef object build_trimesh(self):
        visual = trimesh.visual.texture.TextureVisuals(uv=Box.VertexUV[self.uv_map])
        return trimesh.base.Trimesh(vertices=Box.Vertices, vertex_normals=Box.VertexNormals, faces=Box.Faces, visual=visual)


cdef class Sphere(PrimitiveModel):
    cdef int64_t segments

    @staticmethod
    cdef Sphere _get(int64_t segments):
        segments = max(4, <int64_t>ceil(<double>segments / 4.0) * 4)
        cdef str name = f'!sphere-{segments}' if segments != DefaultSegments else '!sphere'
        cdef Sphere model = <Sphere>ModelCache.get(name, None)
        if model is None:
            model = Sphere.__new__(Sphere)
            model.name = name
            model.segments = segments
            ModelCache[name] = model
        return model

    @cython.cdivision(True)
    @cython.boundscheck(False)
    cpdef object build_trimesh(self):
        cdef int64_t nrows = self.segments / 4, nvertices = (nrows + 1) * (nrows + 2) * 4, nfaces = nrows * nrows * 8
        cdef object vertices_array = np.empty((nvertices, 3), dtype='f4')
        cdef float[:, :] vertices = vertices_array
        cdef object vertex_uv_array = np.empty((nvertices, 2), dtype='f4')
        cdef float[:, :] vertex_uv = vertex_uv_array
        cdef object faces_array = np.empty((nfaces, 3), dtype='i4')
        cdef int32_t[:, :] faces = faces_array
        cdef float x, y, z, r, th, u, v
        cdef int64_t side, hemisphere, row, col, i = 0, j = 0
        for side in range(4):
            for hemisphere in range(-1, 2, 2):
                for row in range(nrows + 1):
                    if row == 0:
                        r, z, v = 0, hemisphere, hemisphere * 0.5 + 0.5
                    elif row == nrows:
                        r, z, v = 1, 0, 0.5
                    else:
                        th = hemisphere * (1 - <float>row / nrows) / 4
                        v, th = 2 * th + 0.5, th * Tau
                        r, z = cos(th), sin(th)
                    for col in range(row + 1):
                        if row == 0:
                            u = (side + 0.5) / 4
                            x = y = 0
                        elif col == 0:
                            u = side / 4.0
                            x = r if side == 0 else -r if side == 2 else 0
                            y = r if side == 1 else -r if side == 3 else 0
                        elif col == row:
                            u = (side + 1.0) / 4
                            x = r if side == 3 else -r if side == 1 else 0
                            y = r if side == 0 else -r if side == 2 else 0
                        else:
                            u = (side + (<float>col / row)) / 4
                            th = Tau * u
                            x, y = r * cos(th), r * sin(th)
                        vertices[i, 0], vertices[i, 1], vertices[i, 2] = x, y, z
                        vertex_uv[i, 0], vertex_uv[i, 1] = u, v
                        if col:
                            faces[j, 0] = i
                            if hemisphere == -1:
                                faces[j, 1], faces[j, 2] = i-1, i-row-1
                            else:
                                faces[j, 1], faces[j, 2] = i-row-1, i-1
                            j += 1
                            if col < row:
                                faces[j, 0] = i
                                if hemisphere == -1:
                                    faces[j, 1], faces[j, 2] = i-row-1, i-row
                                else:
                                    faces[j, 1], faces[j, 2] = i-row, i-row-1
                                j += 1
                        i += 1
        visual = trimesh.visual.texture.TextureVisuals(uv=vertex_uv_array)
        return trimesh.base.Trimesh(vertices=vertices_array, vertex_normals=vertices_array, faces=faces_array, visual=visual)


cdef class Cylinder(PrimitiveModel):
    cdef int64_t segments

    @staticmethod
    cdef Cylinder _get(int64_t segments):
        segments = max(2, segments)
        cdef str name = f'!cylinder-{segments}' if segments != DefaultSegments else '!cylinder'
        cdef Cylinder model = <Cylinder>ModelCache.get(name, None)
        if model is None:
            model = Cylinder.__new__(Cylinder)
            model.name = name
            model.segments = segments
            ModelCache[name] = model
        return model

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object build_trimesh(self):
        cdef int64_t i, j, k, n = self.segments, m = (n+1)*6
        cdef object vertices_array = np.empty((m, 3), dtype='f4')
        cdef float[:, :] vertices = vertices_array
        cdef object vertex_normals_array = np.empty((m, 3), dtype='f4')
        cdef float[:, :] vertex_normals = vertex_normals_array
        cdef object vertex_uv_array = np.empty((m, 2), dtype='f4')
        cdef float[:, :] vertex_uv = vertex_uv_array
        cdef object faces_array = np.empty((n*4, 3), dtype='i4')
        cdef int32_t[:, :] faces = faces_array
        cdef float x, y, th, u, u_
        for i in range(n+1):
            j = k = i * 6
            u = <float>i / n
            u_ = (i+0.5) / n
            if i == 0 or i == n:
                x, y = 1, 0
            else:
                th = Tau * u
                x, y = cos(th), sin(th)
            # bottom centre (k):
            vertices[j, 0], vertices[j, 1], vertices[j, 2] = 0, 0, -0.5
            vertex_normals[j, 0], vertex_normals[j, 1], vertex_normals[j, 2] = 0, 0, -1
            vertex_uv[j, 0], vertex_uv[j, 1] = u_, 0
            j += 1
            # bottom edge (k+1):
            vertices[j, 0], vertices[j, 1], vertices[j, 2] = x, y, -0.5
            vertex_normals[j, 0], vertex_normals[j, 1], vertex_normals[j, 2] = 0, 0, -1
            vertex_uv[j, 0], vertex_uv[j, 1] = u, 0.25
            j += 1
            # side bottom (k+2):
            vertices[j, 0], vertices[j, 1], vertices[j, 2] = x, y, -0.5
            vertex_normals[j, 0], vertex_normals[j, 1], vertex_normals[j, 2] = x, y, 0
            vertex_uv[j, 0], vertex_uv[j, 1] = u, 0.25
            j += 1
            # side top (k+3):
            vertices[j, 0], vertices[j, 1], vertices[j, 2] = x, y, 0.5
            vertex_normals[j, 0], vertex_normals[j, 1], vertex_normals[j, 2] = x, y, 0
            vertex_uv[j, 0], vertex_uv[j, 1] = u, 0.75
            j += 1
            # top edge (k+4):
            vertices[j, 0], vertices[j, 1], vertices[j, 2] = x, y, 0.5
            vertex_normals[j, 0], vertex_normals[j, 1], vertex_normals[j, 2] = 0, 0, 1
            vertex_uv[j, 0], vertex_uv[j, 1] = u, 0.75
            j += 1
            # top centre (k+5):
            vertices[j, 0], vertices[j, 1], vertices[j, 2] = 0, 0, 0.5
            vertex_normals[j, 0], vertex_normals[j, 1], vertex_normals[j, 2] = 0, 0, 1
            vertex_uv[j, 0], vertex_uv[j, 1] = u_, 1
            if i < n:
                j = i * 4
                # bottom face
                faces[j, 0], faces[j, 1], faces[j, 2] = k, k+1+6, k+1
                j += 1
                # side face 1
                faces[j, 0], faces[j, 1], faces[j, 2] = k+2+6, k+3, k+2
                j += 1
                # side face 2
                faces[j, 0], faces[j, 1], faces[j, 2] = k+3, k+2+6, k+3+6
                j += 1
                # top face
                faces[j, 0], faces[j, 1], faces[j, 2] = k+5, k+4, k+4+6
        visual = trimesh.visual.texture.TextureVisuals(uv=vertex_uv_array)
        return trimesh.base.Trimesh(vertices=vertices_array, vertex_normals=vertex_normals_array, faces=faces_array, visual=visual)


cdef class Cone(PrimitiveModel):
    cdef int64_t segments

    @staticmethod
    cdef Cone _get(int64_t segments):
        segments = max(2, segments)
        cdef str name = f'!cone-{segments}' if segments != DefaultSegments else '!cone'
        cdef Cone model = <Cone>ModelCache.get(name, None)
        if model is None:
            model = Cone.__new__(Cone)
            model.name = name
            model.segments = segments
            ModelCache[name] = model
        return model

    @cython.cdivision(True)
    @cython.boundscheck(False)
    cpdef object build_trimesh(self):
        cdef int64_t i, j, k, n = self.segments, m = (n+1)*4
        cdef object vertices_array = np.empty((m, 3), dtype='f4')
        cdef float[:, :] vertices = vertices_array
        cdef object vertex_normals_array = np.empty((m, 3), dtype='f4')
        cdef float[:, :] vertex_normals = vertex_normals_array
        cdef object vertex_uv_array = np.empty((m, 2), dtype='f4')
        cdef float[:, :] vertex_uv = vertex_uv_array
        cdef object faces_array = np.empty((n*2, 3), dtype='i4')
        cdef int32_t[:, :] faces = faces_array
        cdef float x, y, th, u, u_
        for i in range(n+1):
            j = k = i * 4
            u = <double>i / n
            u_ = (i+0.5) / n
            th_ = Tau * u_
            if i == 0 or i == n:
                x, y = 1, 0
            else:
                th = Tau * u
                x, y = cos(th), sin(th)
            # bottom centre (k):
            vertices[j, 0], vertices[j, 1], vertices[j, 2] = 0, 0, -0.5
            vertex_normals[j, 0], vertex_normals[j, 1], vertex_normals[j, 2] = 0, 0, -1
            vertex_uv[j, 0], vertex_uv[j, 1] = u_, 0
            j += 1
            # bottom edge (k+1):
            vertices[j, 0], vertices[j, 1], vertices[j, 2] = x, y, -0.5
            vertex_normals[j, 0], vertex_normals[j, 1], vertex_normals[j, 2] = 0, 0, -1
            vertex_uv[j, 0], vertex_uv[j, 1] = u, 0.25
            j += 1
            # side bottom (k+2):
            vertices[j, 0], vertices[j, 1], vertices[j, 2] = x, y, -0.5
            vertex_normals[j, 0], vertex_normals[j, 1], vertex_normals[j, 2] = x*RootHalf, y*RootHalf, RootHalf
            vertex_uv[j, 0], vertex_uv[j, 1] = u, 0.25
            j += 1
            # side top (k+3):
            vertices[j, 0], vertices[j, 1], vertices[j, 2] = 0, 0, 0.5
            vertex_normals[j, 0], vertex_normals[j, 1], vertex_normals[j, 2] = cos(th_)*RootHalf, sin(th_)*RootHalf, RootHalf
            vertex_uv[j, 0], vertex_uv[j, 1] = u_, 1
            if i < n:
                j = i * 2
                # bottom face
                faces[j, 0], faces[j, 1], faces[j, 2] = k, k+1+4, k+1
                j += 1
                # side face
                faces[j, 0], faces[j, 1], faces[j, 2] = k+3, k+2, k+2+4
        visual = trimesh.visual.texture.TextureVisuals(uv=vertex_uv_array)
        return trimesh.base.Trimesh(vertices=vertices_array, vertex_normals=vertex_normals_array, faces=faces_array, visual=visual)


cdef class ExternalModel(Model):
    cdef object cache_path

    @staticmethod
    cdef ExternalModel _get(str filename):
        if filename is None:
            return None
        cdef ExternalModel model = <ExternalModel>ModelCache.get(filename, None)
        if model is None:
            model = ExternalModel.__new__(ExternalModel)
            model.name = filename
            model.cache_path = SharedCache[filename]
            ModelCache[filename] = model
        return model

    cpdef bint is_manifold(self):
        return False

    cpdef void check_for_changes(self):
        if self.valid and self.trimesh_model is not self.cache_path.read_trimesh_model():
            self.invalidate()

    cpdef object build_trimesh(self):
        return self.cache_path.read_trimesh_model()
