
import cython
from loguru import logger
import manifold3d
import numpy as np
import trimesh

from libc.math cimport cos, sin, sqrt
from libc.stdint cimport int32_t, int64_t

from ... import name_patch
from ...cache import SharedCache


logger = name_patch(logger, __name__)

cdef dict ModelCache = {}
cdef double Tau = 6.283185307179586
cdef double RootHalf = sqrt(0.5)
cdef double DefaultSnapAngle = 0.05
cdef int64_t DefaultSegments = 64


cdef class Model:
    def __hash__(self):
        return <Py_hash_t>(<void*>self)

    def __eq__(self, other):
        return self is other

    cdef bint is_manifold(self):
        return False

    cdef bint check_valid(self):
        raise NotImplementedError()

    cdef void build_trimesh_model(self):
        raise NotImplementedError()

    cdef Vector get_bounds(self):
        if self.check_valid() and self.bounds is not None:
            return self.bounds
        self.build_trimesh_model()
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
        if self.check_valid():
            if name in objects:
                return objects[name]
        else:
            self.build_trimesh_model()
        trimesh_model = self.trimesh_model
        if trimesh_model is None:
            if name in objects:
                del objects[name]
            return None, None
        cdef tuple buffers
        cdef bint has_uv = trimesh_model.visual is not None and isinstance(trimesh_model.visual, trimesh.visual.texture.TextureVisuals) and \
            trimesh_model.visual.uv is not None
        vertex_uvs = trimesh_model.visual.uv if has_uv else np.zeros((len(trimesh_model.vertices), 2))
        vertex_data = np.hstack((trimesh_model.vertices, trimesh_model.vertex_normals, vertex_uvs)).astype('f4')
        index_data = trimesh_model.faces.astype('i4')
        buffers = (glctx.buffer(vertex_data), glctx.buffer(index_data))
        logger.trace("Prepared model {} with {} vertices and {} faces", name, len(trimesh_model.vertices), len(trimesh_model.faces))
        objects[name] = buffers
        return buffers

    cdef Model manifold(self):
        return Manifold.get(self)

    cdef Model flatten(self):
        return Flatten.get(self)

    cdef Model invert(self):
        return Invert.get(self)

    cdef Model repair(self):
        return Repair.get(self)

    cdef Model snap_edges(self, double snap_angle, double minimum_area):
        return SnappedEdgesModel.get(self, snap_angle, minimum_area)

    cdef Model transform(self, Matrix44 transform_matrix):
        return Transform.get(self, transform_matrix)

    cdef Model slice(self, Vector origin, Vector normal):
        return Slice.get(self, origin, normal)

    @staticmethod
    cdef Model intersect(list models):
        return BooleanOperation.get('intersection', models)

    @staticmethod
    cdef Model union(list models):
        return BooleanOperation.get('union', models)

    @staticmethod
    cdef Model difference(list models):
        return BooleanOperation.get('difference', models)

    @staticmethod
    cdef Model get_box(Node node):
        return Box.get(node)

    @staticmethod
    cdef Model get_sphere(Node node):
        return Sphere.get(node)

    @staticmethod
    cdef Model get_cylinder(Node node):
        return Cylinder.get(node)

    @staticmethod
    cdef Model get_cone(Node node):
        return Cone.get(node)

    @staticmethod
    cdef Model get_external(Node node):
        return ExternalModel.get(node)


cdef class UnaryOperation(Model):
    cdef Model original

    cdef bint is_manifold(self):
        return self.original.is_manifold()

    cdef bint check_valid(self):
        if not self.valid:
            return False
        if self.original.check_valid():
            return True
        self.valid = False
        return False


cdef class Manifold(UnaryOperation):
    @staticmethod
    cdef Manifold get(Model original):
        assert not isinstance(original, Manifold)
        cdef str name = f'manifold({original.name})'
        cdef Manifold model = <Manifold>ModelCache.get(name, None)
        if model is None:
            model = Manifold.__new__(Manifold)
            model.name = name
            model.original = original
            ModelCache[name] = model
        return model

    cdef bint is_manifold(self):
        return True

    cdef Model manifold(self):
        return self

    cdef Model flatten(self):
        return self.original.flatten()

    cdef Model snap_edges(self, double snap_angle, double minimum_area):
        return self.original.snap_edges(snap_angle, minimum_area)

    cdef void build_trimesh_model(self):
        if not self.original.check_valid():
            self.original.build_trimesh_model()
        cdef bint merged=False, filled=False, hull=False
        trimesh_model = self.original.trimesh_model
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
        self.trimesh_model = trimesh_model
        self.valid = True


cdef class Flatten(UnaryOperation):
    @staticmethod
    cdef Flatten get(Model original):
        cdef str name = f'flat({original.name})'
        cdef Flatten model = <Flatten>ModelCache.get(name, None)
        if model is None:
            model = Flatten.__new__(Flatten)
            model.name = name
            model.original = original
            ModelCache[name] = model
        return model

    cdef void build_trimesh_model(self):
        if not self.original.check_valid():
            self.original.build_trimesh_model()
        trimesh_model = self.original.trimesh_model.copy()
        trimesh_model.unmerge_vertices()
        self.trimesh_model = trimesh_model
        self.valid = True


cdef class Invert(UnaryOperation):
    @staticmethod
    cdef Invert get(Model original):
        cdef str name = f'invert({original.name})'
        cdef Invert model = <Invert>ModelCache.get(name, None)
        if model is None:
            model = Invert.__new__(Invert)
            model.name = name
            model.original = original
            ModelCache[name] = model
        return model

    cdef Model manifold(self):
        return Invert.get(self.original.manifold())

    cdef void build_trimesh_model(self):
        if not self.original.check_valid():
            self.original.build_trimesh_model()
        trimesh_model = self.original.trimesh_model
        self.trimesh_model = trimesh.base.Trimesh(vertices=trimesh_model.vertices,
                                                  vertex_normals=-trimesh_model.vertex_normals,
                                                  faces=trimesh_model.faces[:, ::-1],
                                                  visual=trimesh_model.visual)
        self.valid = True


cdef class SnappedEdgesModel(UnaryOperation):
    cdef double snap_angle
    cdef double minimum_area

    @staticmethod
    cdef SnappedEdgesModel get(Model original, double snap_angle, double minimum_area):
        cdef str name = 'snap(' + original.name
        if snap_angle != DefaultSnapAngle:
            name += f', {snap_angle:g}'
        if minimum_area:
            name += f', {minimum_area:g}'
        name += ')'
        cdef SnappedEdgesModel model = <SnappedEdgesModel>ModelCache.get(name, None)
        if model is None:
            model = SnappedEdgesModel.__new__(SnappedEdgesModel)
            model.name = name
            model.original = original
            model.snap_angle = snap_angle
            model.minimum_area = minimum_area
            ModelCache[name] = model
        return model

    cdef void build_trimesh_model(self):
        if not self.original.check_valid():
            self.original.build_trimesh_model()
        if self.original.trimesh_model is not None:
            self.trimesh_model = trimesh.graph.smooth_shade(self.original.trimesh_model, angle=self.snap_angle*Tau,
                                                            facet_minarea=1/self.minimum_area if self.minimum_area else None)
        else:
            self.trimesh_model = None
        self.valid = True


cdef class Repair(UnaryOperation):
    @staticmethod
    cdef Repair get(Model original):
        cdef str name = f'repair({original.name})'
        cdef Repair model = <Repair>ModelCache.get(name, None)
        if model is None:
            model = Repair.__new__(Repair)
            model.name = name
            model.original = original
            ModelCache[name] = model
        return model

    cdef bint is_manifold(self):
        return True

    cdef Model manifold(self):
        return self

    cdef void build_trimesh_model(self):
        if not self.original.check_valid():
            self.original.build_trimesh_model()
        trimesh_model = self.original.trimesh_model
        if trimesh_model is not None:
            trimesh_model = trimesh_model.copy()
            trimesh_model.process(validate=True, merge_tex=True, merge_norm=True)
            trimesh_model.remove_unreferenced_vertices()
            trimesh_model.fill_holes()
            trimesh_model.fix_normals()
            self.trimesh_model = trimesh_model
        else:
            self.trimesh_model = None
        self.valid = True


cdef class Transform(UnaryOperation):
    cdef Matrix44 transform_matrix

    @staticmethod
    cdef Model get(Model original, Matrix44 transform_matrix):
        cdef str name = f'{original.name}@{hex(transform_matrix.hash(False))[3:]}'
        cdef Transform model = <Transform>ModelCache.get(name, None)
        if model is None:
            model = Transform.__new__(Transform)
            model.name = name
            model.original = original
            model.transform_matrix = transform_matrix
            ModelCache[name] = model
        return model

    cdef Model transform(self, Matrix44 transform_matrix):
        return Transform.get(self.original, transform_matrix.mmul(self.transform_matrix))

    cdef Model flatten(self):
        return Flatten.get(self.original).transform(self.transform_matrix)

    cdef Model invert(self):
        return Invert.get(self.original).transform(self.transform_matrix)

    cdef Model repair(self):
        return Repair.get(self.original).transform(self.transform_matrix)

    cdef Model snap_edges(self, double snap_angle, double minimum_area):
        return SnappedEdgesModel.get(self.original, snap_angle, minimum_area).transform(self.transform_matrix)

    cdef void build_trimesh_model(self):
        if not self.original.check_valid():
            self.original.build_trimesh_model()
        if self.original.trimesh_model is not None:
            transform_array = np.array(self.transform_matrix, dtype='float64').reshape((4, 4)).transpose()
            trimesh_model = self.original.trimesh_model.copy().apply_transform(transform_array)
            self.trimesh_model = trimesh_model if len(trimesh_model.vertices) and len(trimesh_model.faces) else None
        else:
            self.trimesh_model = None
        self.valid = True


cdef class Slice(UnaryOperation):
    cdef Vector origin
    cdef Vector normal

    @staticmethod
    cdef Slice get(Model original, Vector origin, Vector normal):
        original = original.manifold()
        cdef str name = f'slice({original.name}, {hex(origin.hash(False) ^ normal.hash(False))[3:]})'
        cdef Slice model = <Slice>ModelCache.get(name, None)
        if model is None:
            model = Slice.__new__(Slice)
            model.name = name
            model.original = original
            model.origin = origin
            model.normal = normal.normalize()
            ModelCache[name] = model
        return model

    cdef bint is_manifold(self):
        return True

    cdef Model manifold(self):
        return self

    cdef Model transform(self, Matrix44 transform_matrix):
        cdef Vector origin = transform_matrix.vmul(self.origin)
        cdef Vector normal = transform_matrix.inverse_transpose_matrix33().vmul(self.normal).normalize()
        return Slice.get(self.original.transform(transform_matrix), origin, normal)

    cdef void build_trimesh_model(self):
        if not self.original.check_valid():
            self.original.build_trimesh_model()
        if self.original.trimesh_model is not None:
            trimesh_model = self.original.trimesh_model
            manifold = manifold3d.Manifold(mesh=manifold3d.Mesh(vert_properties=np.array(trimesh_model.vertices, dtype=np.float32),
                                                                tri_verts=np.array(trimesh_model.faces, dtype=np.uint32)))
            normal = self.normal.neg()
            mesh = manifold.trim_by_plane(normal=tuple(normal), origin_offset=self.origin.dot(normal)).to_mesh()
            if len(mesh.vert_properties) and len(mesh.tri_verts):
                self.trimesh_model = trimesh.base.Trimesh(vertices=mesh.vert_properties, faces=mesh.tri_verts)
            else:
                self.trimesh_model = None
                logger.warning("Result of slice was empty mesh: {}", self.name)
        else:
            self.trimesh_model = None
        self.valid = True


cdef class BooleanOperation(Model):
    cdef str operation
    cdef list models

    @staticmethod
    cdef Model get(str operation, list models):
        cdef Model child_model
        cdef list collected_models
        if operation is 'union':
            collected_models = []
            for child_model in models:
                if isinstance(child_model, BooleanOperation) and (<BooleanOperation>child_model).operation is 'union':
                    collected_models.extend((<BooleanOperation>child_model).models)
                else:
                    collected_models.append(child_model)
            models = collected_models
        cdef set existing = set()
        collected_models = []
        for child_model in models:
            if child_model is None:
                continue
            child_model = child_model.manifold()
            if child_model in existing:
                if operation is 'difference' and len(collected_models) and child_model is collected_models[0]:
                    return None
                continue
            existing.add(child_model)
            collected_models.append(child_model)
        models = collected_models
        if len(models) == 0:
            return None
        if len(models) == 1:
            return models[0]
        cdef str name = operation + '('
        cdef int64_t i
        for i, child_model in enumerate(models):
            if i:
                name += ', '
            name += child_model.name
        name += ')'
        cdef BooleanOperation model = <BooleanOperation>ModelCache.get(name, None)
        if model is None:
            model = BooleanOperation.__new__(BooleanOperation)
            model.name = name
            model.operation = operation
            model.models = models
            ModelCache[name] = model
        return model

    cdef bint is_manifold(self):
        return True

    cdef Model manifold(self):
        return self

    cdef Model transform(self, Matrix44 transform_matrix):
        cdef Model model
        cdef list models = []
        for model in self.models:
            models.append(model.transform(transform_matrix))
        return BooleanOperation.get(self.operation, models)

    cdef Model slice(self, Vector origin, Vector normal):
        cdef Model model
        cdef list models = []
        cdef int64_t i
        for i, model in enumerate(self.models):
            if i == 0 or self.operation is 'union':
                models.append(model.slice(origin, normal))
            else:
                models.append(model)
        return BooleanOperation.get(self.operation, models)

    cdef bint check_valid(self):
        if not self.valid:
            return False
        cdef Model model
        for model in self.models:
            if not model.check_valid():
                break
        else:
            return True
        self.valid = False
        return False

    cdef void build_trimesh_model(self):
        cdef list trimesh_models = []
        cdef Model model
        for model in self.models:
            if not model.check_valid():
                model.build_trimesh_model()
            if model.trimesh_model is not None:
                trimesh_models.append(model.trimesh_model)
        if len(trimesh_models) == 1:
            self.trimesh_model = trimesh_models[0]
        elif trimesh_models:
            if self.operation is 'difference' and len(trimesh_models) > 2:
                union_models = trimesh.boolean.boolean_manifold(trimesh_models[1:], 'union')
                trimesh_model = trimesh.boolean.boolean_manifold([trimesh_models[0], union_models], 'difference')
            else:
                trimesh_model = trimesh.boolean.boolean_manifold(trimesh_models, self.operation)
            if len(trimesh_model.vertices) and len(trimesh_model.faces):
                self.trimesh_model = trimesh_model
            else:
                logger.warning("Result of {} was empty mesh: {}", self.operation, self.name)
                self.trimesh_model = None
        self.valid = True


cdef class PrimitiveModel(Model):
    cdef bint check_valid(self):
        return self.valid


cdef class Box(PrimitiveModel):
    Vertices = np.array([
        (-.5, -.5, +.5), (+.5, -.5, +.5), (+.5, +.5, +.5), (-.5, +.5, +.5),
        (-.5, +.5, +.5), (+.5, +.5, +.5), (+.5, +.5, -.5), (-.5, +.5, -.5),
        (+.5, +.5, +.5), (+.5, -.5, +.5), (+.5, -.5, -.5), (+.5, +.5, -.5),
        (+.5, +.5, -.5), (+.5, -.5, -.5), (-.5, -.5, -.5), (-.5, +.5, -.5),
        (-.5, +.5, -.5), (-.5, -.5, -.5), (-.5, -.5, +.5), (-.5, +.5, +.5),
        (-.5, -.5, -.5), (+.5, -.5, -.5), (+.5, -.5, +.5), (-.5, -.5, +.5),
    ], dtype='f4')
    VertexNormals = np.array([
        (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1),
        (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0),
        (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0),
        (0, 0, -1), (0, 0, -1), (0, 0, -1), (0, 0, -1),
        (-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0),
        (0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0),
    ], dtype='f4')
    VertexUV = np.array([
        (0, 0), (1/6, 0), (1/6, 1), (0, 1),
        (1/6, 0), (2/6, 0), (2/6, 1), (1/6, 1),
        (2/6, 0), (3/6, 0), (3/6, 1), (2/6, 1),
        (3/6, 0), (4/6, 0), (4/6, 1), (3/6, 1),
        (4/6, 0), (5/6, 0), (5/6, 1), (4/6, 1),
        (5/6, 0), (6/6, 0), (6/6, 1), (5/6, 1),
    ], dtype='f4')
    Faces = np.array([
        (0, 1, 2), (2, 3, 0),
        (4, 5, 6), (6, 7, 4),
        (8, 9, 10), (10, 11, 8),
        (12, 13, 14), (14, 15, 12),
        (16, 17, 18), (18, 19, 16),
        (20, 21, 22), (22, 23, 20),
    ], dtype='i4')

    @staticmethod
    cdef Box get(Node node):
        cdef str name = '!box'
        cdef Box model = <Box>ModelCache.pop(name, None)
        if model is None:
            model = Box.__new__(Box)
            model.name = name
            model.trimesh_model = None
        ModelCache[name] = model
        return model

    cdef void build_trimesh_model(self):
        visual = trimesh.visual.texture.TextureVisuals(uv=Box.VertexUV)
        self.trimesh_model = trimesh.base.Trimesh(vertices=Box.Vertices, vertex_normals=Box.VertexNormals, faces=Box.Faces, visual=visual)
        self.valid = True


cdef class Sphere(PrimitiveModel):
    cdef int64_t segments

    @staticmethod
    cdef Sphere get(Node node):
        cdef int64_t segments = max(4, node.get_int('segments', DefaultSegments) // 4 * 4)
        cdef str name = f'!sphere-{segments}' if segments != DefaultSegments else '!sphere'
        cdef Sphere model = <Sphere>ModelCache.get(name, None)
        if model is None:
            model = Sphere.__new__(Sphere)
            model.name = name
            model.segments = segments
            model.trimesh_model = None
            ModelCache[name] = model
        return model

    @cython.cdivision(True)
    @cython.boundscheck(False)
    cdef void build_trimesh_model(self):
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
                            u = (side + 0.125) / 4
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
        self.trimesh_model = trimesh.base.Trimesh(vertices=vertices_array, vertex_normals=vertices_array, faces=faces_array, visual=visual)
        self.valid = True


cdef class Cylinder(PrimitiveModel):
    cdef int64_t segments

    @staticmethod
    cdef Cylinder get(Node node):
        cdef int64_t segments = max(2, node.get_int('segments', DefaultSegments))
        cdef str name = f'!cylinder-{segments}' if segments != DefaultSegments else '!cylinder'
        cdef Cylinder model = <Cylinder>ModelCache.get(name, None)
        if model is None:
            model = Cylinder.__new__(Cylinder)
            model.name = name
            model.segments = segments
            model.trimesh_model = None
            ModelCache[name] = model
        return model

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void build_trimesh_model(self):
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
        self.trimesh_model = trimesh.base.Trimesh(vertices=vertices_array, vertex_normals=vertex_normals_array, faces=faces_array, visual=visual)
        self.valid = True


cdef class Cone(PrimitiveModel):
    cdef int64_t segments

    @staticmethod
    cdef Cone get(Node node):
        cdef int64_t segments = max(2, node.get_int('segments', DefaultSegments))
        cdef str name = f'!cone-{segments}' if segments != DefaultSegments else '!cone'
        cdef Cone model = <Cone>ModelCache.get(name, None)
        if model is None:
            model = Cone.__new__(Cone)
            model.name = name
            model.segments = segments
            model.trimesh_model = None
            ModelCache[name] = model
        return model

    @cython.cdivision(True)
    @cython.boundscheck(False)
    cdef void build_trimesh_model(self):
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
        self.trimesh_model = trimesh.base.Trimesh(vertices=vertices_array, vertex_normals=vertex_normals_array, faces=faces_array, visual=visual)
        self.valid = True


cdef class ExternalModel(Model):
    cdef object cache_path

    @staticmethod
    cdef ExternalModel get(Node node):
        cdef str name = node.get_str('filename', None)
        if not name:
            return None
        cdef ExternalModel model = <ExternalModel>ModelCache.get(name, None)
        if model is None:
            model = ExternalModel.__new__(ExternalModel)
            model.name = name
            model.cache_path = SharedCache[name]
            model.trimesh_model = None
            ModelCache[name] = model
        return model

    cdef bint check_valid(self):
        return self.trimesh_model is self.cache_path.read_trimesh_model()

    cdef void build_trimesh_model(self):
        self.trimesh_model = self.cache_path.read_trimesh_model()
        self.valid = True
