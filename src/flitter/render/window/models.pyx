
import cython
from loguru import logger
import manifold3d
import numpy as np
import trimesh
import trimesh.proximity

from libc.math cimport cos, sin, sqrt, atan2, abs
from libc.stdint cimport int32_t, int64_t
from cpython.object cimport PyObject
from cpython.dict cimport PyDict_GetItem

from ... import name_patch
from ...cache import SharedCache
from ...model cimport true_, Context, Vector, StateDict, HASH_START, HASH_UPDATE, HASH_STRING, double_long
from ...timer cimport perf_counter
from ...language.vm cimport VectorStack


logger = name_patch(logger, __name__)

cdef dict ModelCache = {}
cdef double Tau = 6.283185307179586231995926937088370323181152343750231995926937088370323181152343750
cdef double RootHalf = sqrt(0.5)
cdef double DefaultSnapAngle = 0.05
cdef int64_t DefaultSegments = 64
cdef Matrix44 IdentityTransform = Matrix44._identity()
cdef double NaN = float("nan")

cdef uint64_t FLATTEN = HASH_UPDATE(HASH_START, HASH_STRING('flatten'))
cdef uint64_t INVERT = HASH_UPDATE(HASH_START, HASH_STRING('invert'))
cdef uint64_t REPAIR = HASH_UPDATE(HASH_START, HASH_STRING('repair'))
cdef uint64_t SNAP_EDGES = HASH_UPDATE(HASH_START, HASH_STRING('snap_edges'))
cdef uint64_t TRANSFORM = HASH_UPDATE(HASH_START, HASH_STRING('transform'))
cdef uint64_t UV_REMAP = HASH_UPDATE(HASH_START, HASH_STRING('uv_remap'))
cdef uint64_t TRIM = HASH_UPDATE(HASH_START, HASH_STRING('trim'))
cdef uint64_t BOX = HASH_UPDATE(HASH_START, HASH_STRING('box'))
cdef uint64_t SPHERE = HASH_UPDATE(HASH_START, HASH_STRING('sphere'))
cdef uint64_t CYLINDER = HASH_UPDATE(HASH_START, HASH_STRING('cylinder'))
cdef uint64_t CONE = HASH_UPDATE(HASH_START, HASH_STRING('cone'))
cdef uint64_t EXTERNAL = HASH_UPDATE(HASH_START, HASH_STRING('external'))
cdef uint64_t VECTOR = HASH_UPDATE(HASH_START, HASH_STRING('vector'))
cdef uint64_t SDF = HASH_UPDATE(HASH_START, HASH_STRING('sdf'))
cdef uint64_t MIX = HASH_UPDATE(HASH_START, HASH_STRING('mix'))


cdef class Model:
    @staticmethod
    def flush_caches(double min_age=300, int64_t min_size=2000):
        cdef double now = perf_counter()
        cdef double cutoff = now - min_age
        cdef Model model
        cdef int64_t unload_count = 0
        cdef list unloaded = []
        while len(ModelCache) > min_size:
            for model in ModelCache.values():
                if model.touch_timestamp == 0:
                    model.touch_timestamp = now
                if model.touch_timestamp <= cutoff and not model.dependents:
                    unloaded.append(model)
            if not unloaded:
                break
            while unloaded:
                model = unloaded.pop()
                del ModelCache[model.id]
                model.unload()
                unload_count += 1
        if unload_count:
            logger.trace("Unloaded {} models from cache, {} remaining", unload_count, len(ModelCache))

    @staticmethod
    def by_id(uint64_t id):
        return ModelCache.get(id)

    @staticmethod
    def from_node(Node node):
        raise NotImplementedError()

    def __init__(self, id=None):
        if id is None:
            raise ValueError("id cannot be None")
        if id in ModelCache:
            raise ValueError(f"Model already exists with id: '{id}'")
        self.id = id
        ModelCache[id] = self

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<{self.__class__.__name__}(0x{self.id:x})>'

    @property
    def name(self):
        raise NotImplementedError()

    cpdef void unload(self):
        assert not self.dependents
        self.dependents = None
        self.cache = None
        if self.buffer_caches is not None:
            for cache in self.buffer_caches:
                if self.id in cache:
                    del cache[self.id]
            self.buffer_caches = None

    cpdef bint is_smooth(self):
        raise NotImplementedError()

    cpdef double signed_distance(self, double x, double y, double z) noexcept:
        raise NotImplementedError()

    def inverse_signed_distance(self, x, y, z):
        return -self.signed_distance(x, y, z)

    cpdef void check_for_changes(self):
        raise NotImplementedError()

    cpdef object build_trimesh(self):
        raise NotImplementedError()

    cpdef object build_manifold(self):
        cdef bint merged=False, filled=False, hull=False
        trimesh_model = self.get_trimesh()
        if trimesh_model is None:
            return None
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
            logger.error("Mesh is not a volume: {}", self.name)
            return None
        elif hull:
            logger.warning("Computed convex hull of non-manifold mesh: {}", self.name)
        elif filled:
            logger.debug("Filled holes in non-manifold mesh: {}", self.name)
        elif merged:
            logger.trace("Merged vertices of non-manifold mesh: {}", self.name)
        return manifold3d.Manifold(mesh=manifold3d.Mesh(vert_properties=np.array(trimesh_model.vertices, dtype='f4'),
                                                        tri_verts=np.array(trimesh_model.faces, dtype=np.uint32)))

    cpdef void add_dependent(self, Model model):
        if self.dependents is None:
            self.dependents = set()
        self.dependents.add(model)

    cpdef void remove_dependent(self, Model model):
        self.dependents.remove(model)

    cpdef void invalidate(self):
        cdef Model model
        cdef dict cache
        if self.cache:
            self.cache = None
        if self.dependents is not None:
            for model in self.dependents:
                model.invalidate()
        cdef object model_id
        if self.buffer_caches is not None:
            model_id = self.id
            for cache in self.buffer_caches:
                cache.pop(model_id)
            self.buffer_caches = None

    cpdef object get_trimesh(self):
        self.cache_timestamp = perf_counter()
        cdef PyObject* objptr
        if self.cache is None:
            self.cache = {}
        elif (objptr := PyDict_GetItem(self.cache, 'trimesh')) != NULL:
            return <object>objptr
        trimesh_model = self.build_trimesh()
        self.cache['trimesh'] = trimesh_model
        return trimesh_model

    cpdef object get_manifold(self):
        self.cache_timestamp = perf_counter()
        cdef PyObject* objptr
        if self.cache is None:
            self.cache = {}
        elif (objptr := PyDict_GetItem(self.cache, 'manifold')) != NULL:
            return <object>objptr
        manifold = self.build_manifold()
        self.cache['manifold'] = manifold
        return manifold

    cpdef Vector get_bounds(self):
        self.cache_timestamp = perf_counter()
        cdef PyObject* objptr
        if self.cache is None:
            self.cache = {}
        elif (objptr := PyDict_GetItem(self.cache, 'bounds')) != NULL:
            return <Vector>objptr
        cdef object trimesh_model = self.get_trimesh()
        cdef Vector bounds_vector = Vector.__new__(Vector)
        cdef const double[:, :] bounds
        cdef int64_t i, j
        if trimesh_model is not None:
            bounds = trimesh_model.bounds
            bounds_vector.allocate_numbers(6)
            for i in range(2):
                for j in range(3):
                    bounds_vector.numbers[i*3 + j] = bounds[i, j]
        self.cache['bounds'] = bounds_vector
        return bounds_vector

    cpdef tuple get_buffers(self, object glctx, dict objects):
        self.cache_timestamp = perf_counter()
        cdef object model_id = self.id
        cdef PyObject* objptr
        if (objptr := PyDict_GetItem(objects, model_id)) != NULL:
            return <tuple>objptr
        cdef trimesh_model = self.get_trimesh()
        cdef tuple buffers
        if self.buffer_caches is None:
            self.buffer_caches = []
        self.buffer_caches.append(objects)
        if trimesh_model is None:
            buffers = None, None
            objects[model_id] = buffers
            return buffers
        if (visual := trimesh_model.visual) is not None and isinstance(visual, trimesh.visual.texture.TextureVisuals) \
                and visual.uv is not None and len(visual.uv) == len(trimesh_model.vertices):
            vertex_uvs = visual.uv
        else:
            vertex_uvs = np.zeros((len(trimesh_model.vertices), 2))
        vertex_data = np.hstack((trimesh_model.vertices, trimesh_model.vertex_normals, vertex_uvs)).astype('f4', copy=False)
        index_data = trimesh_model.faces.astype('i4', copy=False)
        buffers = (glctx.buffer(vertex_data), glctx.buffer(index_data))
        logger.trace("Constructed model {} with {} vertices and {} faces", self.name, len(vertex_data), len(index_data))
        objects[model_id] = buffers
        return buffers

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
        if transform_matrix.eq(IdentityTransform) is true_:
            return self
        return Transform._get(self, transform_matrix)

    def transform(self, transform_matrix):
        return self._transform(Matrix44._coerce(transform_matrix))

    cdef Model _uv_remap(self, str mapping):
        return UVRemap._get(self, mapping)

    def uv_remap(self, mapping):
        return self._uv_remap(str(mapping))

    cdef Model _trim(self, Vector origin, Vector normal, double smooth, double fillet, double chamfer):
        return Trim._get(self, origin, normal, smooth, fillet, chamfer)

    def trim(self, origin, normal, smooth=0, fillet=0, chamfer=0):
        return self._trim(Vector._coerce(origin), Vector._coerce(normal), float(smooth), float(fillet), float(chamfer))

    @staticmethod
    cdef Model _boolean(str operation, list models, double smooth, double fillet, double chamfer):
        return BooleanOperation._get(operation, models, smooth, fillet, chamfer)

    @staticmethod
    def union(*models, smooth=0, fillet=0, chamfer=0):
        return BooleanOperation._get('union', list(models), smooth, fillet, chamfer)

    @staticmethod
    def intersect(*models, smooth=0, fillet=0, chamfer=0):
        return BooleanOperation._get('intersect', list(models), smooth, fillet, chamfer)

    @staticmethod
    def difference(*models, smooth=0, fillet=0, chamfer=0):
        return BooleanOperation._get('difference', list(models), smooth, fillet, chamfer)

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

    @staticmethod
    cdef Model _vector(Vector vertices, Vector faces):
        return VectorModel._get(vertices, faces)

    @staticmethod
    def vector(vertices, faces):
        return VectorModel._get(Vector._coerce(vertices), Vector._coerce(faces))

    @staticmethod
    cdef Model _sdf(Function function, Model original, Vector minimum, Vector maximum, double resolution):
        return SignedDistanceFunction._get(function, original, minimum, maximum, resolution)

    @staticmethod
    def sdf(Function function, Model original, minimum, maximum, resolution):
        return SignedDistanceFunction._get(function, original, Vector._coerce(minimum), Vector._coerce(maximum), float(resolution))

    @staticmethod
    cdef Model _mix(list models, Vector weights):
        return Mix._get(models, weights)

    @staticmethod
    def mix(list models, weights):
        return Mix._get(list(models), Vector._coerce(weights))


cdef class UnaryOperation(Model):
    cdef Model original

    cpdef void unload(self):
        if self.original is not None:
            self.original.remove_dependent(self)
        super(UnaryOperation, self).unload()

    cpdef bint is_smooth(self):
        return self.original.is_smooth() if self.original is not None else False

    cpdef double signed_distance(self, double x, double y, double z) noexcept:
        return self.original.signed_distance(x, y, z) if self.original is not None else NaN

    cpdef void check_for_changes(self):
        if self.original is not None:
            self.original.check_for_changes()


cdef class Flatten(UnaryOperation):
    @staticmethod
    cdef Flatten _get(Model original):
        cdef uint64_t id = HASH_UPDATE(FLATTEN, original.id)
        cdef Flatten model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = Flatten.__new__(Flatten)
            model.id = id
            model.original = original
            model.original.add_dependent(model)
            ModelCache[id] = model
        else:
            model = <Flatten>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        return f'flatten({self.original.name})'

    cpdef bint is_smooth(self):
        return False

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

    cpdef object build_manifold(self):
        return self.original.get_manifold()


cdef class Invert(UnaryOperation):
    @staticmethod
    cdef Invert _get(Model original):
        cdef uint64_t id = HASH_UPDATE(INVERT, original.id)
        cdef Invert model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = Invert.__new__(Invert)
            model.id = id
            model.original = original
            model.original.add_dependent(model)
            ModelCache[id] = model
        else:
            model = <Invert>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        return f'invert({self.original.name})'

    cpdef Model invert(self):
        return self.original

    cpdef Model repair(self):
        return self.original.repair().invert()

    cdef Model _snap_edges(self, double snap_angle, double minimum_area):
        return self.original._snap_edges(snap_angle, minimum_area).invert()

    cdef Model _transform(self, Matrix44 transform_matrix):
        return self.original._transform(transform_matrix).invert()

    cpdef double signed_distance(self, double x, double y, double z) noexcept:
        return -self.original.signed_distance(x, y, z)

    cpdef object build_trimesh(self):
        trimesh_model = self.original.get_trimesh()
        if trimesh_model is not None:
            trimesh_model = trimesh.base.Trimesh(vertices=trimesh_model.vertices,
                                                 vertex_normals=-trimesh_model.vertex_normals,
                                                 faces=trimesh_model.faces[:, ::-1],
                                                 visual=trimesh_model.visual,
                                                 process=False)
        return trimesh_model

    cpdef object build_manifold(self):
        return self.original.get_manifold()


cdef class Repair(UnaryOperation):
    @staticmethod
    cdef Repair _get(Model original):
        cdef uint64_t id = HASH_UPDATE(REPAIR, original.id)
        cdef Repair model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = Repair.__new__(Repair)
            model.id = id
            model.original = original
            model.original.add_dependent(model)
            ModelCache[id] = model
        else:
            model = <Repair>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        return f'repair({self.original.name})'

    cpdef bint is_smooth(self):
        return True

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
        cdef uint64_t id = HASH_UPDATE(SNAP_EDGES, original.id)
        id = HASH_UPDATE(id, double_long(f=snap_angle).l)
        id = HASH_UPDATE(id, double_long(f=minimum_area).l)
        cdef SnapEdges model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = SnapEdges.__new__(SnapEdges)
            model.id = id
            model.original = original
            model.original.add_dependent(model)
            model.snap_angle = snap_angle
            model.minimum_area = minimum_area
            ModelCache[id] = model
        else:
            model = <SnapEdges>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        cdef str name = f'snap_edges({self.original.name}'
        if self.snap_angle != DefaultSnapAngle or self.minimum_area:
            name += f', {self.snap_angle:g}'
            if self.minimum_area:
                name += f', {self.minimum_area:g}'
        return name + ')'

    cpdef bint is_smooth(self):
        return False

    cpdef Model flatten(self):
        return self.original.flatten()

    cpdef Model repair(self):
        return self.original.repair()._snap_edges(self.snap_angle, self.minimum_area)

    cdef Model _snap_edges(self, double snap_angle, double minimum_area):
        return self.original._snap_edges(snap_angle, minimum_area)

    cdef Model _transform(self, Matrix44 transform_matrix):
        return self.original._transform(transform_matrix).snap_edges(self.snap_angle, self.minimum_area)

    cdef Model _trim(self, Vector origin, Vector normal, double smooth, double fillet, double chamfer):
        return self.original._trim(origin, normal, smooth, fillet, chamfer)

    cpdef object build_trimesh(self):
        trimesh_model = self.original.get_trimesh()
        if trimesh_model is not None:
            trimesh_model = trimesh.graph.smooth_shade(trimesh_model, angle=self.snap_angle*Tau,
                                                       facet_minarea=1/self.minimum_area if self.minimum_area else None)
        return trimesh_model

    cpdef object build_manifold(self):
        return self.original.get_manifold()


cdef class Transform(UnaryOperation):
    cdef Matrix44 transform_matrix
    cdef Matrix44 inverse_matrix
    cdef double scale

    @staticmethod
    cdef Model _get(Model original, Matrix44 transform_matrix):
        cdef uint64_t id = HASH_UPDATE(TRANSFORM, original.id)
        id = HASH_UPDATE(id, transform_matrix.hash(False))
        cdef Transform model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = Transform.__new__(Transform)
            model.id = id
            model.original = original
            model.original.add_dependent(model)
            model.transform_matrix = transform_matrix
            ModelCache[id] = model
        else:
            model = <Transform>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        return f'{self.original.name}@{self.transform_matrix.hash(False):x}'

    cpdef Model repair(self):
        return self.original.repair()._transform(self.transform_matrix)

    cdef Model _transform(self, Matrix44 transform_matrix):
        return self.original._transform(transform_matrix.mmul(self.transform_matrix))

    cpdef double signed_distance(self, double x, double y, double z) noexcept:
        cdef double s
        if self.inverse_matrix is None:
            self.inverse_matrix = self.transform_matrix.inverse()
            s = sqrt(self.transform_matrix.numbers[0]*self.transform_matrix.numbers[0] +
                     self.transform_matrix.numbers[1]*self.transform_matrix.numbers[1] +
                     self.transform_matrix.numbers[2]*self.transform_matrix.numbers[2])
            s = min(sqrt(self.transform_matrix.numbers[4]*self.transform_matrix.numbers[4] +
                         self.transform_matrix.numbers[5]*self.transform_matrix.numbers[5] +
                         self.transform_matrix.numbers[6]*self.transform_matrix.numbers[6]), s)
            s = min(sqrt(self.transform_matrix.numbers[8]*self.transform_matrix.numbers[8] +
                         self.transform_matrix.numbers[9]*self.transform_matrix.numbers[9] +
                         self.transform_matrix.numbers[10]*self.transform_matrix.numbers[10]), s)
            self.scale = s
        cdef Vector pos=Vector.__new__(Vector)
        pos.allocate_numbers(3)
        pos.numbers[0] = x
        pos.numbers[1] = y
        pos.numbers[2] = z
        cdef Vector ipos=self.inverse_matrix.vmul(pos)
        cdef double distance=self.original.signed_distance(ipos.numbers[0], ipos.numbers[1], ipos.numbers[2])
        return distance * self.scale

    cpdef object build_trimesh(self):
        trimesh_model = self.original.get_trimesh()
        if trimesh_model is not None:
            transform_array = np.array(self.transform_matrix, dtype='f8').reshape((4, 4)).transpose()
            trimesh_model = trimesh_model.copy().apply_transform(transform_array)
        return trimesh_model

    cpdef object build_manifold(self):
        manifold = self.original.get_manifold()
        if manifold is not None:
            transform_matrix = np.array(self.transform_matrix, dtype='f8').reshape((4, 4)).transpose()[:3].tolist()
            manifold = manifold.transform(transform_matrix)
        return manifold


cdef class UVRemap(UnaryOperation):
    cdef str mapping

    @staticmethod
    cdef UVRemap _get(Model original, str mapping):
        cdef uint64_t id = HASH_UPDATE(UV_REMAP, original.id)
        id = HASH_UPDATE(id, HASH_STRING(mapping))
        cdef UVRemap model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = UVRemap.__new__(UVRemap)
            model.id = id
            model.original = original
            model.original.add_dependent(model)
            model.mapping = mapping
            ModelCache[id] = model
        else:
            model = <UVRemap>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        return f'uv_remap({self.original.name}, {self.mapping})'

    cpdef Model repair(self):
        return self.original.repair()._uv_remap(self.mapping)

    cpdef Model _uv_remap(self, str mapping):
        return self.original._uv_remap(mapping)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef object remap_sphere(self, trimesh_model):
        cdef const float[:, :] vertices
        cdef object vertex_uv_array
        cdef float[:, :] vertex_uv
        cdef int64_t i, n
        cdef float x, y, z
        n = len(trimesh_model.vertices)
        vertices = trimesh_model.vertices.astype('f4', copy=False)
        vertex_uv_array = np.zeros((n, 2), dtype='f4')
        vertex_uv = vertex_uv_array
        for i in range(n):
            x, y, z = vertices[i][0], vertices[i][1], vertices[i][2]
            vertex_uv[i][0] = atan2(y, x) / Tau % 1
            vertex_uv[i][1] = atan2(z, sqrt(x*x + y*y)) / Tau * 2 + 0.5
        visual = trimesh.visual.texture.TextureVisuals(uv=vertex_uv_array)
        return trimesh.base.Trimesh(vertices=trimesh_model.vertices, vertex_normals=trimesh_model.vertex_normals,
                                    faces=trimesh_model.faces, visual=visual, process=False)

    cpdef object build_trimesh(self):
        trimesh_model = self.original.get_trimesh()
        if trimesh_model is not None:
            if self.mapping is 'sphere':
                trimesh_model = self.remap_sphere(trimesh_model)
        return trimesh_model

    cpdef object build_manifold(self):
        return self.original.get_manifold()


cdef class Trim(UnaryOperation):
    cdef Vector origin
    cdef Vector normal
    cdef double smooth
    cdef double fillet
    cdef double chamfer

    @staticmethod
    cdef Trim _get(Model original, Vector origin, Vector normal, double smooth, double fillet, double chamfer):
        if origin.numbers == NULL or origin.length != 3 or normal.numbers == NULL or normal.length != 3:
            return None
        cdef uint64_t id = HASH_UPDATE(TRIM, original.id)
        id = HASH_UPDATE(id, origin.hash(False))
        id = HASH_UPDATE(id, normal.hash(False))
        id = HASH_UPDATE(id, double_long(f=smooth).l)
        id = HASH_UPDATE(id, double_long(f=fillet).l)
        id = HASH_UPDATE(id, double_long(f=chamfer).l)
        cdef Trim model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = Trim.__new__(Trim)
            model.id = id
            model.original = original
            model.original.add_dependent(model)
            model.origin = origin
            model.normal = normal.normalize()
            model.smooth = smooth
            model.fillet = fillet
            model.chamfer = chamfer
            ModelCache[id] = model
        else:
            model = <Trim>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        cdef str name = f'trim({self.original.name}, {self.origin.hash(False) ^ self.normal.hash(False):x}'
        if self.smooth:
            name += f', smooth={self.smooth:g}'
        if self.fillet:
            name += f', fillet={self.fillet:g}'
        if self.chamfer:
            name += f', chamfer={self.chamfer:g}'
        return name + ')'

    cpdef bint is_smooth(self):
        return True

    cpdef Model repair(self):
        return self.original.repair()._trim(self.origin, self.normal, self.smooth, self.fillet, self.chamfer)

    cdef Model _transform(self, Matrix44 transform_matrix):
        return self.original._transform(transform_matrix)._trim(transform_matrix.vmul(self.origin),
                                                                transform_matrix.matrix33_cofactor().vmul(self.normal).normalize(),
                                                                self.smooth, self.fillet, self.chamfer)

    @cython.cdivision(True)
    cpdef double signed_distance(self, double x, double y, double z) noexcept:
        cdef double distance=self.original.signed_distance(x, y, z)
        x -= self.origin.numbers[0]
        y -= self.origin.numbers[1]
        z -= self.origin.numbers[2]
        cdef double h, d=x*self.normal.numbers[0] + y*self.normal.numbers[1] + z*self.normal.numbers[2]
        if self.smooth:
            h = min(max(0, 0.5+0.5*(d-distance)/self.smooth), 1)
            distance = h*d + (1-h)*distance + self.smooth*h*(1-h)
        if self.fillet:
            g = max(0, self.fillet+distance)
            h = max(0, self.fillet+d)
            distance = min(-self.fillet, max(distance, d)) + sqrt(g*g + h*h)
        elif self.chamfer:
            distance = max(max(distance, d), (distance + self.chamfer + d)*RootHalf)
        else:
            distance = max(distance, d)
        return distance

    cpdef object build_trimesh(self):
        manifold = self.get_manifold()
        if manifold is not None:
            mesh = manifold.to_mesh()
            return trimesh.base.Trimesh(vertices=mesh.vert_properties, faces=mesh.tri_verts)
        return None

    cpdef object build_manifold(self):
        manifold = self.original.get_manifold()
        if manifold is not None:
            normal = self.normal.neg()
            manifold = manifold.trim_by_plane(normal=tuple(normal), origin_offset=self.origin.dot(normal))
            if manifold.is_empty():
                logger.warning("Result of trim was empty: {}", self.name)
                manifold = None
        return manifold


cdef class BooleanOperation(Model):
    cdef str operation
    cdef list models
    cdef double smooth
    cdef double fillet
    cdef double chamfer

    @staticmethod
    cdef Model _get(str operation, list models, double smooth, double fillet, double chamfer):
        cdef Model child_model, first=None
        cdef set existing = set()
        models = list(models)
        models.reverse()
        cdef list collected_models = []
        cdef uint64_t id = HASH_UPDATE(HASH_START, HASH_STRING(operation))
        while models:
            child_model = models.pop()
            if child_model is None:
                continue
            if type(child_model) is BooleanOperation \
                    and operation is not 'difference' \
                    and (<BooleanOperation>child_model).operation is operation \
                    and (<BooleanOperation>child_model).smooth == smooth \
                    and (<BooleanOperation>child_model).fillet == fillet \
                    and (<BooleanOperation>child_model).chamfer == chamfer:
                models.extend(reversed((<BooleanOperation>child_model).models))
                continue
            if first is None:
                first = child_model
            elif operation is 'difference' and child_model is first:
                return None
            elif child_model in existing:
                continue
            id = HASH_UPDATE(id, child_model.id)
            existing.add(child_model)
            collected_models.append(child_model)
        cdef int64_t n = len(collected_models)
        if n == 0:
            return None
        if n == 1:
            return collected_models[0]
        id = HASH_UPDATE(id, double_long(f=smooth).l)
        id = HASH_UPDATE(id, double_long(f=fillet).l)
        id = HASH_UPDATE(id, double_long(f=chamfer).l)
        cdef BooleanOperation model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = BooleanOperation.__new__(BooleanOperation)
            model.id = id
            model.operation = operation
            model.models = collected_models
            model.smooth = smooth
            model.fillet = fillet
            model.chamfer = chamfer
            for child_model in collected_models:
                child_model.add_dependent(model)
            ModelCache[id] = model
        else:
            model = <BooleanOperation>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        cdef str name = f'{self.operation}('
        cdef Model model
        cdef int64_t i
        for i, model in enumerate(self.models):
            if i:
                name += ', '
            name += model.name
        if self.smooth:
            name += f', smooth={self.smooth:g}'
        if self.fillet:
            name += f', fillet={self.fillet:g}'
        if self.chamfer:
            name += f', chamfer={self.chamfer:g}'
        return name + ')'

    cpdef void unload(self):
        for model in self.models:
            model.remove_dependent(self)
        super(BooleanOperation, self).unload()

    cpdef bint is_smooth(self):
        return True

    cpdef void check_for_changes(self):
        cdef Model model
        for model in self.models:
            model.check_for_changes()

    cpdef Model repair(self):
        return self

    cdef Model _transform(self, Matrix44 transform_matrix):
        cdef Model model
        models = [model._transform(transform_matrix) for model in self.models]
        return BooleanOperation._get(self.operation, models, self.smooth, self.fillet, self.chamfer)

    cdef Model _trim(self, Vector origin, Vector normal, double smooth, double fillet, double chamfer):
        if smooth or fillet or chamfer or self.operation is 'union' or self.smooth or self.fillet or self.chamfer:
            return Trim._get(self, origin, normal, smooth, fillet, chamfer)
        cdef Model model
        cdef list models = []
        cdef int64_t i
        for i, model in enumerate(self.models):
            if i == 0:
                models.append(model._trim(origin, normal, 0, 0, 0))
            else:
                models.append(model)
        return BooleanOperation._get(self.operation, models, self.smooth, self.fillet, self.chamfer)

    @cython.cdivision(True)
    cpdef double signed_distance(self, double x, double y, double z) noexcept:
        cdef double g, h, d, distance=(<Model>self.models[0]).signed_distance(x, y, z)
        cdef int64_t i
        for i in range(1, len(self.models)):
            d = (<Model>self.models[i]).signed_distance(x, y, z)
            if self.operation is 'union':
                if self.smooth:
                    g = min(max(0, 0.5+0.5*(d-distance)/self.smooth), 1)
                    h = 1-g
                    distance = g*distance + h*d - self.smooth*g*h
                elif self.fillet:
                    g = max(0, self.fillet-distance)
                    h = max(0, self.fillet-d)
                    distance = max(self.fillet, min(distance, d)) - sqrt(g*g + h*h)
                elif self.chamfer:
                    distance = min(min(distance, d), (distance - self.chamfer + d)*RootHalf)
                else:
                    distance = min(distance, d)
            elif self.operation is 'intersect':
                if self.smooth:
                    g = min(max(0, 0.5+0.5*(d-distance)/self.smooth), 1)
                    h = 1-g
                    distance = g*d + h*distance + self.smooth*g*h
                if self.fillet:
                    g = max(0, self.fillet+distance)
                    h = max(0, self.fillet+d)
                    distance = min(-self.fillet, max(distance, d)) + sqrt(g*g + h*h)
                elif self.chamfer:
                    distance = max(max(distance, d), (distance + self.chamfer + d)*RootHalf)
                else:
                    distance = max(distance, d)
            elif self.operation is 'difference':
                if self.smooth:
                    g = min(max(0, 0.5+0.5*(-d-distance)/self.smooth), 1)
                    h = 1-g
                    distance = -g*d + h*distance + self.smooth*g*h
                if self.fillet:
                    g = max(0, self.fillet+distance)
                    h = max(0, self.fillet-d)
                    distance = min(-self.fillet, max(distance, -d)) + sqrt(g*g + h*h)
                elif self.chamfer:
                    distance = max(max(distance, -d), (distance + self.chamfer - d)*RootHalf)
                else:
                    distance = max(distance, -d)
        return distance

    cpdef object build_trimesh(self):
        manifold = self.get_manifold()
        if manifold is not None:
            mesh = manifold.to_mesh()
            if len(mesh.vert_properties) and len(mesh.tri_verts):
                return trimesh.base.Trimesh(vertices=mesh.vert_properties, faces=mesh.tri_verts, process=False)
        return None

    cpdef object build_manifold(self):
        cdef list manifolds=[]
        cdef Model model
        cdef int64_t i
        for i, model in enumerate(self.models):
            manifold = model.get_manifold()
            if manifold is None:
                if self.operation is 'difference' and i == 0:
                    return None
                if self.operation is 'intersect':
                    return None
            else:
                manifolds.append(manifold)
        if not manifolds:
            return None
        if len(manifolds) == 1:
            return manifolds[0]
        if self.operation is 'union':
            manifold = manifold3d.Manifold.batch_boolean(manifolds, manifold3d.OpType.Add)
        elif self.operation is 'intersect':
            manifold = manifold3d.Manifold.batch_boolean(manifolds, manifold3d.OpType.Intersect)
        elif self.operation is 'difference':
            manifold = manifold3d.Manifold.batch_boolean(manifolds, manifold3d.OpType.Subtract)
        if manifold.is_empty():
            logger.warning("Result of {} was empty: {}", self.operation, self.name)
            manifold = None
        return manifold


cdef class PrimitiveModel(Model):
    cpdef void check_for_changes(self):
        pass

    cpdef bint is_smooth(self):
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
        cdef uint64_t id = BOX
        if uv_map is not 'standard':
            if uv_map not in Box.VertexUV:
                uv_map = 'standard'
            else:
                id = HASH_UPDATE(id, HASH_STRING(uv_map))
        cdef Box model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = Box.__new__(Box)
            model.id = id
            model.uv_map = uv_map
            ModelCache[id] = model
        else:
            model = <Box>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        return '!box' if self.uv_map is 'standard' else f'!box-{self.uv_map}'

    cpdef double signed_distance(self, double x, double y, double z) noexcept:
        cdef double xp=abs(x)-0.5, yp=abs(y)-0.5, zp=abs(z)-0.5
        cdef double xm=max(xp, 0), ym=max(yp, 0), zm=max(zp, 0)
        cdef double h=xm*xm + ym*ym + zm*zm
        return sqrt(h) + min(0, max(xp, max(yp, zp)))

    cpdef object build_trimesh(self):
        visual = trimesh.visual.texture.TextureVisuals(uv=Box.VertexUV[self.uv_map])
        return trimesh.base.Trimesh(vertices=Box.Vertices, vertex_normals=Box.VertexNormals, faces=Box.Faces, visual=visual, process=False)


cdef class Sphere(PrimitiveModel):
    cdef int64_t segments

    @staticmethod
    cdef Sphere _get(int64_t segments):
        if segments < 4:
            segments = 4
        elif segments % 4:
            segments += 4 - segments % 4
        cdef uint64_t id = HASH_UPDATE(SPHERE, segments)
        cdef Sphere model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = Sphere.__new__(Sphere)
            model.id = id
            model.segments = segments
            ModelCache[id] = model
        else:
            model = <Sphere>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        return '!sphere' if self.segments == DefaultSegments else f'!sphere-{self.segments}'

    cpdef double signed_distance(self, double x, double y, double z) noexcept:
        return sqrt(x*x + y*y + z*z) - 1

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
        return trimesh.base.Trimesh(vertices=vertices_array, vertex_normals=vertices_array, faces=faces_array, visual=visual, process=False)


cdef class Cylinder(PrimitiveModel):
    cdef int64_t segments

    @staticmethod
    cdef Cylinder _get(int64_t segments):
        if segments < 2:
            segments = 2
        cdef uint64_t id = HASH_UPDATE(CYLINDER, segments)
        cdef Cylinder model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = Cylinder.__new__(Cylinder)
            model.id = id
            model.segments = segments
            ModelCache[id] = model
        else:
            model = <Cylinder>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        return '!cylinder' if self.segments == DefaultSegments else f'!cylinder-{self.segments}'

    cpdef double signed_distance(self, double x, double y, double z) noexcept:
        return max(sqrt(x*x+y*y)-1, abs(z)-0.5)

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
        return trimesh.base.Trimesh(vertices=vertices_array, vertex_normals=vertex_normals_array, faces=faces_array, visual=visual, process=False)


cdef class Cone(PrimitiveModel):
    cdef int64_t segments

    @staticmethod
    cdef Cone _get(int64_t segments):
        if segments < 2:
            segments = 2
        cdef uint64_t id = HASH_UPDATE(CONE, segments)
        cdef Cone model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = Cone.__new__(Cone)
            model.id = id
            model.segments = segments
            ModelCache[id] = model
        else:
            model = <Cone>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        return '!cone' if self.segments == DefaultSegments else f'!cone-{self.segments}'

    cpdef double signed_distance(self, double x, double y, double z) noexcept:
        return max(sqrt(x*x+y*y)+z, abs(z))-0.5

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
        return trimesh.base.Trimesh(vertices=vertices_array, vertex_normals=vertex_normals_array, faces=faces_array, visual=visual, process=False)


cdef class ExternalModel(Model):
    cdef object cache_path

    @staticmethod
    cdef ExternalModel _get(str filename):
        if filename is None:
            return None
        cdef uint64_t id = HASH_UPDATE(EXTERNAL, HASH_STRING(filename))
        cdef ExternalModel model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = ExternalModel.__new__(ExternalModel)
            model.id = id
            model.cache_path = SharedCache[filename]
            ModelCache[id] = model
        else:
            model = <ExternalModel>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        return str(self.cache_path)

    cpdef bint is_smooth(self):
        return False

    cpdef void check_for_changes(self):
        if self.cache and 'trimesh' in self.cache and self.cache['trimesh'] is not self.cache_path.read_trimesh_model():
            self.invalidate()

    cpdef double signed_distance(self, double x, double y, double z) noexcept:
        if self.cache is None:
            self.cache = {}
        try:
            if 'proximity_query' not in self.cache:
                mesh = self.get_trimesh()
                proximity_query = trimesh.proximity.ProximityQuery(mesh) if mesh else None
                self.cache['proximity_query'] = proximity_query
            else:
                proximity_query = self.cache['proximity_query']
            if proximity_query is not None:
                return -(proximity_query.signed_distance([(x, y, z)])[0])
        except Exception:
            logger.exception("Unable to do SDF proximity query of mesh: {}", self.name)
            self.cache['proximity_query'] = None
        return NaN

    cpdef object build_trimesh(self):
        return self.cache_path.read_trimesh_model()


cdef class VectorModel(Model):
    cdef Vector vertices
    cdef Vector faces

    @staticmethod
    cdef VectorModel _get(Vector vertices, Vector faces):
        if vertices is None or vertices.numbers == NULL or faces is None or faces.numbers == NULL:
            return None
        cdef uint64_t id = VECTOR
        id = HASH_UPDATE(id, vertices.hash(False))
        id = HASH_UPDATE(id, faces.hash(False))
        cdef VectorModel model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = VectorModel.__new__(VectorModel)
            model.id = id
            model.vertices = vertices
            model.faces = faces
            ModelCache[id] = model
        else:
            model = <VectorModel>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        return f'vector({self.vertices.hash(False):x}, {self.faces.hash(False):x})'

    cpdef bint is_smooth(self):
        return False

    cpdef void check_for_changes(self):
        pass

    cpdef double signed_distance(self, double x, double y, double z) noexcept:
        if self.cache is None:
            self.cache = {}
        try:
            if 'proximity_query' not in self.cache:
                mesh = self.get_trimesh()
                proximity_query = trimesh.proximity.ProximityQuery(mesh) if mesh else None
                self.cache['proximity_query'] = proximity_query
            else:
                proximity_query = self.cache['proximity_query']
            if proximity_query is not None:
                return -(proximity_query.signed_distance([(x, y, z)])[0])
        except Exception:
            logger.exception("Unable to do SDF proximity query of mesh: {}", self.name)
            self.cache['proximity_query'] = None
        return NaN

    cpdef object build_trimesh(self):
        if self.vertices.length % 3 != 0:
            return None
        vertices_array = np.array(self.vertices, dtype='f4').reshape((-1, 3))
        if self.faces.length % 3 != 0:
            return None
        faces_array = np.array(self.faces, dtype='i4').reshape((-1, 3))
        return trimesh.base.Trimesh(vertices=vertices_array, faces=faces_array, process=False)


cdef class SignedDistanceFunction(UnaryOperation):
    cdef Function function
    cdef Vector minimum
    cdef Vector maximum
    cdef double resolution
    cdef Context context

    @staticmethod
    cdef SignedDistanceFunction _get(Function function, Model original, Vector minimum, Vector maximum, double resolution):
        if function is None and original is None:
            return None
        cdef uint64_t id = SDF
        if function is not None:
            id = HASH_UPDATE(id, function.hash())
        else:
            id = HASH_UPDATE(id, original.id)
        id = HASH_UPDATE(id, minimum.hash(False))
        id = HASH_UPDATE(id, maximum.hash(False))
        id = HASH_UPDATE(id, double_long(f=resolution).l)
        cdef SignedDistanceFunction model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = SignedDistanceFunction.__new__(SignedDistanceFunction)
            model.id = id
            model.function = function
            model.original = original
            model.minimum = minimum
            model.maximum = maximum
            model.resolution = resolution
            ModelCache[id] = model
            if original is not None:
                original.add_dependent(model)
        else:
            model = <SignedDistanceFunction>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        if self.function is not None:
            return f'sdf({self.function}, {self.minimum!r}, {self.maximum!r}, {self.resolution})'
        return f'sdf({self.original.name}, {self.minimum!r}, {self.maximum!r}, {self.resolution:g})'

    cpdef bint is_smooth(self):
        return False

    cpdef double signed_distance(self, double x, double y, double z) noexcept:
        if self.context is None:
            self.context = Context()
            self.context.state = StateDict()
            self.context.stack = VectorStack()
            self.context.lnames = VectorStack()
        cdef Vector pos = Vector.__new__(Vector)
        pos.allocate_numbers(3)
        pos.numbers[0] = x
        pos.numbers[1] = y
        pos.numbers[2] = z
        cdef Vector result = self.function.call_one_fast(self.context, pos)
        if result.length == 1 and result.numbers != NULL:
            return result.numbers[0]
        return NaN

    cpdef object build_trimesh(self):
        manifold = self.get_manifold()
        if manifold is not None:
            mesh = manifold.to_mesh()
            if len(mesh.vert_properties) and len(mesh.tri_verts):
                return trimesh.base.Trimesh(vertices=mesh.vert_properties, faces=mesh.tri_verts, process=False)
        return None

    cpdef object build_manifold(self):
        box = (*self.minimum, *self.maximum)
        if self.original is not None:
            manifold = manifold3d.Manifold.level_set(self.original.inverse_signed_distance, box, self.resolution)
        else:
            manifold = manifold3d.Manifold.level_set(self.inverse_signed_distance, box, self.resolution)
        if manifold.is_empty():
            logger.warning("Result of SDF was empty: {}", self.name)
            manifold = None
        return manifold


cdef class Mix(Model):
    cdef list models
    cdef Vector weights

    @staticmethod
    cdef Model _get(list models, Vector weights):
        cdef Model child_model
        cdef list collected_models = []
        cdef uint64_t id = MIX
        for child_model in models:
            if child_model is not None:
                collected_models.append(child_model)
                id = HASH_UPDATE(id, child_model.id)
        id = HASH_UPDATE(id, weights.hash(False))
        if len(collected_models) == 0:
            return None
        if len(collected_models) == 1:
            return collected_models[0]
        cdef Mix model
        cdef PyObject* objptr = PyDict_GetItem(ModelCache, id)
        if objptr == NULL:
            model = Mix.__new__(Mix)
            model.id = id
            model.models = collected_models
            model.weights = weights
            for child_model in collected_models:
                child_model.add_dependent(model)
            ModelCache[id] = model
        else:
            model = <Mix>objptr
            model.touch_timestamp = 0
        return model

    @property
    def name(self):
        cdef str name = 'mix('
        cdef Model model
        cdef int64_t i
        for i, model in enumerate(self.models):
            if i:
                name += ', '
            name += model.name
        name += f', {self.weights!r})'
        return name

    cpdef void unload(self):
        for model in self.models:
            model.remove_dependent(self)
        super(Mix, self).unload()

    cpdef bint is_smooth(self):
        return False

    cpdef void check_for_changes(self):
        cdef Model model
        for model in self.models:
            model.check_for_changes()

    @cython.cdivision(True)
    cpdef double signed_distance(self, double x, double y, double z) noexcept:
        cdef double distance=0, weight=0, d, w
        cdef int64_t i
        cdef Model model
        for i, model in enumerate(self.models):
            d = model.signed_distance(x, y, z)
            w = self.weights.numbers[i % self.weights.length]
            distance += d * w
            weight += w
        return distance / weight

    cpdef object build_trimesh(self):
        logger.warning("Cannot use !mix node outside of !sdf")
        return None

    cpdef object build_manifold(self):
        logger.warning("Cannot use !mix node outside of !sdf")
        return None
