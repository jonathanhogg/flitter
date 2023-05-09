# cython: language_level=3, profile=True

"""
Flitter OpenGL 3D drawing canvas
"""

import cython
from cython cimport view
from loguru import logger
import moderngl
import numpy as np
import trimesh

from libc.math cimport cos

from ... import name_patch
from ...cache import SharedCache
from ...clock import system_clock
from ...model cimport Node, Vector, Matrix44, null_
from .glsl import TemplateLoader


logger = name_patch(logger, __name__)

cdef Vector Zero3 = Vector((0, 0, 0))
cdef Vector One3 = Vector((1, 1, 1))
cdef dict ModelCache = {}
cdef int DEFAULT_MAX_LIGHTS = 50
cdef double Pi = 3.141592653589793


cdef enum LightType:
    Ambient = 1
    Directional = 2
    Point = 3
    Spot = 4


@cython.dataclasses.dataclass
cdef class Light:
    type: LightType
    inner_cone: float
    outer_cone: float
    color: Vector
    position: Vector
    direction: Vector


@cython.dataclasses.dataclass
cdef class Textures:
    diffuse_id: str = None
    specular_id: str = None
    emissive_id: str = None

    def __hash__(self):
        return hash(self.diffuse_id) ^ hash(self.specular_id) ^ hash(self.emissive_id)


@cython.dataclasses.dataclass
cdef class Material:
    diffuse: Vector = Zero3
    specular: Vector = One3
    emissive: Vector = Zero3
    shininess: cython.double = 0
    transparency: cython.double = 0
    textures: Textures = None


@cython.dataclasses.dataclass
cdef class Instance:
    model_matrix: Matrix44
    material: Material


cdef class Model:
    cdef str name
    cdef bint flat
    cdef bint invert
    cdef object trimesh_model

    def __hash__(self):
        return <Py_hash_t>(<void*>self)

    def __eq__(self, other):
        return self is other

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


@cython.dataclasses.dataclass
cdef class RenderSet:
    lights: list[list[Light]]
    instances: dict[Model, list[Instance]]


cdef class Box(Model):
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


cdef class Sphere(Model):
    cdef int subdivisions

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


cdef class Cylinder(Model):
    cdef int segments

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


cdef class LoadedModel(Model):
    cdef str filename

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


cdef object StandardVertexSource = TemplateLoader.get_template("standard_lighting.vert")
cdef object StandardFragmentSource = TemplateLoader.get_template("standard_lighting.frag")


def draw(Node node, tuple size, glctx, dict objects, dict references):
    cdef int width, height
    width, height = size
    cdef Vector viewpoint = node.get_fvec('viewpoint', 3, Vector((0, 0, width/2)))
    cdef Vector focus = node.get_fvec('focus', 3, Zero3)
    cdef Vector up = node.get_fvec('up', 3, Vector((0, 1, 0)))
    cdef double fov = node.get('fov', 1, float, 0.25)
    cdef double near = node.get('near', 1, float, 1)
    cdef double far = node.get('far', 1, float, width)
    cdef double fog_min = node.get('fog_min', 1, float, 0)
    cdef double fog_max = node.get('fog_max', 1, float, 0)
    cdef Vector fog_color = node.get_fvec('fog_color', 3, Zero3)
    cdef int max_lights = node.get_int('max_lights', DEFAULT_MAX_LIGHTS)
    cdef Matrix44 pv_matrix = Matrix44._project(width/height, fov, near, far).mmul(Matrix44._look(viewpoint, focus, up))
    cdef Matrix44 model_matrix = update_model_matrix(Matrix44.__new__(Matrix44), node)
    cdef Node child = node.first_child
    cdef RenderSet render_set = RenderSet(lights=[[]], instances={})
    cdef list render_sets = [render_set]
    while child is not None:
        collect(child, model_matrix, Material(), render_set, render_sets)
        child = child.next_sibling
    for render_set in render_sets:
        if render_set.instances:
            render(render_set, pv_matrix, viewpoint, max_lights, fog_min, fog_max, fog_color, glctx, objects, references)


cdef Matrix44 update_model_matrix(Matrix44 model_matrix, Node node):
    cdef Matrix44 matrix
    cdef str attribute
    cdef Vector vector
    for attribute, vector in node._attributes.items():
        if attribute == 'translate':
            if (matrix := Matrix44._translate(vector)) is not None:
                model_matrix = model_matrix.mmul(matrix)
        elif attribute == 'scale':
            if (matrix := Matrix44._scale(vector)) is not None:
                model_matrix = model_matrix.mmul(matrix)
        elif attribute == 'rotate':
            if (matrix := Matrix44._rotate(vector)) is not None:
                model_matrix = model_matrix.mmul(matrix)
        elif attribute == 'rotate_x':
            if vector.numbers !=  NULL and vector.length == 1 and (matrix := Matrix44._rotate_x(vector.numbers[0])) is not None:
                model_matrix = model_matrix.mmul(matrix)
        elif attribute == 'rotate_y':
            if vector.numbers !=  NULL and vector.length == 1 and (matrix := Matrix44._rotate_y(vector.numbers[0])) is not None:
                model_matrix = model_matrix.mmul(matrix)
        elif attribute == 'rotate_z':
            if vector.numbers !=  NULL and vector.length == 1 and (matrix := Matrix44._rotate_z(vector.numbers[0])) is not None:
                model_matrix = model_matrix.mmul(matrix)
        elif attribute == 'matrix':
            if (matrix := Matrix44(vector)) is not None:
                model_matrix = model_matrix.mmul(matrix)
    return model_matrix


cdef void collect(Node node, Matrix44 model_matrix, Material material, RenderSet render_set, list render_sets):
    cdef str kind = node.kind
    cdef Light light
    cdef list lights, instances
    cdef Vector color, position, direction, emissive, diffuse, specular
    cdef double shininess, inner, outer
    cdef Node child
    cdef str filename
    cdef int subdivisions, segments
    cdef bint flat
    cdef Model model
    cdef Material new_material

    if node.kind == 'transform':
        model_matrix = update_model_matrix(model_matrix, node)
        child = node.first_child
        while child is not None:
            collect(child, model_matrix, material, render_set, render_sets)
            child = child.next_sibling

    elif node.kind == 'group':
        model_matrix = update_model_matrix(model_matrix, node)
        lights = list(render_set.lights)
        lights.append([])
        render_set = RenderSet(lights, {})
        render_sets.append(render_set)
        child = node.first_child
        while child is not None:
            collect(child, model_matrix, material, render_set, render_sets)
            child = child.next_sibling

    elif node.kind == 'light':
        color = node.get_fvec('color', 3)
        if color.as_bool():
            position = node.get_fvec('position', 3)
            direction = node.get_fvec('direction', 3)
            light = Light.__new__(Light)
            light.color = color
            if position.length and direction.as_bool():
                light.type = LightType.Spot
                inner = max(0, node.get_float('inner', 0))
                outer = max(inner, node.get_float('outer', 0.5))
                light.inner_cone = cos(inner * Pi)
                light.outer_cone = cos(outer * Pi)
                light.position = model_matrix.vmul(position)
                light.direction = model_matrix.inverse().transpose().matrix33().vmul(direction).normalize()
            elif position.length:
                light.type = LightType.Point
                light.position = model_matrix.vmul(position)
                light.direction = None
            elif direction.as_bool():
                light.type = LightType.Directional
                light.position = None
                light.direction = model_matrix.inverse().transpose().matrix33().vmul(direction).normalize()
            else:
                light.type = LightType.Ambient
                light.position = None
                light.direction = None
            lights = render_set.lights[-1]
            lights.append(light)

    elif node.kind == 'material':
        new_material = Material.__new__(Material)
        new_material.diffuse = node.get_fvec('color', 3, material.diffuse)
        new_material.specular = node.get_fvec('specular', 3, material.specular)
        new_material.emissive = node.get_fvec('emissive', 3, material.emissive)
        new_material.shininess = node.get_float('shininess', material.shininess)
        new_material.transparency = node.get_float('transparency', material.transparency)
        diffuse_id = node.get('texture_id', 1, str)
        specular_id = node.get('specular_texture_id', 1, str)
        emissive_id = node.get('emissive_texture_id', 1, str)
        if diffuse_id is not None or specular_id is not None or emissive_id is not None:
            new_material.textures = Textures(diffuse_id, specular_id, emissive_id)
        else:
            new_material.textures = None
        child = node.first_child
        while child is not None:
            collect(child, model_matrix, new_material, render_set, render_sets)
            child = child.next_sibling

    elif node.kind == 'box':
        flat = node.get_bool('flat', False)
        invert = node.get_bool('invert', False)
        model = Box.get(flat, invert)
        add_instance(render_set.instances, model, node, model_matrix, material)

    elif node.kind == 'sphere':
        flat = node.get_bool('flat', False)
        invert = node.get_bool('invert', False)
        subdivisions = node.get_int('subdivisions', 2)
        model = Sphere.get(flat, invert, subdivisions)
        add_instance(render_set.instances, model, node, model_matrix, material)

    elif node.kind == 'cylinder':
        flat = node.get_bool('flat', False)
        invert = node.get_bool('invert', False)
        segments = node.get_int('segments', 32)
        model = Cylinder.get(flat, invert, segments)
        add_instance(render_set.instances, model, node, model_matrix, material)

    elif node.kind == 'model':
        filename = node.get('filename', 1, str)
        if filename:
            flat = node.get_bool('flat', False)
            invert = node.get_bool('invert', False)
            model = LoadedModel.get(flat, invert, filename)
            add_instance(render_set.instances, model, node, model_matrix, material)


cdef void add_instance(dict render_instances, Model model, Node node, Matrix44 model_matrix, Material material):
    cdef dict attrs = node._attributes
    cdef Matrix44 matrix
    if (matrix := Matrix44._translate(attrs.get('position'))) is not None:
        model_matrix = model_matrix.mmul(matrix)
    if (matrix := Matrix44._rotate(attrs.get('rotation'))) is not None:
        model_matrix = model_matrix.mmul(matrix)
    if (matrix := Matrix44._scale(attrs.get('size'))) is not None:
        model_matrix = model_matrix.mmul(matrix)
    cdef Instance instance = Instance.__new__(Instance)
    instance.model_matrix = model_matrix
    instance.material = material
    cdef list instances
    cdef tuple model_textures = (model, material.textures)
    if (instances := render_instances.get(model_textures)) is not None:
        instances.append(instance)
    else:
        render_instances[model_textures] = [instance]


cdef void render(RenderSet render_set, Matrix44 pv_matrix, Vector viewpoint, int max_lights,
                 double fog_min, double fog_max, Vector fog_color, glctx, dict objects, dict references):
    cdef list instances, lights, buffers
    cdef cython.float[:, :] matrices, materials, lights_data
    cdef Material material
    cdef Light light
    cdef Model model
    cdef int i, j, k, n
    cdef double z
    cdef double* src
    cdef float* dest
    cdef Instance instance
    cdef tuple transparent_object
    cdef list transparent_objects = []
    cdef double[:] zs
    cdef long[:] indices
    cdef str shader_name = f'!standard_shader/{max_lights}'
    if (standard_shader := objects.get(shader_name)) is None:
        logger.debug("Compiling standard lighting shader for {} max lights", max_lights)
        variables = {'max_lights': max_lights, 'Ambient': LightType.Ambient, 'Directional': LightType.Directional,
                     'Point': LightType.Point, 'Spot': LightType.Spot}
        standard_shader = glctx.program(vertex_shader=StandardVertexSource.render(**variables),
                                        fragment_shader=StandardFragmentSource.render(**variables))
        objects[shader_name] = standard_shader
    standard_shader['pv_matrix'] = pv_matrix
    standard_shader['view_position'] = viewpoint
    standard_shader['fog_min'] = fog_min
    standard_shader['fog_max'] = fog_max
    standard_shader['fog_color'] = fog_color
    lights_data = view.array((max_lights, 12), 4, 'f')
    i = 0
    for lights in render_set.lights:
        for light in lights:
            if i == max_lights:
                break
            dest = &lights_data[i, 0]
            dest[0] = <cython.float>(<int>light.type)
            dest[1] = light.inner_cone
            dest[2] = light.outer_cone
            for j in range(3):
                dest[j+3] = light.color.numbers[j]
            if light.position is not None:
                for j in range(3):
                    dest[j+6] = light.position.numbers[j]
            if light.direction is not None:
                for j in range(3):
                    dest[j+9] = light.direction.numbers[j]
            i += 1
    standard_shader['nlights'] = i
    standard_shader['lights'].write(lights_data)
    for (model, textures), instances in render_set.instances.items():
        n = len(instances)
        matrices = view.array((n, 16), 4, 'f')
        materials = view.array((n, 11), 4, 'f')
        k = 0
        zs_array = np.empty(n)
        zs = zs_array
        for i in range(n):
            instance = instances[i]
            zs[i] = pv_matrix.mmul(instance.model_matrix).numbers[14]
        indices = zs_array.argsort()
        for i in indices:
            instance = instances[i]
            material = instance.material
            if material.transparency > 0:
                transparent_objects.append((-zs[i], model, instance))
            else:
                src = instance.model_matrix.numbers
                dest = &matrices[k, 0]
                for j in range(16):
                    dest[j] = src[j]
                dest = &materials[k, 0]
                for j in range(3):
                    dest[j] = material.diffuse.numbers[j]
                    dest[j+3] = material.specular.numbers[j]
                    dest[j+6] = material.emissive.numbers[j]
                dest[9] = material.shininess
                dest[10] = 0
                k += 1
        dispatch_instances(glctx, objects, standard_shader, model, k, matrices, materials, textures, references)
    if transparent_objects:
        transparent_objects.sort()
        matrices = view.array((1, 16), 4, 'f')
        materials = view.array((1, 11), 4, 'f')
        for transparent_object in transparent_objects:
            model = transparent_object[1]
            instance = transparent_object[2]
            material = instance.material
            src = instance.model_matrix.numbers
            dest = &matrices[0, 0]
            for j in range(16):
                dest[j] = src[j]
            dest = &materials[0, 0]
            for j in range(3):
                dest[j] = material.diffuse.numbers[j]
                dest[j+3] = material.specular.numbers[j]
                dest[j+6] = material.emissive.numbers[j]
            dest[9] = material.shininess
            dest[10] = material.transparency
            dispatch_instances(glctx, objects, standard_shader, model, 1, matrices, materials, material.textures, references)


cdef void dispatch_instances(glctx, dict objects, shader, Model model, int count, cython.float[:, :] matrices,
                             cython.float[:, :] materials, Textures textures, dict references):
    vertex_buffer, index_buffer = model.get_buffers(glctx, objects)
    if vertex_buffer is None:
        return
    shader['use_diffuse_texture'] = False
    shader['use_specular_texture'] = False
    shader['use_emissive_texture'] = False
    cdef dict unit_ids = {}
    if references is not None and textures is not None:
        if (scene_node := references.get(textures.diffuse_id)) is not None and scene_node.texture is not None:
            unit_id = unit_ids.setdefault(textures.diffuse_id, len(unit_ids))
            scene_node.texture.use(location=unit_id)
            shader['use_diffuse_texture'] = True
            shader['diffuse_texture'] = unit_id
        if (scene_node := references.get(textures.specular_id)) is not None and scene_node.texture is not None:
            unit_id = unit_ids.setdefault(textures.specular_id, len(unit_ids))
            scene_node.texture.use(location=unit_id)
            shader['use_specular_texture'] = True
            shader['specular_texture'] = unit_id
        if (scene_node := references.get(textures.emissive_id)) is not None and scene_node.texture is not None:
            unit_id = unit_ids.setdefault(textures.emissive_id, len(unit_ids))
            scene_node.texture.use(location=unit_id)
            shader['use_emissive_texture'] = True
            shader['emissive_texture'] = unit_id
    matrices_buffer = glctx.buffer(matrices)
    materials_buffer = glctx.buffer(materials)
    buffers = [(vertex_buffer, '3f 3f 2f', 'model_position', 'model_normal', 'model_uv'),
               (matrices_buffer, '16f/i', 'model_matrix'),
               (materials_buffer, '9f 1f 1f/i', 'material_colors', 'material_shininess', 'material_transparency')]
    render_array = glctx.vertex_array(shader, buffers, index_buffer=index_buffer, mode=moderngl.TRIANGLES)
    render_array.render(instances=count)
