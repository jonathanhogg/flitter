# cython: language_level=3, profile=True

"""
Flitter OpenGL 3D drawing canvas
"""

import cython
from cython cimport view
from loguru import logger
import moderngl
import numpy as np

from libc.math cimport cos

from ... import name_patch
from ...clock import system_clock
from ...model cimport Node, Vector, Matrix44, null_
from .glsl import TemplateLoader
from .models cimport Model, Box, Cylinder, Sphere, LoadedModel


logger = name_patch(logger, __name__)

cdef Vector Zero3 = Vector((0, 0, 0))
cdef Vector One3 = Vector((1, 1, 1))
cdef int DEFAULT_MAX_LIGHTS = 50
cdef double Pi = 3.141592653589793
cdef tuple MaterialAttributes = ('color', 'specular', 'emissive', 'shininess', 'transparency',
                                 'texture_id', 'specular_texture_id', 'emissive_texture_id')


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


@cython.dataclasses.dataclass
cdef class RenderSet:
    lights: list[list[Light]]
    instances: dict[Model, list[Instance]]
    depth_test: bool
    cull_face: bool
    composite: str


cdef object StandardVertexSource = TemplateLoader.get_template("standard_lighting.vert")
cdef object StandardFragmentSource = TemplateLoader.get_template("standard_lighting.frag")


cdef void set_blend(glctx, str composite):
    glctx.blend_equation = moderngl.FUNC_ADD
    if composite == 'source':
        glctx.blend_func = moderngl.ONE, moderngl.ZERO
    elif composite == 'dest':
        glctx.blend_func = moderngl.ZERO, moderngl.ONE
    elif composite == 'dest_over':
        glctx.blend_func = moderngl.ONE_MINUS_DST_ALPHA, moderngl.ONE
    elif composite == 'add':
        glctx.blend_func = moderngl.ONE, moderngl.ONE
    elif composite == 'subtract':
        glctx.blend_equation = moderngl.FUNC_SUBTRACT
        glctx.blend_func = moderngl.ONE, moderngl.ONE
    elif composite == 'lighten':
        glctx.blend_equation = moderngl.MAX
        glctx.blend_func = moderngl.ONE, moderngl.ONE
    elif composite == 'darken':
        glctx.blend_equation = moderngl.MIN
        glctx.blend_func = moderngl.ONE, moderngl.ONE
    else: # over
        glctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA


cdef Material update_material(Node node, Material material):
    for attr in MaterialAttributes:
        if attr in node._attributes:
            break
    else:
        return material
    cdef Material new_material = Material.__new__(Material)
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
    return new_material


def draw(Node node, tuple size, glctx, dict objects, dict references):
    cdef int width, height
    width, height = size
    cdef Vector viewpoint = node.get_fvec('viewpoint', 3, Vector((0, 0, width/2)))
    cdef Vector focus = node.get_fvec('focus', 3, Zero3)
    cdef Vector up = node.get_fvec('up', 3, Vector((0, 1, 0)))
    cdef double fov = node.get_float('fov', 0.25)
    cdef bint orthographic = node.get_bool('orthographic', False)
    cdef double ortho_width = node.get('width', 1, float, width)
    cdef double near = node.get_float('near', 1)
    cdef double far = node.get_float('far', width)
    cdef double fog_min = node.get_float('fog_min', 0)
    cdef double fog_max = node.get_float('fog_max', 0)
    cdef Vector fog_color = node.get_fvec('fog_color', 3, Zero3)
    cdef int max_lights = node.get_int('max_lights', DEFAULT_MAX_LIGHTS)
    cdef Matrix44 pv_matrix
    if orthographic:
        pv_matrix = Matrix44._ortho(width/height, ortho_width, near, far)
    else:
        pv_matrix = Matrix44._project(width/height, fov, near, far)
    pv_matrix = pv_matrix.mmul(Matrix44._look(viewpoint, focus, up))
    cdef Matrix44 model_matrix = update_model_matrix(Matrix44.__new__(Matrix44), node)
    cdef Node child = node.first_child
    cdef bint depth_test = node.get_bool('depth_test', True)
    cdef bint cull_face = node.get_bool('cull_face', True)
    cdef str composite = node.get_str('composite', 'over')
    cdef RenderSet render_set = RenderSet(lights=[[]], instances={}, depth_test=depth_test, cull_face=cull_face,
                                          composite=composite.lower())
    cdef list render_sets = [render_set]
    while child is not None:
        collect(child, model_matrix, Material(), render_set, render_sets)
        child = child.next_sibling
    for render_set in render_sets:
        if render_set.instances:
            render(render_set, pv_matrix, orthographic, viewpoint, focus, max_lights, fog_min, fog_max, fog_color,
                   glctx, objects, references)


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
    cdef str filename, composite
    cdef int subdivisions, segments
    cdef bint flat, depth_test, cull_face
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
        depth_test = node.get_bool('depth_test', True)
        cull_face = node.get_bool('cull_face', True)
        composite = node.get_str('composite', 'over')
        render_set = RenderSet(lights=lights, instances={}, depth_test=depth_test, cull_face=cull_face, composite=composite.lower())
        render_sets.append(render_set)
        child = node.first_child
        while child is not None:
            collect(child, model_matrix, material, render_set, render_sets)
            child = child.next_sibling

    elif node.kind == 'light':
        color = node.get_fvec('color', 3, null_)
        if color.as_bool():
            position = node.get_fvec('position', 3, null_)
            direction = node.get_fvec('direction', 3, null_)
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
        new_material = update_material(node, material)
        child = node.first_child
        while child is not None:
            collect(child, model_matrix, new_material, render_set, render_sets)
            child = child.next_sibling

    elif node.kind == 'box':
        flat = node.get_bool('flat', False)
        invert = node.get_bool('invert', False)
        model = Box.get(flat, invert)
        material = update_material(node, material)
        add_instance(render_set.instances, model, node, model_matrix, material)

    elif node.kind == 'sphere':
        flat = node.get_bool('flat', False)
        invert = node.get_bool('invert', False)
        subdivisions = node.get_int('subdivisions', 2)
        model = Sphere.get(flat, invert, subdivisions)
        material = update_material(node, material)
        add_instance(render_set.instances, model, node, model_matrix, material)

    elif node.kind == 'cylinder':
        flat = node.get_bool('flat', False)
        invert = node.get_bool('invert', False)
        segments = node.get_int('segments', 32)
        model = Cylinder.get(flat, invert, segments)
        material = update_material(node, material)
        add_instance(render_set.instances, model, node, model_matrix, material)

    elif node.kind == 'model':
        filename = node.get('filename', 1, str)
        if filename:
            flat = node.get_bool('flat', False)
            invert = node.get_bool('invert', False)
            model = LoadedModel.get(flat, invert, filename)
            material = update_material(node, material)
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


def fst(tuple ab):
    return ab[0]


cdef void render(RenderSet render_set, Matrix44 pv_matrix, bint orthographic, Vector viewpoint, Vector focus, int max_lights,
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
    cdef str shader_name = f'!shader/{max_lights}'
    if (shader := objects.get(shader_name)) is None:
        logger.debug("Compiling standard lighting shader for {} max lights", max_lights)
        variables = {'max_lights': max_lights, 'Ambient': LightType.Ambient, 'Directional': LightType.Directional,
                     'Point': LightType.Point, 'Spot': LightType.Spot}
        shader = glctx.program(vertex_shader=StandardVertexSource.render(**variables),
                               fragment_shader=StandardFragmentSource.render(**variables))
        objects[shader_name] = shader
    shader['pv_matrix'] = pv_matrix
    shader['orthographic'] = orthographic
    shader['view_position'] = viewpoint
    shader['focus'] = focus
    shader['fog_min'] = fog_min
    shader['fog_max'] = fog_max
    shader['fog_color'] = fog_color
    shader['use_diffuse_texture'] = False
    shader['use_specular_texture'] = False
    shader['use_emissive_texture'] = False
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
    shader['nlights'] = i
    shader['lights'].write(lights_data)
    cdef int flags = moderngl.BLEND
    if render_set.depth_test:
        flags |= moderngl.DEPTH_TEST
    if render_set.cull_face:
        flags |= moderngl.CULL_FACE
    glctx.enable(flags)
    set_blend(glctx, render_set.composite)
    for (model, textures), instances in render_set.instances.items():
        n = len(instances)
        matrices = view.array((n, 16), 4, 'f')
        materials = view.array((n, 11), 4, 'f')
        k = 0
        if render_set.depth_test:
            zs_array = np.empty(n)
            zs = zs_array
            for i, instance in enumerate(instances):
                zs[i] = pv_matrix.mmul(instance.model_matrix).numbers[14]
            indices = zs_array.argsort()
        else:
            indices = np.arange(n, dtype='long')
        for i in indices:
            instance = instances[i]
            material = instance.material
            if material.transparency < 1:
                if material.transparency > 0 and render_set.depth_test:
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
                    dest[10] = material.transparency
                    k += 1
        dispatch_instances(glctx, objects, shader, model, k, matrices, materials, textures, references)
    if transparent_objects:
        transparent_objects.sort(key=fst)
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
            dispatch_instances(glctx, objects, shader, model, 1, matrices, materials, material.textures, references)
    glctx.disable(flags)


cdef void dispatch_instances(glctx, dict objects, shader, Model model, int count, cython.float[:, :] matrices,
                             cython.float[:, :] materials, Textures textures, dict references):
    vertex_buffer, index_buffer = model.get_buffers(glctx, objects)
    if vertex_buffer is None:
        return
    cdef dict unit_ids
    cdef list samplers = None
    cdef int unit_id
    if references is not None and textures is not None:
        unit_ids = {}
        samplers = []
        if (scene_node := references.get(textures.diffuse_id)) is not None and scene_node.texture is not None:
            if textures.diffuse_id in unit_ids:
                unit_id = unit_ids[textures.diffuse_id]
            else:
                unit_id = len(unit_ids) + 1
                unit_ids[textures.diffuse_id] = unit_id
                sampler = glctx.sampler(texture=scene_node.texture, filter=(moderngl.LINEAR, moderngl.LINEAR))
                sampler.use(unit_id)
                samplers.append(sampler)
            shader['use_diffuse_texture'] = True
            shader['diffuse_texture'] = unit_id
        if (scene_node := references.get(textures.specular_id)) is not None and scene_node.texture is not None:
            if textures.diffuse_id in unit_ids:
                unit_id = unit_ids[textures.specular_id]
            else:
                unit_id = len(unit_ids) + 1
                unit_ids[textures.specular_id] = unit_id
                sampler = glctx.sampler(texture=scene_node.texture, filter=(moderngl.LINEAR, moderngl.LINEAR))
                sampler.use(unit_id)
                samplers.append(sampler)
            shader['use_specular_texture'] = True
            shader['specular_texture'] = unit_id
        if (scene_node := references.get(textures.emissive_id)) is not None and scene_node.texture is not None:
            if textures.diffuse_id in unit_ids:
                unit_id = unit_ids[textures.emissive_id]
            else:
                unit_id = len(unit_ids) + 1
                unit_ids[textures.emissive_id] = unit_id
                sampler = glctx.sampler(texture=scene_node.texture, filter=(moderngl.LINEAR, moderngl.LINEAR))
                sampler.use(unit_id)
                samplers.append(sampler)
            shader['use_emissive_texture'] = True
            shader['emissive_texture'] = unit_id
    matrices_buffer = glctx.buffer(matrices)
    materials_buffer = glctx.buffer(materials)
    buffers = [(vertex_buffer, '3f 3f 2f', 'model_position', 'model_normal', 'model_uv'),
               (matrices_buffer, '16f/i', 'model_matrix'),
               (materials_buffer, '9f 1f 1f/i', 'material_colors', 'material_shininess', 'material_transparency')]
    render_array = glctx.vertex_array(shader, buffers, index_buffer=index_buffer, mode=moderngl.TRIANGLES)
    render_array.render(instances=count)
    if samplers is not None:
        for sampler in samplers:
            sampler.clear()
        shader['use_diffuse_texture'] = False
        shader['use_specular_texture'] = False
        shader['use_emissive_texture'] = False
