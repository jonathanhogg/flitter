# cython: language_level=3, profile=True

"""
Flitter OpenGL 3D drawing canvas
"""

import cython
from cython cimport view
from loguru import logger
from mako.template import Template
import moderngl
import numpy as np

from libc.math cimport cos, log2, sqrt

from . import SceneNode, COLOR_FORMATS
from ... import name_patch
from ...clock import system_clock
from ...model cimport Node, Vector, Matrix44, null_, true_
from .glsl import TemplateLoader
from .models cimport Model, Box, Cylinder, Cone, Sphere, ExternalModel


logger = name_patch(logger, __name__)

cdef Vector Zero3 = Vector((0, 0, 0))
cdef Vector One3 = Vector((1, 1, 1))
cdef Vector Xaxis = Vector((1, 0, 0))
cdef Vector Yaxis = Vector((0, 1, 0))
cdef Vector Black = Vector((0, 0, 0, 1))
cdef int DEFAULT_MAX_LIGHTS = 50
cdef double Pi = 3.141592653589793
cdef tuple MaterialAttributes = ('color', 'specular', 'emissive', 'shininess', 'transparency',
                                 'texture_id', 'specular_texture_id', 'emissive_texture_id', 'transparency_texture_id')

cdef object StandardVertexTemplate = TemplateLoader.get_template("standard_lighting.vert")
cdef object StandardFragmentTemplate = TemplateLoader.get_template("standard_lighting.frag")


cdef enum LightType:
    Ambient = 1
    Directional = 2
    Point = 3
    Spot = 4


cdef class Light:
    cdef LightType type
    cdef float inner_cone
    cdef float outer_cone
    cdef Vector color
    cdef Vector position
    cdef Vector direction


cdef class Textures:
    cdef str diffuse_id
    cdef str specular_id
    cdef str emissive_id
    cdef str transparency_id

    def __eq__(self, Textures other):
        return other.diffuse_id == self.diffuse_id and \
               other.specular_id == self.specular_id and \
               other.emissive_id == self.emissive_id and \
               other.transparency_id == self.transparency_id

    def __hash__(self):
        return hash(self.diffuse_id) ^ hash(self.specular_id) ^ hash(self.emissive_id) ^ hash(self.transparency_id)


cdef class Material:
    cdef Vector diffuse
    cdef Vector specular
    cdef Vector emissive
    cdef double shininess
    cdef double transparency
    cdef Textures textures

    cdef Material update(Material self, Node node):
        for attr in MaterialAttributes:
            if attr in node._attributes:
                break
        else:
            return self
        cdef Material material = Material.__new__(Material)
        material.diffuse = node.get_fvec('color', 3, self.diffuse)
        material.specular = node.get_fvec('specular', 3, self.specular)
        material.emissive = node.get_fvec('emissive', 3, self.emissive)
        material.shininess = node.get_float('shininess', self.shininess)
        material.transparency = node.get_float('transparency', self.transparency)
        if self.textures is not None:
            diffuse_id = node.get_str('texture_id', self.textures.diffuse_id)
            specular_id = node.get_str('specular_texture_id', self.textures.specular_id)
            emissive_id = node.get_str('emissive_texture_id', self.textures.emissive_id)
            transparency_id = node.get_str('transparency_texture_id', self.textures.transparency_id)
        else:
            diffuse_id = node.get_str('texture_id', None)
            specular_id = node.get_str('specular_texture_id', None)
            emissive_id = node.get_str('emissive_texture_id', None)
            transparency_id = node.get_str('transparency_texture_id', None)
        cdef Textures textures
        if diffuse_id is not None or specular_id is not None or emissive_id is not None or transparency_id is not None:
            textures = Textures.__new__(Textures)
            textures.diffuse_id = diffuse_id
            textures.specular_id = specular_id
            textures.emissive_id = emissive_id
            textures.transparency_id = transparency_id
            material.textures = textures
        return material


cdef class Instance:
    cdef Matrix44 model_matrix
    cdef Material material


cdef class RenderSet:
    cdef int max_lights
    cdef list lights
    cdef dict instances
    cdef bint depth_test
    cdef bint cull_face
    cdef str composite
    cdef object vertex_shader_template
    cdef object fragment_shader_template

    cdef void set_blend(self, glctx):
        glctx.blend_equation = moderngl.FUNC_ADD
        if self.composite == 'source':
            glctx.blend_func = moderngl.ONE, moderngl.ZERO
        elif self.composite == 'dest':
            glctx.blend_func = moderngl.ZERO, moderngl.ONE
        elif self.composite == 'dest_over':
            glctx.blend_func = moderngl.ONE_MINUS_DST_ALPHA, moderngl.ONE
        elif self.composite == 'add':
            glctx.blend_func = moderngl.ONE, moderngl.ONE
        elif self.composite == 'subtract':
            glctx.blend_equation = moderngl.FUNC_SUBTRACT
            glctx.blend_func = moderngl.ONE, moderngl.ONE
        elif self.composite == 'lighten':
            glctx.blend_equation = moderngl.MAX
            glctx.blend_func = moderngl.ONE, moderngl.ONE
        elif self.composite == 'darken':
            glctx.blend_equation = moderngl.MIN
            glctx.blend_func = moderngl.ONE, moderngl.ONE
        else: # over
            glctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA


cdef Matrix44 update_model_matrix(Node node, Matrix44 model_matrix):
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
        elif attribute == 'shear_x':
            if (matrix := Matrix44._shear_x(vector)) is not None:
                model_matrix = model_matrix.mmul(matrix)
        elif attribute == 'shear_y':
            if (matrix := Matrix44._shear_y(vector)) is not None:
                model_matrix = model_matrix.mmul(matrix)
        elif attribute == 'shear_z':
            if (matrix := Matrix44._shear_z(vector)) is not None:
                model_matrix = model_matrix.mmul(matrix)
        elif attribute == 'matrix':
            if (matrix := Matrix44(vector)) is not None:
                model_matrix = model_matrix.mmul(matrix)
    return model_matrix


def draw(Node node, tuple size, glctx, dict objects, dict references):
    cdef int width, height
    width, height = size
    cdef Vector viewpoint = node.get_fvec('viewpoint', 3, Vector((0, 0, width/2)))
    cdef Vector focus = node.get_fvec('focus', 3, Zero3)
    cdef Vector up = node.get_fvec('up', 3, Vector((0, 1, 0)))
    cdef double fov = node.get_float('fov', 0.25)
    cdef bint orthographic = node.get_bool('orthographic', False)
    cdef double ortho_width = node.get_float('width', width)
    cdef double near = node.get_float('near', 1)
    cdef double far = node.get_float('far', width)
    cdef double fog_min = node.get_float('fog_min', 0)
    cdef double fog_max = node.get_float('fog_max', 0)
    cdef Vector fog_color = node.get_fvec('fog_color', 3, Black)
    cdef double fog_curve = max(0, node.get_float('fog_curve', 1))
    cdef Matrix44 pv_matrix
    if orthographic:
        pv_matrix = Matrix44._ortho(width/height, ortho_width, near, far)
    else:
        pv_matrix = Matrix44._project(width/height, fov, near, far)
    pv_matrix = pv_matrix.mmul(Matrix44._look(viewpoint, focus, up))
    cdef Material material = Material.__new__(Material)
    material.diffuse = Zero3
    material.specular = One3
    material.emissive = Zero3
    cdef Matrix44 model_matrix = Matrix44.__new__(Matrix44)
    cdef list render_sets = []
    collect(node, model_matrix, material, None, render_sets)
    cdef RenderSet render_set
    for render_set in render_sets:
        if render_set.instances:
            render(render_set, pv_matrix, orthographic, viewpoint, focus, fog_min, fog_max, fog_color, fog_curve,
                   glctx, objects, references)


cdef void collect(Node node, Matrix44 model_matrix, Material material, RenderSet render_set, list render_sets):
    cdef str kind = node.kind
    cdef Light light
    cdef list lights, instances
    cdef Vector color, position, direction, emissive, diffuse, specular
    cdef double shininess, inner, outer
    cdef Node child
    cdef str filename, vertex_shader, fragment_shader
    cdef Model model

    if node.kind == 'box':
        model = Box.get(node)
        if model is not None:
            material = material.update(node)
            add_instance(render_set.instances, model, node, model_matrix, material)

    elif node.kind == 'sphere':
        model = Sphere.get(node)
        if model is not None:
            material = material.update(node)
            add_instance(render_set.instances, model, node, model_matrix, material)

    elif node.kind == 'cylinder':
        model = Cylinder.get(node)
        if model is not None:
            material = material.update(node)
            add_instance(render_set.instances, model, node, model_matrix, material)

    elif node.kind == 'cone':
        model = Cone.get(node)
        if model is not None:
            material = material.update(node)
            add_instance(render_set.instances, model, node, model_matrix, material)

    elif node.kind == 'model':
        model = ExternalModel.get(node)
        if model is not None:
            material = material.update(node)
            add_instance(render_set.instances, model, node, model_matrix, material)

    elif node.kind == 'material':
        material = material.update(node)
        child = node.first_child
        while child is not None:
            collect(child, model_matrix, material, render_set, render_sets)
            child = child.next_sibling

    elif node.kind == 'transform':
        model_matrix = update_model_matrix(node, model_matrix)
        child = node.first_child
        while child is not None:
            collect(child, model_matrix, material, render_set, render_sets)
            child = child.next_sibling

    elif node.kind == 'group' or node.kind == 'canvas3d':
        model_matrix = update_model_matrix(node, model_matrix)
        material = material.update(node)
        lights = list(render_set.lights) if render_set is not None else []
        lights.append([])
        render_set = RenderSet.__new__(RenderSet)
        render_set.max_lights = node.get_int('max_lights', DEFAULT_MAX_LIGHTS)
        render_set.lights = lights
        render_set.instances = {}
        render_set.depth_test = node.get_bool('depth_test', True)
        render_set.cull_face = node.get_bool('cull_face', True)
        render_set.composite = node.get_str('composite', 'over').lower()
        vertex_shader = node.get_str('vertex', None)
        fragment_shader = node.get_str('fragment', None)
        render_set.vertex_shader_template = Template(vertex_shader) if vertex_shader is not None else StandardVertexTemplate
        render_set.fragment_shader_template = Template(fragment_shader) if fragment_shader is not None else StandardFragmentTemplate
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


cdef Matrix44 instance_start_end_matrix(Vector start, Vector end, double radius):
    cdef Vector direction = end.sub(start);
    cdef double length = sqrt(direction.squared_sum())
    if length == 0 or radius <= 0:
        return None
    cdef Vector up = Xaxis if direction.numbers[0] == 0 and direction.numbers[2] == 0 else Yaxis
    cdef Vector middle = Vector.__new__(Vector)
    middle.allocate_numbers(3)
    middle.numbers[0] = (start.numbers[0] + end.numbers[0]) / 2
    middle.numbers[1] = (start.numbers[1] + end.numbers[1]) / 2
    middle.numbers[2] = (start.numbers[2] + end.numbers[2]) / 2
    cdef Vector size = Vector.__new__(Vector)
    size.allocate_numbers(3)
    size.numbers[0] = radius
    size.numbers[1] = radius
    size.numbers[2] = length
    return Matrix44._look(middle, start, up).inverse().mmul(Matrix44._scale(size))


cdef void add_instance(dict render_instances, Model model, Node node, Matrix44 model_matrix, Material material):
    cdef Matrix44 matrix = None
    cdef Vector vec=None, start=None, end=None
    if (start := node.get_fvec('start', 3, None)) is not None and (end := node.get_fvec('end', 3, None)) is not None \
            and (matrix := instance_start_end_matrix(start, end, node.get_float('radius', 1))) is not None:
        model_matrix = model_matrix.mmul(matrix)
    else:
        if (vec := node.get_fvec('position', 3, None)) is not None and (matrix := Matrix44._translate(vec)) is not None:
            model_matrix = model_matrix.mmul(matrix)
        if (vec := node.get_fvec('rotation', 3, None)) is not None and (matrix := Matrix44._rotate(vec)) is not None:
            model_matrix = model_matrix.mmul(matrix)
        if (vec := node.get_fvec('size', 3, None)) is not None and (matrix := Matrix44._scale(vec)) is not None:
            model_matrix = model_matrix.mmul(matrix)
    cdef Instance instance = Instance.__new__(Instance)
    instance.model_matrix = model_matrix
    instance.material = material
    cdef tuple model_textures = (model, material.textures)
    (<list>render_instances.setdefault(model_textures, [])).append(instance)


def fst(tuple ab):
    return ab[0]


cdef void render(RenderSet render_set, Matrix44 pv_matrix, bint orthographic, Vector viewpoint, Vector focus,
                 double fog_min, double fog_max, Vector fog_color, float fog_curve, glctx, dict objects, dict references):
    cdef list instances, lights, buffers
    cdef cython.float[:, :] matrices, materials, lights_data
    cdef Material material
    cdef Light light
    cdef Model model
    cdef Textures textures
    cdef int i, j, k, n
    cdef double z
    cdef double* src
    cdef float* dest
    cdef Instance instance
    cdef Matrix44 matrix
    cdef bint has_transparency_texture
    cdef tuple transparent_object
    cdef list transparent_objects = []
    cdef double[:] zs
    cdef long[:] indices
    cdef dict shaders = objects.setdefault('canvas3d_shaders', {})
    cdef dict variables = {'max_lights': render_set.max_lights, 'Ambient': LightType.Ambient, 'Directional': LightType.Directional,
                           'Point': LightType.Point, 'Spot': LightType.Spot}
    cdef str vertex_shader = render_set.vertex_shader_template.render(**variables)
    cdef str fragment_shader = render_set.fragment_shader_template.render(**variables)
    cdef tuple source = (vertex_shader, fragment_shader)
    shader = shaders.get(source)
    if shader is None:
        try:
            shader = glctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        except Exception as exc:
            logger.error("Shader program compile failed:\n{}", '\n'.join(str(exc).strip().split('\n')[4:]))
            shader = False
        else:
            logger.debug("Compiled shader program")
        shaders[source] = shader
    if shader is False:
        return
    shader['pv_matrix'] = pv_matrix
    shader['orthographic'] = orthographic
    shader['view_position'] = viewpoint
    shader['focus'] = focus
    shader['fog_min'] = fog_min
    shader['fog_max'] = fog_max
    shader['fog_color'] = fog_color
    shader['fog_curve'] = fog_curve
    shader['use_diffuse_texture'] = False
    shader['use_specular_texture'] = False
    shader['use_emissive_texture'] = False
    shader['use_transparency_texture'] = False
    lights_data = view.array((render_set.max_lights, 12), 4, 'f')
    i = 0
    for lights in render_set.lights:
        for light in lights:
            if i == render_set.max_lights:
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
    render_set.set_blend(glctx)
    for (model, textures), instances in render_set.instances.items():
        has_transparency_texture = textures is not None and textures.transparency_id is not None
        n = len(instances)
        matrices = view.array((n, 16), 4, 'f')
        materials = view.array((n, 11), 4, 'f')
        k = 0
        if render_set.depth_test:
            zs_array = np.empty(n)
            zs = zs_array
            for i, instance in enumerate(instances):
                matrix = pv_matrix.mmul(instance.model_matrix)
                zs[i] = matrix.numbers[14] / matrix.numbers[15]
            indices = zs_array.argsort()
        else:
            indices = np.arange(n, dtype='long')
        for i in indices:
            instance = instances[i]
            material = instance.material
            if material.shininess > 0 or material.transparency < 1 or has_transparency_texture:
                if (material.transparency > 0 or has_transparency_texture) and render_set.depth_test:
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
        n = len(transparent_objects)
        transparent_objects.sort(key=fst)
        matrices = view.array((n, 16), 4, 'f')
        materials = view.array((n, 11), 4, 'f')
        k = 0
        for i, transparent_object in enumerate(transparent_objects):
            model = transparent_object[1]
            instance = transparent_object[2]
            material = instance.material
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
            if i == n-1 or (<tuple>transparent_objects[i+1])[1] is not model:
                dispatch_instances(glctx, objects, shader, model, k, matrices, materials, material.textures, references)
                k = 0
        if k:
            dispatch_instances(glctx, objects, shader, model, k, matrices, materials, material.textures, references)
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
        if (scene_node := references.get(textures.diffuse_id)) is not None and hasattr(scene_node, 'texture') and scene_node.texture is not None:
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
        if (scene_node := references.get(textures.specular_id)) is not None and hasattr(scene_node, 'texture') and scene_node.texture is not None:
            if textures.specular_id in unit_ids:
                unit_id = unit_ids[textures.specular_id]
            else:
                unit_id = len(unit_ids) + 1
                unit_ids[textures.specular_id] = unit_id
                sampler = glctx.sampler(texture=scene_node.texture, filter=(moderngl.LINEAR, moderngl.LINEAR))
                sampler.use(unit_id)
                samplers.append(sampler)
            shader['use_specular_texture'] = True
            shader['specular_texture'] = unit_id
        if (scene_node := references.get(textures.emissive_id)) is not None and hasattr(scene_node, 'texture') and scene_node.texture is not None:
            if textures.emissive_id in unit_ids:
                unit_id = unit_ids[textures.emissive_id]
            else:
                unit_id = len(unit_ids) + 1
                unit_ids[textures.emissive_id] = unit_id
                sampler = glctx.sampler(texture=scene_node.texture, filter=(moderngl.LINEAR, moderngl.LINEAR))
                sampler.use(unit_id)
                samplers.append(sampler)
            shader['use_emissive_texture'] = True
            shader['emissive_texture'] = unit_id
        if (scene_node := references.get(textures.transparency_id)) is not None and hasattr(scene_node, 'texture') and scene_node.texture is not None:
            if textures.transparency_id in unit_ids:
                unit_id = unit_ids[textures.transparency_id]
            else:
                unit_id = len(unit_ids) + 1
                unit_ids[textures.transparency_id] = unit_id
                sampler = glctx.sampler(texture=scene_node.texture, filter=(moderngl.LINEAR, moderngl.LINEAR))
                sampler.use(unit_id)
                samplers.append(sampler)
            shader['use_transparency_texture'] = True
            shader['transparency_texture'] = unit_id
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
        shader['use_transparency_texture'] = False


class Canvas3D(SceneNode):
    def __init__(self, glctx):
        super().__init__(glctx)
        self._image_texture = None
        self._image_framebuffer = None
        self._color_renderbuffer = None
        self._depth_renderbuffer = None
        self._render_framebuffer = None
        self._colorbits = None
        self._samples = None
        self._total_duration = 0
        self._total_count = 0

    @property
    def texture(self):
        return self._image_texture

    def release(self):
        self._colorbits = None
        self._samples = None
        self._render_framebuffer = None
        self._image_texture = None
        self._image_framebuffer = None
        self._color_renderbuffer = None
        self._depth_renderbuffer = None

    def purge(self):
        logger.info("{} draw stats - {:d} x {:.1f}ms = {:.1f}s", self.name, self._total_count,
                    1e3 * self._total_duration / self._total_count, self._total_duration)
        self._total_duration = 0
        self._total_count = 0

    def create(self, engine, node, resized, **kwargs):
        colorbits = node.get('colorbits', 1, int, self.glctx.extra['colorbits'])
        if colorbits not in COLOR_FORMATS:
            colorbits = self.glctx.extra['colorbits']
        samples = max(0, node.get('samples', 1, int, 0))
        if samples:
            samples = min(1 << int(log2(samples)), self.glctx.info['GL_MAX_SAMPLES'])
        if resized or colorbits != self._colorbits or samples != self._samples:
            self.release()
            format = COLOR_FORMATS[colorbits]
            self._image_texture = self.glctx.texture((self.width, self.height), 4, dtype=format.moderngl_dtype)
            self._depth_renderbuffer = self.glctx.depth_renderbuffer((self.width, self.height), samples=samples)
            if samples:
                self._color_renderbuffer = self.glctx.renderbuffer((self.width, self.height), 4, samples=samples, dtype=format.moderngl_dtype)
                self._render_framebuffer = self.glctx.framebuffer(color_attachments=(self._color_renderbuffer,), depth_attachment=self._depth_renderbuffer)
                self._image_framebuffer = self.glctx.framebuffer(self._image_texture)
                logger.debug("Created canvas3d {}x{}/{}-bit render target with {}x sampling", self.width, self.height, colorbits, samples)
            else:
                self._render_framebuffer = self.glctx.framebuffer(color_attachments=(self._image_texture,), depth_attachment=self._depth_renderbuffer)
                logger.debug("Created canvas3d {}x{}/{}-bit render target", self.width, self.height, colorbits)
            self._colorbits = colorbits
            self._samples = samples

    async def descend(self, engine, node, **kwargs):
        # A canvas3d is a leaf node from the perspective of the OpenGL world
        pass

    def render(self, node, references=None, **kwargs):
        self._total_duration -= system_clock()
        self._render_framebuffer.use()
        fog_min = node.get('fog_min', 1, float, 0)
        fog_max = node.get('fog_max', 1, float, 0)
        if fog_max > fog_min:
            fog_color = node.get('fog_color', 3, float, (0, 0, 0))
            self._render_framebuffer.clear(*fog_color)
        else:
            self._render_framebuffer.clear()
        objects = self.glctx.extra.setdefault('canvas3d_objects', {})
        draw(node, (self.width, self.height), self.glctx, objects, references)
        if self._image_framebuffer is not None:
            self.glctx.copy_framebuffer(self._image_framebuffer, self._render_framebuffer)
        self._total_duration += system_clock()
        self._total_count += 1


SCENE_NODE_CLASS = Canvas3D
