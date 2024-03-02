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

from libc.math cimport cos, log2, sqrt, tan
from libc.stdint cimport int64_t

from . import SceneNode, COLOR_FORMATS, set_uniform_vector
from ... import name_patch
from ...clock import system_clock
from ...model cimport Node, Vector, Matrix44, Matrix33, null_, true_
from .glsl import TemplateLoader
from .models cimport Model, DefaultSnapAngle


logger = name_patch(logger, __name__)

cdef Vector Zero3 = Vector((0, 0, 0))
cdef Vector Zero4 = Vector((0, 0, 0, 0))
cdef Vector Greyscale = Vector((0.299, 0.587, 0.114))
cdef Vector One3 = Vector((1, 1, 1))
cdef Vector Xaxis = Vector((1, 0, 0))
cdef Vector Yaxis = Vector((0, 1, 0))
cdef Vector DefaultFalloff = Vector((0, 0, 1, 0))
cdef Matrix44 IdentityTransform = Matrix44.__new__(Matrix44)
cdef int DEFAULT_MAX_LIGHTS = 50
cdef double Pi = 3.141592653589793
cdef tuple MaterialAttributes = ('color', 'metal', 'roughness', 'shininess', 'occlusion', 'emissive', 'transparency',
                                 'texture_id', 'metal_texture_id', 'roughness_texture_id', 'occlusion_texture_id',
                                 'emissive_texture_id', 'transparency_texture_id')
cdef tuple TransformAttributes = ('translate', 'scale', 'rotate', 'rotate_x', 'rotate_y', 'rotate_z', 'shear_x', 'shear_y', 'shear_z')
cdef set GroupAttributes = set(MaterialAttributes)
GroupAttributes.update(TransformAttributes)
GroupAttributes.update(('max_lights', 'depth_test', 'cull_face', 'composite', 'vertex', 'fragment'))

cdef object StandardVertexTemplate = TemplateLoader.get_template("standard_lighting.vert")
cdef object StandardFragmentTemplate = TemplateLoader.get_template("standard_lighting.frag")


cdef enum LightType:
    Ambient = 1
    Directional = 2
    Point = 3
    Spot = 4
    Line = 5


cdef class Light:
    cdef LightType type
    cdef double inner_cone
    cdef double outer_cone
    cdef Vector color
    cdef Vector position
    cdef Vector direction
    cdef Vector falloff


cdef class Textures:
    cdef str albedo_id
    cdef str metal_id
    cdef str roughness_id
    cdef str occlusion_id
    cdef str emissive_id
    cdef str transparency_id

    def __eq__(self, Textures other):
        return other.albedo_id == self.albedo_id and \
               other.metal_id == self.metal_id and \
               other.roughness_id == self.roughness_id and \
               other.occlusion_id == self.occlusion_id and \
               other.emissive_id == self.emissive_id and \
               other.transparency_id == self.transparency_id

    def __hash__(self):
        return (hash(self.albedo_id) ^ hash(self.metal_id) ^ hash(self.roughness_id) ^ hash(self.occlusion_id) ^
                hash(self.emissive_id) ^ hash(self.transparency_id))


cdef class Material:
    cdef Vector albedo
    cdef double ior
    cdef double metal
    cdef double roughness
    cdef double occlusion
    cdef Vector emissive
    cdef double transparency
    cdef Textures textures

    cdef Material update(Material self, Node node):
        for attr in MaterialAttributes:
            if node._attributes and attr in node._attributes:
                break
        else:
            return self
        cdef Material material = Material.__new__(Material)
        material.albedo = node.get_fvec('color', 3, self.albedo)
        material.ior = max(1, node.get_float('ior', 1.5))
        cdef double shininess = max(0, node.get_float('shininess', (10 / self.roughness - 10)**2))
        material.roughness = min(max(1e-6, node.get_float('roughness', 10 / (10 + sqrt(shininess)))), 1)
        cdef Vector specular = node.get_fvec('specular', 3, One3)
        material.metal = min(max(0, node.get_float('metal', self.metal)), 1)
        cdef Vector k
        if specular.ne(One3) is true_ and material.roughness < 1 and material.metal == 0:
            k = Vector.__new__(Vector, 0.001)
            material.metal = min(max(0, k.add(specular.sub(material.albedo)).truediv(k.add(specular.add(material.albedo))).squared_sum() / 3), 1)
            k = Vector.__new__(Vector, material.metal)
            material.albedo = material.albedo.mul(true_.sub(k)).mul_add(k, specular)
            material.roughness = material.roughness ** 1.5
        material.occlusion = min(max(0, node.get_float('occlusion', self.occlusion)), 1)
        material.emissive = node.get_fvec('emissive', 3, self.emissive)
        material.transparency = min(max(0, node.get_float('transparency', self.transparency)), 1)
        if self.textures is not None:
            albedo_id = node.get_str('texture_id', self.textures.albedo_id)
            metal_id = node.get_str('metal_texture_id', self.textures.metal_id)
            roughness_id = node.get_str('roughness_texture_id', self.textures.roughness_id)
            occlusion_id = node.get_str('occlusion_texture_id', self.textures.occlusion_id)
            emissive_id = node.get_str('emissive_texture_id', self.textures.emissive_id)
            transparency_id = node.get_str('transparency_texture_id', self.textures.transparency_id)
        else:
            albedo_id = node.get_str('texture_id', None)
            metal_id = node.get_str('metal_texture_id', None)
            roughness_id = node.get_str('roughness_texture_id', None)
            occlusion_id = node.get_str('occlusion_texture_id', None)
            emissive_id = node.get_str('emissive_texture_id', None)
            transparency_id = node.get_str('transparency_texture_id', None)
        cdef Textures textures
        if (albedo_id is not None or metal_id is not None or roughness_id is not None or occlusion_id is not None or
                emissive_id is not None or transparency_id is not None):
            textures = Textures.__new__(Textures)
            textures.albedo_id = albedo_id
            textures.metal_id = metal_id
            textures.roughness_id = roughness_id
            textures.occlusion_id = occlusion_id
            textures.emissive_id = emissive_id
            textures.transparency_id = transparency_id
            material.textures = textures
        return material


cdef class Instance:
    cdef Matrix44 model_matrix
    cdef Material material


cdef class RenderGroup:
    cdef RenderGroup parent_group
    cdef int max_lights
    cdef list lights
    cdef dict instances
    cdef bint depth_test
    cdef bint cull_face
    cdef str composite
    cdef object vertex_shader_template
    cdef object fragment_shader_template
    cdef dict names

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
        else:   # over
            glctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA


cdef class Camera:
    cdef str id
    cdef bint secondary
    cdef Vector position
    cdef Vector focus
    cdef Vector up
    cdef double fov
    cdef str fov_ref
    cdef bint monochrome
    cdef Vector tint
    cdef bint orthographic
    cdef double ortho_width
    cdef double near
    cdef double far
    cdef double fog_min
    cdef double fog_max
    cdef Vector fog_color
    cdef double fog_curve
    cdef Matrix44 pv_matrix
    cdef int width
    cdef int height
    cdef int colorbits
    cdef int samples

    cdef Camera derive(self, Node node, Matrix44 transform_matrix, int max_samples):
        cdef Camera camera = Camera.__new__(Camera)
        camera.id = node.get_str('id', None)
        camera.secondary = node.get_bool('secondary', False)
        cdef Vector position = node.get_fvec('position', 3, node.get_fvec('viewpoint', 3, None))
        if position is None:
            camera.position = self.position
        else:
            camera.position = transform_matrix.vmul(position)
        cdef Vector focus = node.get_fvec('focus', 3, None)
        if focus is None:
            camera.focus = self.focus
        else:
            camera.focus = transform_matrix.vmul(focus)
        cdef Vector up = node.get_fvec('up', 3, None)
        if up is None:
            camera.up = self.up
        else:
            camera.up = transform_matrix.inverse_transpose_matrix33().vmul(up).normalize()
        camera.fov = node.get_float('fov', self.fov)
        camera.fov_ref = node.get_str('fov_ref', self.fov_ref).lower()
        camera.monochrome = node.get_bool('monochrome', self.monochrome)
        camera.tint = node.get_fvec('tint', 3, self.tint)
        camera.orthographic = node.get_bool('orthographic', self.orthographic)
        camera.ortho_width = node.get_float('width', self.ortho_width)
        camera.near = max(1e-9, node.get_float('near', self.near))
        camera.far = max(camera.near+1e-9, node.get_float('far', self.far))
        camera.fog_min = node.get_float('fog_min', self.fog_min)
        camera.fog_max = node.get_float('fog_max', self.fog_max)
        camera.fog_color = node.get_fvec('fog_color', 3, self.fog_color)
        camera.fog_curve = max(0, node.get_float('fog_curve', self.fog_curve))
        cdef Vector size = node.get_fvec('size', 2, Vector((self.width, self.height)))
        camera.width = max(1, int(size.numbers[0]))
        camera.height = max(1, int(size.numbers[1]))
        cdef int colorbits = node.get_int('colorbits', self.colorbits)
        camera.colorbits = colorbits if colorbits in COLOR_FORMATS else self.colorbits
        cdef double samples = node.get_float('samples', <double>self.samples)
        camera.samples = min(1 << int(log2(samples)), max_samples) if samples >= 2 else 0
        cdef double aspect_ratio = (<double>camera.width) / camera.height
        cdef Matrix44 projection_matrix
        cdef double gradient, diagonal_ratio
        if camera.orthographic:
            projection_matrix = Matrix44._ortho(aspect_ratio, camera.ortho_width, camera.near, camera.far)
        else:
            gradient = tan(Pi*camera.fov)
            if camera.fov_ref == 'diagonal':
                diagonal_ratio = sqrt(1 + aspect_ratio*aspect_ratio)
                projection_matrix = Matrix44._project(aspect_ratio*gradient/diagonal_ratio, gradient/diagonal_ratio, camera.near, camera.far)  # diagonal
            elif camera.fov_ref == 'vertical':
                projection_matrix = Matrix44._project(aspect_ratio*gradient, gradient, camera.near, camera.far)  # vertical
            elif camera.fov_ref == 'wide':
                if aspect_ratio > 1:  # widescreen
                    projection_matrix = Matrix44._project(gradient, gradient/aspect_ratio, camera.near, camera.far)  # horizontal
                else:
                    projection_matrix = Matrix44._project(aspect_ratio*gradient, gradient, camera.near, camera.far)  # vertical
            elif camera.fov_ref == 'narrow':
                if aspect_ratio > 1:  # widescreen
                    projection_matrix = Matrix44._project(aspect_ratio*gradient, gradient, camera.near, camera.far)  # vertical
                else:
                    projection_matrix = Matrix44._project(gradient, gradient/aspect_ratio, camera.near, camera.far)  # horizontal
            else:
                projection_matrix = Matrix44._project(gradient, gradient/aspect_ratio, camera.near, camera.far)  # horizontal
        camera.pv_matrix = projection_matrix.mmul(Matrix44._look(camera.position, camera.focus, camera.up))
        return camera


cdef Matrix44 update_transform_matrix(Node node, Matrix44 transform_matrix):
    cdef Matrix44 matrix
    cdef str attribute
    cdef Vector vector
    if node._attributes:
        for attribute, vector in node._attributes.items():
            if attribute == 'translate':
                if (matrix := Matrix44._translate(vector)) is not None:
                    transform_matrix = transform_matrix.mmul(matrix)
            elif attribute == 'scale':
                if (matrix := Matrix44._scale(vector)) is not None:
                    transform_matrix = transform_matrix.mmul(matrix)
            elif attribute == 'rotate':
                if (matrix := Matrix44._rotate(vector)) is not None:
                    transform_matrix = transform_matrix.mmul(matrix)
            elif attribute == 'rotate_x':
                if vector.numbers != NULL and vector.length == 1 and (matrix := Matrix44._rotate_x(vector.numbers[0])) is not None:
                    transform_matrix = transform_matrix.mmul(matrix)
            elif attribute == 'rotate_y':
                if vector.numbers != NULL and vector.length == 1 and (matrix := Matrix44._rotate_y(vector.numbers[0])) is not None:
                    transform_matrix = transform_matrix.mmul(matrix)
            elif attribute == 'rotate_z':
                if vector.numbers != NULL and vector.length == 1 and (matrix := Matrix44._rotate_z(vector.numbers[0])) is not None:
                    transform_matrix = transform_matrix.mmul(matrix)
            elif attribute == 'shear_x':
                if (matrix := Matrix44._shear_x(vector)) is not None:
                    transform_matrix = transform_matrix.mmul(matrix)
            elif attribute == 'shear_y':
                if (matrix := Matrix44._shear_y(vector)) is not None:
                    transform_matrix = transform_matrix.mmul(matrix)
            elif attribute == 'shear_z':
                if (matrix := Matrix44._shear_z(vector)) is not None:
                    transform_matrix = transform_matrix.mmul(matrix)
            elif attribute == 'matrix':
                if (matrix := Matrix44(vector)) is not None:
                    transform_matrix = transform_matrix.mmul(matrix)
    return transform_matrix


cdef Matrix44 instance_start_end_matrix(Vector start, Vector end, double radius):
    cdef Vector direction = end.sub(start)
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


cdef Matrix44 get_model_transform(Node node, Matrix44 transform_matrix):
    cdef Vector vec=None, start=None, end=None
    cdef Matrix44 matrix = None
    if (start := node.get_fvec('start', 3, None)) is not None and (end := node.get_fvec('end', 3, None)) is not None \
            and (matrix := instance_start_end_matrix(start, end, node.get_float('radius', 1))) is not None:
        transform_matrix = transform_matrix.mmul(matrix)
    else:
        if (vec := node.get_fvec('position', 3, None)) is not None and (matrix := Matrix44._translate(vec)) is not None:
            transform_matrix = transform_matrix.mmul(matrix)
        if (vec := node.get_fvec('rotation', 3, None)) is not None and (matrix := Matrix44._rotate(vec)) is not None:
            transform_matrix = transform_matrix.mmul(matrix)
        if (vec := node.get_fvec('size', 3, None)) is not None and (matrix := Matrix44._scale(vec)) is not None:
            transform_matrix = transform_matrix.mmul(matrix)
    return transform_matrix


cdef Model get_model(Node node, bint top):
    cdef Node child
    cdef Model model = None
    cdef Vector origin, normal
    cdef list models
    cdef double snap_angle, minimum_area
    if node.kind == 'intersect':
        models = []
        for child in node._children:
            models.append(get_model(child, False))
        model = Model.intersect(models)
    elif node.kind == 'union' or node.kind == 'transform':
        models = []
        for child in node._children:
            models.append(get_model(child, False))
        model = Model.union(models)
        if node.kind == 'transform':
            model = model.transform(update_transform_matrix(node, IdentityTransform))
    elif node.kind == 'difference':
        models = []
        for child in node._children:
            models.append(get_model(child, False))
        model = Model.difference(models)
    elif node.kind == 'slice':
        normal = node.get_fvec('normal', 3, null_)
        origin = node.get_fvec('origin', 3, Zero3)
        models = []
        for child in node._children:
            models.append(get_model(child, False))
        model = Model.union(models)
        if model is not None and normal.as_bool():
            model = model.slice(origin, normal)
    else:
        if node.kind == 'box':
            model = Model.get_box(node)
        elif node.kind == 'sphere':
            model = Model.get_sphere(node)
        elif node.kind == 'cylinder':
            model = Model.get_cylinder(node)
        elif node.kind == 'cone':
            model = Model.get_cone(node)
        elif node.kind == 'model':
            model = Model.get_external(node)
    if model is not None:
        if top:
            if node.get_bool('flat', False):
                model = model.flatten()
            elif (snap_angle := min(max(0, node.get_float('snap_edges', DefaultSnapAngle if model.is_constructed() else 0.5)), 0.5)) < 0.5:
                minimum_area = min(max(0, node.get_float('minimum_area', 0)), 1)
                model = model.snap_edges(snap_angle, minimum_area)
            if node.get_bool('invert', False):
                model = model.invert()
        elif node.kind != 'transform':
            model = model.transform(get_model_transform(node, IdentityTransform))
    return model


cdef void collect(Node node, Matrix44 transform_matrix, Material material, RenderGroup render_group, list render_groups,
                  Camera default_camera, dict cameras, int max_samples):
    cdef Light light
    cdef Vector color, position, direction, focus, start, end
    cdef double inner, outer
    cdef Node child
    cdef str camera_id, vertex_shader, fragment_shader
    cdef Model model
    cdef Instance instance
    cdef tuple model_textures
    cdef RenderGroup new_render_group
    cdef Matrix44 model_matrix

    if node.kind == 'material':
        material = material.update(node)
        for child in node._children:
            collect(child, transform_matrix, material, render_group, render_groups, default_camera, cameras, max_samples)

    elif node.kind == 'transform':
        transform_matrix = update_transform_matrix(node, transform_matrix)
        for child in node._children:
            collect(child, transform_matrix, material, render_group, render_groups, default_camera, cameras, max_samples)

    elif node.kind == 'group' or node.kind == 'canvas3d':
        transform_matrix = update_transform_matrix(node, transform_matrix)
        material = material.update(node)
        new_render_group = RenderGroup.__new__(RenderGroup)
        new_render_group.parent_group = render_group
        new_render_group.max_lights = node.get_int('max_lights', DEFAULT_MAX_LIGHTS if render_group is None else render_group.max_lights)
        new_render_group.lights = []
        new_render_group.instances = {}
        new_render_group.depth_test = node.get_bool('depth_test', True if render_group is None else render_group.depth_test)
        new_render_group.cull_face = node.get_bool('cull_face', True if render_group is None else render_group.cull_face)
        new_render_group.composite = node.get_str('composite', 'over' if render_group is None else render_group.composite).lower()
        vertex_shader = node.get_str('vertex', None)
        fragment_shader = node.get_str('fragment', None)
        new_render_group.vertex_shader_template = Template(vertex_shader) if vertex_shader is not None else None
        new_render_group.fragment_shader_template = Template(fragment_shader) if fragment_shader is not None else None
        new_render_group.names = {}
        if node._attributes:
            for name, value in node._attributes.items():
                if name not in GroupAttributes:
                    new_render_group.names[name] = value
        render_groups.append(new_render_group)
        for child in node._children:
            collect(child, transform_matrix, material, new_render_group, render_groups, default_camera, cameras, max_samples)

    elif node.kind == 'light':
        color = node.get_fvec('color', 3, null_)
        if color.as_bool():
            position = node.get_fvec('position', 3, null_)
            focus = node.get_fvec('focus', 3, null_)
            direction = node.get_fvec('direction', 3, focus.sub(position) if position.length and focus.length else null_)
            start = node.get_fvec('start', 3, null_)
            end = node.get_fvec('end', 3, null_)
            light = Light.__new__(Light)
            light.color = color
            light.falloff = node.get_fvec('falloff', 4, DefaultFalloff)
            if start.length and end.length:
                light.type = LightType.Line
                light.outer_cone = node.get_float('radius', 0)
                light.position = transform_matrix.vmul(start)
                light.direction = transform_matrix.vmul(end) - light.position
            elif position.length and direction.as_bool():
                light.type = LightType.Spot
                outer = min(max(0, node.get_float('outer', 0.25)), 0.5)
                inner = min(max(0, node.get_float('inner', 0)), outer)
                light.inner_cone = cos(inner * Pi)
                light.outer_cone = cos(outer * Pi)
                light.position = transform_matrix.vmul(position)
                light.direction = transform_matrix.inverse_transpose_matrix33().vmul(direction).normalize()
            elif position.length:
                light.type = LightType.Point
                light.outer_cone = node.get_float('radius', 0)
                light.position = transform_matrix.vmul(position)
                light.direction = None
            elif direction.as_bool():
                light.type = LightType.Directional
                light.position = None
                light.direction = transform_matrix.inverse_transpose_matrix33().vmul(direction).normalize()
            else:
                light.type = LightType.Ambient
                light.position = None
                light.direction = None
            render_group.lights.append(light)

    elif node.kind == 'camera':
        if (camera_id := node.get_str('id', None)) is not None:
            cameras[camera_id] = default_camera.derive(node, transform_matrix, max_samples)

    elif (model := get_model(node, True)) is not None:
        model, model_matrix = model.instance(get_model_transform(node, transform_matrix))
        material = material.update(node)
        instance = Instance.__new__(Instance)
        instance.model_matrix = model_matrix
        instance.material = material
        model_textures = (model, material.textures)
        (<list>render_group.instances.setdefault(model_textures, [])).append(instance)


def fst(tuple ab):
    return ab[0]


cdef object get_shader(object glctx, dict shaders, dict names, object vertex_shader_template, object fragment_shader_template):
    cdef str vertex_shader = vertex_shader_template.render(**names)
    cdef str fragment_shader = fragment_shader_template.render(**names)
    cdef tuple source = (vertex_shader, fragment_shader)
    shader = shaders.get(source, False)
    if shader is not False:
        return shader
    try:
        shader = glctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    except Exception as exc:
        logger.error("Group shader compile failed:\n{}", '\n'.join(str(exc).strip().split('\n')[4:]))
        shader = None
    else:
        logger.debug("Compiled group shader program")
    shaders[source] = shader
    return shader


cdef void render(RenderTarget render_target, RenderGroup render_group, Camera camera, glctx, dict objects, dict references):
    cdef list instances
    cdef cython.float[:, :] instances_data, lights_data
    cdef Material material
    cdef Light light
    cdef Model model
    cdef Textures textures
    cdef int i, j, k, n
    cdef double z, w
    cdef double* src
    cdef float* dest
    cdef Instance instance
    cdef Matrix44 matrix
    cdef Matrix33 normal_matrix
    cdef bint has_transparency_texture
    cdef tuple transparent_object
    cdef list transparent_objects = []
    cdef double[:] zs
    cdef int64_t[:] indices
    cdef dict shaders = objects.setdefault('canvas3d_shaders', {})
    cdef dict names = render_group.names.copy()
    names.update({'max_lights': render_group.max_lights, 'Ambient': LightType.Ambient, 'Directional': LightType.Directional,
                  'Point': LightType.Point, 'Spot': LightType.Spot, 'Line': LightType.Line})
    shader = None
    if render_group.vertex_shader_template is not None or render_group.fragment_shader_template is not None:
        shader = get_shader(glctx, shaders, names,
                            render_group.vertex_shader_template or StandardVertexTemplate,
                            render_group.fragment_shader_template or StandardFragmentTemplate)
    if shader is None:
        shader = get_shader(glctx, shaders, names, StandardVertexTemplate, StandardFragmentTemplate)
    cdef str name
    cdef Vector value
    cdef int base_unit_id = 1
    for name, value in render_group.names.items():
        if name in shader:
            member = shader[name]
            if isinstance(member, moderngl.Uniform):
                if member.fmt == '1i' and (scene_node := references.get(value.as_string())) is not None \
                        and hasattr(scene_node, 'texture') and scene_node.texture is not None:
                    sampler = glctx.sampler(texture=scene_node.texture, filter=(moderngl.NEAREST, moderngl.NEAREST))
                    sampler.use(base_unit_id)
                    member.value = base_unit_id
                    base_unit_id += 1
                elif value.numbers != NULL:
                    set_uniform_vector(member, value)
    shader['pv_matrix'] = camera.pv_matrix
    shader['orthographic'] = camera.orthographic
    if 'monochrome' in shader:
        shader['monochrome'] = camera.monochrome
    if 'tint' in shader:
        shader['tint'] = camera.tint
    shader['view_position'] = camera.position
    shader['view_focus'] = camera.focus
    if 'fog_color' in shader:
        shader['fog_min'] = camera.fog_min
        shader['fog_max'] = camera.fog_max
        shader['fog_color'] = camera.fog_color
        shader['fog_curve'] = camera.fog_curve
    cdef RenderGroup group
    if 'nlights' in shader:
        lights_data = view.array((render_group.max_lights, 16), 4, 'f')
        i = 0
        group = render_group
        while group is not None:
            for light in group.lights:
                if i == render_group.max_lights:
                    break
                dest = &lights_data[i, 0]
                dest[3] = <cython.float>(<int>light.type)
                dest[7] = light.inner_cone
                dest[11] = light.outer_cone
                for j in range(3):
                    dest[j] = light.color.numbers[j]
                if light.position is not None:
                    for j in range(3):
                        dest[j+4] = light.position.numbers[j]
                if light.direction is not None:
                    for j in range(3):
                        dest[j+8] = light.direction.numbers[j]
                for j in range(4):
                    dest[j+12] = light.falloff.numbers[j]
                i += 1
            group = group.parent_group
        shader['nlights'] = i
        shader['lights_data'].binding = 1
        name = f'canvas3d_lights-{render_group.max_lights}'
        lights_buffer = objects.get(name)
        if lights_buffer is None:
            lights_buffer = glctx.buffer(lights_data)
            objects[name] = lights_buffer
        else:
            lights_buffer.write(lights_data)
        lights_buffer.bind_to_uniform_block(1)
    cdef int flags = moderngl.BLEND
    if render_group.depth_test:
        flags |= moderngl.DEPTH_TEST
    if render_group.cull_face:
        flags |= moderngl.CULL_FACE
    glctx.enable(flags)
    render_group.set_blend(glctx)
    cdef bint shader_supports_textures = 'use_albedo_texture' in shader
    if shader_supports_textures:
        shader['use_albedo_texture'] = False
        shader['use_metal_texture'] = False
        shader['use_roughness_texture'] = False
        shader['use_occlusion_texture'] = False
        shader['use_emissive_texture'] = False
        shader['use_transparency_texture'] = False
    for (model, textures), instances in render_group.instances.items():
        has_transparency_texture = textures is not None and textures.transparency_id is not None
        n = len(instances)
        instances_data = view.array((n, 36), 4, 'f')
        k = 0
        if render_group.depth_test:
            zs_array = np.empty(n)
            zs = zs_array
            for i, instance in enumerate(instances):
                matrix = camera.pv_matrix.mmul(instance.model_matrix)
                z = matrix.numbers[14]
                w = matrix.numbers[15]
                if matrix.numbers[2] > 0:
                    z -= matrix.numbers[2]
                    w -= matrix.numbers[3]
                else:
                    z += matrix.numbers[2]
                    w += matrix.numbers[3]
                if matrix.numbers[6] > 0:
                    z -= matrix.numbers[6]
                    w -= matrix.numbers[7]
                else:
                    z += matrix.numbers[6]
                    w += matrix.numbers[7]
                if matrix.numbers[10] > 0:
                    z -= matrix.numbers[10]
                    w -= matrix.numbers[11]
                else:
                    z += matrix.numbers[10]
                    w += matrix.numbers[11]
                zs[i] = z / w if w != 0 else -1
            indices = zs_array.argsort().astype('int64')
        else:
            indices = np.arange(n, dtype='int64')
        for i in indices:
            instance = instances[i]
            material = instance.material
            if (material.transparency > 0 or has_transparency_texture) and render_group.depth_test:
                transparent_objects.append((-zs[i], model, instance))
            else:
                src = instance.model_matrix.numbers
                dest = &instances_data[k, 0]
                for j in range(16):
                    dest[j] = src[j]
                normal_matrix = instance.model_matrix.inverse_transpose_matrix33()
                src = normal_matrix.numbers
                dest = &instances_data[k, 16]
                for j in range(9):
                    dest[j] = src[j]
                dest = &instances_data[k, 25]
                for j in range(3):
                    dest[j] = material.albedo.numbers[j]
                    dest[j+4] = material.emissive.numbers[j]
                dest[3] = material.transparency
                dest[7] = material.ior
                dest[8] = material.metal
                dest[9] = material.roughness
                dest[10] = material.occlusion
                k += 1
        dispatch_instances(glctx, objects, shader, model, k, instances_data,
                           textures if shader_supports_textures else None, references, base_unit_id)
    if transparent_objects:
        render_target.depth_write(False)
        n = len(transparent_objects)
        transparent_objects.sort(key=fst)
        instances_data = view.array((n, 36), 4, 'f')
        k = 0
        for i, transparent_object in enumerate(transparent_objects):
            model = transparent_object[1]
            instance = transparent_object[2]
            material = instance.material
            src = instance.model_matrix.numbers
            dest = &instances_data[k, 0]
            for j in range(16):
                dest[j] = src[j]
            normal_matrix = instance.model_matrix.inverse_transpose_matrix33()
            src = normal_matrix.numbers
            dest = &instances_data[k, 16]
            for j in range(9):
                dest[j] = src[j]
            dest = &instances_data[k, 25]
            for j in range(3):
                dest[j] = material.albedo.numbers[j]
                dest[j+4] = material.emissive.numbers[j]
            dest[3] = material.transparency
            dest[7] = material.ior
            dest[8] = material.metal
            dest[9] = material.roughness
            dest[10] = material.occlusion
            k += 1
            if i == n-1 or (<tuple>transparent_objects[i+1])[1] is not model:
                dispatch_instances(glctx, objects, shader, model, k, instances_data,
                                   material.textures if shader_supports_textures else None, references, base_unit_id)
                k = 0
        if k:
            dispatch_instances(glctx, objects, shader, model, k, instances_data,
                               material.textures if shader_supports_textures else None, references, base_unit_id)
        render_target.depth_write(True)
    glctx.disable(flags)


cdef void dispatch_instances(glctx, dict objects, shader, Model model, int count, cython.float[:, :] instances_data,
                             Textures textures, dict references, int base_unit_id):
    vertex_buffer, index_buffer = model.get_buffers(glctx, objects)
    if vertex_buffer is None:
        return
    cdef dict unit_ids
    cdef list samplers = None
    cdef int unit_id
    if references is not None and textures is not None:
        unit_ids = {}
        samplers = []
        if (scene_node := references.get(textures.albedo_id)) is not None and hasattr(scene_node, 'texture') and scene_node.texture is not None:
            if textures.albedo_id in unit_ids:
                unit_id = unit_ids[textures.albedo_id]
            else:
                unit_id = base_unit_id
                unit_ids[textures.albedo_id] = unit_id
                sampler = glctx.sampler(texture=scene_node.texture, filter=(moderngl.LINEAR, moderngl.LINEAR))
                sampler.use(unit_id)
                samplers.append(sampler)
                base_unit_id += 1
            shader['use_albedo_texture'] = True
            shader['albedo_texture'] = unit_id
        if (scene_node := references.get(textures.metal_id)) is not None and hasattr(scene_node, 'texture') and scene_node.texture is not None:
            if textures.metal_id in unit_ids:
                unit_id = unit_ids[textures.metal_id]
            else:
                unit_id = base_unit_id
                unit_ids[textures.metal_id] = unit_id
                sampler = glctx.sampler(texture=scene_node.texture, filter=(moderngl.LINEAR, moderngl.LINEAR))
                sampler.use(unit_id)
                samplers.append(sampler)
                base_unit_id += 1
            shader['use_metal_texture'] = True
            shader['metal_texture'] = unit_id
        if (scene_node := references.get(textures.roughness_id)) is not None and hasattr(scene_node, 'texture') and scene_node.texture is not None:
            if textures.roughness_id in unit_ids:
                unit_id = unit_ids[textures.roughness_id]
            else:
                unit_id = base_unit_id
                unit_ids[textures.roughness_id] = unit_id
                sampler = glctx.sampler(texture=scene_node.texture, filter=(moderngl.LINEAR, moderngl.LINEAR))
                sampler.use(unit_id)
                samplers.append(sampler)
                base_unit_id += 1
            shader['use_roughness_texture'] = True
            shader['roughness_texture'] = unit_id
        if (scene_node := references.get(textures.occlusion_id)) is not None and hasattr(scene_node, 'texture') and scene_node.texture is not None:
            if textures.occlusion_id in unit_ids:
                unit_id = unit_ids[textures.occlusion_id]
            else:
                unit_id = base_unit_id
                unit_ids[textures.occlusion_id] = unit_id
                sampler = glctx.sampler(texture=scene_node.texture, filter=(moderngl.LINEAR, moderngl.LINEAR))
                sampler.use(unit_id)
                samplers.append(sampler)
                base_unit_id += 1
            shader['use_occlusion_texture'] = True
            shader['occlusion_texture'] = unit_id
        if (scene_node := references.get(textures.emissive_id)) is not None and hasattr(scene_node, 'texture') and scene_node.texture is not None:
            if textures.emissive_id in unit_ids:
                unit_id = unit_ids[textures.emissive_id]
            else:
                unit_id = base_unit_id
                unit_ids[textures.emissive_id] = unit_id
                sampler = glctx.sampler(texture=scene_node.texture, filter=(moderngl.LINEAR, moderngl.LINEAR))
                sampler.use(unit_id)
                samplers.append(sampler)
                base_unit_id += 1
            shader['use_emissive_texture'] = True
            shader['emissive_texture'] = unit_id
        if (scene_node := references.get(textures.transparency_id)) is not None and hasattr(scene_node, 'texture') and scene_node.texture is not None:
            if textures.transparency_id in unit_ids:
                unit_id = unit_ids[textures.transparency_id]
            else:
                unit_id = base_unit_id
                unit_ids[textures.transparency_id] = unit_id
                sampler = glctx.sampler(texture=scene_node.texture, filter=(moderngl.LINEAR, moderngl.LINEAR))
                sampler.use(unit_id)
                samplers.append(sampler)
                base_unit_id += 1
            shader['use_transparency_texture'] = True
            shader['transparency_texture'] = unit_id
    instances_buffer = glctx.buffer(instances_data)
    cdef list buffers = []
    cdef str format = '3f'
    cdef list names = ['model_position']
    if 'model_normal' in shader:
        format += ' 3f'
        names.append('model_normal')
    else:
        format += ' 3x4'
    if 'model_uv' in shader:
        format += ' 2f'
        names.append('model_uv')
    else:
        format += ' 2x4'
    buffers.append((vertex_buffer, format) + tuple(names))
    format = '16f'
    names = ['model_matrix']
    if 'model_normal_matrix' in shader:
        format += ' 9f'
        names.append('model_normal_matrix')
    else:
        format += ' 9x4'
    if 'material_albedo' in shader:
        format += ' 4f'
        names.append('material_albedo')
    else:
        format += ' 4x4'
    if 'material_emissive' in shader:
        format += ' 3f'
        names.append('material_emissive')
    else:
        format += ' 3x4'
    if 'material_properties' in shader:
        format += ' 4f'
        names.append('material_properties')
    else:
        format += ' 4x4'
    format += '/i'
    buffers.append((instances_buffer, format) + tuple(names))
    render_array = glctx.vertex_array(shader, buffers, index_buffer=index_buffer, mode=moderngl.TRIANGLES)
    render_array.render(instances=count)
    if samplers is not None:
        for sampler in samplers:
            sampler.clear()
        shader['use_albedo_texture'] = False
        shader['use_metal_texture'] = False
        shader['use_roughness_texture'] = False
        shader['use_occlusion_texture'] = False
        shader['use_emissive_texture'] = False
        shader['use_transparency_texture'] = False


cdef class RenderTarget:
    cdef int width
    cdef int height
    cdef int colorbits
    cdef int samples
    cdef object image_texture
    cdef object image_framebuffer
    cdef object color_renderbuffer
    cdef object depth_renderbuffer
    cdef object render_framebuffer

    @property
    def texture(self):
        return self.image_texture

    def release(self):
        self.width = 0
        self.height = 0
        self.colorbits = 0
        self.samples = 0
        self.image_texture = None
        self.image_framebuffer = None
        self.color_renderbuffer = None
        self.depth_renderbuffer = None
        self.render_framebuffer = None

    def prepare(self, glctx, Camera camera):
        if self.render_framebuffer is None or self.width != camera.width or self.height != camera.height \
                or self.colorbits != camera.colorbits or self.samples != camera.samples:
            self.width = camera.width
            self.height = camera.height
            self.colorbits = camera.colorbits
            self.samples = camera.samples
            format = COLOR_FORMATS[self.colorbits]
            self.image_texture = glctx.texture((self.width, self.height), 4, dtype=format.moderngl_dtype)
            self.depth_renderbuffer = glctx.depth_renderbuffer((self.width, self.height), samples=self.samples)
            if self.samples:
                self.color_renderbuffer = glctx.renderbuffer((self.width, self.height), 4, samples=self.samples, dtype=format.moderngl_dtype)
                self.render_framebuffer = glctx.framebuffer(color_attachments=(self.color_renderbuffer,), depth_attachment=self.depth_renderbuffer)
                self.image_framebuffer = glctx.framebuffer(self.image_texture)
                logger.debug("Created canvas3d {}x{} {}-bit render targets with {}x sampling", self.width, self.height, self.colorbits, self.samples)
            else:
                self.render_framebuffer = glctx.framebuffer(color_attachments=(self.image_texture,), depth_attachment=self.depth_renderbuffer)
                logger.debug("Created canvas3d {}x{} {}-bit render targets", self.width, self.height, self.colorbits)
        cdef Vector clear_color
        if camera.fog_max > camera.fog_min:
            clear_color = camera.fog_color
            if camera.monochrome:
                clear_color = clear_color.dot(Greyscale)
            clear_color = clear_color.mul(camera.tint).concat(true_)
        else:
            clear_color = Zero4
        self.render_framebuffer.use()
        self.render_framebuffer.clear(*tuple(clear_color))

    def finalize(self, glctx):
        if self.image_framebuffer is not None:
            glctx.copy_framebuffer(self.image_framebuffer, self.render_framebuffer)

    def depth_write(self, bint enabled):
        self.render_framebuffer.depth_mask = enabled


class Canvas3D(SceneNode):
    def __init__(self, glctx):
        super().__init__(glctx)
        self._primary_render_target = None
        self._secondary_render_targets = {}
        self._total_duration = 0
        self._total_count = 0

    @property
    def texture(self):
        return (<RenderTarget>self._primary_render_target).texture if self._primary_render_target is not None else None

    def release(self):
        self._render_target = None
        cdef RenderTarget render_target
        for render_target in self._secondary_render_targets.values():
            render_target.release()
        self._secondary_render_targets = {}

    def purge(self):
        logger.info("{} draw stats - {:d} x {:.1f}ms = {:.1f}s", self.name, self._total_count,
                    1e3 * self._total_duration / self._total_count, self._total_duration)
        self._total_duration = 0
        self._total_count = 0

    async def descend(self, engine, node, **kwargs):
        # A canvas3d is a leaf node from the perspective of the OpenGL world
        pass

    def render(self, Node node, dict references=None, **kwargs):
        self._total_duration -= system_clock()
        cdef int width = self.width, height = self.height
        cdef dict objects = self.glctx.extra.setdefault('canvas3d_objects', {})
        cdef Matrix44 transform_matrix = IdentityTransform
        cdef Camera default_camera = Camera.__new__(Camera)
        cdef int max_samples = self.glctx.info['GL_MAX_SAMPLES']
        default_camera.position = Vector((0, 0, width/2))
        default_camera.focus = Zero3
        default_camera.up = Vector((0, 1, 0))
        default_camera.fov = 0.25
        default_camera.fov_ref = 'horizontal'
        default_camera.monochrome = False
        default_camera.tint = One3
        default_camera.orthographic = False
        default_camera.ortho_width = width
        default_camera.near = 1
        default_camera.far = width
        default_camera.fog_min = 0
        default_camera.fog_max = 0
        default_camera.fog_color = Zero3
        default_camera.fog_curve = 1
        default_camera.width = width
        default_camera.height = height
        default_camera.colorbits = self.glctx.extra['colorbits']
        default_camera.samples = 0
        default_camera = default_camera.derive(node, transform_matrix, max_samples)
        cdef Material material = Material.__new__(Material)
        material.albedo = Zero3
        material.roughness = 1
        material.occlusion = 1
        material.emissive = Zero3
        cdef list render_groups = []
        cdef dict cameras = {}
        cameras[default_camera.id] = default_camera
        collect(node, transform_matrix, material, None, render_groups, default_camera, cameras, max_samples)
        cdef Camera primary_camera = cameras.get(node.get_str('camera_id', default_camera.id), default_camera)
        if self._primary_render_target is None:
            self._primary_render_target = RenderTarget.__new__(RenderTarget)
        cdef RenderTarget primary_render_target = self._primary_render_target
        primary_render_target.prepare(self.glctx, primary_camera)
        cdef RenderGroup render_group
        for render_group in render_groups:
            if render_group.instances:
                render(primary_render_target, render_group, primary_camera, self.glctx, objects, references)
        primary_render_target.finalize(self.glctx)
        cdef Camera camera
        cdef RenderTarget secondary_render_target
        if references is not None:
            for camera in cameras.values():
                if camera.secondary and camera.id:
                    if camera is not primary_camera:
                        secondary_render_target = self._secondary_render_targets.get(camera.id)
                        if secondary_render_target is None:
                            secondary_render_target = RenderTarget.__new__(RenderTarget)
                            self._secondary_render_targets[camera.id] = secondary_render_target
                        secondary_render_target.prepare(self.glctx, camera)
                        for render_group in render_groups:
                            if render_group.instances:
                                render(secondary_render_target, render_group, camera, self.glctx, objects, references)
                        secondary_render_target.finalize(self.glctx)
                        references[camera.id] = secondary_render_target
                    else:
                        references[camera.id] = primary_render_target
        self._total_duration += system_clock()
        self._total_count += 1


SCENE_NODE_CLASS = Canvas3D
