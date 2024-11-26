
"""
Flitter OpenGL 3D drawing canvas
"""

import cython
from cython cimport view
from loguru import logger
from mako.template import Template
import moderngl
import numpy as np

from libc.math cimport cos, log2, sqrt, tan, floor
from libc.stdint cimport int32_t, int64_t, uint64_t
from cpython.dict cimport PyDict_GetItem
from cpython.object cimport PyObject

from . import WindowNode
from ... import name_patch
from ...clock import system_clock
from .glsl import TemplateLoader
from ...model cimport Node, Vector, Matrix44, Matrix33, Quaternion, null_, true_, false_, HASH_START, HASH_UPDATE, HASH_STRING
from .models cimport Model, DefaultSegments, DefaultSnapAngle
from .target import RenderTarget, COLOR_FORMATS
from ...plugins import get_plugin


logger = name_patch(logger, __name__)

cdef Vector Zero3 = Vector((0, 0, 0))
cdef Vector Zero4 = Vector((0, 0, 0, 0))
cdef Vector Greyscale = Vector((0.299, 0.587, 0.114))
cdef Vector MinusOne3 = Vector((-1, -1, -1))
cdef Vector One3 = Vector((1, 1, 1))
cdef Vector Xaxis = Vector((1, 0, 0))
cdef Vector Yaxis = Vector((0, 1, 0))
cdef Vector Zaxis = Vector((0, 0, 1))
cdef Vector DefaultFalloff = Vector((0, 0, 1, 0))
cdef Matrix44 IdentityTransform = Matrix44._identity()
cdef int DEFAULT_MAX_LIGHTS = 50
cdef double Pi = 3.141592653589793115997963468544185161590576171875
cdef set MaterialAttributes = {'color', 'metal', 'roughness', 'ao', 'emissive', 'transparency',
                               'color_id', 'metal_id', 'roughness_id', 'ao_id', 'emissive_id', 'transparency_id',
                               'texture_id', 'metal_texture_id', 'roughness_texture_id', 'ao_texture_id',
                               'emissive_texture_id', 'transparency_texture_id'}
cdef set GroupAttributes = set(MaterialAttributes)
GroupAttributes.update(('translate', 'scale', 'rotate', 'rotate_q', 'rotate_x', 'rotate_y', 'rotate_z', 'shear_x', 'shear_y', 'shear_z'))
GroupAttributes.update(('max_lights', 'depth_sort', 'depth_test', 'face_cull', 'cull_face', 'composite', 'vertex', 'fragment'))

cdef object StandardVertexTemplate = TemplateLoader.get_template("standard_lighting.vert")
cdef object StandardFragmentTemplate = TemplateLoader.get_template("standard_lighting.frag")
cdef object BackfaceFragmentTemplate = TemplateLoader.get_template("backface_lighting.frag")


@cython.cdivision(True)
@cython.boundscheck(False)
cdef bint set_uniform_vector(uniform, Vector vector):
    cdef str fmt = uniform.fmt
    cdef str dtype = fmt[-1]
    cdef int64_t n=uniform.array_length, m=uniform.dimension, o=n*m, p=vector.length, i, j, k
    if vector.numbers == NULL or (p != o and p != m and p != 1):
        return False
    cdef float[:] float_values
    if dtype == 'f':
        float_values = view.array((o,), 4, 'f')
        i = 0
        for j in range(n):
            for k in range(m):
                float_values[i] = <float>vector.numbers[i % p]
                i += 1
        uniform.write(float_values)
        return True
    cdef double[:] double_values
    if dtype == 'd':
        double_values = view.array((o,), 8, 'f')
        i = 0
        for j in range(n):
            for k in range(m):
                double_values[i] = vector.numbers[i % p]
                i += 1
        uniform.write(double_values)
        return True
    cdef int32_t[:] int32_values
    if dtype == 'i':
        int32_values = view.array((o,), 4, 'i')
        i = 0
        for j in range(n):
            for k in range(m):
                int32_values[i] = <int32_t>floor(vector.numbers[i % p])
                i += 1
        uniform.write(int32_values)
        return True
    cdef int64_t[:] int64_values
    if dtype == 'I':
        int64_values = view.array((o,), 8, 'i')
        i = 0
        for j in range(n):
            for k in range(m):
                int64_values[i] = <int64_t>floor(vector.numbers[i % p])
                i += 1
        uniform.write(int64_values)
        return True
    return False


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
    cdef int64_t _hash
    cdef str albedo_id
    cdef str metal_id
    cdef str roughness_id
    cdef str ao_id
    cdef str emissive_id
    cdef str transparency_id
    cdef Vector border_color
    cdef bint repeat_x
    cdef bint repeat_y

    def __eq__(self, Textures other):
        return other.albedo_id == self.albedo_id and \
               other.metal_id == self.metal_id and \
               other.roughness_id == self.roughness_id and \
               other.ao_id == self.ao_id and \
               other.emissive_id == self.emissive_id and \
               other.transparency_id == self.transparency_id and \
               other.border_color == self.border_color and \
               other.repeat_x == self.repeat_x and \
               other.repeat_y == self.repeat_y

    def __hash__(self):
        if self._hash:
            return self._hash
        cdef uint64_t _hash = HASH_START
        _hash = HASH_UPDATE(_hash, HASH_STRING(self.albedo_id) if self.albedo_id is not None else 0)
        _hash = HASH_UPDATE(_hash, HASH_STRING(self.metal_id) if self.metal_id is not None else 0)
        _hash = HASH_UPDATE(_hash, HASH_STRING(self.roughness_id) if self.roughness_id is not None else 0)
        _hash = HASH_UPDATE(_hash, HASH_STRING(self.ao_id) if self.ao_id is not None else 0)
        _hash = HASH_UPDATE(_hash, HASH_STRING(self.emissive_id) if self.emissive_id is not None else 0)
        _hash = HASH_UPDATE(_hash, HASH_STRING(self.transparency_id) if self.transparency_id is not None else 0)
        _hash = HASH_UPDATE(_hash, <uint64_t>self.border_color.hash(False) if self.border_color is not None else 0)
        _hash = HASH_UPDATE(_hash, <uint64_t>self.repeat_x)
        _hash = HASH_UPDATE(_hash, <uint64_t>self.repeat_y)
        self._hash = _hash
        return _hash


cdef class Material:
    cdef Vector albedo
    cdef double ior
    cdef double metal
    cdef double roughness
    cdef double ao
    cdef Vector emissive
    cdef double transparency
    cdef double translucency
    cdef Textures textures

    cdef Material update(Material self, Node node):
        if node._attributes is None:
            return self
        cdef Material material = Material.__new__(Material)
        cdef Vector border_color, repeat
        material.albedo = node.get_fvec('color', 3, self.albedo)
        material.ior = max(1, node.get_float('ior', self.ior))
        material.roughness = min(max(1e-6, node.get_float('roughness', 1)), self.roughness)
        material.metal = min(max(0, node.get_float('metal', self.metal)), 1)
        material.ao = min(max(0, node.get_float('ao', self.ao)), 1)
        material.emissive = node.get_fvec('emissive', 3, self.emissive)
        material.transparency = min(max(0, node.get_float('transparency', self.transparency)), 1)
        material.translucency = max(0, node.get_float('translucency', self.translucency))
        albedo_id = node.get_str('color_id', node.get_str('texture_id', None))
        metal_id = node.get_str('metal_id', node.get_str('metal_texture_id', None))
        roughness_id = node.get_str('roughness_id', node.get_str('roughness_texture_id', None))
        ao_id = node.get_str('ao_id', node.get_str('ao_texture_id', None))
        emissive_id = node.get_str('emissive_id', node.get_str('emissive_texture_id', None))
        transparency_id = node.get_str('transparency_id', node.get_str('transparency_texture_id', None))
        border_color = node.get_fvec('border', 3, None)
        if border_color is not None:
            border_color = border_color.concat(true_)
        else:
            border_color = node.get_fvec('border', 4, None)
        repeat = node.get_fvec('repeat', 2, None)
        cdef Textures textures
        if albedo_id is not None or metal_id is not None or roughness_id is not None or ao_id is not None \
                or emissive_id is not None or transparency_id is not None or border_color is not None or repeat is not None:
            textures = Textures.__new__(Textures)
            if self.textures is not None:
                textures.albedo_id = albedo_id or self.textures.albedo_id
                textures.metal_id = metal_id or self.textures.metal_id
                textures.roughness_id = roughness_id or self.textures.roughness_id
                textures.ao_id = ao_id or self.textures.ao_id
                textures.emissive_id = emissive_id or self.textures.emissive_id
                textures.transparency_id = transparency_id or self.textures.transparency_id
                textures.border_color = border_color or self.textures.border_color
                textures.repeat_x = repeat.numbers[0] != 0 if repeat is not None else self.textures.repeat_x
                textures.repeat_y = repeat.numbers[1] != 0 if repeat is not None else self.textures.repeat_y
            else:
                textures.albedo_id = albedo_id
                textures.metal_id = metal_id
                textures.roughness_id = roughness_id
                textures.ao_id = ao_id
                textures.emissive_id = emissive_id
                textures.transparency_id = transparency_id
                textures.border_color = border_color
                textures.repeat_x = repeat.numbers[0] != 0 if repeat is not None else False
                textures.repeat_y = repeat.numbers[1] != 0 if repeat is not None else False
            material.textures = textures
        else:
            material.textures = self.textures
        return material


cdef class Instance:
    cdef Matrix44 model_matrix
    cdef Material material


cdef class RenderGroup:
    cdef RenderGroup parent_group
    cdef int64_t max_lights
    cdef list lights
    cdef dict instances
    cdef bint depth_sort
    cdef bint depth_test
    cdef bint face_cull
    cdef bint cull_front_face
    cdef str composite
    cdef object vertex_shader_template
    cdef object fragment_shader_template
    cdef dict names
    cdef int64_t cached_nlights
    cdef object cached_lights_buffer

    cdef void set_blend(self, glctx):
        glctx.blend_equation = moderngl.FUNC_ADD
        if self.composite is 'source':
            glctx.blend_func = moderngl.ONE, moderngl.ZERO
        elif self.composite is 'dest':
            glctx.blend_func = moderngl.ZERO, moderngl.ONE
        elif self.composite is 'dest_over':
            glctx.blend_func = moderngl.ONE_MINUS_DST_ALPHA, moderngl.ONE
        elif self.composite is 'add':
            glctx.blend_func = moderngl.ONE, moderngl.ONE
        elif self.composite is 'subtract':
            glctx.blend_equation = moderngl.FUNC_SUBTRACT
            glctx.blend_func = moderngl.ONE, moderngl.ONE
        elif self.composite is 'lighten':
            glctx.blend_equation = moderngl.MAX
            glctx.blend_func = moderngl.ONE, moderngl.ONE
        elif self.composite is 'darken':
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
    cdef Matrix44 view_matrix
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
        camera.fov_ref = node.get_str('fov_ref', self.fov_ref)
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
            if camera.fov_ref is 'diagonal':
                diagonal_ratio = sqrt(1 + aspect_ratio*aspect_ratio)
                projection_matrix = Matrix44._project(aspect_ratio*gradient/diagonal_ratio, gradient/diagonal_ratio, camera.near, camera.far)  # diagonal
            elif camera.fov_ref is 'vertical':
                projection_matrix = Matrix44._project(aspect_ratio*gradient, gradient, camera.near, camera.far)  # vertical
            elif camera.fov_ref is 'wide':
                if aspect_ratio > 1:  # widescreen
                    projection_matrix = Matrix44._project(gradient, gradient/aspect_ratio, camera.near, camera.far)  # horizontal
                else:
                    projection_matrix = Matrix44._project(aspect_ratio*gradient, gradient, camera.near, camera.far)  # vertical
            elif camera.fov_ref is 'narrow':
                if aspect_ratio > 1:  # widescreen
                    projection_matrix = Matrix44._project(aspect_ratio*gradient, gradient, camera.near, camera.far)  # vertical
                else:
                    projection_matrix = Matrix44._project(gradient, gradient/aspect_ratio, camera.near, camera.far)  # horizontal
            else:
                projection_matrix = Matrix44._project(gradient, gradient/aspect_ratio, camera.near, camera.far)  # horizontal
        camera.view_matrix = Matrix44._look(camera.position, camera.focus, camera.up)
        if camera.view_matrix is None:
            camera.view_matrix = Matrix44._identity()
        camera.pv_matrix = projection_matrix.mmul(camera.view_matrix)
        return camera

    cdef Vector get_clear_color(self):
        cdef Vector color
        if self.fog_max > self.fog_min:
            color = self.fog_color
            if self.monochrome:
                color = color.dot(Greyscale)
            color = color.mul(self.tint).concat(true_)
        else:
            color = Zero4
        return color


cdef Matrix44 update_transform_matrix(Node node, Matrix44 transform_matrix):
    cdef str attribute
    cdef Vector vector
    if node._attributes:
        transform_matrix = transform_matrix.copy()
        for attribute, vector in node._attributes.items():
            if attribute is 'translate':
                transform_matrix.immul(Matrix44._translate(vector))
            elif attribute is 'scale':
                transform_matrix.immul(Matrix44._scale(vector))
            elif attribute is 'rotate':
                if vector.length == 4:
                    transform_matrix.immul(Quaternion._coerce(vector).matrix44())
                else:
                    transform_matrix.immul(Matrix44._rotate(vector))
            elif attribute is 'rotate_x':
                if vector.numbers != NULL and vector.length == 1:
                    transform_matrix.immul(Matrix44._rotate_x(vector.numbers[0]))
            elif attribute is 'rotate_y':
                if vector.numbers != NULL and vector.length == 1:
                    transform_matrix.immul(Matrix44._rotate_y(vector.numbers[0]))
            elif attribute is 'rotate_z':
                if vector.numbers != NULL and vector.length == 1:
                    transform_matrix.immul(Matrix44._rotate_z(vector.numbers[0]))
            elif attribute is 'shear_x':
                transform_matrix.immul(Matrix44._shear_x(vector))
            elif attribute is 'shear_y':
                transform_matrix.immul(Matrix44._shear_y(vector))
            elif attribute is 'shear_z':
                transform_matrix.immul(Matrix44._shear_z(vector))
            elif attribute is 'matrix' and vector.length == 16:
                transform_matrix.immul(Matrix44(vector))
    return transform_matrix


cdef Matrix44 instance_start_end_matrix(Vector start, Vector end, double radius):
    cdef Vector direction = end.sub(start)
    cdef double length = sqrt(direction.squared_sum())
    if length == 0 or radius <= 0:
        return None
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
    cdef Matrix44 matrix = Matrix44._translate(middle)
    matrix.immul(Quaternion._between(Zaxis, direction).matrix44())
    matrix.immul(Matrix44._scale(size))
    return matrix


cdef inline Matrix44 get_model_transform(Node node, Matrix44 transform_matrix):
    transform_matrix = transform_matrix.copy()
    cdef Vector end=node.get_fvec('end', 3, None)
    if end is not None:
        transform_matrix.immul(instance_start_end_matrix(node.get_fvec('start', 3, Zero3), end, node.get_float('radius', 1)))
    else:
        transform_matrix.immul(Matrix44._translate(node.get_fvec('position', 3, None)))
        transform_matrix.immul(Matrix44._rotate(node.get_fvec('rotation', 3, None)))
        transform_matrix.immul(Matrix44._scale(node.get_fvec('size', 3, None)))
    return transform_matrix


cdef Model get_model(Node node, bint top):
    cdef Node child
    cdef Model model = None
    cdef Vector origin, normal, function, minimum, maximum
    cdef double snap_angle, resolution
    cdef str mapping
    if node.kind is 'box':
        model = Model._box(node.get_str('uv_map', 'standard'))
    elif node.kind is 'sphere':
        model = Model._sphere(node.get_int('segments', DefaultSegments))
    elif node.kind is 'cylinder':
        model = Model._cylinder(node.get_int('segments', DefaultSegments))
    elif node.kind is 'cone':
        model = Model._cone(node.get_int('segments', DefaultSegments))
    elif node.kind is 'model':
        model = Model._external(node.get_str('filename', None))
        if model is not None and node.get_bool('repair', False):
            model = model.repair()
    elif node.kind is 'sdf':
        minimum = node.get_fvec('minimum', 3, MinusOne3)
        maximum = node.get_fvec('maximum', 3, One3)
        resolution = node.get_float('resolution', 0.1)
        if 'function' in node and (function := node['function']) and function.length == 1 and \
                function.objects is not None and callable(f := function.objects[0]):
            model = Model._sdf(f, None, minimum, maximum, resolution)
        else:
            model = Model._boolean('union', [get_model(child, False) for child in node._children], 0, 0, 0)
            model = Model._sdf(None, model, minimum, maximum, resolution)
    elif not top and node.kind is 'transform':
        transform_matrix = update_transform_matrix(node, IdentityTransform)
        model = Model._boolean('union', [get_model(child, False)._transform(transform_matrix) for child in node._children], 0, 0, 0)
    elif node.kind in ('union', 'intersect', 'difference'):
        model = Model._boolean(node.kind, [get_model(child, False) for child in node._children],
                               node.get_float('smooth', 0),
                               node.get_float('fillet', 0),
                               node.get_float('chamfer', 0))
    elif node.kind is 'trim' or node.kind is 'slice':
        normal = node.get_fvec('normal', 3, null_)
        origin = node.get_fvec('origin', 3, Zero3)
        model = Model._boolean('union', [get_model(child, False) for child in node._children], 0, 0, 0)
        if model is not None and normal.as_bool():
            model = model._trim(origin, normal)
    elif (cls := get_plugin('flitter.render.window.models', node.kind)) is not None:
        model = cls.from_node(node)
    if model is not None:
        if top:
            if node.get_bool('flat', False):
                model = model.flatten()
            elif (snap_angle := node.get_float('snap_edges', DefaultSnapAngle if model.is_smooth() else 0.5)) < 0.5:
                model = model._snap_edges(snap_angle, node.get_float('minimum_area', 0))
            if node.get_bool('invert', False):
                model = model.invert()
            if (mapping := node.get_str('uv_remap', None)) is not None:
                model = model._uv_remap(mapping)
        elif node.kind is not 'transform':
            if (transform_matrix := get_model_transform(node, IdentityTransform)) is not IdentityTransform:
                model = model._transform(transform_matrix)
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
    cdef PyObject* instances

    if node.kind is 'transform':
        transform_matrix = update_transform_matrix(node, transform_matrix)
        for child in node._children:
            collect(child, transform_matrix, material, render_group, render_groups, default_camera, cameras, max_samples)

    elif node.kind is 'material':
        material = material.update(node)
        for child in node._children:
            collect(child, transform_matrix, material, render_group, render_groups, default_camera, cameras, max_samples)

    elif node.kind is 'light':
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

    elif node.kind is 'camera':
        if (camera_id := node.get_str('id', None)) is not None:
            cameras[camera_id] = default_camera.derive(node, transform_matrix, max_samples)

    elif node.kind is 'group' or node.kind is 'canvas3d':
        transform_matrix = update_transform_matrix(node, transform_matrix)
        material = material.update(node)
        new_render_group = RenderGroup.__new__(RenderGroup)
        new_render_group.parent_group = render_group
        new_render_group.max_lights = max(1, node.get_int('max_lights', DEFAULT_MAX_LIGHTS if render_group is None else render_group.max_lights))
        new_render_group.lights = []
        new_render_group.instances = {}
        new_render_group.depth_sort = node.get_bool('depth_sort', True if render_group is None else render_group.depth_sort)
        new_render_group.depth_test = node.get_bool('depth_test', True if render_group is None else render_group.depth_test)
        new_render_group.face_cull = node.get_bool('face_cull', True if render_group is None else render_group.face_cull)
        if 'cull_face' in node:
            new_render_group.cull_front_face = node.get_str('cull_face', 'back') is 'front'
        else:
            new_render_group.cull_front_face = False if render_group is None else render_group.cull_front_face
        new_render_group.composite = node.get_str('composite', 'over' if render_group is None else render_group.composite)
        vertex_shader = node.get_str('vertex', None)
        fragment_shader = node.get_str('fragment', None)
        new_render_group.vertex_shader_template = Template(vertex_shader, lookup=TemplateLoader) if vertex_shader is not None else None
        new_render_group.fragment_shader_template = Template(fragment_shader, lookup=TemplateLoader) if fragment_shader is not None else None
        new_render_group.names = {}
        if node._attributes:
            for name, value in node._attributes.items():
                if name not in GroupAttributes:
                    new_render_group.names[name] = value
        render_groups.append(new_render_group)
        for child in node._children:
            collect(child, transform_matrix, material, new_render_group, render_groups, default_camera, cameras, max_samples)

    elif (model := get_model(node, True)) is not None:
        instance = Instance.__new__(Instance)
        instance.model_matrix = get_model_transform(node, transform_matrix)
        instance.material = material
        if node._attributes is not None:
            for attr in node._attributes:
                if attr in MaterialAttributes:
                    instance.material = material.update(node)
                    break
        model_textures = (model, instance.material.textures)
        if (instances := PyDict_GetItem(render_group.instances, model_textures)) != NULL:
            (<list>instances).append(instance)
        else:
            render_group.instances[model_textures] = [instance]


def fst(tuple ab):
    return ab[0]


cdef object get_shader(object glctx, dict shaders, dict names, object vertex_shader_template, object fragment_shader_template):
    cdef str vertex_shader = vertex_shader_template.render(**names)
    cdef str fragment_shader = fragment_shader_template.render(**names)
    cdef tuple source_pair = (vertex_shader, fragment_shader)
    cdef list error
    cdef str source, line
    cdef int64_t i
    shader = shaders.get(source_pair, False)
    if shader is not False:
        return shader
    try:
        shader = glctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    except Exception as exc:
        error = str(exc).strip().split('\n')
        logger.error("Group shader program compile failed: {}", error[-1])
        logger.trace("Full error: \n{}", str(exc))
        source = fragment_shader if 'fragment_shader' in error else vertex_shader
        source = '\n'.join(f'{i+1:3d}|{line}' for i, line in enumerate(source.split('\n')))
        logger.trace("Failing source:\n{}", source)
        shader = None
    else:
        logger.debug("Compiled group shader program")
    shaders[source_pair] = shader
    return shader


cdef inline double nearest_corner(Vector bounds, Matrix44 view_model_matrix):
    cdef int64_t i, j, k
    cdef double d, min_d
    for i in range(8):
        d = -view_model_matrix.numbers[14]
        for j in range(3):
            k = (i >> j) & 1
            d -= view_model_matrix.numbers[4*j + 2] * bounds.numbers[k*3 + j]
        if i == 0 or d < min_d:
            min_d = d
    return min_d


cdef void render(render_target, RenderGroup render_group, Camera camera, glctx, dict objects, dict references):
    cdef list instances
    cdef cython.float[:, :] instances_data, lights_data
    cdef Material material
    cdef Light light
    cdef Model model
    cdef Textures textures
    cdef int i, j, k, n, nlights
    cdef double* src
    cdef float* dest
    cdef Instance instance
    cdef Matrix33 normal_matrix
    cdef Vector bounds
    cdef bint has_transparency_texture, depth_write
    cdef tuple transparent_object, translucent_object
    cdef list transparent_objects = []
    cdef list translucent_objects = []
    cdef bint depth_sorted
    cdef double[:] zs
    cdef int64_t[:] indices
    cdef dict shaders = objects.setdefault('canvas3d_shaders', {})
    cdef dict names = render_group.names.copy()
    names.update({'max_lights': render_group.max_lights, 'Ambient': LightType.Ambient, 'Directional': LightType.Directional,
                  'Point': LightType.Point, 'Spot': LightType.Spot, 'Line': LightType.Line, 'HEADER': glctx.extra['HEADER']})
    shader = None
    if render_group.vertex_shader_template is not None or render_group.fragment_shader_template is not None:
        shader = get_shader(glctx, shaders, names,
                            render_group.vertex_shader_template or StandardVertexTemplate,
                            render_group.fragment_shader_template or StandardFragmentTemplate)
    if shader is None:
        shader = get_shader(glctx, shaders, names, StandardVertexTemplate, StandardFragmentTemplate)
    cdef str name
    cdef Vector value
    glctx.extra['zero'].use(0)
    cdef dict texture_unit_ids = {None: 0}
    cdef list samplers = []
    cdef int unit_id
    for name, value in render_group.names.items():
        if (member := shader.get(name, None)) is not None and isinstance(member, moderngl.Uniform):
            if member.fmt == '1i' and (ref_id := value.as_string()) and (ref := references.get(ref_id)) is not None \
                    and hasattr(ref, 'texture') and (texture := ref.texture) is not None:
                if ref_id in texture_unit_ids:
                    unit_id = texture_unit_ids[ref_id]
                else:
                    unit_id = len(texture_unit_ids)
                    texture_unit_ids[ref_id] = unit_id
                    sampler = glctx.sampler(texture=texture, filter=(moderngl.LINEAR, moderngl.LINEAR))
                    sampler.use(unit_id)
                    samplers.append(sampler)
                    member.value = unit_id
            elif not set_uniform_vector(member, value):
                set_uniform_vector(member, false_)
    set_uniform_vector(shader['pv_matrix'], camera.pv_matrix)
    shader['orthographic'] = camera.orthographic
    if (member := shader.get('monochrome', None)) is not None:
        member.value = camera.monochrome
    if (member := shader.get('tint', None)) is not None:
        member.value = camera.tint
    set_uniform_vector(shader['view_position'], camera.position)
    set_uniform_vector(shader['view_focus'], camera.focus)
    if (member := shader.get('fog_color', None)) is not None:
        set_uniform_vector(member, camera.fog_color)
        shader['fog_min'] = camera.fog_min
        shader['fog_max'] = camera.fog_max
        shader['fog_curve'] = camera.fog_curve
    cdef RenderGroup group
    if (member := shader.get('nlights', None)) is not None:
        if render_group.cached_lights_buffer is not None:
            lights_buffer = render_group.cached_lights_buffer
            nlights = render_group.cached_nlights
        else:
            lights_data = view.array((render_group.max_lights, 16), 4, 'f')
            nlights = 0
            group = render_group
            while group is not None:
                for light in group.lights:
                    if nlights == render_group.max_lights:
                        break
                    dest = &lights_data[nlights, 0]
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
                    nlights += 1
                group = group.parent_group
            lights_buffer = glctx.buffer(lights_data)
            render_group.cached_lights_buffer = lights_buffer
            render_group.cached_nlights = nlights
        lights_buffer.bind_to_uniform_block(1)
        shader['lights_data'].binding = 1
        member.value = nlights
    if (backface_data := shader.get('backface_data', None)) is not None:
        backface_data.value = 0
    cdef int flags = moderngl.BLEND
    if render_group.depth_test:
        flags |= moderngl.DEPTH_TEST
    if render_group.face_cull:
        flags |= moderngl.CULL_FACE
        glctx.cull_face = 'front' if render_group.cull_front_face else 'back'
    glctx.enable(flags)
    render_group.set_blend(glctx)
    for (model, textures), instances in render_group.instances.items():
        model.check_for_changes()
        bounds = model.get_bounds()
        if bounds.length != 6 or bounds.numbers == NULL:
            continue
        has_transparency_texture = textures is not None and textures.transparency_id is not None
        n = len(instances)
        instances_data = view.array((n, 37), 4, 'f')
        k = 0
        depth_sorted = render_group.depth_sort and render_group.depth_test
        if depth_sorted:
            zs_array = np.empty(n)
            zs = zs_array
            for i, instance in enumerate(instances):
                zs[i] = nearest_corner(bounds, camera.view_matrix.mmul(instance.model_matrix))
            indices = zs_array.argsort().astype('int64')
        else:
            indices = np.arange(n, dtype='int64')
        for i in indices:
            instance = instances[i]
            material = instance.material
            if material.translucency > 0:
                translucent_objects.append((zs[i] if depth_sorted else 0, model, instance))
                transparent_objects.append((-zs[i] if depth_sorted else 0, model, instance))
            elif (material.transparency > 0 or has_transparency_texture):
                transparent_objects.append((-zs[i] if depth_sorted else 0, model, instance))
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
                dest[7] = material.translucency
                dest[8] = material.ior
                dest[9] = material.metal
                dest[10] = material.roughness
                dest[11] = material.ao
                k += 1
        dispatch_instances(glctx, objects, shader, model, k, instances_data, textures, references, texture_unit_ids)

    cdef object backface_target = None
    if translucent_objects:
        if backface_data is not None:
            backface_shader = get_shader(glctx, shaders, names, StandardVertexTemplate, BackfaceFragmentTemplate)
            set_uniform_vector(backface_shader['pv_matrix'], camera.pv_matrix)
            backface_shader['orthographic'] = camera.orthographic
            set_uniform_vector(backface_shader['view_position'], camera.position)
            set_uniform_vector(backface_shader['view_focus'], camera.focus)
            backface_shader['nlights'] = nlights
            backface_shader['lights_data'].binding = 1
            glctx.disable(flags)
            glctx.enable(moderngl.BLEND | moderngl.DEPTH_TEST | moderngl.CULL_FACE)
            glctx.cull_face = 'front'
            glctx.blend_equation = moderngl.FUNC_ADD
            glctx.blend_func = moderngl.ONE, moderngl.ZERO
            backface_target = RenderTarget.get(glctx, camera.width, camera.height, 16, has_depth=True)
            backface_target.use()
            backface_target.clear(Zero4)
            n = len(translucent_objects)
            if render_group.depth_sort and render_group.depth_test:
                translucent_objects.sort(key=fst)
            instances_data = view.array((n, 37), 4, 'f')
            k = 0
            for i, translucent_object in enumerate(translucent_objects):
                model = translucent_object[1]
                instance = translucent_object[2]
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
                dest[7] = material.translucency
                dest[8] = material.ior
                dest[9] = material.metal
                dest[10] = material.roughness
                dest[11] = material.ao
                k += 1
                if i == n-1 or (<tuple>translucent_objects[i+1])[1] is not model:
                    dispatch_instances(glctx, objects, backface_shader, model, k, instances_data, material.textures, references, texture_unit_ids)
                    k = 0
            glctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST | moderngl.CULL_FACE)
            glctx.enable(flags)
            glctx.cull_face = 'front' if render_group.cull_front_face else 'back'
            render_group.set_blend(glctx)
            backface_target.finish()
            render_target.use()
            unit_id = len(texture_unit_ids)
            texture_unit_ids['_backface_data'] = unit_id
            sampler = glctx.sampler(texture=backface_target.texture, filter=(moderngl.NEAREST, moderngl.NEAREST))
            sampler.use(unit_id)
            backface_data.value = unit_id
            samplers.append(sampler)
        else:
            backface_data.value = 0
    elif backface_data is not None:
        backface_data.value = 0

    if transparent_objects:
        n = len(transparent_objects)
        if render_group.depth_sort and render_group.depth_test:
            transparent_objects.sort(key=fst)
        instances_data = view.array((n, 37), 4, 'f')
        i = k = 0
        transparent_object = transparent_objects[0]
        while i < n:
            model = transparent_object[1]
            instance = transparent_object[2]
            material = instance.material
            depth_write = material.translucency > 0
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
            dest[7] = material.translucency
            dest[8] = material.ior
            dest[9] = material.metal
            dest[10] = material.roughness
            dest[11] = material.ao
            k += 1
            i += 1
            transparent_object = transparent_objects[i] if i < n else None
            if i == n or transparent_object[1] is not model or (<Instance>transparent_object[2]).material.textures != material.textures \
                    or ((<Instance>transparent_object[2]).material.translucency > 0) != depth_write:
                render_target.depth_write(depth_write)
                dispatch_instances(glctx, objects, shader, model, k, instances_data, material.textures, references, texture_unit_ids)
                k = 0
        render_target.depth_write(True)

    if backface_target is not None:
        backface_target.release()
    glctx.disable(flags)
    for sampler in samplers:
        sampler.clear()


cdef int linear_sampler(glctx, str texture_id, dict sampler_args, dict references, list samplers, dict texture_unit_ids):
    cdef int unit_id = 0
    if texture_id:
        if texture_id in texture_unit_ids:
            unit_id = texture_unit_ids[texture_id]
        elif (ref := references.get(texture_id)) is not None and hasattr(ref, 'texture') and (texture := ref.texture) is not None:
            unit_id = len(texture_unit_ids)
            texture_unit_ids[texture_id] = unit_id
            sampler = glctx.sampler(texture=texture, **sampler_args)
            sampler.use(unit_id)
            samplers.append(sampler)
    return unit_id


cdef list configure_textures(glctx, shader, Textures textures, dict references, dict texture_unit_ids):
    cdef list samplers = []
    cdef dict sampler_args
    cdef int unit_id
    if textures is not None:
        sampler_args = {'filter': (moderngl.LINEAR, moderngl.LINEAR)}
        if textures.border_color is not None:
            sampler_args['border_color'] = tuple(textures.border_color)
        else:
            sampler_args['repeat_x'] = textures.repeat_x
            sampler_args['repeat_y'] = textures.repeat_y
        texture_unit_ids = texture_unit_ids.copy()
        if (texture_uniform := shader.get('albedo_texture', None)) is not None:
            unit_id = linear_sampler(glctx, textures.albedo_id, sampler_args, references, samplers, texture_unit_ids)
            if (use_texture_uniform := shader.get('use_albedo_texture', None)) is not None:
                use_texture_uniform.value = unit_id != 0
                if unit_id:
                    texture_uniform.value = unit_id
            else:
                texture_uniform.value = unit_id
        if (texture_uniform := shader.get('metal_texture', None)) is not None:
            unit_id = linear_sampler(glctx, textures.metal_id, sampler_args, references, samplers, texture_unit_ids)
            if (use_texture_uniform := shader.get('use_metal_texture', None)) is not None:
                use_texture_uniform.value = unit_id != 0
                if unit_id:
                    texture_uniform.value = unit_id
            else:
                texture_uniform.value = unit_id
        if (texture_uniform := shader.get('roughness_texture', None)) is not None:
            unit_id = linear_sampler(glctx, textures.roughness_id, sampler_args, references, samplers, texture_unit_ids)
            if (use_texture_uniform := shader.get('use_roughness_texture', None)) is not None:
                use_texture_uniform.value = unit_id != 0
                if unit_id:
                    texture_uniform.value = unit_id
            else:
                texture_uniform.value = unit_id
        if (texture_uniform := shader.get('ao_texture', None)) is not None:
            unit_id = linear_sampler(glctx, textures.ao_id, sampler_args, references, samplers, texture_unit_ids)
            if (use_texture_uniform := shader.get('use_ao_texture', None)) is not None:
                use_texture_uniform.value = unit_id != 0
                if unit_id:
                    texture_uniform.value = unit_id
            else:
                texture_uniform.value = unit_id
        if (texture_uniform := shader.get('emissive_texture', None)) is not None:
            unit_id = linear_sampler(glctx, textures.emissive_id, sampler_args, references, samplers, texture_unit_ids)
            if (use_texture_uniform := shader.get('use_emissive_texture', None)) is not None:
                use_texture_uniform.value = unit_id != 0
                if unit_id:
                    texture_uniform.value = unit_id
            else:
                texture_uniform.value = unit_id
        if (texture_uniform := shader.get('transparency_texture', None)) is not None:
            unit_id = linear_sampler(glctx, textures.transparency_id, sampler_args, references, samplers, texture_unit_ids)
            if (use_texture_uniform := shader.get('use_transparency_texture', None)) is not None:
                use_texture_uniform.value = unit_id != 0
                if unit_id:
                    texture_uniform.value = unit_id
            else:
                texture_uniform.value = unit_id
    else:
        if (texture_uniform := shader.get('albedo_texture', None)) is not None:
            if (use_texture_uniform := shader.get('use_albedo_texture', None)) is not None:
                use_texture_uniform.value = False
            else:
                texture_uniform.value = 0
        if (texture_uniform := shader.get('metal_texture', None)) is not None:
            if (use_texture_uniform := shader.get('use_metal_texture', None)) is not None:
                use_texture_uniform.value = False
            else:
                texture_uniform.value = 0
        if (texture_uniform := shader.get('roughness_texture', None)) is not None:
            if (use_texture_uniform := shader.get('use_roughness_texture', None)) is not None:
                use_texture_uniform.value = False
            else:
                texture_uniform.value = 0
        if (texture_uniform := shader.get('ao_texture', None)) is not None:
            if (use_texture_uniform := shader.get('use_ao_texture', None)) is not None:
                use_texture_uniform.value = False
            else:
                texture_uniform.value = 0
        if (texture_uniform := shader.get('emissive_texture', None)) is not None:
            if (use_texture_uniform := shader.get('use_emissive_texture', None)) is not None:
                use_texture_uniform.value = False
            else:
                texture_uniform.value = 0
        if (texture_uniform := shader.get('transparency_texture', None)) is not None:
            if (use_texture_uniform := shader.get('use_transparency_texture', None)) is not None:
                use_texture_uniform.value = False
            else:
                texture_uniform.value = 0
    return samplers


cdef void dispatch_instances(glctx, dict objects, shader, Model model, int count, cython.float[:, :] instances_data,
                             Textures textures, dict references, dict texture_unit_ids):
    vertex_buffer, index_buffer = model.get_buffers(glctx, objects)
    if vertex_buffer is None:
        return
    cdef list samplers = configure_textures(glctx, shader, textures, references, texture_unit_ids)
    instances_buffer = glctx.buffer(instances_data)
    cdef str format = '3f'
    cdef list buffer_config = [vertex_buffer, None, 'model_position']
    if shader.get('model_normal', None) is not None:
        format += ' 3f'
        buffer_config.append('model_normal')
    else:
        format += ' 3x4'
    if shader.get('model_uv', None) is not None:
        format += ' 2f'
        buffer_config.append('model_uv')
    else:
        format += ' 2x4'
    buffer_config[1] = format
    cdef list buffers = [tuple(buffer_config)]
    format = '16f'
    buffer_config = [instances_buffer, None, 'model_matrix']
    if shader.get('model_normal_matrix', None) is not None:
        format += ' 9f'
        buffer_config.append('model_normal_matrix')
    else:
        format += ' 9x4'
    if shader.get('material_albedo', None) is not None:
        format += ' 4f'
        buffer_config.append('material_albedo')
    else:
        format += ' 4x4'
    if shader.get('material_emissive', None) is not None:
        format += ' 4f'
        buffer_config.append('material_emissive')
    else:
        format += ' 4x4'
    if shader.get('material_properties', None) is not None:
        format += ' 4f'
        buffer_config.append('material_properties')
    else:
        format += ' 4x4'
    format += '/i'
    buffer_config[1] = format
    buffers.append(tuple(buffer_config))
    render_array = glctx.vertex_array(shader, buffers, index_buffer=index_buffer, mode=moderngl.TRIANGLES)
    render_array.render(instances=count)
    for sampler in samplers:
        sampler.clear()


class Canvas3D(WindowNode):
    def __init__(self, glctx):
        super().__init__(glctx)
        self._primary_render_target = None
        self._secondary_render_targets = []
        self._total_duration = 0
        self._total_count = 0

    @property
    def texture(self):
        return self._primary_render_target.texture if self._primary_render_target is not None else None

    def release(self):
        if self._primary_render_target is not None:
            self._primary_render_target.release()
            self._primary_render_target = None
        while self._secondary_render_targets:
            self._secondary_render_targets.pop().release()

    async def purge(self):
        logger.info("{} draw stats - {:d} x {:.1f}ms = {:.1f}s", self.name, self._total_count,
                    1e3 * self._total_duration / self._total_count, self._total_duration)
        self._total_duration = 0
        self._total_count = 0

    async def descend(self, engine, node, references, **kwargs):
        # A canvas3d is a leaf node from the perspective of the OpenGL world
        pass

    def render(self, Node node, dict references, **kwargs):
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
        material.ao = 1
        material.emissive = Zero3
        material.ior = 1.5
        cdef list render_groups = []
        cdef dict cameras = {}
        cameras[default_camera.id] = default_camera
        collect(node, transform_matrix, material, None, render_groups, default_camera, cameras, max_samples)
        cdef Camera primary_camera = cameras.get(node.get_str('camera_id', default_camera.id), default_camera)
        if self._primary_render_target is not None:
            self._primary_render_target.release()
        self._primary_render_target = RenderTarget.get(self.glctx, primary_camera.width, primary_camera.height, primary_camera.colorbits,
                                                       has_depth=True, samples=primary_camera.samples)
        cdef RenderGroup render_group
        with self._primary_render_target:
            self._primary_render_target.clear(primary_camera.get_clear_color())
            for render_group in render_groups:
                if render_group.instances:
                    render(self._primary_render_target, render_group, primary_camera, self.glctx, objects, references)
        if primary_camera.id:
            references[primary_camera.id] = self._primary_render_target
        cdef Camera camera
        while self._secondary_render_targets:
            self._secondary_render_targets.pop().release()
        for camera in cameras.values():
            if camera.secondary and camera.id and camera is not primary_camera:
                secondary_render_target = RenderTarget.get(self.glctx, camera.width, camera.height, camera.colorbits,
                                                           has_depth=True, samples=camera.samples)
                self._secondary_render_targets.append(secondary_render_target)
                with secondary_render_target:
                    secondary_render_target.clear(camera.get_clear_color())
                    for render_group in render_groups:
                        if render_group.instances:
                            render(secondary_render_target, render_group, camera, self.glctx, objects, references)
                references[camera.id] = secondary_render_target
        self._total_duration += system_clock()
        self._total_count += 1
