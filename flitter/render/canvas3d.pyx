# cython: language_level=3, profile=True

"""
Flitter OpenGL 3D drawing canvas
"""

import time

import cython
from cython cimport view
from loguru import logger
import moderngl
import numpy as np
import trimesh

from .. import name_patch
from ..cache import SharedCache
from ..model cimport Node, Vector, Matrix44, null_


logger = name_patch(logger, __name__)

cdef Vector Zero3 = Vector((0, 0, 0))
cdef Vector One3 = Vector((1, 1, 1))
cdef dict ModelCache = {}
cdef int DEFAULT_MAX_LIGHTS = 50


cdef enum LightType:
    Ambient = 1
    Directional = 2
    Point = 3


@cython.dataclasses.dataclass
cdef class Light:
    type: LightType
    color: Vector
    position: Vector
    direction: Vector


@cython.dataclasses.dataclass
cdef class Material:
    diffuse: Vector = Zero3
    specular: Vector = One3
    emissive: Vector = Zero3
    shininess: cython.double = 0


@cython.dataclasses.dataclass
cdef class Model:
    name: str
    vertex_data: np.ndarray = None
    index_data: np.ndarray = None


@cython.dataclasses.dataclass
cdef class Instance:
    model_matrix: Matrix44
    material: Material


@cython.dataclasses.dataclass
cdef class RenderSet:
    lights: list[list[Light]]
    instances: dict[str, list[Instance]]



cdef str StandardVertexSource = """
#version 410

in vec3 model_position;
in vec3 model_normal;
in mat4 model_matrix;
in mat3 material_colors;
in float material_shininess;

out vec3 world_position;
out vec3 world_normal;
flat out mat3 colors;
flat out float shininess;

uniform mat4 pv_matrix;

void main() {
    world_position = (model_matrix * vec4(model_position, 1)).xyz;
    gl_Position = pv_matrix * vec4(world_position, 1);
    mat3 normal_matrix = mat3(transpose(inverse(model_matrix)));
    world_normal = normal_matrix * model_normal;
    colors = material_colors;
    shininess = material_shininess;
}
"""

cdef str StandardFragmentSource = """
#version 410

const int MAX_LIGHTS = @@max_lights@@;

in vec3 world_position;
in vec3 world_normal;
flat in mat3 colors;
flat in float shininess;

out vec4 fragment_color;

uniform int nlights;
uniform vec3 lights[MAX_LIGHTS * 4];
uniform vec3 view_position;

void main() {
    vec3 view_direction = normalize(view_position - world_position);
    vec3 color = colors * vec3(0, 0, 1);
    vec3 normal = normalize(world_normal);
    for (int i = 0; i < nlights*4; i += 4) {
        float light_type = lights[i].x;
        vec3 light_color = lights[i+1];
        vec3 light_position = lights[i+2];
        vec3 light_direction = lights[i+3];
        if (light_type == """ + str(LightType.Ambient) + """) {
            color += (colors * vec3(1, 0, 1)) * light_color;
        } else if (light_type == """ + str(LightType.Directional) + """) {
            light_direction = normalize(light_direction);
            vec3 reflection_direction = reflect(light_direction, normal);
            float specular_strength = shininess > 0 ? pow(max(dot(view_direction, reflection_direction), 0), shininess) : 0;
            float diffuse_strength = max(dot(normal, -light_direction), 0);
            color += (colors * vec3(diffuse_strength, specular_strength, 0)) * light_color;
        } else if (light_type == """ + str(LightType.Point) + """) {
            light_direction = world_position - light_position;
            float light_distance = length(light_direction);
            light_direction = normalize(light_direction);
            float light_attenuation = 1 / (1 + pow(light_distance, 2));
            vec3 reflection_direction = reflect(light_direction, normal);
            float specular_strength = shininess > 0 ? pow(max(dot(view_direction, reflection_direction), 0), shininess) : 0;
            float diffuse_strength = max(dot(normal, -light_direction), 0);
            color += (colors * vec3(diffuse_strength, specular_strength, 0)) * light_color * light_attenuation;
        }
    }
    fragment_color = vec4(color, 1);
}
"""


def draw(Node node, tuple size, glctx, dict objects):
    cdef int width, height
    width, height = size
    cdef Vector viewpoint = node.get_fvec('viewpoint', 3, Vector((0, 0, width/2)))
    cdef Vector focus = node.get_fvec('focus', 3, Zero3)
    cdef Vector up = node.get_fvec('up', 3, Vector((0, 1, 0)))
    cdef double fov = node.get('fov', 1, float, 0.25)
    cdef double near = node.get('near', 1, float, 1)
    cdef double far = node.get('far', 1, float, width)
    cdef int max_lights = node.get_int('max_lights', DEFAULT_MAX_LIGHTS)
    cdef Matrix44 pv_matrix = Matrix44._project(width/height, fov, near, far).mmul(Matrix44._look(viewpoint, focus, up))
    cdef Matrix44 model_matrix = Matrix44.__new__(Matrix44)
    cdef Node child = node.first_child
    cdef RenderSet no_lights_render_set = RenderSet(lights=[[]], instances={})
    cdef RenderSet render_set = RenderSet(lights=[[]], instances={})
    cdef list render_sets = [no_lights_render_set, render_set]
    while child is not None:
        collect(child, model_matrix, Material(), render_set, render_sets)
        child = child.next_sibling
    for render_set in render_sets:
        if render_set.instances:
            render(render_set, pv_matrix, viewpoint, max_lights, glctx, objects)


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
            if vector.numbers !=  NULL and vector.length == 1:
                model_matrix = model_matrix.mmul(Matrix44._rotate_x(vector.numbers[0]))
        elif attribute == 'rotate_y':
            if vector.numbers !=  NULL and vector.length == 1:
                model_matrix = model_matrix.mmul(Matrix44._rotate_y(vector.numbers[0]))
        elif attribute == 'rotate_z':
            if vector.numbers !=  NULL and vector.length == 1:
                model_matrix = model_matrix.mmul(Matrix44._rotate_z(vector.numbers[0]))
    return model_matrix


cdef void collect(Node node, Matrix44 model_matrix, Material material, RenderSet render_set, list render_sets):
    cdef str kind = node.kind
    cdef Light light
    cdef list lights, instances
    cdef Vector color, position, direction, emissive, diffuse, specular
    cdef double shininess
    cdef Node child
    cdef str model_name, filename
    cdef int subdivisions, sections
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
        position = node.get_fvec('position', 3)
        direction = node.get_fvec('direction', 3)
        if color.as_bool():
            light = Light.__new__(Light)
            light.color = color
            if position.length:
                light.type = LightType.Point
                light.position = model_matrix.vmul(position)
                light.direction = None
            elif direction.as_bool():
                light.type = LightType.Directional
                light.position = None
                light.direction = model_matrix.inverse().transpose().matrix33().vmul(direction)
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
        child = node.first_child
        while child is not None:
            collect(child, model_matrix, new_material, render_set, render_sets)
            child = child.next_sibling

    elif node.kind == 'box':
        flat = node.get_bool('flat', False)
        if flat:
            model_name = '!box/flat'
        else:
            model_name = '!box'
        if model_name not in ModelCache:
            logger.debug("Building primitive model {}", model_name)
            trimesh_model = trimesh.primitives.Box()
            ModelCache[model_name] = build_model(model_name, trimesh_model, flat)
        add_instance(render_sets, render_set, model_name, node, model_matrix, material)

    elif node.kind == 'sphere':
        subdivisions = node.get_int('subdivisions', 2)
        model_name = f'!sphere/{subdivisions}'
        flat = node.get_bool('flat', False)
        if flat:
            model_name += '/flat'
        if model_name not in ModelCache:
            logger.debug("Building primitive model {}", model_name)
            trimesh_model = trimesh.primitives.Sphere(subdivisions=subdivisions)
            ModelCache[model_name] = build_model(model_name, trimesh_model, flat)
        add_instance(render_sets, render_set, model_name, node, model_matrix, material)

    elif node.kind == 'cylinder':
        sections = node.get_int('sections', 32)
        model_name = f'!cylinder/{sections}'
        flat = node.get_bool('flat', False)
        if flat:
            model_name += '/flat'
        if model_name not in ModelCache:
            logger.debug("Building primitive model {}", model_name)
            trimesh_model = trimesh.primitives.Cylinder(sections=sections)
            ModelCache[model_name] = build_model(model_name, trimesh_model, flat)
        add_instance(render_sets, render_set, model_name, node, model_matrix, material)

    elif node.kind == 'model':
        filename = node.get('filename', 1, str)
        if filename:
            flat = node.get_bool('flat', False)
            model_name = filename
            if flat:
                model_name += '/flat'
            if model_name not in ModelCache:
                trimesh_model = SharedCache[filename].read_trimesh_model()
                if trimesh_model is not None:
                    ModelCache[model_name] = build_model(model_name, trimesh_model, flat)
                    logger.debug("Loaded model {} with {} faces", filename, len(trimesh_model.faces))
            add_instance(render_sets, render_set, model_name, node, model_matrix, material)


cdef Model build_model(str model_name, trimesh_model, bint flat):
    cdef Model model = Model(model_name)
    if flat:
        model.vertex_data = np.empty((len(trimesh_model.faces), 3, 2, 3), dtype='f4')
        model.vertex_data[:,:,0] = trimesh_model.vertices[trimesh_model.faces]
        model.vertex_data[:,:,1] = trimesh_model.face_normals[:,:,None]
    else:
        model.vertex_data = np.hstack((trimesh_model.vertices, trimesh_model.vertex_normals)).astype('f4')
        model.index_data = trimesh_model.faces.astype('i4')
    return model


cdef void add_instance(list render_sets, RenderSet render_set, str model_name, Node node, Matrix44 model_matrix, Material material):
    if 'position' in node._attributes:
        model_matrix = model_matrix.mmul(Matrix44._translate(node.get_fvec('position', 3, Zero3)))
    if 'rotation' in node._attributes:
        model_matrix = model_matrix.mmul(Matrix44._rotate(node.get_fvec('rotation', 3, Zero3)))
    if 'size' in node._attributes:
        model_matrix = model_matrix.mmul(Matrix44._scale(node.get_fvec('size', 3, One3)))
    if not material.diffuse.as_bool() and (not material.specular.as_bool() or material.shininess == 0):
        render_set = render_sets[0]
    cdef dict render_instances = render_set.instances
    cdef Instance instance = Instance.__new__(Instance)
    instance.model_matrix = model_matrix
    instance.material = material
    cdef list instances
    if (instances := render_instances.get(model_name)) is not None:
        instances.append(instance)
    else:
        render_instances[model_name] = [instance]


cdef void render(RenderSet render_set, Matrix44 pv_matrix, Vector viewpoint, int max_lights, glctx, dict objects):
    cdef str model_name
    cdef list instances, lights, buffers
    cdef cython.float[:, :] matrices, materials, lights_data
    cdef Material material
    cdef Light light
    cdef Model model
    cdef int i, j, n
    cdef double* src
    cdef float* dest
    cdef Instance instance;
    cdef str shader_name = f'!standard_shader/{max_lights}'
    if (standard_shader := objects.get(shader_name)) is None:
        logger.debug("Compiling standard lighting shader for {} max lights", max_lights)
        standard_shader = compile(glctx, max_lights)
        objects[shader_name] = standard_shader
    standard_shader['pv_matrix'] = pv_matrix
    standard_shader['view_position'] = viewpoint
    lights_data = view.array((max_lights, 12), 4, 'f')
    i = 0
    for lights in render_set.lights:
        for light in lights:
            if i == max_lights:
                break
            dest = &lights_data[i, 0]
            dest[0] = <cython.float>(<int>light.type)
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
    for model_name, instances in render_set.instances.items():
        n = len(instances)
        matrices = view.array((n, 16), 4, 'f')
        materials = view.array((n, 10), 4, 'f')
        for i in range(n):
            instance = instances[i]
            material = instance.material
            src = instance.model_matrix.numbers
            dest = &matrices[i, 0]
            for j in range(16):
                dest[j] = src[j]
            dest = &materials[i, 0]
            for j in range(3):
                dest[j] = material.diffuse.numbers[j]
                dest[j+3] = material.specular.numbers[j]
                dest[j+6] = material.emissive.numbers[j]
            dest[9] = material.shininess
        matrices_buffer = glctx.buffer(matrices)
        materials_buffer = glctx.buffer(materials)
        if model_name in objects:
            vertex_buffer, index_buffer = objects[model_name]
        elif model_name in ModelCache:
            model = ModelCache[model_name]
            vertex_buffer = glctx.buffer(model.vertex_data)
            index_buffer = glctx.buffer(model.index_data) if model.index_data is not None else None
            objects[model_name] = vertex_buffer, index_buffer
        else:
            continue
        buffers = [(vertex_buffer, '3f 3f', 'model_position', 'model_normal'),
                   (matrices_buffer, '16f/i', 'model_matrix'),
                   (materials_buffer, '9f 1f/i', 'material_colors', 'material_shininess')]
        render_array = glctx.vertex_array(standard_shader, buffers, index_buffer=index_buffer, mode=moderngl.TRIANGLES)
        render_array.render(instances=n)

cdef object compile(glctx, int max_lights):
    fragment = StandardFragmentSource.replace('@@max_lights@@', str(max_lights))
    return glctx.program(vertex_shader=StandardVertexSource, fragment_shader=fragment)
