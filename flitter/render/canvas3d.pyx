# cython: language_level=3, profile=True

"""
Flitter OpenGL 3D drawing canvas
"""

import cython
from loguru import logger
import moderngl
import numpy as np
import trimesh

from .. import name_patch
from ..cache import SharedCache
from ..model cimport Node, Vector, Matrix44, null_


logger = name_patch(logger, __name__)


cdef enum LightType:
    Ambient = 1
    Directional = 2
    Point = 3


cdef Vector Black = Vector((0, 0, 0))


@cython.dataclasses.dataclass
cdef class Light:
    type: LightType
    color: Vector
    position: Vector
    direction: Vector


@cython.dataclasses.dataclass
cdef class Material:
    emissive: Vector = Black
    diffuse: Vector = Black
    specular: Vector = Black
    shininess: cython.double = 0


@cython.dataclasses.dataclass
cdef class Model:
    name: str
    vertex_data: np.ndarray = None
    index_data: np.ndarray = None


@cython.dataclasses.dataclass
cdef class Instance:
    model_matrix: Matrix44


@cython.dataclasses.dataclass
cdef class RenderSet:
    lights: list[Light]
    material: Material
    instances: dict[str, list[Instance]]



cdef dict ModelCache = {}

cdef int MAX_LIGHTS = 50

cdef str StandardVertexSource = """
#version 410

in vec3 model_position;
in vec3 model_normal;
in mat4 model_matrix;

out vec2 surface_coord;
out vec3 world_position;
out vec3 world_normal;

uniform mat4 pv_matrix;

void main() {
    world_position = (model_matrix * vec4(model_position, 1)).xyz;
    gl_Position = pv_matrix * vec4(world_position, 1);
    mat3 normal_matrix = mat3(transpose(inverse(model_matrix)));
    world_normal = normal_matrix * model_normal;
}
"""

cdef str StandardFragmentSource = """
#version 410

const int MAX_LIGHTS = """ + str(MAX_LIGHTS) + """;

struct Light {
    int type;
    vec3 color;
    vec3 position;
    vec3 direction;
};

in vec3 world_position;
in vec3 world_normal;

out vec4 fragment_color;

uniform int nlights = 0;
uniform Light lights[MAX_LIGHTS];
uniform vec3 diffuse = vec3(0);
uniform vec3 specular = vec3(0);
uniform vec3 emissive = vec3(0);
uniform float shininess = 0;
uniform vec3 view_position;

void main() {
    vec3 view_direction = normalize(view_position - world_position);
    vec3 color = emissive;
    vec3 normal = normalize(world_normal);
    for (int i = 0; i < nlights; i++) {
        Light light = lights[i];
        if (light.type == """ + str(LightType.Ambient) + """) {
            color += diffuse * light.color;
        } else if (light.type == """ + str(LightType.Directional) + """) {
            vec3 light_direction = normalize(light.direction);
            vec3 reflection_direction = reflect(light_direction, normal);
            float specular_strength = shininess > 0 ? pow(max(dot(view_direction, reflection_direction), 0), shininess) : 0;
            float diffuse_strength = max(dot(normal, -light_direction), 0);
            color += light.color * (diffuse * diffuse_strength + specular * specular_strength);
        } else if (light.type == """ + str(LightType.Point) + """) {
            vec3 light_direction = world_position - light.position;
            float light_distance = length(light_direction);
            light_direction = normalize(light_direction);
            float light_attenuation = 1 / (1 + pow(light_distance, 2));
            vec3 reflection_direction = reflect(light_direction, normal);
            float specular_strength = shininess > 0 ? pow(max(dot(view_direction, reflection_direction), 0), shininess) : 0;
            float diffuse_strength = max(dot(normal, -light_direction), 0);
            color += light.color * light_attenuation * (diffuse * diffuse_strength + specular * specular_strength);
        }
    }
    fragment_color = vec4(color, 1);
}
"""

def draw(Node node, tuple size, glctx, dict objects):
    cdef int width, height
    width, height = size
    cdef Vector viewpoint = Vector._coerce(node.get('viewpoint', 3, float, (0, 0, width/2)))
    cdef Vector focus = Vector._coerce(node.get('focus', 3, float, (0, 0, 0)))
    cdef Vector up = Vector._coerce(node.get('up', 3, float, (0, 1, 0)))
    cdef double fov = node.get('fov', 1, float, 0.25)
    cdef double near = node.get('near', 1, float, 1)
    cdef double far = node.get('far', 1, float, width)
    cdef Matrix44 pv_matrix = Matrix44._project(width/height, fov, near, far).mmul(Matrix44._look(viewpoint, focus, up))
    cdef Matrix44 model_matrix = Matrix44.__new__(Matrix44)
    cdef Node child = node.first_child
    cdef RenderSet render_set = RenderSet(lights=[], material=Material(), instances={})
    cdef list render_sets = [render_set]
    while child is not None:
        render_set = collect(child, model_matrix, render_set, render_sets)
        child = child.next_sibling
    for render_set in render_sets:
        if render_set.instances:
            render(render_set, pv_matrix, viewpoint, glctx, objects)


cdef RenderSet collect(Node node, Matrix44 model_matrix, RenderSet render_set, list render_sets):
    cdef str kind = node.kind
    cdef Light light
    cdef list lights, instances
    cdef Vector vector, color, position, size, direction, emissive, diffuse, specular
    cdef double shininess
    cdef Material material
    cdef Instance instance
    cdef RenderSet new_render_set
    cdef Node child
    cdef str attribute, model_name, filename
    cdef Matrix44 matrix
    cdef int subdivisions, sections
    cdef bint smooth
    cdef Model model

    if node.kind == 'transform':
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
        child = node.first_child
        while child is not None:
            render_set = collect(child, model_matrix, render_set, render_sets)
            child = child.next_sibling

    elif node.kind == 'light':
        color = Vector._coerce(node.get('color', 3, float))
        position = Vector._coerce(node.get('position', 3, float))
        direction = Vector._coerce(node.get('direction', 3, float))
        if color.as_bool():
            if position.length:
                position = model_matrix.vmul(position)
                light = Light(LightType.Point, color, position, None)
            elif direction.length:
                direction = model_matrix.inverse().transpose().matrix33().vmul(direction)
                light = Light(LightType.Directional, color, None, direction)
            else:
                light = Light(LightType.Ambient, color, None, None)
            lights = list(render_set.lights)
            lights.append(light)
            new_render_set = RenderSet(lights, render_set.material, {})
            render_sets.append(new_render_set)
            render_set = new_render_set

    elif node.kind == 'material':
        emissive = Vector._coerce(node.get('emissive', 3, float, Black))
        diffuse = Vector._coerce(node.get('diffuse', 3, float, Black))
        specular = Vector._coerce(node.get('specular', 3, float, Black))
        shininess = node.get('shininess', 1, float, 0)
        material = Material(emissive, diffuse, specular, shininess)
        new_render_set = RenderSet(render_set.lights, material, {})
        render_sets.append(new_render_set)
        child = node.first_child
        while child is not None:
            new_render_set = collect(child, model_matrix, new_render_set, render_sets)
            child = child.next_sibling

    elif node.kind in ('box', 'sphere', 'cylinder'):
        position = Vector._coerce(node.get('position', 3, float, (0, 0, 0)))
        size = Vector._coerce(node.get('size', 3, float, (1, 1, 1)))
        smooth = node.get('smooth', 1, bool, True)
        subdivisions = node.get('subdivisions', 1, int, 2)
        sections = node.get('sections', 1, int, 32)
        if size.as_bool():
            model_name = f'!{node.kind}'
            if node.kind == 'sphere':
                model_name += f'/{subdivisions}'
            elif node.kind == 'cylinder':
                model_name += f'/{sections}'
            if not smooth:
                model_name += '/flat'
            if model_name in ModelCache:
                model = ModelCache[model_name]
            else:
                logger.debug("Building primitive model {}", model_name)
                if node.kind == 'box':
                    trimesh_model = trimesh.primitives.Box()
                elif node.kind == 'sphere':
                    trimesh_model = trimesh.primitives.Sphere(subdivisions=subdivisions)
                elif node.kind == 'cylinder':
                    trimesh_model = trimesh.primitives.Cylinder(sections=sections)
                ModelCache[model_name] = build_model(model_name, trimesh_model, smooth)
            instance = Instance(model_matrix.mmul(Matrix44._translate(position).mmul(Matrix44._scale(size))))
            if (instances := render_set.instances.get(model_name)) is not None:
                instances.append(instance)
            else:
                render_set.instances[model_name] = [instance]

    elif node.kind == 'model':
        filename = node.get('filename', 1, str)
        if filename:
            position = Vector._coerce(node.get('position', 3, float, (0, 0, 0)))
            size = Vector._coerce(node.get('size', 3, float, (1, 1, 1)))
            smooth = node.get('smooth', 1, bool, True)
            model_name = filename
            if not smooth:
                model_name += '/flat'
            if model_name not in ModelCache:
                trimesh_model = SharedCache[filename].read_trimesh_model()
                if trimesh_model is not None:
                    ModelCache[model_name] = build_model(model_name, trimesh_model, smooth)
                    logger.debug("Loaded model {} with {} faces", filename, len(trimesh_model.faces))
            if model_name in ModelCache:
                instance = Instance(model_matrix.mmul(Matrix44._translate(position).mmul(Matrix44._scale(size))))
                if (instances := render_set.instances.get(model_name)) is not None:
                    instances.append(instance)
                else:
                    render_set.instances[model_name] = [instance]

    return render_set


cdef Model build_model(str model_name, trimesh_model, bint smooth):
    cdef Model model = Model(model_name)
    if smooth:
        model.vertex_data = np.hstack((trimesh_model.vertices, trimesh_model.vertex_normals)).astype('f4')
        model.index_data = trimesh_model.faces.astype('i4')
    else:
        model.vertex_data = np.empty((len(trimesh_model.faces), 3, 2, 3), dtype='f4')
        model.vertex_data[:,:,0] = trimesh_model.vertices[trimesh_model.faces]
        model.vertex_data[:,:,1] = trimesh_model.face_normals[:,:,None]
    return model


cdef void render(RenderSet render_set, Matrix44 pv_matrix, Vector viewpoint, glctx, dict objects):
    cdef str model_name
    cdef list instances
    cdef cython.float[:, :] matrices
    cdef Matrix44 model_matrix
    cdef Instance instance
    cdef Material material
    cdef Light light
    cdef Model model
    cdef int i, j, n
    if 'standard_shader' in objects:
        standard_shader, = objects['standard_shader']
    else:
        logger.debug("Compiling standard shader")
        standard_shader = glctx.program(vertex_shader=StandardVertexSource, fragment_shader=StandardFragmentSource)
        objects['standard_shader'] = (standard_shader,)
    standard_shader['pv_matrix'] = pv_matrix
    standard_shader['view_position'] = viewpoint
    material = render_set.material
    standard_shader['emissive'] = material.emissive
    standard_shader['diffuse'] = material.diffuse
    standard_shader['specular'] = material.specular
    standard_shader['shininess'] = material.shininess
    standard_shader['nlights'] = len(render_set.lights)
    cdef str prefix
    for i, light in enumerate(render_set.lights):
        prefix = f'lights[{i}].'
        standard_shader[prefix + 'type'] = light.type
        standard_shader[prefix + 'color'] = light.color
        if light.type == LightType.Directional:
            standard_shader[prefix + 'direction'] = light.direction
        elif light.type == LightType.Point:
            standard_shader[prefix + 'position'] = light.position
    cdef double* src
    cdef float* dest
    for model_name, instances in render_set.instances.items():
        n = len(instances)
        matrices = np.empty((n, 16), dtype='f4')
        for i in range(n):
            instance = instances[i]
            src = instance.model_matrix.numbers
            dest = &matrices[i, 0]
            for j in range(16):
                dest[j] = src[j]
        matrices_buffer = glctx.buffer(matrices)
        if model_name in objects:
            vertex_buffer, index_buffer = objects[model_name]
        else:
            model = ModelCache[model_name]
            vertex_buffer = glctx.buffer(model.vertex_data)
            index_buffer = glctx.buffer(model.index_data) if model.index_data is not None else None
            objects[model_name] = vertex_buffer, index_buffer
        if index_buffer:
            render_array = glctx.vertex_array(standard_shader, [(vertex_buffer, '3f 3f', 'model_position', 'model_normal'),
                                                                (matrices_buffer, '16f/i', 'model_matrix')],
                                              index_buffer=index_buffer, index_element_size=4, mode=moderngl.TRIANGLES)
        else:
            render_array = glctx.vertex_array(standard_shader, [(vertex_buffer, '3f 3f', 'model_position', 'model_normal'),
                                                                (matrices_buffer, '16f/i', 'model_matrix')])
        render_array.render(instances=n)
