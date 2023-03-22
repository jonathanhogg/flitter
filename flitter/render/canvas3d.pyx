# cython: language_level=3, profile=True

"""
Flitter OpenGL drawing canvas
"""

import cython
from cython.view cimport array
from loguru import logger
import moderngl
import numpy as np

from ..model cimport Node, Vector, Matrix44, null_


cdef enum LightType:
    Ambient = 1
    Directional = 2
    Point = 3


cdef Vector Black = Vector((0, 0, 0))

cdef int MAX_LIGHTS = 50

cdef str StandardVertexSource = """
#version 410

in vec3 model_position;
in vec3 model_normal;
in vec2 model_uv;
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
    surface_coord = model_uv;
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

in vec2 surface_coord;
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
            vec3 reflection_direction = reflect(-light_direction, normal);
            float specular_strength = shininess > 0 ? pow(max(dot(view_direction, reflection_direction), 0), shininess) : 0;
            float diffuse_strength = max(dot(normal, light_direction), 0);
            color += light.color * (diffuse * diffuse_strength + specular * specular_strength);
        } else if (light.type == """ + str(LightType.Point) + """) {
            vec3 light_direction = light.position - world_position;
            float light_distance = length(light_direction);
            light_direction = normalize(light_direction);
            float light_attenuation = 1 / (1 + pow(light_distance, 2));
            float diffuse_strength = max(dot(normal, light_direction), 0);
            vec3 reflection_direction = reflect(-light_direction, normal);
            float specular_strength = shininess > 0 ? pow(max(dot(view_direction, reflection_direction), 0), shininess) : 0;
            color += light.color * light_attenuation * (diffuse * diffuse_strength + specular * specular_strength);
        }
    }
    fragment_color = vec4(color, 1);
}
"""

BoxVertices = np.array([
                [0,0,1, 0,0,1, 0,0,], [1,1,1, 0,0,1, 1,1,], [0,1,1, 0,0,1, 0,1,], [0,0,1, 0,0,1, 0,0,], [1,0,1, 0,0,1, 1,0,], [1,1,1, 0,0,1, 1,1,],
                [1,0,1, 1,0,0, 0,0,], [1,1,0, 1,0,0, 1,1,], [1,1,1, 1,0,0, 0,1,], [1,0,1, 1,0,0, 0,0,], [1,0,0, 1,0,0, 1,0,], [1,1,0, 1,0,0, 1,1,],
                [1,0,0, 0,0,-1, 0,0,], [0,1,0, 0,0,-1, 1,1,], [1,1,0, 0,0,-1, 0,1,], [1,0,0, 0,0,-1, 0,0,], [0,0,0, 0,0,-1, 1,0,], [0,1,0, 0,0,-1, 1,1,],
                [0,0,0, -1,0,0, 0,0,], [0,1,1, -1,0,0, 1,1,], [0,1,0, -1,0,0, 0,1,], [0,0,0, -1,0,0, 0,0,], [0,0,1, -1,0,0, 1,0,], [0,1,1, -1,0,0, 1,1,],
                [0,1,1, 0,1,0, 0,0,], [1,1,0, 0,1,0, 1,1,], [0,1,0, 0,1,0, 0,1,], [0,1,1, 0,1,0, 0,0,], [1,1,1, 0,1,0, 1,0,], [1,1,0, 0,1,0, 1,1,],
                [0,0,0, 0,-1,0, 0,0,], [1,0,1, 0,-1,0, 1,1,], [0,0,1, 0,-1,0, 0,1,], [0,0,0, 0,-1,0, 0,0,], [1,0,0, 0,-1,0, 1,0,], [1,0,1, 0,-1,0, 1,1,],
            ], dtype='f4')

@cython.dataclasses.dataclass
cdef class Light:
    type: LightType
    color: Vector
    position: Vector = None
    direction: Vector = None


@cython.dataclasses.dataclass
cdef class Material:
    emissive: Vector = Black
    diffuse: Vector = Black
    specular: Vector = Black
    shininess: cython.double = 0


@cython.dataclasses.dataclass
cdef class Instance:
    model_matrix: Matrix44


@cython.dataclasses.dataclass
cdef class RenderSet:
    lights: list[Light]
    material: Material
    instances: dict[str, list[Instance]]


def render(Node node, tuple size, glctx, dict objects):
    cdef int width, height
    width, height = size
    cdef Vector viewpoint = Vector._coerce(node.get('viewpoint', 3, float, (0, 0, width/2)))
    cdef Vector focus = Vector._coerce(node.get('focus', 3, float, (0, 0, 0)))
    cdef Vector up = Vector._coerce(node.get('up', 3, float, (0, 1, 0)))
    cdef double fov = node.get('fov', 1, float, 0.25)
    cdef double near = node.get('near', 1, float, 1)
    cdef double far = node.get('far', 1, float, width)
    cdef Matrix44 pv_matrix = Matrix44._project(width/height, fov, near, far) @ Matrix44._look(viewpoint, focus, up)
    cdef Matrix44 model_matrix = Matrix44.__new__(Matrix44)
    cdef Node child = node.first_child
    cdef RenderSet render_set = RenderSet(lights=[], material=Material(), instances={})
    cdef list render_sets = [render_set]
    while child is not None:
        render_set = process(child, model_matrix, render_set, render_sets)
        child = child.next_sibling
    for render_set in render_sets:
        if render_set.instances:
            draw(render_set, pv_matrix, viewpoint, glctx, objects)


cdef void draw(RenderSet render_set, Matrix44 pv_matrix, Vector viewpoint, glctx, dict objects):
    cdef str model
    cdef list instances
    cdef cython.float[:, :] matrices
    cdef Matrix44 model_matrix
    cdef Instance instance
    cdef Material material
    cdef Light light
    cdef int i, j, n
    if 'standard_shader' in objects:
        standard_shader = objects['standard_shader']
    else:
        logger.debug("Compiling standard shader")
        standard_shader = glctx.program(vertex_shader=StandardVertexSource, fragment_shader=StandardFragmentSource)
        objects['standard_shader'] = standard_shader
    standard_shader['pv_matrix'] = pv_matrix
    standard_shader['view_position'] = viewpoint
    material = render_set.material
    standard_shader['emissive'] = material.emissive
    standard_shader['diffuse'] = material.diffuse
    standard_shader['specular'] = material.specular
    standard_shader['shininess'] = material.shininess
    standard_shader['nlights'] = len(render_set.lights)
    for i, light in enumerate(render_set.lights):
        standard_shader[f'lights[{i}].type'] = light.type
        standard_shader[f'lights[{i}].color'] = light.color
        if light.type == LightType.Directional:
            standard_shader[f'lights[{i}].direction'] = light.direction
        elif light.type == LightType.Point:
            standard_shader[f'lights[{i}].position'] = light.position
    for model, instances in render_set.instances.items():
        n = len(instances)
        matrices = array((n, 16), itemsize=sizeof(cython.float), format='f')
        for i in range(n):
            instance = instances[i]
            model_matrix = instance.model_matrix
            for j in range(16):
                matrices[i, j] = model_matrix.numbers[j]
        matrices_buffer = glctx.buffer(matrices)
        # render
        if 'box' in objects:
            vertices_buffer = objects['box']
        else:
            logger.debug("Initialising vertex buffer for box")
            vertices_buffer = glctx.buffer(BoxVertices.data)
            objects['box'] = vertices_buffer
        render_array = glctx.vertex_array(standard_shader, [(vertices_buffer, '3f 3f 2f', 'model_position', 'model_normal', 'model_uv'),
                                                            (matrices_buffer, '16f/i', 'model_matrix')])
        render_array.render(mode=moderngl.TRIANGLES, instances=n)
        matrices_buffer.release()
        render_array.release()


cdef RenderSet process(Node node, Matrix44 model_matrix, RenderSet render_set, list render_sets):
    cdef str kind = node.kind
    cdef Light light
    cdef list lights
    cdef Vector vector, color, position, direction, emissive, diffuse, specular
    cdef double shininess
    cdef Material material
    cdef Instance instance
    cdef RenderSet new_render_set
    cdef Node child
    cdef str attribute
    cdef Matrix44 matrix

    if node.kind == 'transform':
        for attribute, vector in node._attributes.items():
            if attribute == 'translate':
                if (matrix := Matrix44._translate(vector)) is not None:
                    model_matrix = model_matrix @ matrix
            elif attribute == 'scale':
                if (matrix := Matrix44._scale(vector)) is not None:
                    model_matrix = model_matrix @ matrix
            elif attribute == 'rotate':
                if (matrix := Matrix44._rotate(vector)) is not None:
                    model_matrix = model_matrix @ matrix
            elif attribute == 'rotate_x':
                if vector.numbers !=  NULL and vector.length == 1:
                    model_matrix = model_matrix @ Matrix44._rotate_x(vector.numbers[0])
            elif attribute == 'rotate_y':
                if vector.numbers !=  NULL and vector.length == 1:
                    model_matrix = model_matrix @ Matrix44._rotate_y(vector.numbers[0])
            elif attribute == 'rotate_z':
                if vector.numbers !=  NULL and vector.length == 1:
                    model_matrix = model_matrix @ Matrix44._rotate_z(vector.numbers[0])
        child = node.first_child
        while child is not None:
            render_set = process(child, model_matrix, render_set, render_sets)
            child = child.next_sibling

    elif node.kind == 'light':
        color = Vector._coerce(node.get('color', 3, float))
        position = model_matrix @ Vector._coerce(node.get('position', 3, float))
        direction = Vector._coerce(node.get('direction', 3, float))
        if color.as_bool():
            if position.length:
                light = Light(type=LightType.Point, color=color, position=position)
            elif direction.length:
                light = Light(type=LightType.Directional, color=color, direction=direction)
            else:
                light = Light(type=LightType.Ambient, color=color)
            lights = render_set.lights.copy()
            lights.append(light)
            new_render_set = RenderSet(lights=lights, material=render_set.material, instances={})
            render_sets.append(new_render_set)
            render_set = new_render_set

    elif node.kind == 'material':
        emissive = Vector._coerce(node.get('emissive', 3, float, Black))
        diffuse = Vector._coerce(node.get('diffuse', 3, float, Black))
        specular = Vector._coerce(node.get('specular', 3, float, Black))
        shininess = node.get('shininess', 1, float, 0)
        material = Material(emissive=emissive, diffuse=diffuse, specular=specular, shininess=shininess)
        new_render_set = RenderSet(lights=render_set.lights, material=material, instances={})
        render_sets.append(new_render_set)
        child = node.first_child
        while child is not None:
            new_render_set = process(child, model_matrix, new_render_set, render_sets)
            child = child.next_sibling

    elif node.kind == 'box':
        position = Vector._coerce(node.get('position', 3, float, (0, 0, 0)))
        size = Vector._coerce(node.get('size', 3, float))
        if size.as_bool():
            instance = Instance(model_matrix=model_matrix.mmul(Matrix44._translate(position).mmul(Matrix44._scale(size))))
            render_set.instances.setdefault('box', []).append(instance)

    return render_set
