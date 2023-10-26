#version 330

in vec3 model_position;
in vec3 model_normal;
in vec2 model_uv;
in mat4 model_matrix;
in mat3 material_colors;
in float material_shininess;
in float material_transparency;

out vec3 world_position;
out vec3 world_normal;
out vec2 uv;
flat out mat3 colors;
flat out float shininess;
flat out float transparency;

uniform mat4 pv_matrix;

void main() {
    world_position = (model_matrix * vec4(model_position, 1)).xyz;
    gl_Position = pv_matrix * vec4(world_position, 1);
    mat3 normal_matrix = mat3(transpose(inverse(model_matrix)));
    world_normal = normal_matrix * model_normal;
    uv = model_uv;
    colors = material_colors;
    shininess = material_shininess;
    transparency = material_transparency;
}
