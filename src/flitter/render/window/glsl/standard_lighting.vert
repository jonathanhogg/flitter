#version 330

in vec3 model_position;
in vec3 model_normal;
in vec2 model_uv;
in mat4 model_matrix;

in vec3 material_albedo;
in vec3 material_emissive;
in float material_ior;
in float material_metal;
in float material_roughness;
in float material_occlusion;
in float material_transparency;

out vec3 world_position;
out vec3 world_normal;
out vec2 texture_uv;

flat out vec3 fragment_albedo;
flat out vec3 fragment_emissive;
flat out float fragment_ior;
flat out float fragment_metal;
flat out float fragment_roughness;
flat out float fragment_occlusion;
flat out float fragment_transparency;

uniform mat4 pv_matrix;

void main() {
    world_position = (model_matrix * vec4(model_position, 1)).xyz;
    gl_Position = pv_matrix * vec4(world_position, 1);
    mat3 normal_matrix = mat3(transpose(inverse(model_matrix)));
    world_normal = normal_matrix * model_normal;
    texture_uv = model_uv;
    fragment_albedo = material_albedo;
    fragment_emissive = material_emissive;
    fragment_ior = material_ior;
    fragment_metal = material_metal;
    fragment_roughness = material_roughness;
    fragment_occlusion = material_occlusion;
    fragment_transparency = material_transparency;
}
