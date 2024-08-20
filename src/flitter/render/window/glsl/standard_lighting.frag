${HEADER}

const vec3 greyscale = vec3(0.299, 0.587, 0.114);

in vec3 world_position;
in vec3 world_normal;
in vec2 texture_uv;

flat in vec4 fragment_albedo;
flat in vec4 fragment_emissive;
flat in vec4 fragment_properties;

out vec4 fragment_color;

uniform vec3 view_position;
uniform vec3 view_focus;
uniform bool monochrome;
uniform vec3 tint;
uniform bool orthographic;
uniform float fog_max;
uniform float fog_min;
uniform vec3 fog_color;
uniform float fog_curve;
uniform mat4 pv_matrix;
uniform sampler2D backface_data;

uniform bool use_albedo_texture;
uniform bool use_metal_texture;
uniform bool use_roughness_texture;
uniform bool use_ao_texture;
uniform bool use_emissive_texture;
uniform bool use_transparency_texture;

uniform sampler2D albedo_texture;
uniform sampler2D metal_texture;
uniform sampler2D roughness_texture;
uniform sampler2D ao_texture;
uniform sampler2D emissive_texture;
uniform sampler2D transparency_texture;

<%include file="pbr_lighting.glsl"/>


void main() {
    vec3 view_direction;
    float view_distance;
    if (orthographic) {
        view_direction = normalize(view_position - view_focus);
        view_distance = dot(view_position - world_position, view_direction);
    } else {
        view_direction = view_position - world_position;
        view_distance = length(view_direction);
        view_direction /= view_distance;
    }
    float fog_alpha = (fog_max > fog_min) && (fog_curve > 0.0) ? pow(clamp((view_distance - fog_min) / (fog_max - fog_min), 0.0, 1.0), 1.0/fog_curve) : 0.0;
    vec3 albedo = fragment_albedo.rgb;
    if (use_albedo_texture) {
        vec4 texture_color = texture(albedo_texture, texture_uv);
        albedo = albedo * (1.0 - clamp(texture_color.a, 0.0, 1.0)) + texture_color.rgb;
    }
    float transparency = fragment_albedo.a;
    if (use_transparency_texture) {
        vec4 texture_color = texture(transparency_texture, texture_uv);
        float mono = clamp(dot(texture_color.rgb, greyscale), 0.0, 1.0);
        transparency = transparency * (1.0 - clamp(texture_color.a, 0.0, 1.0)) + mono;
    }
    vec3 emissive = fragment_emissive.rgb;
    if (use_emissive_texture) {
        vec4 emissive_texture_color = texture(emissive_texture, texture_uv);
        emissive = emissive * (1.0 - clamp(emissive_texture_color.a, 0.0, 1.0)) + emissive_texture_color.rgb;
    }
    float translucency = fragment_emissive.a;
    float ior = fragment_properties.x;
    float metal = fragment_properties.y;
    if (use_metal_texture) {
        vec4 texture_color = texture(metal_texture, texture_uv);
        float mono = clamp(dot(texture_color.rgb, greyscale), 0.0, 1.0);
        metal = metal * (1.0 - clamp(texture_color.a, 0.0, 1.0)) + mono;
    }
    float roughness = fragment_properties.z;
    if (use_roughness_texture) {
        vec4 texture_color = texture(roughness_texture, texture_uv);
        float mono = clamp(dot(texture_color.rgb, greyscale), 0.0, 1.0);
        roughness = roughness * (1.0 - clamp(texture_color.a, 0.0, 1.0)) + mono;
    }
    float ao = fragment_properties.w;
    if (use_ao_texture) {
        vec4 texture_color = texture(ao_texture, texture_uv);
        float mono = clamp(dot(texture_color.rgb, greyscale), 0.0, 1.0);
        ao = ao * (1.0 - clamp(texture_color.a, 0.0, 1.0)) + mono;
    }

    vec3 transmission_color = vec3(0.0);
    vec3 diffuse_color = vec3(0.0);
    vec3 specular_color = emissive;
    compute_pbr_lighting(world_position, world_normal, view_direction, ior, roughness, metal, ao, albedo, transmission_color, diffuse_color, specular_color);

    float opacity = 1.0 - transparency;
    if (translucency > 0.0) {
        vec4 position = pv_matrix * vec4(world_position, 1);
        compute_translucency(world_position, view_direction, view_distance, pv_matrix, backface_data, albedo, translucency, opacity, specular_color);
    }
    vec3 final_color = mix(diffuse_color, fog_color, fog_alpha) * opacity + specular_color * (1.0 - fog_alpha);
    if (monochrome) {
        float grey = dot(final_color, greyscale);
        final_color = vec3(grey);
    }
    fragment_color = vec4(final_color * tint, opacity);
}
