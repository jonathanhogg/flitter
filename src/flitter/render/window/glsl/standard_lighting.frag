${HEADER}

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

<%include file="color_functions.glsl"/>
<%include file="lighting_functions.glsl"/>


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
    float transparency = fragment_albedo.a;
    if (use_transparency_texture) {
        overlay_luminance_texture(transparency_texture, texture_uv, transparency);
    }
    float opacity = 1.0 - transparency;

    vec3 final_color;
    if (fog_alpha < 1.0) {
        vec3 albedo = fragment_albedo.rgb;
        vec3 emissive = fragment_emissive.rgb;
        float translucency = fragment_emissive.a;
        float ior = fragment_properties.x;
        float metal = fragment_properties.y;
        float roughness = fragment_properties.z;
        float ao = fragment_properties.w;

        if (use_albedo_texture) {
            overlay_color_texture(albedo_texture, texture_uv, albedo);
        }
        if (use_emissive_texture) {
            overlay_color_texture(emissive_texture, texture_uv, emissive);
        }
        if (use_metal_texture) {
            overlay_luminance_texture(metal_texture, texture_uv, metal);
        }
        if (use_roughness_texture) {
            overlay_luminance_texture(roughness_texture, texture_uv, roughness);
        }
        if (use_ao_texture) {
            overlay_luminance_texture(ao_texture, texture_uv, ao);
        }

        vec3 transmission_color = vec3(0.0);
        vec3 diffuse_color = vec3(0.0);
        vec3 specular_color = vec3(0.0);
        compute_emissive_lighting(world_normal, view_direction, emissive, specular_color);
        compute_pbr_lighting(world_position, world_normal, view_direction, ior, roughness, metal, ao, albedo, transmission_color, diffuse_color, specular_color);
        if (translucency > 0.0) {
            compute_translucency(world_position, view_direction, view_distance, pv_matrix, backface_data, albedo, translucency, opacity, specular_color);
        }
        final_color = mix(diffuse_color, fog_color, fog_alpha) * opacity + specular_color * (1.0 - fog_alpha);
    } else {
        final_color = fog_color * opacity;
    }

    if (monochrome) {
        final_color = vec3(srgb_luminance(final_color.rgb));
    }
    final_color *= tint;
    if (any(isinf(final_color)) || any(isnan(final_color))) {
        fragment_color = vec4(0.0);
    } else {
        fragment_color = vec4(final_color, opacity);
    }
}
