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
uniform bool orthographic;

uniform bool use_albedo_texture;
uniform bool use_metal_texture;
uniform bool use_roughness_texture;

uniform sampler2D albedo_texture;
uniform sampler2D metal_texture;
uniform sampler2D roughness_texture;

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
    vec3 albedo = fragment_albedo.rgb;
    if (use_albedo_texture) {
        vec4 texture_color = texture(albedo_texture, texture_uv);
        albedo = albedo * (1.0 - clamp(texture_color.a, 0.0, 1.0)) + texture_color.rgb;
    }
    float transparency = fragment_albedo.a;
    float translucency = fragment_emissive.a;
    float ior = fragment_properties.x;
    float metal = fragment_properties.y;
    if (use_metal_texture) {
        vec4 texture_color = texture(metal_texture, texture_uv);
        float mono = clamp(srgb_luminance(texture_color.rgb), 0.0, 1.0);
        metal = metal * (1.0 - clamp(texture_color.a, 0.0, 1.0)) + mono;
    }
    float roughness = fragment_properties.z;
    if (use_roughness_texture) {
        vec4 texture_color = texture(roughness_texture, texture_uv);
        float mono = clamp(srgb_luminance(texture_color.rgb), 0.0, 1.0);
        roughness = roughness * (1.0 - clamp(texture_color.a, 0.0, 1.0)) + mono;
    }

    vec3 transmission_color = vec3(0.0);
    vec3 diffuse_color = vec3(0.0);
    vec3 specular_color = vec3(0.0);
    compute_pbr_lighting(world_position, world_normal, world_normal,
                         ior, roughness, metal, 0.0, albedo,
                         transmission_color, diffuse_color, specular_color);

    fragment_color = vec4(transmission_color, view_distance);
}
