#version 330

const vec3 greyscale = vec3(0.299, 0.587, 0.114);

in vec3 world_position;
in vec3 world_normal;
in vec2 texture_uv;
flat in vec3 fragment_albedo;
flat in float fragment_ior;
flat in float fragment_metal;
flat in float fragment_roughness;
flat in float fragment_occlusion;
flat in vec3 fragment_emissive;
flat in float fragment_transparency;

out vec4 fragment_color;

uniform int nlights;
uniform vec3 lights[${max_lights * 4}];
uniform vec3 view_position;
uniform vec3 focus;
uniform bool orthographic;
uniform float fog_max;
uniform float fog_min;
uniform vec3 fog_color;
uniform float fog_curve;

uniform bool use_albedo_texture;
uniform bool use_metal_texture;
uniform bool use_roughness_texture;
uniform bool use_occlusion_texture;
uniform bool use_emissive_texture;
uniform bool use_transparency_texture;

uniform sampler2D albedo_texture;
uniform sampler2D metal_texture;
uniform sampler2D roughness_texture;
uniform sampler2D occlusion_texture;
uniform sampler2D emissive_texture;
uniform sampler2D transparency_texture;


void main() {
    vec3 V;
    float view_distance;
    if (orthographic) {
        V = normalize(view_position - focus);
        view_distance = dot(view_position - world_position, V);
    } else {
        V = view_position - world_position;
        view_distance = length(V);
        V /= view_distance;
    }
    float fog_alpha = (fog_max > fog_min) && (fog_curve > 0) ? pow(clamp((view_distance - fog_min) / (fog_max - fog_min), 0, 1), 1/fog_curve) : 0;
    if (fog_alpha == 1) {
        discard;
    }
    vec3 albedo = fragment_albedo;
    if (use_albedo_texture) {
        vec4 texture_color = texture(albedo_texture, texture_uv);
        albedo = albedo * (1 - clamp(texture_color.a, 0, 1)) + texture_color.rgb;
    }
    float metal = fragment_metal;
    if (use_metal_texture) {
        vec4 texture_color = texture(metal_texture, texture_uv);
        float mono = clamp(dot(texture_color.rgb, greyscale), 0, 1);
        metal = metal * (1 - clamp(texture_color.a, 0, 1)) + mono;
    }
    float roughness = fragment_roughness;
    if (use_roughness_texture) {
        vec4 texture_color = texture(roughness_texture, texture_uv);
        float mono = clamp(dot(texture_color.rgb, greyscale), 0, 1);
        roughness = roughness * (1 - clamp(texture_color.a, 0, 1)) + mono;
    }
    float occlusion = fragment_occlusion ;
    if (use_occlusion_texture) {
        vec4 texture_color = texture(occlusion_texture, texture_uv);
        float mono = clamp(dot(texture_color.rgb, greyscale), 0, 1);
        occlusion = occlusion * (1 - clamp(texture_color.a, 0, 1)) + mono;
    }
    vec3 emissive = fragment_emissive;
    if (use_emissive_texture) {
        vec4 emissive_texture_color = texture(emissive_texture, texture_uv);
        emissive = emissive * (1 - clamp(emissive_texture_color.a, 0, 1)) + emissive_texture_color.rgb;
    }
    float transparency = fragment_transparency;
    if (use_transparency_texture) {
        vec4 texture_color = texture(transparency_texture, texture_uv);
        float mono = clamp(dot(texture_color.rgb, greyscale), 0, 1);
        transparency = transparency * (1 - clamp(texture_color.a, 0, 1)) + mono;
    }
    vec3 diffuse_color = vec3(0);
    vec3 specular_color = emissive;
    vec3 N = normalize(world_normal);
    float rf0 = (fragment_ior - 1) / (fragment_ior + 1);
    vec3 F0 = mix(vec3(rf0*rf0), albedo, metal);
    for (int i = 0; i < nlights * 4; i += 4) {
        int light_type = int(lights[i].x);
        float inner_cone = lights[i].y;
        float outer_cone = lights[i].z;
        vec3 light_color = lights[i+1];
        vec3 light_position = lights[i+2];
        vec3 light_direction = lights[i+3];
        if (light_type == ${Ambient}) {
            diffuse_color += (1 - F0) * (1 - metal) * albedo * light_color * occlusion;
        } else {
            vec3 L = -light_direction;
            float attenuation = 1;
            if (light_type == ${Point}) {
                L = light_position - world_position;
                float light_distance = length(L);
                L /= light_distance;
                attenuation = 1 / (1 + light_distance*light_distance);
            } else if (light_type == ${Spot}) {
                L = light_position - world_position;
                float light_distance = length(L);
                L /= light_distance;
                float spot_cosine = dot(L, -light_direction);
                attenuation = (1 - clamp((inner_cone-spot_cosine) / (inner_cone-outer_cone), 0, 1)) / (1 + light_distance*light_distance);
            }
            vec3 H = normalize(V + L);
            float NdotL = max(dot(N, L), 0);
            float NdotV = max(dot(N, V), 0);
            float NdotH = max(dot(N, H), 0);
            float HdotV = max(dot(H, V), 0);
            vec3 radiance = light_color * attenuation * NdotL;
            float a = roughness * roughness;
            float a2 = a * a;
            float denom = NdotH * NdotH * (a2-1) + 1;
            float NDF = a2 / (denom * denom);
            float r = roughness + 1;
            float k = (r*r) / 8;
            float G = (NdotV / (NdotV * (1 - k) + k)) * (NdotL / (NdotL * (1 - k) + k));
            vec3 F = F0 + (1 - F0) * pow(1 - HdotV, 5);
            vec3 diffuse = (1 - F) * (1 - metal) * albedo;
            vec3 specular = (NDF * G * F) / (4 * NdotV * NdotL + 1e-6);
            diffuse_color += diffuse * radiance;
            specular_color += specular * radiance;
        }
    }
    float opacity = 1 - transparency;
    fragment_color = vec4(mix(diffuse_color, fog_color, fog_alpha) * opacity + specular_color * (1 - fog_alpha), opacity);
}
