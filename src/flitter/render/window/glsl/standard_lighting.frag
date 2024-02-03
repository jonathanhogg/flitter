#version 330

const vec3 greyscale = vec3(0.299, 0.587, 0.114);

in vec3 world_position;
in vec3 world_normal;
in vec2 texture_uv;

flat in vec4 fragment_albedo;
flat in vec3 fragment_emissive;
flat in vec4 fragment_properties;

out vec4 fragment_color;

uniform int nlights;
uniform lights_data {
    vec4 lights[${max_lights * 4}];
};
uniform vec3 view_position;
uniform vec3 view_focus;
uniform bool monochrome;
uniform vec3 tint;
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
        V = normalize(view_position - view_focus);
        view_distance = dot(view_position - world_position, V);
    } else {
        V = view_position - world_position;
        view_distance = length(V);
        V /= view_distance;
    }
    float fog_alpha = (fog_max > fog_min) && (fog_curve > 0) ? pow(clamp((view_distance - fog_min) / (fog_max - fog_min), 0, 1), 1/fog_curve) : 0;
    vec3 albedo = fragment_albedo.rgb;
    if (use_albedo_texture) {
        vec4 texture_color = texture(albedo_texture, texture_uv);
        albedo = albedo * (1 - clamp(texture_color.a, 0, 1)) + texture_color.rgb;
    }
    float transparency = fragment_albedo.a;
    if (use_transparency_texture) {
        vec4 texture_color = texture(transparency_texture, texture_uv);
        float mono = clamp(dot(texture_color.rgb, greyscale), 0, 1);
        transparency = transparency * (1 - clamp(texture_color.a, 0, 1)) + mono;
    }
    vec3 emissive = fragment_emissive;
    if (use_emissive_texture) {
        vec4 emissive_texture_color = texture(emissive_texture, texture_uv);
        emissive = emissive * (1 - clamp(emissive_texture_color.a, 0, 1)) + emissive_texture_color.rgb;
    }
    float ior = fragment_properties.x;
    float metal = fragment_properties.y;
    if (use_metal_texture) {
        vec4 texture_color = texture(metal_texture, texture_uv);
        float mono = clamp(dot(texture_color.rgb, greyscale), 0, 1);
        metal = metal * (1 - clamp(texture_color.a, 0, 1)) + mono;
    }
    float roughness = fragment_properties.z;
    if (use_roughness_texture) {
        vec4 texture_color = texture(roughness_texture, texture_uv);
        float mono = clamp(dot(texture_color.rgb, greyscale), 0, 1);
        roughness = roughness * (1 - clamp(texture_color.a, 0, 1)) + mono;
    }
    float occlusion = fragment_properties.z;
    if (use_occlusion_texture) {
        vec4 texture_color = texture(occlusion_texture, texture_uv);
        float mono = clamp(dot(texture_color.rgb, greyscale), 0, 1);
        occlusion = occlusion * (1 - clamp(texture_color.a, 0, 1)) + mono;
    }
    vec3 diffuse_color = vec3(0);
    vec3 specular_color = emissive;
    vec3 N = normalize(world_normal);
    float rf0 = (ior - 1) / (ior + 1);
    vec3 F0 = mix(vec3(rf0*rf0), albedo, metal);
    for (int i = 0; i < nlights * 4; i += 4) {
        int light_type = int(lights[i].w);
        float inner_cone = lights[i+1].w;
        float outer_cone = lights[i+2].w;
        vec3 light_color = lights[i].xyz;
        vec3 light_position = lights[i+1].xyz;
        vec3 light_direction = lights[i+2].xyz;
        vec4 light_falloff = lights[i+3];
        if (light_type == ${Ambient}) {
            diffuse_color += (1 - F0) * (1 - metal) * albedo * light_color * occlusion;
        } else {
            vec3 L;
            float attenuation = 1;
            float light_distance = 1;
            if (light_type == ${Point}) {
                L = light_position - world_position;
                light_distance = length(L);
                L /= light_distance;
            } else if (light_type == ${Spot}) {
                L = light_position - world_position;
                light_distance = length(L);
                L /= light_distance;
                float spot_cosine = dot(L, -light_direction);
                attenuation = 1 - clamp((inner_cone-spot_cosine) / (inner_cone-outer_cone), 0, 1);
            } else {
                L = -light_direction;
            }
            float ld2 = light_distance * light_distance;
            vec4 ds = vec4(1, light_distance, ld2, light_distance * ld2);
            attenuation /= dot(ds, light_falloff);
            vec3 H = normalize(V + L);
            float NdotL = clamp(dot(N, L), 0, 1);
            float NdotV = clamp(dot(N, V), 0, 1);
            float NdotH = clamp(dot(N, H), 0, 1);
            float HdotV = clamp(dot(H, V), 0, 1);
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
            vec3 radiance = light_color * attenuation * NdotL;
            diffuse_color += diffuse * radiance;
            specular_color += specular * radiance;
        }
    }
    float opacity = 1 - transparency;
    vec3 final_color = mix(diffuse_color, fog_color, fog_alpha) * opacity + specular_color * (1 - fog_alpha);
    if (monochrome) {
        float grey = dot(final_color, greyscale);
        final_color = vec3(grey);
    }
    fragment_color = vec4(final_color * tint, opacity);
}
