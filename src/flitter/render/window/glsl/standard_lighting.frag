#version 330

const vec3 greyscale = vec3(0.299, 0.587, 0.114);
const float Tau = 6.283185307179586;

in vec3 world_position;
in vec3 world_normal;
in vec2 texture_uv;
noperspective in vec2 screen_coord;

flat in vec4 fragment_albedo;
flat in vec4 fragment_emissive;
flat in vec4 fragment_properties;

out vec4 fragment_color;

uniform int nlights;
uniform lights_data {
    mat4 lights[${max_lights}];
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
uniform mat4 pv_matrix;
uniform sampler2D backface_data;

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


const vec4 RANDOM_SCALE = vec4(443.897, 441.423, .0973, .1099);

vec3 random3(vec3 p) {
    p = fract(p * RANDOM_SCALE.xyz);
    p += dot(p, p.yxz + 19.19);
    return fract((p.xxy + p.yzz) * p.zyx);
}

vec3 normal3(vec2 c, int i) {
    vec3 u1 = random3(vec3(c, i*2));
    vec3 u2 = random3(vec3(c, i*2+1));
    vec3 R = sqrt(-2 * log(u1));
    vec3 th = Tau * u2;
    return sin(th) * R;
}


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
    vec3 emissive = fragment_emissive.rgb;
    if (use_emissive_texture) {
        vec4 emissive_texture_color = texture(emissive_texture, texture_uv);
        emissive = emissive * (1 - clamp(emissive_texture_color.a, 0, 1)) + emissive_texture_color.rgb;
    }
    float translucency = fragment_emissive.a;
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
    float occlusion = fragment_properties.w;
    if (use_occlusion_texture) {
        vec4 texture_color = texture(occlusion_texture, texture_uv);
        float mono = clamp(dot(texture_color.rgb, greyscale), 0, 1);
        occlusion = occlusion * (1 - clamp(texture_color.a, 0, 1)) + mono;
    }
    vec3 diffuse_color = vec3(0);
    vec3 specular_color = emissive;
    vec3 transmission_color = vec3(0);
    vec4 backface;
    vec3 backface_normal;
    float backface_distance;
    float thickness;
    if (translucency > 0) {
        vec4 backface = texture(backface_data, screen_coord);
        backface_distance = backface.w;
        vec3 backface_position = view_position + V*backface_distance;
        float k = backface_distance / translucency;
        thickness = backface_distance - view_distance;
        int count = 1;
        for (int i = 0; i < 50; i++) {
            vec3 offset = normal3(screen_coord, i) / 10;
            vec4 pos = pv_matrix * vec4(backface_position + offset*thickness*k, 1);
            vec2 c = (pos.xy / pos.w + 1) / 2;
            vec4 backface_sample = texture(backface_data, c);
            if (backface_sample.w > view_distance) {
                backface += backface_sample;
                count += 1;
            }
        }
        backface /= count;
        backface_distance = backface.w;
        backface_normal = backface.xyz;
        thickness = backface_distance - view_distance;
    }
    vec3 N = normalize(world_normal);
    float rf0 = (ior - 1) / (ior + 1);
    vec3 F0 = mix(vec3(rf0*rf0), albedo, metal);
    float a = roughness * roughness;
    float a2 = a * a;
    float r = roughness + 1;
    float k = (r*r) / 8;
    float NdotV = clamp(dot(N, V), 0, 1);
    float Gnom = (NdotV / (NdotV * (1 - k) + k));
    for (int i = 0; i < nlights; i++) {
        mat4 light = lights[i];
        int light_type = int(light[0].w);
        vec3 light_color = light[0].xyz;
        int passes = 1;
        for (int pass = 0; pass < passes; pass++) {
            vec3 L;
            float attenuation = 1;
            float light_distance = 1;
            if (light_type == ${Point}) {
                vec3 light_position = light[1].xyz;
                float light_radius = light[2].w;
                L = light_position - world_position;
                light_distance = length(L);
                if (light_radius > 0) {
                    passes = 2;
                    if (pass == 0) {
                        attenuation = clamp(1 - (light_radius / light_distance), 0, 1);
                        light_distance = max(0, light_distance - light_radius*0.99);
                    } else {
                        attenuation = 1 / (1 + light_radius*light_radius);
                        vec3 R = reflect(V, N);
                        vec3 l = dot(L, R) * R - L;
                        L += l * min(0.99, light_radius/length(l));
                    }
                }
                L = normalize(L);
            } else if (light_type == ${Spot}) {
                vec3 light_position = light[1].xyz;
                vec3 light_direction = light[2].xyz;
                float inner_cone = light[1].w;
                float outer_cone = light[2].w;
                L = light_position - world_position;
                light_distance = length(L);
                L /= light_distance;
                float spot_cosine = dot(L, -light_direction);
                attenuation = 1 - clamp((inner_cone-spot_cosine) / (inner_cone-outer_cone), 0, 1);
            } else if (light_type == ${Line}) {
                passes = 2;
                vec3 light_position = light[1].xyz;
                float light_length = length(light[2].xyz);
                vec3 light_direction = light[2].xyz / light_length;
                float light_radius = light[2].w;
                L = light_position - world_position;
                if (pass == 0) {
                    float LdotN = dot(L, N);
                    float cp = clamp(dot(-L, light_direction), 0, light_length);
                    float ip = clamp(-LdotN / dot(light_direction, N), 0, light_length);
                    float m = light_length / 2;
                    if (LdotN < 0) {
                        m = (ip + light_length) / 2;
                        cp = max(cp, ip);
                    } else if (dot(L + light_direction*light_length, N) < 0) {
                        m = ip / 2;
                        cp = min(cp, ip);
                    }
                    L += light_direction * (cp*3 + m) / 4;
                    light_distance = length(L);
                    L /= light_distance;
                    light_distance -= min(light_radius, light_distance*0.99);
                } else {
                    attenuation = 1 / (1 + light_radius);
                    vec3 R = reflect(V, N);
                    mat3 M = mat3(R, light_direction, cross(R, light_direction));
                    L += clamp(-(inverse(M) * L).y, 0, light_length) * light_direction;
                    vec3 l = dot(L, R) * R - L;
                    L += l * min(0.99, light_radius/length(l));
                    light_distance = length(L);
                    L /= light_distance;
                }
            } else if (light_type == ${Directional}) {
                vec3 light_direction = light[2].xyz;
                L = -light_direction;
            } else { // (light_type == ${Ambient})
                diffuse_color += (1 - F0) * (1 - metal) * albedo * light_color * occlusion;
                break;
            }
            vec4 light_falloff = light[3];
            float ld2 = light_distance * light_distance;
            vec4 ds = vec4(1, light_distance, ld2, light_distance * ld2);
            attenuation /= dot(ds, light_falloff);
            vec3 H = normalize(V + L);
            float NdotL = clamp(dot(N, L), 0, 1);
            float NdotH = clamp(dot(N, H), 0, 1);
            float HdotV = clamp(dot(H, V), 0, 1);
            float denom = NdotH * NdotH * (a2-1) + 1;
            float NDF = a2 / (denom * denom);
            float G = Gnom * (NdotL / (NdotL * (1 - k) + k));
            vec3 F = F0 + (1 - F0) * pow(1 - HdotV, 5);
            vec3 radiance = light_color * attenuation * NdotL;
            if (pass == 0) {
                diffuse_color += radiance * (1 - F) * (1 - metal) * albedo;
                if (translucency > 0 && backface_distance > view_distance) {
                    float k = thickness / translucency;
                    vec3 radiance = light_color * attenuation * dot(backface_normal, L);
                    transmission_color += radiance * (1 - F0) * (1 - metal) * (1 + albedo)/2;
                }
            }
            if (pass == passes-1) {
                specular_color += radiance * (NDF * G * F) / (4 * NdotV * NdotL + 1e-6);
            }
        }
    }
    float opacity = 1 - transparency;
    if (translucency > 0) {
        float absorption = clamp(pow(0.5, thickness / translucency), 0, 1);
        specular_color += transmission_color * albedo * absorption * (1 - absorption);
        opacity *= 1 - absorption;
    }
    vec3 final_color = mix(diffuse_color, fog_color, fog_alpha) * opacity + specular_color * (1 - fog_alpha);
    if (monochrome) {
        float grey = dot(final_color, greyscale);
        final_color = vec3(grey);
    }
    fragment_color = vec4(final_color * tint, opacity);
}
