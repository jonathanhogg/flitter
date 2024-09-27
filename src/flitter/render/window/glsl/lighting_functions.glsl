
uniform int nlights;
uniform lights_data {
    mat4 lights[${max_lights}];
};


void overlay_color_texture(sampler2D tex, vec2 coord, inout vec3 color) {
    vec4 texture_color = texture(tex, coord);
    color = color * (1.0 - clamp(texture_color.a, 0.0, 1.0)) + texture_color.rgb;
}

void overlay_luminance_texture(sampler2D tex, vec2 coord, inout float luminance) {
    vec4 texture_color = texture(tex, coord);
    float mono = clamp(srgb_luminance(texture_color.rgb), 0.0, 1.0);
    luminance = luminance * (1.0 - clamp(texture_color.a, 0.0, 1.0)) + mono;
}


void compute_emissive_lighting(vec3 world_normal, vec3 view_direction, vec3 emissive, inout vec3 specular_color) {
    vec3 N = normalize(world_normal);
    vec3 V = normalize(view_direction);
    float l = srgb_luminance(emissive);
    if (l > 1.0) {
        vec3 base = emissive / l;
        specular_color += base + (emissive - base) * dot(N, V);
    } else {
        specular_color += emissive;
    }
}


void compute_pbr_lighting(vec3 world_position, vec3 world_normal, vec3 view_direction,
                          float ior, float roughness, float metal, float ao, vec3 albedo,
                          inout vec3 transmission_color, inout vec3 diffuse_color, inout vec3 specular_color) {
    vec3 N = normalize(world_normal);
    vec3 V = normalize(view_direction);
    float rf0 = (ior - 1.0) / (ior + 1.0);
    vec3 F0 = mix(vec3(rf0*rf0), albedo, metal);
    float a = roughness * roughness;
    float a2 = a * a;
    float r = roughness + 1.0;
    float k = (r*r) / 8.0;
    float NdotV = clamp(dot(N, view_direction), 0.0, 1.0);
    float Gnom = (NdotV / (NdotV * (1.0 - k) + k));
    for (int i = 0; i < nlights; i++) {
        mat4 light = lights[i];
        int light_type = int(light[0].w);
        vec3 light_color = light[0].xyz;
        int passes = 1;
        for (int pass = 0; pass < passes; pass++) {
            vec3 L;
            float attenuation = 1.0;
            float light_distance = 1.0;
            if (light_type == ${Point}) {
                vec3 light_position = light[1].xyz;
                float light_radius = light[2].w;
                L = light_position - world_position;
                light_distance = length(L);
                if (light_radius > 0.0) {
                    passes = 2;
                    attenuation = clamp(1.0 - (light_radius / light_distance), 0.005, 1.0);
                    if (pass == 0) {
                        light_distance -= min(light_radius, light_distance*0.99);
                    } else {
                        vec3 R = reflect(V, N);
                        vec3 l = dot(L, R) * R - L;
                        L += l * min(0.99, light_radius/length(l));
                        light_distance = length(L);
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
                attenuation = 1.0 - clamp((inner_cone-spot_cosine) / (inner_cone-outer_cone), 0.0, 1.0);
            } else if (light_type == ${Line}) {
                passes = 2;
                vec3 light_position = light[1].xyz;
                float light_length = length(light[2].xyz);
                vec3 light_direction = light[2].xyz / light_length;
                float light_radius = light[2].w;
                L = light_position - world_position;
                if (pass == 0) {
                    float LdotN = dot(L, N);
                    float cp = clamp(dot(-L, light_direction), 0.0, light_length);
                    float ip = clamp(-LdotN / dot(light_direction, N), 0.0, light_length);
                    float m = light_length / 2.0;
                    if (LdotN < 0.0) {
                        m = (ip + light_length) / 2.0;
                        cp = max(cp, ip);
                    } else if (dot(L + light_direction*light_length, N) < 0.0) {
                        m = ip / 2.0;
                        cp = min(cp, ip);
                    }
                    L += light_direction * (cp*3.0 + m) / 4.0;
                    light_distance = length(L);
                    L /= light_distance;
                    attenuation = clamp(1.0 - (light_radius / light_distance), 0.0, 1.0);
                    light_distance -= min(light_radius, light_distance*0.99);
                } else {
                    vec3 R = reflect(V, N);
                    mat3 M = mat3(R, light_direction, cross(R, light_direction));
                    L += clamp(-(inverse(M) * L).y, 0.0, light_length) * light_direction;
                    vec3 l = dot(L, R) * R - L;
                    light_distance = length(L);
                    L += l * min(0.99, light_radius/light_distance);
                    attenuation = clamp(1.0 - (light_radius / light_distance), 0.0, 1.0);
                    light_distance = length(L);
                    L /= light_distance;
                }
            } else if (light_type == ${Directional}) {
                vec3 light_direction = light[2].xyz;
                L = -light_direction;
            } else { // (light_type == ${Ambient})
                diffuse_color += (1.0 - F0) * (1.0 - metal) * albedo * light_color * ao;
                break;
            }
            vec4 light_falloff = light[3];
            float ld2 = light_distance * light_distance;
            vec4 ds = vec4(1.0, light_distance, ld2, light_distance * ld2);
            attenuation /= dot(ds, light_falloff);
            vec3 H = normalize(V + L);
            float NdotL = clamp(dot(N, L), 0.0, 1.0);
            float NdotH = clamp(dot(N, H), 0.0, 1.0);
            float HdotV = clamp(dot(H, V), 0.0, 1.0);
            float denom = NdotH * NdotH * (a2-1.0) + 1.0;
            float NDF = a2 / (denom * denom);
            float G = Gnom * (NdotL / (NdotL * (1.0 - k) + k));
            vec3 F = F0 + (1.0 - F0) * pow(1.0 - HdotV, 5.0);
            vec3 radiance = light_color * attenuation * NdotL;
            if (pass == 0) {
                transmission_color += radiance * (1.0 - F0) * (1.0 - metal) * (1.0 + albedo) * 0.5;
                diffuse_color += radiance * (1.0 - F) * (1.0 - metal) * albedo;
            }
            if (pass == passes-1) {
                specular_color += radiance * (NDF * G * F) / (4.0 * NdotV * NdotL + 1e-6);
            }
        }
    }
}


const float Tau = 6.283185307179586231995926937088370323181152343750;

uint hash(uint state)
{
  state = state * 747796405u + 2891336453u;
  state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (state >> 22u) ^ state;
}

float random(inout uint state)
{
  state = hash(state);
  return float(state) * uintBitsToFloat(0x2f800004u);
}

vec3 random3(inout uint state) {
    return vec3(random(state), random(state), random(state));
}

const float MaxTranslucencySamples = 500.0;

void compute_translucency(vec3 world_position, vec3 view_direction, float view_distance, mat4 pv_matrix, sampler2D backface_data,
                          vec3 albedo, float translucency, inout float opacity, inout vec3 color) {
    vec4 position = pv_matrix * vec4(world_position, 1);
    vec2 screen_coord = (position.xy / position.w + 1.0) / 2.0;
    vec4 backface = texture(backface_data, screen_coord);
    float backface_distance = backface.w;
    if (backface_distance > view_distance) {
        vec3 backface_position = view_position + view_direction*backface_distance;
        float thickness = backface_distance - view_distance;
        float s = thickness / (translucency * 5.0);
        float cs = clamp(s, 0.01, 0.99);
        float k = thickness * s;
        int n = int(round(MaxTranslucencySamples * 2.0 * cs * (1.0 - cs)));
        int count = 1;
        uvec2 size = uvec2(textureSize(backface_data, 0));
        uvec2 p = uvec2(screen_coord * vec2(size));
        uint seed = p.y * size.x + p.x;
        for (int i = 0; i < n; i++) {
            vec3 radius = sqrt(-2.0 * log(random3(seed))) * k;
            vec3 theta = Tau * random3(seed);
            vec4 pos = pv_matrix * vec4(backface_position + sin(theta) * radius, 1.0);
            vec4 backface_sample = texture(backface_data, (pos.xy / pos.w + 1.0) / 2.0);
            if (backface_sample.w > view_distance) {
                backface += backface_sample;
                count += 1;
            }
            pos = pv_matrix * vec4(backface_position + cos(theta) * radius, 1.0);
            backface_sample = texture(backface_data, (pos.xy / pos.w + 1.0) / 2.0);
            if (backface_sample.w > view_distance) {
                backface += backface_sample;
                count += 1;
            }
        }
        backface /= float(count);
        thickness = backface.w - view_distance;
        k = thickness / translucency;
        float transmission = pow(0.5, k);
        color += backface.rgb * albedo * transmission * (1.0 - transmission);
        opacity *= 1.0 - pow(transmission, 5.0);
    } else {
        opacity = 0.0;
    }
}
