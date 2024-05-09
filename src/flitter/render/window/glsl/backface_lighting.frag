${HEADER}

const vec3 greyscale = vec3(0.299, 0.587, 0.114);

in vec3 world_position;
in vec3 world_normal;
in vec2 texture_uv;

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
uniform bool orthographic;

uniform bool use_albedo_texture;
uniform bool use_metal_texture;

uniform sampler2D albedo_texture;
uniform sampler2D metal_texture;


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
        float mono = clamp(dot(texture_color.rgb, greyscale), 0.0, 1.0);
        metal = metal * (1.0 - clamp(texture_color.a, 0.0, 1.0)) + mono;
    }
    vec3 transmission_color = vec3(0.0);
    vec3 N = normalize(world_normal);
    float rf0 = (ior - 1.0) / (ior + 1.0);
    vec3 F0 = mix(vec3(rf0*rf0), albedo, metal);
    for (int i = 0; i < nlights; i++) {
        mat4 light = lights[i];
        int light_type = int(light[0].w);
        vec3 light_color = light[0].xyz;
        vec3 L;
        float attenuation = 1.0;
        float light_distance = 1.0;
        if (light_type == ${Point}) {
            vec3 light_position = light[1].xyz;
            float light_radius = light[2].w;
            L = light_position - world_position;
            light_distance = length(L);
            if (light_radius > 0.0) {
                attenuation = clamp(1.0 - (light_radius / light_distance), 0.0, 1.0);
                light_distance = max(0.0, light_distance - light_radius*0.99);
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
            vec3 light_position = light[1].xyz;
            float light_length = length(light[2].xyz);
            vec3 light_direction = light[2].xyz / light_length;
            float light_radius = light[2].w;
            L = light_position - world_position;
            float NdotL = dot(L, N);
            float cp = clamp(dot(-L, light_direction), 0.0, light_length);
            float ip = clamp(-NdotL / dot(light_direction, N), 0.0, light_length);
            float m = light_length / 2.0;
            if (NdotL < 0.0) {
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
        } else if (light_type == ${Directional}) {
            vec3 light_direction = light[2].xyz;
            L = -light_direction;
        } else { // (light_type == ${Ambient})
            continue;
        }
        vec4 light_falloff = light[3];
        float ld2 = light_distance * light_distance;
        vec4 ds = vec4(1.0, light_distance, ld2, light_distance * ld2);
        attenuation /= dot(ds, light_falloff);
        float NdotL = clamp(dot(N, L), 0.0, 1.0);
        vec3 radiance = light_color * attenuation * NdotL;
        transmission_color += radiance * (1.0 - F0) * (1.0 - metal);
    }
    fragment_color = vec4(transmission_color * (albedo + 1.0) / 2.0, view_distance);
}
