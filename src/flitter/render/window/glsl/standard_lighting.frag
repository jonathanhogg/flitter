#version 410

const float min_shininess = 50;

in vec3 world_position;
in vec3 world_normal;
in vec2 uv;
flat in mat3 colors;
flat in float shininess;
flat in float transparency;

out vec4 fragment_color;

uniform int nlights;
uniform vec3 lights[${max_lights * 4}];
uniform vec3 view_position;
uniform vec3 focus;
uniform bool orthographic;
uniform float fog_max;
uniform float fog_min;
uniform vec3 fog_color;

uniform bool use_diffuse_texture;
uniform bool use_specular_texture;
uniform bool use_emissive_texture;
uniform bool use_transparency_texture;
uniform sampler2D diffuse_texture;
uniform sampler2D specular_texture;
uniform sampler2D emissive_texture;
uniform sampler2D transparency_texture;


void main() {
    vec3 view_direction;
    float view_distance;
    if (orthographic) {
        view_direction = normalize(view_position - focus);
        view_distance = dot(view_position - world_position, view_direction);
    } else {
        view_direction = view_position - world_position;
        view_distance = length(view_direction);
        view_direction = normalize(view_direction);
    }
    float fog_alpha = (fog_max > fog_min) ? clamp((view_distance - fog_min) / (fog_max - fog_min), 0, 1) : 0;
    if (fog_alpha == 1) {
        discard;
    }
    vec3 diffuse_color = colors[0];
    if (use_diffuse_texture) {
        vec4 diffuse_texture_color = texture(diffuse_texture, uv);
        diffuse_color = diffuse_color * (1 - diffuse_texture_color.a) + diffuse_texture_color.rgb;
    }
    vec3 specular_color = colors[1];
    if (use_specular_texture) {
        vec4 specular_texture_color = texture(specular_texture, uv);
        specular_color = specular_color * (1 - specular_texture_color.a) + specular_texture_color.rgb;
    }
    vec3 color = colors[2];
    if (use_emissive_texture) {
        vec4 emissive_texture_color = texture(emissive_texture, uv);
        color = color * (1 - emissive_texture_color.a) + emissive_texture_color.rgb;
    }
    float opacity = 1 - transparency;
    if (use_transparency_texture) {
        vec4 transparency_texture_color = texture(transparency_texture, uv);
        float mono = clamp(dot(transparency_texture_color.rgb, vec3(0.299, 0.587, 0.114)), 0, 1);
        opacity = opacity * (1 - clamp(transparency_texture_color.a, 0, 1)) + mono;
    }
    vec3 normal = normalize(world_normal);
    int n = shininess == 0 && colors[0] == vec3(0) ? 0 : nlights * 4;
    for (int i = 0; i < n; i += 4) {
        float light_type = lights[i].x;
        float inner_cone = lights[i].y;
        float outer_cone = lights[i].z;
        vec3 light_color = lights[i+1];
        vec3 light_position = lights[i+2];
        vec3 light_direction = lights[i+3];
        if (light_type == ${Ambient}) {
            color += (colors * vec3(1, 0, 0)) * light_color;
        } else if (light_type == ${Directional}) {
            vec3 reflection_direction = reflect(light_direction, normal);
            float specular_strength = pow(max(dot(view_direction, reflection_direction), 0), shininess) * min(shininess, min_shininess) / min_shininess;
            float diffuse_strength = max(dot(normal, -light_direction), 0);
            color += (diffuse_color * diffuse_strength + specular_color * specular_strength) * light_color;
        } else if (light_type == ${Point}) {
            light_direction = world_position - light_position;
            float light_distance = length(light_direction);
            light_direction = normalize(light_direction);
            float light_attenuation = 1 / (1 + light_distance*light_distance);
            vec3 reflection_direction = reflect(light_direction, normal);
            float specular_strength = pow(max(dot(view_direction, reflection_direction), 0), shininess) * min(shininess, min_shininess) / min_shininess;
            float diffuse_strength = max(dot(normal, -light_direction), 0);
            color += (diffuse_color * diffuse_strength + specular_color * specular_strength) * light_color * light_attenuation;
        } else if (light_type == ${Spot}) {
            vec3 spot_direction = world_position - light_position;
            float spot_distance = length(spot_direction);
            spot_direction = normalize(spot_direction);
            float light_attenuation = 1 / (1 + spot_distance*spot_distance);
            vec3 reflection_direction = reflect(spot_direction, normal);
            float specular_strength = pow(max(dot(view_direction, reflection_direction), 0), shininess) * min(shininess, min_shininess) / min_shininess;
            float diffuse_strength = max(dot(normal, -spot_direction), 0);
            float spot_cosine = dot(spot_direction, light_direction);
            light_attenuation *= 1 - clamp((inner_cone - spot_cosine) / (inner_cone - outer_cone), 0, 1);
            color += (diffuse_color * diffuse_strength + specular_color * specular_strength) * light_color * light_attenuation;
        }
    }
    vec4 model_color = vec4(color * opacity, opacity);
    fragment_color = mix(model_color, vec4(fog_color, 1), fog_alpha);
}
