#version 330

const float min_shininess = 50;
const vec3 greyscale = vec3(0.299, 0.587, 0.114);

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
uniform float fog_curve;

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
        view_direction /= view_distance;
    }
    float fog_alpha = (fog_max > fog_min) && (fog_curve > 0) ? pow(clamp((view_distance - fog_min) / (fog_max - fog_min), 0, 1), 1/fog_curve) : 0;
    if (fog_alpha == 1) {
        discard;
    }
    float opacity = 1 - transparency;
    if (use_transparency_texture) {
        vec4 transparency_texture_color = texture(transparency_texture, uv);
        float mono = clamp(dot(transparency_texture_color.rgb, greyscale), 0, 1);
        opacity = opacity * (1 - clamp(transparency_texture_color.a, 0, 1)) + mono;
    }
    vec3 diffuse_color = colors[0];
    if (use_diffuse_texture) {
        vec4 diffuse_texture_color = texture(diffuse_texture, uv);
        diffuse_color = diffuse_color * (1 - clamp(diffuse_texture_color.a, 0, 1)) + diffuse_texture_color.rgb;
    }
    vec3 specular_color = colors[1];
    if (use_specular_texture) {
        vec4 specular_texture_color = texture(specular_texture, uv);
        specular_color = specular_color * (1 - clamp(specular_texture_color.a, 0, 1)) + specular_texture_color.rgb;
    }
    vec3 color_base = colors[2];
    if (use_emissive_texture) {
        vec4 emissive_texture_color = texture(emissive_texture, uv);
        color_base = color_base * (1 - clamp(emissive_texture_color.a, 0, 1)) + emissive_texture_color.rgb;
    }
    vec3 normal = normalize(world_normal);
    vec3 color_overlay = vec3(0);
    for (int i = 0; i < nlights * 4; i += 4) {
        float light_type = lights[i].x;
        float inner_cone = lights[i].y;
        float outer_cone = lights[i].z;
        vec3 light_color = lights[i+1];
        vec3 light_position = lights[i+2];
        vec3 light_direction = lights[i+3];
        float diffuse_strength, specular_strength;
        if (light_type == ${Ambient}) {
            diffuse_strength = 1;
            specular_strength = 0;
        } else if (light_type == ${Directional}) {
            vec3 reflection_direction = reflect(light_direction, normal);
            specular_strength = pow(max(dot(view_direction, reflection_direction), 0), shininess) * min(shininess, min_shininess) / min_shininess;
            diffuse_strength = max(dot(normal, -light_direction), 0);
        } else if (light_type == ${Point}) {
            light_direction = world_position - light_position;
            float light_distance = length(light_direction);
            light_direction = normalize(light_direction);
            vec3 reflection_direction = reflect(light_direction, normal);
            float light_attenuation = 1 / (1 + light_distance*light_distance);
            light_color *= light_attenuation;
            specular_strength = pow(max(dot(view_direction, reflection_direction), 0), shininess) * min(shininess, min_shininess) / min_shininess;
            diffuse_strength = max(dot(normal, -light_direction), 0);
        } else if (light_type == ${Spot}) {
            vec3 spot_direction = world_position - light_position;
            float spot_distance = length(spot_direction);
            spot_direction = normalize(spot_direction);
            float light_attenuation = 1 / (1 + spot_distance*spot_distance);
            vec3 reflection_direction = reflect(spot_direction, normal);
            float spot_cosine = dot(spot_direction, light_direction);
            light_attenuation *= 1 - clamp((inner_cone - spot_cosine) / (inner_cone - outer_cone), 0, 1);
            light_color *= light_attenuation;
            specular_strength = pow(max(dot(view_direction, reflection_direction), 0), shininess) * min(shininess, min_shininess) / min_shininess;
            diffuse_strength = max(dot(normal, -spot_direction), 0);
        }
        color_base += diffuse_color * diffuse_strength * light_color;
        color_overlay += specular_color * specular_strength * light_color;
    }
    fragment_color = vec4(mix(color_base, fog_color, fog_alpha) * opacity + color_overlay * (1 - fog_alpha), opacity);
}
