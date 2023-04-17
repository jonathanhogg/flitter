#version 410

const float min_shininess = 50;

in vec3 world_position;
in vec3 world_normal;
flat in mat3 colors;
flat in float shininess;
flat in float transparency;

out vec4 fragment_color;

uniform int nlights;
uniform vec3 lights[${max_lights * 4}];
uniform vec3 view_position;

void main() {
    vec3 view_direction = normalize(view_position - world_position);
    vec3 color = colors * vec3(0, 0, 1);
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
            color += (colors * vec3(diffuse_strength, specular_strength, 0)) * light_color;
        } else if (light_type == ${Point}) {
            light_direction = world_position - light_position;
            float light_distance = length(light_direction);
            light_direction = normalize(light_direction);
            float light_attenuation = 1 / (1 + light_distance*light_distance);
            vec3 reflection_direction = reflect(light_direction, normal);
            float specular_strength = pow(max(dot(view_direction, reflection_direction), 0), shininess) * min(shininess, min_shininess) / min_shininess;
            float diffuse_strength = max(dot(normal, -light_direction), 0);
            color += (colors * vec3(diffuse_strength, specular_strength, 0)) * light_color * light_attenuation;
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
            color += (colors * vec3(diffuse_strength, specular_strength, 0)) * light_color * light_attenuation;
        }
    }
    float opacity = 1 - transparency;
    fragment_color = vec4(color * opacity, opacity);
}
