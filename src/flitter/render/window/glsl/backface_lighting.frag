#version 330

in vec3 world_position;
in vec3 world_normal;
in vec2 texture_uv;

flat in vec4 fragment_properties;

out vec4 fragment_color;

uniform vec3 view_position;
uniform vec3 view_focus;
uniform bool orthographic;


void main() {
    vec3 V;
    float view_distance;
    if (orthographic) {
        V = normalize(view_position - view_focus);
        view_distance = dot(view_position - world_position, V);
    } else {
        V = view_position - world_position;
        view_distance = length(V);
    }
    fragment_color = vec4(normalize(world_normal), view_distance);
}
