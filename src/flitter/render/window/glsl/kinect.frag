#version 330

in vec2 coord;
out vec4 fragment_color;

uniform int mode = 0;
uniform float near = 500;
uniform float far = 4500;

uniform sampler2D color;
uniform sampler2D depth;

void main() {
    vec2 inverted = vec2(coord.x, 1 - coord.y);
    vec3 rgb = texture(color, inverted).rgb;
    float d = texture(depth, inverted).r;
    float a = d < near ? 0 : 1 - d / far;
    if (mode == 0) {
        fragment_color = vec4(rgb*a, a);
    } else if (mode == 1) {
        fragment_color = vec4(rgb, 1);
    } else if (mode == 2) {
        fragment_color = vec4(a, a, a, a);
    }
}
