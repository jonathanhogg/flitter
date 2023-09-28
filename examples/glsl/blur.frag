
#version 330

in vec2 coord;
out vec4 color;

uniform sampler2D texture0;
uniform vec2 size;

uniform int horizontal = 0;
uniform int radius = 5;
uniform float sigma = 1;
uniform float spread = 1;

void main() {
    float s = radius * sigma / 2;
    vec3 gaussian;
    gaussian.x = 1;
    gaussian.y = exp(-0.5 / (s * s));
    gaussian.z = gaussian.y * gaussian.y;

    vec4 color_sum = texture(texture0, coord) * gaussian.x;
    float weight_sum = gaussian.x;
    gaussian.xy *= gaussian.yz;

    vec2 offset = (horizontal > 0 ? vec2(1, 0) : vec2(0, 1)) * spread / size;
    for (float i = 1; i < radius; ++i) {
        color_sum += texture(texture0, coord + i * offset) * gaussian.x;
        color_sum += texture(texture0, coord - i * offset) * gaussian.x;
        weight_sum += 2 * gaussian.x;
        gaussian.xy *= gaussian.yz;
    }
    color = color_sum / weight_sum;
}
