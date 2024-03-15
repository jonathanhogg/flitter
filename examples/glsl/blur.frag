${HEADER}

in vec2 coord;
out vec4 color;

uniform sampler2D texture0;
uniform vec2 size;

uniform int horizontal;
uniform int radius;
uniform float sigma;

void main() {
    float s = float(radius) * sigma / 2.0;
    vec3 gaussian;
    gaussian.x = 1.0;
    gaussian.y = exp(-0.5 / (s * s));
    gaussian.z = gaussian.y * gaussian.y;

    vec4 color_sum = texture(texture0, coord) * gaussian.x;
    float weight_sum = gaussian.x;
    gaussian.xy *= gaussian.yz;

    vec2 offset = (horizontal != 0 ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) / size;
    for (int i = 1; i < radius; i++) {
        color_sum += texture(texture0, coord + float(i) * offset) * gaussian.x;
        color_sum += texture(texture0, coord - float(i) * offset) * gaussian.x;
        weight_sum += 2.0 * gaussian.x;
        gaussian.xy *= gaussian.yz;
    }
    color = color_sum / weight_sum;
}
