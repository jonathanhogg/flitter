${HEADER}

in vec2 coord;
out vec4 color;

uniform sampler2D texture0;
uniform sampler2D last;
uniform vec2 size;
uniform int pass;

uniform ivec2 radius;
uniform vec2 sigma;

void main() {
    int n;
    float s;
    vec2 offset;
    if (pass == 0) {
        n = radius.x;
        s = float(n) * sigma.x / 2.0;
        offset = vec2(1.0, 0.0) / size;
    } else {
        n = radius.y;
        s = float(n) * sigma.y / 2.0;
        offset = vec2(0.0, 1.0) / size;
    }
    vec3 gaussian;
    gaussian.x = 1.0;
    gaussian.y = exp(-0.5 / (s * s));
    gaussian.z = gaussian.y * gaussian.y;

    vec4 color_sum = texture(texture0, coord) * gaussian.x;
    float weight_sum = gaussian.x;
    gaussian.xy *= gaussian.yz;

    for (int i = 1; i < n; i++) {
        if (pass == 0) {
            color_sum += texture(texture0, coord + float(i) * offset) * gaussian.x;
            color_sum += texture(texture0, coord - float(i) * offset) * gaussian.x;
        } else {
            color_sum += texture(last, coord + float(i) * offset) * gaussian.x;
            color_sum += texture(last, coord - float(i) * offset) * gaussian.x;
        }
        weight_sum += 2.0 * gaussian.x;
        gaussian.xy *= gaussian.yz;
    }
    color = color_sum / weight_sum;
}
