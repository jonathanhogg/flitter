
#version 400

in vec2 coord;
out vec4 color;

uniform sampler2D texture0;
uniform float contrast = 1;
uniform float brightness = 0;

void main() {
    float offset = brightness + (1 - contrast) / 2;
    vec4 rgba = texture(texture0, coord);
    vec3 c = rgba.rgb / rgba.a;
    color = vec4(max((c * contrast + offset) * rgba.a, 0), rgba.a);
}
