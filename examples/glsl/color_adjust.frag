${HEADER}

in vec2 coord;
out vec4 color;

uniform sampler2D texture0;
uniform float contrast;
uniform float brightness;
uniform float exposure;

void main() {
    float offset = brightness + (1.0 - contrast) / 2.0;
    vec4 rgba = texture(texture0, coord);
    vec3 c = rgba.rgb / rgba.a;
    color = vec4(max((c * pow(2.0, exposure) * contrast + offset) * rgba.a, 0.0), rgba.a);
}
