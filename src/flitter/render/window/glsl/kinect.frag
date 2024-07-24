${HEADER}

in vec2 coord;
out vec4 color;

uniform sampler2D color_frame;
uniform sampler2D depth_frame;
uniform int mode;
uniform float near;
uniform float far;
uniform bool flip_x;
uniform bool flip_y;

void main() {
    vec2 inverted = vec2(flip_x ? 1.0 - coord.x : coord.x, flip_y ? coord.y : 1.0 - coord.y);
    vec3 rgb = texture(color_frame, inverted).rgb;
    float d = texture(depth_frame, inverted).r;
    float a = 1.0 - (clamp(d, near, far) - near) / (far - near);
    if (mode == 0) {
        color = vec4(rgb * a, a);
    } else if (mode == 1) {
        color = vec4(rgb, 1.0);
    } else if (mode == 2) {
        color = vec4(a, a, a, a);
    }
}
