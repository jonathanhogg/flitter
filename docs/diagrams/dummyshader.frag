
#version 330

in vec2 coord;
out vec4 color;

void main() {
    color = vec4(coord.x, coord.y, 1.0 - coord.x, 1.0);
}
