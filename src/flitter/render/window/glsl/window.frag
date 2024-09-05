${HEADER}

in vec2 coord;
out vec4 color;
uniform sampler2D target;

<%include file="color_functions.glsl"/>

void main() {
    color = vec4(srgb_transfer(texture(target, coord).rgb), 1.0);
}
