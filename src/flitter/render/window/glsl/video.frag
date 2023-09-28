#version 330

in vec2 coord;
out vec4 color;

uniform sampler2D ${child_textures[0]};
uniform sampler2D ${child_textures[1]};
uniform float ratio;
uniform float alpha = 1;

void main() {
    vec4 frame0_color = texture(${child_textures[0]}, coord);
    vec4 frame1_color = texture(${child_textures[1]}, coord);
    color = vec4(mix(frame0_color.rgb, frame1_color.rgb, ratio) * alpha, alpha);
}
