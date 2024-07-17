${HEADER}

in vec2 coord;
out vec4 color;

uniform sampler2D ${child_textures[0]};
uniform sampler2D ${child_textures[1]};
uniform float ratio;
uniform float alpha;
uniform float gamma;

void main() {
    vec4 frame0_color = texture(${child_textures[0]}, coord);
    vec4 frame1_color = texture(${child_textures[1]}, coord);
    vec4 merged = mix(frame0_color, frame1_color, ratio);
    color = gamma == 1.0 ? merged * alpha : pow(merged * alpha, vec4(gamma));
}
