${HEADER}

in vec2 coord;
out vec4 color;

uniform sampler2D ${child_textures[0]};
uniform sampler2D ${child_textures[1]};
uniform float ratio;
uniform float alpha;
uniform vec2 size;
uniform vec2 frame_size;
uniform int aspect_mode;

void main() {
    vec2 s = size;
    vec2 o = vec2(0.0);
    if (aspect_mode == 1) {
        if (size.x/size.y > frame_size.x/frame_size.y) {
            s = frame_size * size.y / frame_size.y;
            o.x = (s.x - size.x) / 2.0;
        } else {
            s = frame_size * size.x / frame_size.x;
            o.y = (s.y - size.y) / 2.0;
        }
    } else if (aspect_mode == 2) {
        if (size.x/size.y > frame_size.x/frame_size.y) {
            s = frame_size * size.x / frame_size.x;
            o.y = (s.y - size.y) / 2.0;
        } else {
            s = frame_size * size.y / frame_size.y;
            o.x = (s.x - size.x) / 2.0;
        }
    }
    vec2 point = ((coord * size) + o) / s;
    vec4 frame0_color = texture(${child_textures[0]}, point);
    vec4 frame1_color = texture(${child_textures[1]}, point);
    vec4 merged = mix(frame0_color, frame1_color, ratio);
    color = merged * alpha;
}
