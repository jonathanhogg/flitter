${HEADER}

in vec2 position;
out vec2 coord;
uniform vec2 size;
uniform vec2 frame_size;
uniform int aspect_mode;

void main() {
    gl_Position = vec4(position.x, -position.y, 0.0, 1.0);
    vec2 s = size;
    vec2 o = vec2(0.0);
    if (aspect_mode == 1) {
        if (size.x/size.y > frame_size.x/frame_size.y) {
            s = frame_size * size.y / frame_size.y;
            o.x = (s.x - size.x) / 2;
        } else {
            s = frame_size * size.x / frame_size.x;
            o.y = (s.y - size.y) / 2;
        }
    } else if (aspect_mode == 2) {
        if (size.x/size.y > frame_size.x/frame_size.y) {
            s = frame_size * size.x / frame_size.x;
            o.y = (s.y - size.y) / 2;
        } else {
            s = frame_size * size.y / frame_size.y;
            o.x = (s.x - size.x) / 2;
        }
    }
    coord = (((position + 1.0) / 2.0 * size) + o) / s;
}
