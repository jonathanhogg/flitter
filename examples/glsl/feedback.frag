${HEADER}

const float Tau = 6.283185307179586;

in vec2 coord;
out vec4 color;

uniform sampler2D last;
uniform sampler2D texture0;
uniform vec2 size;
uniform float delta;

uniform float mixer;
uniform float timebase;
uniform float glow;
uniform vec2 translate;
uniform vec2 scale;
uniform float rotate;


void main()
{
    float t = delta/timebase;
    float th = Tau * rotate * t, cth = cos(th), sth = sin(th);
    vec2 last_coord = (coord - 0.5) * size / pow(scale, vec2(t));
    last_coord *= mat2(cth, -sth, sth, cth);
    last_coord -= translate * vec2(t, -t);
    float k = mixer > 0.0 ? pow(1.0/mixer, -t) : 0.0;
    color = mix(texture(texture0, coord) * (1.0 + glow), clamp(texture(last, last_coord / size + 0.5), 0.0, 1.0), k);
}
