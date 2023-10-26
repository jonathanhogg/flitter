
#version 330

precision mediump float;

in vec2 coord;
out vec4 color;

uniform sampler2D last;
uniform sampler2D texture0;
uniform vec2 size;
uniform float delta;

uniform float mixing = 0.0;
uniform float glow = 0.0;
uniform float rotate = 0.0;
uniform vec2 scale = vec2(1.0, 1.0);
uniform vec2 offset = vec2(0.0, 0.0);

const float Tau = 6.283185307179586;


void main()
{
    float th = Tau * rotate * delta;
    float a = pow(mixing, delta);
    vec2 last_coord = (coord - 0.5) * size;
    last_coord /= pow(scale, vec2(delta, delta));
    last_coord *= mat2(cos(th), -sin(th), sin(th), cos(th));
    last_coord -= offset * vec2(1.0, -1.0) * delta;
    color = texture(last, last_coord / size + 0.5) * a + texture(texture0, coord) * (1.0 + glow) * (1 - a);
}
