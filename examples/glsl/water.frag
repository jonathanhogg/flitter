
#version 410

precision mediump float;

in vec2 coord;
out vec4 color;

uniform sampler2D texture0;
uniform vec2 size;

uniform float depth = 0.0;
uniform float strength = 0.0;
uniform float ior = 1.333333;
uniform vec2 window = vec2(0.0, 0.0);
uniform vec4 theta = vec4(0.0, 0.0, 0.0, 0.0);
uniform vec4 lambda = vec4(0.0, 0.0, 0.0, 0.0);
uniform vec4 sigma = vec4(0.0, 0.0, 0.0, 0.0);
uniform vec4 phi = vec4(0.0, 0.0, 0.0, 0.0);

const float Tau = 6.283185307179586;
const vec2 offsetx = vec2(1.0, 0.0);
const vec2 offsety = vec2(0.0, 1.0);
const vec3 incident = vec3(0.0, 0.0, 1.0);
const vec4 sumv = vec4(1.0, 1.0, 1.0, 1.0);

void main() {
    vec2 p = window != vec2(0.0, 0.0) ? coord * window + (size - window) / 2 : coord * size;
    vec4 th = Tau * theta;
    vec4 l = Tau / lambda;
    vec4 cth = l * cos(th);
    vec4 sth = l * sin(th);
    vec4 pp = p.x * cth - p.y * sth + phi;
    float z = dot(sigma * sin(pp), sumv);
    float dx = dot(sigma * sin(pp + cth), sumv) - z;
    float dy = dot(sigma * sin(pp - sth), sumv) - z;
    vec3 refracted = refract(incident, normalize(cross(vec3(offsetx, dx), vec3(offsety, dy))), ior);
    refracted *= strength * (depth + z) / refracted.z;
    color = texture(texture0, (p + refracted.xy) / size);
}
