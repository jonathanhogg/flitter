${HEADER}

in vec2 coord;
out vec4 color;
uniform float alpha;
uniform vec2 size;
uniform float segments = 0.0;
uniform float rotation = 0.0;
uniform float radius = 0.0;
% for name in child_textures:
uniform sampler2D ${name};
% endfor

const float Tau = 6.283185307179586;

<%include file="composite_functions.glsl"/>

void main() {
% if child_textures:
    vec2 p = (coord - 0.5) * size;
    float th = ((atan(p.y, p.x) / Tau) + 0.5 - rotation) * segments;
    float i = floor(th);
    th = ((mod(i, 2.0) == 0.0 ? th - i : i + 1.0 - th) / segments + rotation - 0.5) * Tau;
    p *= p;
    float scale = radius * min(size.x, size.y) / 2;
    float r = sqrt(p.x + p.y) / scale;
    float j = floor(r);
    r = (mod(j, 2.0) == 0.0 ? r - j : j + 1.0 - r) * scale;
    p = vec2(r * cos(th), r * sin(th)) / size + 0.5;
%     for name in child_textures:
%         if loop.index == 0:
    vec4 merged = texture(${name}, p);
%         else:
    merged = composite_${composite}(texture(${name}, p), merged);
%         endif
%     endfor
    color = merged * alpha;
% else:
    color = vec4(0.0);
% endif
}
