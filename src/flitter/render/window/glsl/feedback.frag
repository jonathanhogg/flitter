${HEADER}

const float Tau = 6.283185307179586231995926937088370323181152343750;

in vec2 coord;
out vec4 color;
uniform float alpha;
uniform vec2 size;
uniform float delta;
uniform float mixer;
uniform float timebase;
uniform float glow;
uniform vec2 translate;
uniform vec2 scale;
uniform float rotate;
uniform sampler2D last;
% for name in child_textures:
uniform sampler2D ${name};
% endfor

<%include file="composite_functions.glsl"/>

void main() {
% if child_textures:
%     for name in child_textures:
%         if loop.index == 0:
    vec4 merged = texture(${name}, coord);
%         else:
    merged = composite_${composite}(texture(${name}, coord), merged);
%         endif
%     endfor
    float t = delta/timebase;
    float th = Tau * rotate * t, cth = cos(th), sth = sin(th);
    vec2 last_coord = (coord - 0.5) * size / pow(scale, vec2(t));
    last_coord *= mat2(cth, -sth, sth, cth);
    last_coord -= translate * vec2(t, -t);
    float k = mixer > 0.0 ? pow(1.0/mixer, -t) : 0.0;
    merged = mix(merged * (1.0 + glow), texture(last, last_coord / size + 0.5), k);
    color = merged * alpha;
% else:
    color = vec4(0.0);
% endif
}
