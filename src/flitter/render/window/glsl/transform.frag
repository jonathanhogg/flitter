${HEADER}

const float Tau = 6.283185307179586231995926937088370323181152343750;

in vec2 coord;
out vec4 color;
uniform vec2 scale;
uniform vec2 translate;
uniform float rotate;
uniform vec2 size;
uniform float alpha;
uniform bool flip_x;
uniform bool flip_y;
% for name in child_textures:
uniform sampler2D ${name};
% endfor

<%include file="composite_functions.glsl"/>

void main() {
% if child_textures:
    float th = -rotate * Tau;
    float cth = cos(th);
    float sth = sin(th);
    vec2 point = ((coord - 0.5) * size - translate) * mat2(cth, sth, -sth, cth) / scale / size + 0.5;
%     for name in child_textures:
%         if loop.index == 0:
    vec4 merged = texture(${name}, point);
%         else:
    merged = composite_${composite}(texture(${name}, point), merged);
%         endif
%     endfor
    color = merged * alpha;
% else:
    color = vec4(0.0);
% endif
}
