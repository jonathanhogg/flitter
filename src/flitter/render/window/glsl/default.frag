#version 330

in vec2 coord;
out vec4 color;
uniform float alpha = 1;
uniform float gamma = 1;
% for name in child_textures:
uniform sampler2D ${name};
% endfor

<%include file="composite_functions.glsl"/>

void main() {
% if child_textures:
%     for name in child_textures:
%         if loop.index == 0:
    vec4 merged = texture(${name}, coord);
%         elif loop.index == 1:
    vec4 child = texture(${name}, coord);
    merged = composite_${composite}(child, merged);
%         else:
    child = texture(${name}, coord);
    merged = composite_${composite}(child, merged);
%         endif
%     endfor
    color = gamma == 1 ? merged * alpha : pow(merged * alpha, vec4(gamma));
% else:
    color = vec4(0);
% endif
}
