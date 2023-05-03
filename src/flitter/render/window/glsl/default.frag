#version 410

in vec2 coord;
out vec4 color;
uniform float alpha = 1;
% for name in child_textures:
uniform sampler2D ${name};
% endfor

<%include file="blend_functions.glsl"/>
<%
blend_mode = node.get('blend', 1, str);
if blend_mode not in {'over', 'dest_over', 'lighten', 'darken'}:
    blend_mode = 'over'
%>

void main() {
% if child_textures:
%     for name in child_textures:
%         if loop.index == 0:
    vec4 merged = texture(${name}, coord);
%         elif loop.index == 1:
    vec4 child = texture(${name}, coord);
    merged = blend_${blend_mode}(child, merged);
%         else:
    child = texture(${name}, coord);
    merged = blend_${blend_mode}(child, merged);
%         endif
%     endfor
    color = merged * alpha;
% else:
    color = vec4(0);
% endif
}
