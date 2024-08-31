${HEADER}

in vec2 coord;
out vec4 color;
uniform float alpha;
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
    color = merged * alpha;
% else:
    color = vec4(0.0);
% endif
}
