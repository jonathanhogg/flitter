${HEADER}

in vec2 coord;
out vec4 color;
uniform float alpha;
uniform float gamma;
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
    color = gamma == 1.0 ? merged * alpha : pow(merged * alpha, vec4(gamma));
% else:
    color = vec4(0.0);
% endif
}
