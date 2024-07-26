${HEADER}

in vec2 coord;
out vec4 color;
uniform float alpha;
uniform float gamma;
uniform float inset;
% for name in child_textures:
uniform sampler2D ${name};
% endfor

<%include file="composite_functions.glsl"/>

void main() {
% if child_textures:
    float k = alpha;
    if (coord.x < inset) {
        k *= coord.x / inset;
    } else if (coord.x > 1 - inset) {
        k *= (1 - coord.x) / inset;
    }
    if (coord.y < inset) {
        k *= coord.y / inset;
    } else if (coord.y > 1 - inset) {
        k *= (1 - coord.y) / inset;
    }
%     for name in child_textures:
%         if loop.index == 0:
    vec4 merged = texture(${name}, coord);
%         else:
    merged = composite_${composite}(texture(${name}, coord), merged);
%         endif
%     endfor
    color = gamma == 1.0 ? merged * k : pow(merged * k, vec4(gamma));
% else:
    color = vec4(0.0);
% endif
}
