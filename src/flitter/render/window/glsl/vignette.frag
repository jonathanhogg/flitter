${HEADER}

in vec2 coord;
out vec4 color;
uniform float alpha;
uniform float inset;
% for name in child_textures:
uniform sampler2D ${name};
% endfor

<%include file="composite_functions.glsl"/>
<%include file="ease_functions.glsl"/>

void main() {
% if child_textures:
    float a = 1.0, b = 1.0;
    if (coord.x < inset) {
        a *= coord.x / inset;
    } else if (coord.x > 1.0 - inset) {
        a *= (1.0 - coord.x) / inset;
    }
    if (coord.y < inset) {
        b *= coord.y / inset;
    } else if (coord.y > 1.0 - inset) {
        b *= (1.0 - coord.y) / inset;
    }
    float k = ease_${ease}(a) * ease_${ease}(b) * alpha;
%     for name in child_textures:
%         if loop.index == 0:
    vec4 merged = texture(${name}, coord);
%         else:
    merged = composite_${composite}(texture(${name}, coord), merged);
%         endif
%     endfor
    color = merged * k;
% else:
    color = vec4(0.0);
% endif
}
