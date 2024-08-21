${HEADER}

in vec2 coord;
out vec4 color;
uniform float alpha;
uniform float gamma;
uniform float exposure;
uniform float contrast;
uniform float brightness;
uniform mat3 color_matrix;
% for name in child_textures:
uniform sampler2D ${name};
% endfor

<%include file="composite_functions.glsl"/>
<%include file="filter_functions.glsl"/>

void main() {
% if child_textures:
%     for name in child_textures:
%         if loop.index == 0:
    vec4 merged = texture(${name}, coord);
%         else:
    merged = composite_${composite}(texture(${name}, coord), merged);
%         endif
%     endfor
    merged.rgb = color_matrix * merged.rgb;
    merged = filter_adjust(merged, exposure, contrast, brightness);
    color = gamma == 1.0 ? merged * alpha : pow(merged * alpha, vec4(gamma));
% else:
    color = vec4(0.0);
% endif
}
