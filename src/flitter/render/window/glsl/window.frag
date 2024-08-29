${HEADER}

in vec2 coord;
out vec4 color;
uniform float alpha;
uniform float gamma;
% for name in child_textures:
uniform sampler2D ${name};
% endfor

<%include file="composite_functions.glsl"/>
<%include file="colorspace_functions.glsl"/>

void main() {
% if child_textures:
%     for name in child_textures:
%         if loop.index == 0:
    vec4 merged = texture(${name}, coord);
%         else:
    merged = composite_${composite}(texture(${name}, coord), merged);
%         endif
%     endfor
    merged = gamma == 1.0 ? merged * alpha : pow(merged * alpha, vec4(gamma));
    color = vec4(srgb_transfer(merged.rgb), 1.0);
% else:
    color = vec4(0.0, 0.0, 0.0, 1.0);
% endif
}
