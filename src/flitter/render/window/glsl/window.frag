${HEADER}

in vec2 coord;
out vec4 color;
uniform float alpha;
uniform float gamma;
% for name in child_textures:
uniform sampler2D ${name};
% endfor

<%include file="composite_functions.glsl"/>

float srgb(float c) {
     if (isnan(c))
         return 0.0;
     if (c > 1.0)
         return 1.0;
     if (c < 0.0)
         return 0.0;
     if (c < 0.0031308)
         return 12.92 * c;
     return 1.055 * pow(c, 0.41666) - 0.055;
}

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
    color = vec4(srgb(merged.r), srgb(merged.g), srgb(merged.b), 1);
% else:
    color = vec4(0.0, 0.0, 0.0, 1.0);
% endif
}
