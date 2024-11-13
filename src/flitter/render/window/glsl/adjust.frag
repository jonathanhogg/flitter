${HEADER}

in vec2 coord;
out vec4 color;
uniform float alpha;
uniform float gamma;
uniform float exposure;
uniform float contrast;
uniform float brightness;
uniform float shadows;
uniform float highlights;
uniform float hue;
uniform float saturation;
uniform mat3 color_matrix;
% if tonemap_function == 'reinhard':
uniform float whitepoint;
% endif
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
    vec3 col = merged.a > 0.0 ? merged.rgb / merged.a : vec3(0.0);
    vec3 hsv = rgb_to_hsv(color_matrix * col);
    hsv.x += hue;
    hsv.y *= saturation;
    col = filter_adjust(hsv_to_rgb(hsv), exposure, contrast, brightness, shadows, highlights);
    col = max(vec3(0.0), col);
%     if gamma != 1:
    col = pow(col, vec3(gamma));
%     endif
%     if tonemap_function == 'reinhard':
    col = tonemap_reinhard(col, whitepoint);
%     elif tonemap_function:
    col = tonemap_${tonemap_function}(col);
%     endif
    color = vec4(col * merged.a, merged.a) * alpha;
% else:
    color = vec4(0.0);
% endif
}
