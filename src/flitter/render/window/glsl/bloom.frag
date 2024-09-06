${HEADER}

in vec2 coord;
out vec4 color;
uniform float alpha;
uniform vec2 size;
uniform int pass;
uniform int downsample;
uniform ivec2 radius;
uniform vec2 sigma;
uniform float exposure;
uniform float contrast;
uniform float brightness;
uniform float shadows;
uniform float highlights;
% if passes == 5:
uniform sampler2D first;
% endif
% if passes > 1:
uniform sampler2D last;
%     for name in child_textures:
uniform sampler2D ${name};
%     endfor
% endif

<%include file="composite_functions.glsl"/>
<%include file="filter_functions.glsl"/>

void main() {
% if passes > 1:
    switch (pass) {
%     if passes == 5:
        case 0: {
%         for name in child_textures:
%             if loop.index == 0:
            vec4 merged = texture(${name}, coord);
%             else:
            merged = composite_${composite}(texture(${name}, coord), merged);
%             endif
%         endfor
            color = merged;
            break;
        }
%     endif
        case ${passes - 4}: {
            vec4 merged = texture(${'last' if passes == 5 else 'texture0'}, coord);
            vec3 col = merged.a > 0.0 ? merged.rgb / merged.a : vec3(0.0);
            col = filter_adjust(col, exposure, contrast, brightness, shadows, highlights);
            color = vec4(col * merged.a, merged.a);
            break;
        }
        case ${passes - 3}: {
            int r = radius.x / downsample;
            color = filter_blur(last, coord, r, float(r) * sigma.x, vec2(1.0, 0.0) / size);
            break;
        }
        case ${passes - 2}: {
            int r = radius.y / downsample;
            color = filter_blur(last, coord, r, float(r) * sigma.y, vec2(0.0, 1.0) / size);
            break;
        }
        case ${passes - 1}: {
            int r = radius.y / downsample;
            vec4 merged = composite_lighten(texture(${'first' if passes == 5 else 'texture0'}, coord), texture(last, coord));
            color = merged * alpha;
            break;
        }
    }
% else:
    color = vec4(0.0);
% endif
}
