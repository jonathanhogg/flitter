${HEADER}

in vec2 coord;
out vec4 color;
uniform float alpha;
uniform float gamma;
uniform vec2 size;
uniform int pass;
uniform ivec2 radius;
uniform vec2 sigma;
% if passes == 4:
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
%     if passes == 4:
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
        case ${passes - 3}: {
            color = filter_blur(${'last' if passes == 4 else 'texture0'}, coord, radius.x, float(radius.x) * sigma.x, vec2(1.0, 0.0) / size);
            break;
        }
        case ${passes - 2}: {
            color = filter_blur(last, coord, radius.y, float(radius.y) * sigma.y, vec2(0.0, 1.0) / size);
            break;
        }
        case ${passes - 1}: {
            vec4 merged = composite_difference(texture(${'first' if passes == 4 else 'texture0'}, coord), texture(last, coord));
            color = gamma == 1.0 ? merged * alpha : pow(merged * alpha, vec4(gamma));
            break;
        }
    }
% else:
    color = vec4(0.0);
% endif
}
