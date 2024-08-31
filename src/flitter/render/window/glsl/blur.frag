${HEADER}

in vec2 coord;
out vec4 color;
uniform float alpha;
uniform vec2 size;
uniform int pass;
uniform ivec2 radius;
uniform vec2 sigma;
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
%     if passes == 3:
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
        case ${passes - 2}: {
            color = filter_blur(${'last' if passes == 3 else 'texture0'}, coord, radius.x, float(radius.x) * sigma.x, vec2(1.0, 0.0) / size);
            break;
        }
        case ${passes - 1}: {
            vec4 blurred = filter_blur(last, coord, radius.y, float(radius.y) * sigma.y, vec2(0.0, 1.0) / size);
            color = blurred * alpha;
            break;
        }
    }
% else:
    color = vec4(0.0);
% endif
}
