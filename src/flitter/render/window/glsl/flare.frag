${HEADER}

in vec2 coord;
out vec4 color;
uniform float alpha;
uniform vec2 size;
uniform int pass;
uniform int ghosts;
uniform float upright_length;
uniform float diagonal_length;
uniform float halo_radius;
uniform float halo_attenuation;
uniform float threshold;
uniform float attenuation;
uniform float aberration;
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
        case ${passes-4}: {
            vec3 col = vec3(0.0);
% if ghosts > 0:
            col += filter_lens_ghost(${'last' if passes == 5 else 'texture0'}, coord, size, 1.0, -1.25, threshold, attenuation, aberration);
% endif
% if ghosts > 1:
            col += filter_lens_ghost(${'last' if passes == 5 else 'texture0'}, coord, size, 1.5, -0.25, threshold, attenuation, aberration);
% endif
% if ghosts > 2:
            col += filter_lens_ghost(${'last' if passes == 5 else 'texture0'}, coord, size, 0.5, 0.25, threshold, attenuation + 1.0, aberration);
% endif
% if ghosts > 3:
            col += filter_lens_ghost(${'last' if passes == 5 else 'texture0'}, coord, size, 0.75, -3.0, threshold, attenuation + 2.0, aberration / 2.0);
% endif
% if ghosts > 4:
            col += filter_lens_ghost(${'last' if passes == 5 else 'texture0'}, coord, size, 0.25, 2.0, threshold, attenuation + 3.0, aberration / 2.0);
% endif
% if ghosts > 5:
            col += filter_lens_ghost(${'last' if passes == 5 else 'texture0'}, coord, size, 1.0, -0.75, threshold, attenuation + 1.0, aberration);
% endif
            col += filter_lens_halo(${'last' if passes == 5 else 'texture0'}, coord, size, halo_radius, threshold, attenuation - 0.5 + halo_attenuation);
            color = vec4(col, 1.0);
            break;
        }
        case ${passes-3}: {
            int radius = int(min(size.x, size.y) * 0.05);
            color = filter_blur(last, coord, radius, float(radius) * 0.3, vec2(1.0, 0.0) / size);
            break;
        }
        case ${passes-2}: {
            int radius = int(min(size.x, size.y) * 0.05);
            vec4 merged = filter_blur(last, coord, radius, float(radius) * 0.3, vec2(0.0, 1.0) / size);
            merged.rgb += filter_lens_flare(${'first' if passes == 5 else 'texture0'}, coord, size, upright_length, diagonal_length, threshold, attenuation);
            color = merged;
            break;
        }
        case ${passes-1}: {
            vec4 merged = composite_add(texture(${'first' if passes == 5 else 'texture0'}, coord), texture(last, coord));
            color = merged * alpha;
            break;
        }
    }
% else:
    color = vec4(0.0);
% endif
}
