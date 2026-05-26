${HEADER}

in vec2 coord;
out vec4 color;
uniform float hue;
uniform float range;
uniform float saturation;
uniform float brightness;
uniform float alpha;
uniform vec2 size;
uniform int pass;
uniform int downsample;
uniform float hard;
uniform ivec2 radius;
uniform vec2 sigma;
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
            vec3 hsv = rgb_to_hsv(texture(${'last' if passes == 5 else 'texture0'}, coord).rgb);
            float hue_min = hue - range / 2.0;
            float hue_max = hue + range / 2.0;
            color = (hue_min < hsv.r && hsv.r < hue_max && saturation < hsv.g && brightness < hsv.b) ? vec4(0.0) : vec4(1.0);
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
            vec4 original = texture(${'first' if passes == 5 else 'texture0'}, coord);
            float mask = texture(last, coord).r;
            color = original * pow(mask, hard) * alpha;
            break;
        }
    }
% else:
    color = vec4(0.0);
% endif
}
