${HEADER}

in vec2 coord;
out vec4 color;

uniform vec2 size;
uniform float seed_hash;
uniform int components;
uniform int octaves;
uniform float roughness;
uniform vec2 origin;
uniform float z;
uniform vec3 scale;
uniform vec3 tscale;
uniform float offset;
uniform float multiplier;
uniform vec4 default_values;
uniform float alpha;
% for name in child_textures:
uniform sampler2D ${name};
% endfor


<%include file="composite_functions.glsl"/>
<%include file="noise_functions.glsl"/>


void main() {
% if child_textures:
%     for name in child_textures:
%         if loop.index == 0:
    vec4 c = texture(${name}, coord);
%         else:
    c = composite_${composite}(texture(${name}, coord), c);
%         endif
%     endfor
% else:
    vec4 c = vec4(0.0);
% endif
    vec3 point = (vec3(coord*size + origin, z) + c.xyz*tscale) * scale;
    mat4 hashes;
    hashes[0] = vec4(seed_hash);
    hashes[1] = opensimplex2s_permute(hashes[0] + 1.0);
    hashes[2] = opensimplex2s_permute(hashes[1] + 1.0);
    hashes[3] = opensimplex2s_permute(hashes[2] + 1.0);
    vec4 sum = vec4(0.0);
    float k = 1.0;
    float weight = 0.0;
    for (int i = 0; i < octaves; i++) {
        vec3 p = point / k;
        vec4 c = vec4(0.0);
        for (int j = 0; j < components; j++) {
            c[j] = opensimplex2s_improvexy(hashes[j], p);
        }
        sum += c * k;
        weight += k;
        k *= roughness;
    }
    vec4 merged = default_values;
    for (int i = 0; i < components; i++) {
        merged[i] = sum[i] / weight * multiplier + offset;
    }
    color = merged * alpha;
}
