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


const float Pi = 3.141592653589793115997963468544185161590576171875;
const float Tau = 6.283185307179586231995926937088370323181152343750;


void main() {
% if noise_shape == 'cylinder':
    float th = Tau * coord.x;
    float r = size.x / Tau;
    vec3 point = vec3(vec2(cos(th), sin(th))*r, coord.y*size.y);
% elif noise_shape == 'sphere':
    float th = Tau * coord.x;
    float r = size.x / Tau;
    float lat = Pi * (coord.y - 0.5);
    vec3 point = vec3(vec2(cos(th), sin(th))*cos(lat)*r, sin(lat)*r);
% else:
    vec3 point = vec3(coord*size, 0);
% endif
    point += vec3(origin, z);
% if child_textures:
%     for name in child_textures:
%         if loop.index == 0:
    vec4 s = texture(${name}, coord);
%         else:
    s = composite_${composite}(texture(${name}, coord), s);
%         endif
%     endfor
    point += s * tscale;
% endif
    point *= scale;
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
% if noise_shape in ('cylinder', 'sphere'):
            c[j] = opensimplex2s(hashes[j], p);
% else:
            c[j] = opensimplex2s_improvexy(hashes[j], p);
% endif
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
