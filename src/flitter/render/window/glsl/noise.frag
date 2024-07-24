${HEADER}

// This OpenSimplex 2S (3D ImproveXY) implementation is largely lifted from
// Kurt Spencer's original (which has been placed into the public domain):
// https://github.com/KdotJPG/OpenSimplex2/blob/master/glsl/OpenSimplex2S.glsl

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
uniform float alpha;
uniform float gamma;
% for name in child_textures:
uniform sampler2D ${name};
% endfor


<%include file="composite_functions.glsl"/>


const mat3 orthonormalMap = mat3(0.788675134594813, -0.211324865405187, -0.577350269189626,
                                 -0.211324865405187, 0.788675134594813, -0.577350269189626,
                                 0.577350269189626, 0.577350269189626, 0.577350269189626);

vec4 permute(vec4 t) {
    t = mod(t, 289.0);
    return t * (t * 34.0 + 133.0);
}

vec3 grad(float hash) {
    vec3 cube = mod(floor(hash / vec3(1.0, 2.0, 4.0)), 2.0) * 2.0 - 1.0;
    vec3 cuboct = cube;
    cuboct[int(hash / 16.0)] = 0.0;
    float type = mod(floor(hash / 8.0), 2.0);
    vec3 rhomb = (1.0 - type) * cube + type * (cuboct + cross(cube, cuboct));
    vec3 grad = cuboct * 1.22474487139 + rhomb;
    return grad * (1.0 - 0.042942436724648037 * type) * 3.5946317686139184;
}

float openSimplex2SDerivativesPart(vec4 hashes, vec3 X) {
    vec3 b = floor(X);
    vec4 i4 = vec4(X - b, 2.5);
    vec3 v1 = b + floor(dot(i4, vec4(.25)));
    vec3 v2 = b + vec3(1.0, 0.0, 0.0) + vec3(-1.0, 1.0, 1.0) * floor(dot(i4, vec4(-.25, .25, .25, .35)));
    vec3 v3 = b + vec3(0.0, 1.0, 0.0) + vec3(1.0, -1.0, 1.0) * floor(dot(i4, vec4(.25, -.25, .25, .35)));
    vec3 v4 = b + vec3(0.0, 0.0, 1.0) + vec3(1.0, 1.0, -1.0) * floor(dot(i4, vec4(.25, .25, -.25, .35)));
    hashes = permute(hashes + vec4(v1.x, v2.x, v3.x, v4.x));
    hashes = permute(hashes + vec4(v1.y, v2.y, v3.y, v4.y));
    hashes = mod(permute(hashes + vec4(v1.z, v2.z, v3.z, v4.z)), 48.0);
    vec3 d1 = X - v1; vec3 d2 = X - v2; vec3 d3 = X - v3; vec3 d4 = X - v4;
    vec4 a = max(0.75 - vec4(dot(d1, d1), dot(d2, d2), dot(d3, d3), dot(d4, d4)), 0.0);
    vec4 aa = a * a; vec4 aaaa = aa * aa;
    vec3 g1 = grad(hashes.x); vec3 g2 = grad(hashes.y);
    vec3 g3 = grad(hashes.z); vec3 g4 = grad(hashes.w);
    vec4 extrapolations = vec4(dot(d1, g1), dot(d2, g2), dot(d3, g3), dot(d4, g4));
    return dot(aaaa, extrapolations);
}

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
    hashes[0] = permute(vec4(seed_hash));
    hashes[1] = permute(hashes[0] + 1.0);
    hashes[2] = permute(hashes[1] + 1.0);
    hashes[3] = permute(hashes[2] + 1.0);
    vec3 sum = vec3(0.0);
    float multiplier = 1.0;
    float weight = 0.0;
    for (int i = 0; i < octaves; i++) {
        vec3 p = orthonormalMap * point / multiplier;
        for (int j = 0; j < components; j++) {
            float result = openSimplex2SDerivativesPart(hashes[j], p) + openSimplex2SDerivativesPart(hashes[j], p + 144.5);
            sum[j] += result * multiplier;
        }
        weight += multiplier;
        multiplier *= roughness;
    }
    vec4 merged = vec4(sum/weight*0.5 + 0.5, 1.0);
    color = gamma == 1.0 ? merged * alpha : pow(merged * alpha, vec4(gamma));
}
