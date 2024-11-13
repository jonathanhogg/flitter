
// This OpenSimplex 2S (3D ImprovepY) implementation is lifted from
// Kurt Spencer's original (which has been placed into the public domain):
// https://github.com/KdotJPG/OpenSimplex2/blob/master/glsl/OpenSimplex2S.glsl
//
// Other than some renaming and reformatting, the main change is to allow a
// seed hash to be passed in.

const mat3 opensimplex2s_orthonormal_map = mat3(0.788675134594813, -0.211324865405187, -0.577350269189626,
                                                -0.211324865405187, 0.788675134594813, -0.577350269189626,
                                                0.577350269189626, 0.577350269189626, 0.577350269189626);

vec4 opensimplex2s_permute(vec4 t) {
    t = mod(t, 289.0);
    return t * (t * 34.0 + 133.0);
}

vec3 opensimplex2s_grad(float hash) {
    vec3 cube = mod(floor(hash / vec3(1.0, 2.0, 4.0)), 2.0) * 2.0 - 1.0;
    vec3 cuboct = cube;
    cuboct[int(hash / 16.0)] = 0.0;
    float type = mod(floor(hash / 8.0), 2.0);
    vec3 rhomb = (1.0 - type) * cube + type * (cuboct + cross(cube, cuboct));
    vec3 grad = cuboct * 1.22474487139 + rhomb;
    return grad * (1.0 - 0.042942436724648037 * type) * 3.5946317686139184;
}

float opensimplex2s_derivatives(vec4 seed_hashes, vec3 p) {
    vec3 b = floor(p);
    vec4 i4 = vec4(p - b, 2.5);
    vec3 v1 = b + floor(dot(i4, vec4(.25)));
    vec3 v2 = b + vec3(1.0, 0.0, 0.0) + vec3(-1.0, 1.0, 1.0) * floor(dot(i4, vec4(-.25, .25, .25, .35)));
    vec3 v3 = b + vec3(0.0, 1.0, 0.0) + vec3(1.0, -1.0, 1.0) * floor(dot(i4, vec4(.25, -.25, .25, .35)));
    vec3 v4 = b + vec3(0.0, 0.0, 1.0) + vec3(1.0, 1.0, -1.0) * floor(dot(i4, vec4(.25, .25, -.25, .35)));
    vec4 hashes = opensimplex2s_permute(seed_hashes + vec4(v1.x, v2.x, v3.x, v4.x));
    hashes = opensimplex2s_permute(hashes + vec4(v1.y, v2.y, v3.y, v4.y));
    hashes = mod(opensimplex2s_permute(hashes + vec4(v1.z, v2.z, v3.z, v4.z)), 48.0);
    vec3 d1 = p - v1;
    vec3 d2 = p - v2;
    vec3 d3 = p - v3;
    vec3 d4 = p - v4;
    vec4 a = max(0.75 - vec4(dot(d1, d1), dot(d2, d2), dot(d3, d3), dot(d4, d4)), 0.0);
    vec4 aa = a * a;
    vec4 aaaa = aa * aa;
    vec3 g1 = opensimplex2s_grad(hashes.x);
    vec3 g2 = opensimplex2s_grad(hashes.y);
    vec3 g3 = opensimplex2s_grad(hashes.z);
    vec3 g4 = opensimplex2s_grad(hashes.w);
    vec4 extrapolations = vec4(dot(d1, g1), dot(d2, g2), dot(d3, g3), dot(d4, g4));
    return dot(aaaa, extrapolations);
}

float opensimplex2s(vec4 seed_hashes, vec3 p) {
    p = dot(p, vec3(2.0/3.0)) - p;
    return opensimplex2s_derivatives(seed_hashes, p) + opensimplex2s_derivatives(seed_hashes, p + 144.5);
}

float opensimplex2s_improvexy(vec4 seed_hashes, vec3 p) {
    p = opensimplex2s_orthonormal_map * p;
    return opensimplex2s_derivatives(seed_hashes, p) + opensimplex2s_derivatives(seed_hashes, p + 144.5);
}
