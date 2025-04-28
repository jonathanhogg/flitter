
vec3 srgb_transfer(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, ceil(c - 0.0031308));
}

float srgb_luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 tonemap_reinhard(vec3 color, float whitepoint) {
    vec3 cd = color / (1.0 + color);
    if (whitepoint > 1.0) {
        cd *= (1.0 + color / (whitepoint * whitepoint));
    }
    return cd;
}

// This function based on https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
//
vec3 tonemap_aces(vec3 color)
{
    color = mat3(0.59719, 0.076, 0.0284, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777) * color;
    color = (color * (color + 0.0245786) - 0.000090537) / (color * (0.983729 * color + 0.4329510) + 0.238081);
    return mat3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602) * color;
}

// These functions based on https://iolite-engine.com/blog_posts/minimal_agx_implementation
//
vec3 agx_default_contrast_approx(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return 15.5*x4*x2 - 40.14*x4*x + 31.96*x4 - 6.868*x2*x + 0.4298*x2 + 0.1191*x - 0.00232;
}

vec3 agx(vec3 color) {
    const mat3 agx_mat = mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                              0.0784335999999992, 0.878468636469772, 0.0784336,
                              0.0792237451477643, 0.0791661274605434, 0.879142973793104);
	const float min_ev = -12.47393;
	const float max_ev = 4.026069;
    color = agx_mat * color;
    color = (clamp(log2(color), min_ev, max_ev) - min_ev) / (max_ev - min_ev);
    return agx_default_contrast_approx(color);
}

vec3 agx_eotf(vec3 color) {
    const mat3 agx_mat_inv = mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                                  -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                                  -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    return agx_mat_inv * color;
}

vec3 tonemap_agx(vec3 color)
{
    color = agx(color);
    return agx_eotf(color);
}

vec3 tonemap_agx_punchy(vec3 color)
{
    color = agx(color);
    float luma = srgb_luminance(color);
    color = pow(color, vec3(1.35));
    color = luma + 1.4 * (color - luma);
    return agx_eotf(color);
}

vec3 rgb_to_hsv(vec3 color) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(color.bg, K.wz), vec4(color.gb, K.xy), step(color.b, color.g));
    vec4 q = mix(vec4(p.xyw, color.r), vec4(color.r, p.yzx), step(p.x, color.r));
    float d = q.x - min(q.w, q.y);
    const float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv_to_rgb(vec3 color) {
    const vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(color.xxx + K.xyz) * 6.0 - K.www);
    return color.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), color.y);
}

vec3 hsl_to_rgb(vec3 color) {
    vec3 hsl = clamp(color, vec3(0.0), vec3(1.0));
    if (hsl.y == 0.0) {
        return hsv_to_rgb(hsl);
    } else {
        float v = min(1.0, hsl.z / (1.0 - hsl.y / 2.0));
        float s = (hsl.z == 0.0 || hsl.z == 1.0) ? 0.0 : (v - hsl.z) / min(hsl.z / 2.0, 1.0 - hsl.z / 2.0);
        return hsv_to_rgb(vec3(hsl.x, s, v));
    }
}
