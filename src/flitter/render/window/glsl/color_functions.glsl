
vec3 srgb_transfer(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, ceil(c - 0.0031308));
}

float srgb_luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 tonemap_reinhard(vec3 c, float whitepoint) {
    float l = srgb_luminance(c);
    float ld = l / (1.0 + l);
    if (whitepoint > 0.0) {
        ld *= (1.0 + l / (whitepoint*whitepoint));
    }
    return c * vec3(ld);
}
