
vec3 srgb_transfer(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, ceil(c - 0.0031308));
}

float srgb_luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 tonemap_reinhard(vec3 c, float whitepoint) {
    vec3 cd = c / (1.0 + c);
    if (whitepoint > 1.0) {
        cd *= (1.0 + c / (whitepoint * whitepoint));
    }
    return cd;
}

vec3 tonemap_aces(vec3 c) {
    return c * (2.51 * c + 0.03) / (c * (2.43 * c + 0.59) + 0.14);
}
