
vec4 blend_over(vec4 s, vec4 d) {
    return s + d * (1 - s.a);
}

vec4 blend_dest_over(vec4 s, vec4 d) {
    return s * (1 - d.a) + d;
}

vec4 blend_lighten(vec4 s, vec4 d) {
    return max(s, d);
}

vec4 blend_darken(vec4 s, vec4 d) {
    return min(s, d);
}
