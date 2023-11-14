
vec4 composite_over(vec4 s, vec4 d) {
    return s + d * (1 - s.a);
}

vec4 composite_dest_over(vec4 s, vec4 d) {
    return s * (1 - d.a) + d;
}

vec4 composite_lighten(vec4 s, vec4 d) {
    return max(s, d);
}

vec4 composite_darken(vec4 s, vec4 d) {
    return min(s, d);
}

vec4 composite_add(vec4 s, vec4 d) {
    return vec4(s.rgb + d.rgb, min(s.a + d.a, 1));
}

vec4 composite_difference(vec4 s, vec4 d) {
    return vec4(abs(d.rgb - s.rgb), max(s.a, d.a));
}

vec4 composite_multiply(vec4 s, vec4 d) {
    return s * d;
}
