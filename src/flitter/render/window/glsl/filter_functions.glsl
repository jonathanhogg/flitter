
vec4 filter_blur(sampler2D tex, vec2 coord, int radius, float sigma, vec2 delta) {
    vec3 gaussian;
    gaussian.x = 1.0;
    gaussian.y = exp(-0.5 / (sigma * sigma));
    gaussian.z = gaussian.y * gaussian.y;
    vec4 color_sum = texture(tex, coord) * gaussian.x;
    float weight_sum = gaussian.x;
    gaussian.xy *= gaussian.yz;
    for (int i = 1; i <= radius; i++) {
        color_sum += texture(tex, coord + float(i) * delta) * gaussian.x;
        color_sum += texture(tex, coord - float(i) * delta) * gaussian.x;
        weight_sum += 2.0 * gaussian.x;
        gaussian.xy *= gaussian.yz;
    }
    return color_sum / weight_sum;
}

vec4 filter_adjust(vec4 color, float exposure, float contrast, float brightness) {
    float offset = brightness + (1.0 - contrast) / 2.0;
    return vec4(max((color.rgb / color.a * pow(2.0, exposure) * contrast + offset) * color.a, 0.0), color.a);
}
