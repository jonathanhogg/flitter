
<%include file="color_functions.glsl"/>

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

vec3 filter_adjust(vec3 color, float exposure, float contrast, float brightness, float shadows, float highlights) {
    float offset = brightness + (1.0 - contrast) / 2.0;
    color = color * pow(2.0, exposure) * contrast + offset;
    float l = srgb_luminance(color);
    color *= pow(2.0, shadows * clamp(4.0 * (0.25 - l), 0.0, 1.0));
    color *= pow(2.0, highlights * clamp(2.0 * (l - 0.5), 0.0, 1.0));
    return color;
}

const int SpectrumSize = 6;
const vec3[SpectrumSize] SpectrumWeights = vec3[](vec3(0.1, 0.0, 0.3),
                                                  vec3(0.0, 0.0, 0.5),
                                                  vec3(0.0, 0.25, 0.2),
                                                  vec3(0.0, 0.5, 0.0),
                                                  vec3(0.4, 0.25, 0.0),
                                                  vec3(0.5, 0.0, 0.0));

vec3 filter_lens_ghost(sampler2D tex, vec2 coord, vec2 size, float distort, float scale, float threshold, float attenuation, float aberration) {
    vec2 p = (coord - 0.5) * size;
    float r = length(p);
    p /= r;
    float rmax = length(size) / 2.0;
    r = (1.0 - pow(1.0 - r / rmax, distort)) * rmax / scale;
    vec2 q = r * p;
    float w = clamp(r / rmax, 0.0, 1.0);
    w *= w;
    w = 1.0 - w;
    vec3 color = vec3(0.0);
    if (aberration > 0.0) {
        for (int j = 0; j < SpectrumSize; j++) {
            vec2 qq = q * (1.0 + (float(j)/float(SpectrumSize - 1) - 0.5) * 0.25 * aberration);
            vec3 col = texture(tex, qq / size + 0.5).rgb;
            color += col * smoothstep(0.0, 1.0, srgb_luminance(col) - threshold) * SpectrumWeights[j];
        }
    } else {
        vec3 col = texture(tex, q / size + 0.5).rgb;
        color += col * smoothstep(0.0, 1.0, srgb_luminance(col) - threshold);
    }
    return color * pow(0.5, attenuation) * w;
}

vec3 filter_lens_flare(sampler2D tex, vec2 coord, vec2 size, float upright_length, float diagonal_length, float threshold, float attenuation) {
    float l = min(size.x, size.y) * 0.5;
    vec3 upright_color = vec3(0.0);
    float n = upright_length * l;
    if (n > 0.0) {
        vec2 N = vec2(0.0, 1.0) / size;
        vec2 E = vec2(1.0, 0.0) / size;
        vec2 S = vec2(0.0, -1.0) / size;
        vec2 W = vec2(-1.0, 0.0) / size;
        for (float i = 0.0; i < n; i += 1.0) {
            vec3 col = texture(tex, coord + i * N).rgb;
            col = max(col, texture(tex, coord + i * E).rgb);
            col = max(col, texture(tex, coord + i * S).rgb);
            col = max(col, texture(tex, coord + i * W).rgb);
            float w = 1.0 - i / n;
            w *= w;
            upright_color += col * smoothstep(0.0, 1.0, srgb_luminance(col) - threshold) * w;
        }
        upright_color /= n;
    }
    vec3 diagonal_color = vec3(0.0);
    n = diagonal_length * l * sqrt(2.0) / 2.0;
    if (n > 0.0) {
        vec2 NE = vec2(1.0, 1.0) / size;
        vec2 SE = vec2(1.0, -1.0) / size;
        vec2 SW = vec2(-1.0, -1.0) / size;
        vec2 NW = vec2(-1.0, 1.0) / size;
        for (float i = 0.0; i < n; i += 1.0) {
            vec3 col = texture(tex, coord + i * NE).rgb;
            col = max(col, texture(tex, coord + i * SE).rgb);
            col = max(col, texture(tex, coord + i * SW).rgb);
            col = max(col, texture(tex, coord + i * NW).rgb);
            float w = 1.0 - i / n;
            w *= w;
            diagonal_color += col * smoothstep(0.0, 1.0, srgb_luminance(col) - threshold) * w;
        }
        diagonal_color /= n;
    }
    return max(upright_color, diagonal_color) * pow(0.5, attenuation);
}

vec3 filter_lens_halo(sampler2D tex, vec2 coord, vec2 size, float radius, float threshold, float attenuation) {
    const float Tau = 6.283185307179586231995926937088370323181152343750;
    float l = min(size.x, size.y) * 0.5;
    vec3 color = vec3(0.0);
    float r = radius * l;
    float n = Tau * r;
    if (n > 0.0) {
        vec2 s = r / size;
        for (float i = 0.0; i < n; i += 1.0) {
            float th = i / n * Tau;
            vec3 col = texture(tex, coord + s * vec2(cos(th), sin(th))).rgb;
            color += col * smoothstep(0.0, 1.0, srgb_luminance(col) - threshold);
        }
        color /= n;
    }
    return color * pow(0.5, attenuation);
}
