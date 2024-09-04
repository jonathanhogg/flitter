
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

vec3 filter_adjust(vec3 color, float exposure, float contrast, float brightness) {
    float offset = brightness + (1.0 - contrast) / 2.0;
    return color * pow(2.0, exposure) * contrast + offset;
}

const int SpectrumSize = 6;
const vec3[SpectrumSize] SpectrumWeights = vec3[](vec3(0.05, 0.0, 0.3),
                               vec3(0.0, 0.0, 0.5),
                               vec3(0.0, 0.25, 0.2),
                               vec3(0.0, 0.5, 0.0),
                               vec3(0.25, 0.25, 0.0),
                               vec3(0.5, 0.0, 0.0));

vec3 filter_lens_ghost(sampler2D tex, vec2 coord, vec2 size, float distort, float scale, float threshold, float attenuation, float aberration) {
    vec2 p = (coord - 0.5) * size;
    float r = length(p);
    float th = atan(p.y, p.x);
    float rmax = max(size.x, size.y) / 2.0;
    r = (1.0 - pow(1.0 - r / rmax, distort)) * rmax / scale;
    vec2 q = r * vec2(cos(th), sin(th));
    float w = clamp(r / rmax, 0.0, 1.0);
    w *= w;
    w = 1.0 - w;
    vec3 color = vec3(0.0);
    if (aberration > 0.0) {
        for (int j = 0; j < SpectrumSize; j++) {
            vec2 qq = q * (1.0 + (float(j) - 2.5) * 0.05 * aberration);
            color += texture(tex, qq / size + 0.5).rgb * SpectrumWeights[j];
        }
    } else {
        color += texture(tex, q / size + 0.5).rgb;
    }
    // float k = clamp(dot(color, vec3(1.0)) / 3.0 - threshold, 0.0, 1.0);
    float k = clamp(srgb_luminance(color) - threshold, 0.0, 1.0);
    return color * k / pow(2.0, attenuation) * w;
}

vec3 filter_lens_flare(sampler2D tex, vec2 coord, vec2 size, float threshold, float upright_length, float diagonal_length) {
    vec2 p = (coord - 0.5) * size;
    float th = atan(p.y, p.x);
    float l = length(size) * 0.5;
    vec3 upright_color = vec3(0.0);
    float n = upright_length * l;
    if (n > 0.0) {
        vec2 N = vec2(0.0, 1.0) / size;
        vec2 E = vec2(1.0, 0.0) / size;
        vec2 S = vec2(0.0, -1.0) / size;
        vec2 W = vec2(-1.0, 0.0) / size;
        float k = 0.25 / n;
        for (float i = 0.0; i < n; i += 1.0) {
            float w = 1.0 - i / n;
            w *= w;
            vec3 col = texture(tex, coord + i * N).rgb;
            col += texture(tex, coord + i * E).rgb;
            col += texture(tex, coord + i * S).rgb;
            col += texture(tex, coord + i * W).rgb;
            w *= clamp(dot(col, vec3(1.0)) / 12.0 - threshold, 0.0, 1.0);
            upright_color += col * k * w;
        }
    }
    vec3 diagonal_color = vec3(0.0);
    n = diagonal_length * l;
    if (n > 0.0) {
        vec2 NE = vec2(0.7071, 0.7071) / size;
        vec2 SE = vec2(0.7071, -0.7071) / size;
        vec2 SW = vec2(-0.7071, -0.7071) / size;
        vec2 NW = vec2(-0.7071, 0.7071) / size;
        float k = 0.25 / n;
        for (float i = 0.0; i < n; i += 1.0) {
            float w = 1.0 - i / n;
            w *= w;
            vec3 col = texture(tex, coord + i * NE).rgb;
            col += texture(tex, coord + i * SE).rgb;
            col += texture(tex, coord + i * SW).rgb;
            col += texture(tex, coord + i * NW).rgb;
            w *= clamp(dot(col, vec3(1.0)) / 12.0 - threshold, 0.0, 1.0);
            diagonal_color += col * k * w;

        }
    }
    return max(upright_color, diagonal_color);
}
