
float ease_linear(float x) {
    return clamp(x, 0.0, 1.0);
}

float ease_quad(float x) {
    float x_;
    if (x < 0.5) {
        x_ = max(0.0, x) * 2.0;
        return x_ * x_ / 2.0;
    }
    x_ = (1.0 - min(x, 1.0)) * 2.0;
    return 1.0 - x_ * x_ / 2.0;
}

float ease_cubic(float x) {
    float x_;
    if (x < 0.5) {
        x_ = max(0.0, x) * 2.0;
        return x_ * x_ * x_ / 2.0;
    }
    x_ = (1.0 - min(x, 1.0)) * 2.0;
    return 1.0 - x_ * x_ * x_ / 2.0;
}
