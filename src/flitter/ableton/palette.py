"""
Push LED color management
"""

import colorsys


class Palette:
    def rgb_to_index(self, r, g, b):
        """Translate RGB values in range 0..1 to an index in range 0..127"""
        raise NotImplementedError()

    def index_to_rgb_led(self, i):
        """Translate RGB index in range 0..127 to LED RGB values in range 0..255"""
        raise NotImplementedError()

    def white_to_index(self, w):
        """Translate white value in range 0..1 to an index in range 0..127"""
        raise NotImplementedError()

    def index_to_white_led(self, i):
        """Translate white index in range 0..127 to LED white value in range 0..255"""
        raise NotImplementedError()


class SimplePalette(Palette):
    """
    A palette that provides 128 levels of white for white LEDs and 5x5x5 colors
    for RGB LEDs, with a 2.2 gamma applied.

    The white LED transfer function approximates 2.2 gamma, but with some
    linearity to reduce stepping at the dark end.
    """

    def rgb_to_index(self, r, g, b):
        r, g, b = (int(round(min(max(0, c), 0.8) * 5)) for c in (r, g, b))
        return ((r * 5) + g) * 5 + b

    def index_to_rgb_led(self, i):
        i = min(max(0, int(i)), 124)
        r, g, b = i // 25, (i // 5) % 5, i % 5
        return tuple(int(((c / 4) ** 2.2) * 255) for c in (r, g, b))

    def white_to_index(self, w):
        return int(min(max(0, w), 0.9921875) * 128)

    def index_to_white_led(self, i):
        value = min(max(0, int(i)), 127)
        return value // 4 + int(224 * (value / 127) ** 2.4)


class PrimaryPalette(SimplePalette):
    """
    This palette changes the simple palette for RGB LEDs to provide 32 levels
    of white and 17 levels each of red, green, blue, yellow, cyan and magenta
    (including a shared value for black). This is a good palette if you don't
    plan to use a lot of colours, but do want to do pulsing effects with the
    LED brightness.

    Again, the LED transfer function approximates 2.2 gamma, but with some
    linearity to reduce stepping at the dark end.
    """

    def rgb_to_index(self, r, g, b):
        r, g, b = (min(max(0, c), 1) for c in (r, g, b))
        value = max(r, g, b)
        if value == 0:
            return 0
        r, g, b = (int(round(c / value)) for c in (r, g, b))
        if r == g == b:
            value = int(min(value * 32, 31))
            return value if value < 16 else value + 0b1100000
        value = int(min(value * 17, 16))
        if value == 0:
            return 0
        return (r << 4) + (g << 5) + (b << 6) + (value - 1)

    def index_to_rgb_led(self, i):
        i = min(max(0, int(i)), 127)
        rgb = i >> 4
        if rgb in (0b000, 0b111):
            value = i & 0b11111
            value += int(224 * (value / 31)**2.4)
            return (value, value, value)
        value = (i & 0b1111) + 1
        value += int(239 * (value / 16)**2.4)
        return (value if rgb & 0b001 else 0, value if rgb & 0b010 else 0, value if rgb & 0b100 else 0)


class HuePalette(SimplePalette):
    """
    This palette provides 32 shades of white, but then 24 different hues at
    2 levels of saturation (0.5 and 1) and 2 levels of brightness (0.5 and 1).
    This is a good palette if you're content with just a couple of levels of
    brightness - e.g., to show toggled/touched status - but want lots of
    different colours.
    """

    def rgb_to_index(self, r, g, b):
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        s = int(round(s * 2))
        if s == 0:
            return int(min(max(0, v), 0.96875) * 32)
        h = int(round(h * 24)) % 24
        v = int(round(v * 2))
        if v == 0:
            return 0
        return ((v - 1) * 48) + ((s - 1) * 24) + h + 32

    def index_to_rgb_led(self, i):
        i = min(max(0, int(i)), 127)
        if i < 32:
            v = i + int(224 * (i / 31)**2.4)
            return (v, v, v)
        i -= 32
        h = i % 24 / 24
        s = ((i // 24) % 2 + 1) / 2
        v = ((i // 48) + 1) / 2
        return tuple(int(c**2.2 * 255) for c in colorsys.hsv_to_rgb(h, s, v))
