"""
Push LED color management
"""

# pylama:ignore=C0103


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


class SimplePalete(Palette):
    """A simple gamma-corrected palette"""

    def rgb_to_index(self, r, g, b):
        r, g, b = (int(round(min(max(0, c), 1) * 4)) for c in (r, g, b))
        return ((r * 5) + g) * 5 + b

    def index_to_rgb_led(self, i):
        i = min(max(0, int(i)), 124)
        r, g, b = i // 25, (i // 5) % 5, i % 5
        return tuple(int(((c / 4) ** 2) * 255) for c in (r, g, b))

    def white_to_index(self, w):
        return int(round(min(max(0, w), 1) * 127))

    def index_to_white_led(self, i):
        i = min(max(0, int(i)), 127)
        return int(((i / 127) ** 2) * 255)
