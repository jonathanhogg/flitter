
import math
import unittest


class TestCase(unittest.TestCase):
    def assertAllAlmostEqual(self, xs, ys, places=7, delta=None, msg=None):
        if msg is None:
            msg = f"{xs!r} unexpectedly not (almost) equal to {ys!r}"
        self.assertEqual(len(xs), len(ys), msg=msg)
        for x, y in zip(xs, ys):
            if hasattr(x, '__len__') and hasattr(y, '__len__'):
                self.assertAllAlmostEqual(x, y, places=places, delta=delta, msg=msg)
                continue
            self.assertEqual(math.isnan(x), math.isnan(y), msg=msg)
            if not (math.isnan(x) and math.isnan(y)):
                self.assertAlmostEqual(x, y, places=places, delta=delta, msg=msg)

    def assertNotAllAlmostEqual(self, xs, ys, places=7, delta=None, msg=None):
        if len(xs) != len(ys):
            return
        for x, y in zip(xs, ys):
            exponent = 10**places if places is not None else None
            if math.isnan(x) != math.isnan(y):
                return
            if not (math.isnan(x) and math.isnan(y)):
                if delta is not None:
                    if abs(x - y) > delta:
                        return
                elif round(x*exponent) != round(y*exponent):
                    return
        self.assertFalse(True, msg=msg if msg else f"{xs!r} unexpectedly (almost) equal to {ys!r}")
