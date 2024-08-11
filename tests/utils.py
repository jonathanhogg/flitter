
import math
import unittest


class TestCase(unittest.TestCase):
    def assertAllAlmostEqual(self, xs, ys, places=7, delta=None, msg=None):
        self.assertEqual(len(xs), len(ys), msg=msg)
        for x, y in zip(xs, ys):
            self.assertEqual(math.isnan(x), math.isnan(y), msg=msg)
            if not (math.isnan(x) and math.isnan(y)):
                self.assertAlmostEqual(x, y, places=places, delta=delta, msg=msg)
