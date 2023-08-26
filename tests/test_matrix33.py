"""
Tests of the model.Matrix33 class
"""

import math
import unittest

from flitter.model import Matrix33


def all_isclose(xs, ys, rel_tol=1e-9, abs_tol=0):
    for x, y in zip(xs, ys):
        if not math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol):
            return False
    return True


class TestMatrix33(unittest.TestCase):
    """
    Tests of the Matrix33 class

    Note that matrices are column-major as per OpenGL.
    """

    def test_construct(self):
        self.assertEqual(Matrix33(), [1, 0, 0, 0, 1, 0, 0, 0, 1])
        self.assertEqual(Matrix33(0), [0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(Matrix33(1), [1, 0, 0, 0, 1, 0, 0, 0, 1])
        self.assertEqual(Matrix33(2), [2, 0, 0, 0, 2, 0, 0, 0, 2])
        self.assertEqual(Matrix33(range(9)), range(9))
        self.assertRaises(ValueError, Matrix33, "Hello world!")
        self.assertRaises(ValueError, Matrix33, [1, 2, 3])

    def test_translate(self):
        self.assertIsNone(Matrix33.translate(None))
        self.assertIsNone(Matrix33.translate([0, 1, 2]))
        self.assertIsNone(Matrix33.translate("Hello world!"))
        self.assertEqual(Matrix33.translate([2]), [1, 0, 0, 0, 1, 0, 2, 2, 1])
        self.assertEqual(Matrix33.translate([2, 3]), [1, 0, 0, 0, 1, 0, 2, 3, 1])

    def test_scale(self):
        self.assertIsNone(Matrix33.scale(None))
        self.assertIsNone(Matrix33.scale([0, 1, 2]))
        self.assertIsNone(Matrix33.scale("Hello world!"))
        self.assertEqual(Matrix33.scale(2), [2, 0, 0, 0, 2, 0, 0, 0, 1])
        self.assertEqual(Matrix33.scale([2, 3]), [2, 0, 0, 0, 3, 0, 0, 0, 1])

    def test_rotate(self):
        self.assertIsNone(Matrix33.rotate(math.nan))
        self.assertEqual(Matrix33.rotate(0), Matrix33())
        self.assertTrue(all_isclose(Matrix33.rotate(1), Matrix33(), abs_tol=1e-12))
        self.assertTrue(all_isclose(Matrix33.rotate(0.5), [-1, 0, 0, 0, -1, 0, 0, 0, 1], abs_tol=1e-12))

    def test_mmul(self):
        self.assertEqual(Matrix33() @ Matrix33(), Matrix33())
        self.assertEqual(Matrix33.translate([1, 2]) @ Matrix33(), Matrix33.translate([1, 2]))
        self.assertEqual(Matrix33() @ Matrix33.translate([1, 2]), Matrix33.translate([1, 2]))
        self.assertEqual(Matrix33.translate([1, 2]) @ Matrix33.translate([1, 2]), Matrix33.translate([2, 4]))
        self.assertEqual(Matrix33.translate([1, 2]) @ Matrix33.scale(2), [2, 0, 0, 0, 2, 0, 1, 2, 1])
        self.assertEqual(Matrix33.scale(2) @ Matrix33.translate([1, 2]), [2, 0, 0, 0, 2, 0, 2, 4, 1])

    def test_vmul(self):
        self.assertEqual(Matrix33() @ None, None)
        self.assertEqual(Matrix33() @ [], None)
        self.assertEqual(Matrix33() @ [1], None)
        self.assertEqual(Matrix33() @ [1, 2], [1, 2])
        self.assertEqual(Matrix33() @ [1, 2, 3], [1, 2, 3])
        self.assertEqual(Matrix33() @ [1, 2, 3, 1], None)
        self.assertEqual(Matrix33.translate([1, 2]) @ [0, 0], [1, 2])
        self.assertEqual(Matrix33.translate([1, 2]) @ [-1, -2, 1], [0, 0, 1])
        self.assertEqual(Matrix33.scale(2) @ [-1, -2, 1], [-2, -4, 1])

    def test_inverse(self):
        a = Matrix33.scale([2, 3])
        b = Matrix33.rotate(0.25)
        c = Matrix33.translate([7, 9])
        self.assertTrue(all_isclose(a.inverse() @ a, Matrix33()))
        self.assertTrue(all_isclose(b.inverse() @ b, Matrix33()))
        self.assertTrue(all_isclose(b.inverse() @ (a.inverse() @ (a @ (b @ c))), c, abs_tol=1e-12))
        self.assertTrue(all_isclose((((b.inverse() @ a.inverse()) @ a) @ b) @ c, c))

    def test_transpose(self):
        self.assertEqual(Matrix33().transpose(), Matrix33())
        self.assertEqual(Matrix33(range(9)).transpose(), [0, 3, 6, 1, 4, 7, 2, 5, 8])

    def test_repr(self):
        self.assertEqual(repr(Matrix33()), """|   1.000   0.000   0.000 |
|   0.000   1.000   0.000 |
|   0.000   0.000   1.000 |""")
