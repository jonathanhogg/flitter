"""
Tests of the model.Matrix33 class
"""

import math

import numpy as np

from flitter.model import Matrix33, Matrix44

from . import utils


class TestMatrix33(utils.TestCase):
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
        self.assertEqual(Matrix33([0, 1, 2, 3, 4, 5, 6, 7, 8]), range(9))
        self.assertEqual(Matrix33(np.arange(9)), range(9))
        self.assertRaises(ValueError, Matrix33, "Hello world!")
        self.assertRaises(ValueError, Matrix33, [1, 2, 3])
        self.assertRaises(ValueError, Matrix33, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
        self.assertRaises(ValueError, Matrix33, range(8))
        self.assertRaises(ValueError, Matrix33, np.arange(8))

    def test_copy(self):
        m1 = Matrix33(range(9))
        m2 = m1.copy()
        self.assertEqual(type(m1), type(m2))
        self.assertEqual(m1, m2)
        self.assertIsNot(m1, m2)

    def test_identity(self):
        self.assertEqual(Matrix33.identity(), [1, 0, 0, 0, 1, 0, 0, 0, 1])

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
        self.assertAllAlmostEqual(Matrix33.rotate(1), Matrix33(), places=None, delta=1e-12)
        self.assertAllAlmostEqual(Matrix33.rotate(0.5), [-1, 0, 0, 0, -1, 0, 0, 0, 1], places=None, delta=1e-12)

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
        self.assertAllAlmostEqual(a.inverse() @ a, Matrix33())
        self.assertAllAlmostEqual(b.inverse() @ b, Matrix33())
        self.assertAllAlmostEqual(b.inverse() @ (a.inverse() @ (a @ (b @ c))), c, places=None, delta=1e-12)
        self.assertAllAlmostEqual((((b.inverse() @ a.inverse()) @ a) @ b) @ c, c)
        self.assertAllAlmostEqual(Matrix33([1, 3, -1, 2, 4, 2, 1, 1, -1]).inverse(), [-0.75, 0.25, 1.25, 0.5, 0, -0.5, -0.25, 0.25, -0.25])

    def test_cofactor(self):
        m = Matrix33.scale([2, -3]) @ \
            Matrix33.rotate(1/3) @ \
            Matrix33.translate([7, 9])
        self.assertAllAlmostEqual(m @ m.cofactor().transpose(), Matrix33(m.det()))

    def test_transpose(self):
        self.assertEqual(Matrix33().transpose(), Matrix33())
        self.assertEqual(Matrix33(range(9)).transpose(), [0, 3, 6, 1, 4, 7, 2, 5, 8])

    def test_matrix44(self):
        self.assertEqual(Matrix33().matrix44(), Matrix44())
        self.assertEqual(Matrix33(range(9)).matrix44(), [0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 0, 1])

    def test_repr(self):
        self.assertEqual(repr(Matrix33()), """|   1.000   0.000   0.000 |
|   0.000   1.000   0.000 |
|   0.000   0.000   1.000 |""")
