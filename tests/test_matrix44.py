"""
Tests of the model.Matrix44 class
"""

import math

import numpy as np

from flitter.model import Matrix44, Matrix33

from . import utils


class TestMatrix44(utils.TestCase):
    """
    Tests of the Matrix44 class

    Note that matrices are column-major as per OpenGL.
    """

    def test_construct(self):
        self.assertEqual(Matrix44(), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        self.assertEqual(Matrix44(0), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(Matrix44(1), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        self.assertEqual(Matrix44(2), [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2])
        self.assertEqual(Matrix44(range(16)), range(16))
        self.assertRaises(ValueError, Matrix44, "Hello world!")
        self.assertRaises(ValueError, Matrix44, [1, 2, 3])

        self.assertEqual(Matrix44(), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        self.assertEqual(Matrix44(0), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(Matrix44(1), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        self.assertEqual(Matrix44(2), [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2])
        self.assertEqual(Matrix44(range(16)), range(16))
        self.assertEqual(Matrix44([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]), range(16))
        self.assertEqual(Matrix44(np.arange(16)), range(16))
        self.assertRaises(ValueError, Matrix44, "Hello world!")
        self.assertRaises(ValueError, Matrix44, [1, 2, 3])
        self.assertRaises(ValueError, Matrix44, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'])
        self.assertRaises(ValueError, Matrix44, range(15))
        self.assertRaises(ValueError, Matrix44, np.arange(15))

    def test_copy(self):
        m1 = Matrix44(range(16))
        m2 = m1.copy()
        self.assertEqual(type(m1), type(m2))
        self.assertEqual(m1, m2)
        self.assertIsNot(m1, m2)

    def test_identity(self):
        self.assertEqual(Matrix44.identity(), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    def test_project(self):
        self.assertAllAlmostEqual(Matrix44.project(1, 1, 1, 2), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -3, -1, 0, 0, -4, 0])
        self.assertAllAlmostEqual(Matrix44.project(1, 0.5, 1, 2), [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, -3, -1, 0, 0, -4, 0])
        self.assertAllAlmostEqual(Matrix44.project(1, 0.5, 1, 3), [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, -2, -1, 0, 0, -3, 0])
        self.assertAllAlmostEqual(Matrix44.project(0.5773502691896256, 0.2886751345948128, 1, 3),
                                  [1.7320508075688776, 0, 0, 0, 0, 3.4641016151377553, 0, 0, 0, 0, -2, -1, 0, 0, -3, 0])

    def test_ortho(self):
        self.assertAllAlmostEqual(Matrix44.ortho(1, 2, 0, 2), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 1])
        self.assertAllAlmostEqual(Matrix44.ortho(2, 2, 0, 2), [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, -1, 0, 0, 0, -1, 1])
        self.assertAllAlmostEqual(Matrix44.ortho(2, 1, 0, 2), [2, 0, 0, 0, 0, 4, 0, 0, 0, 0, -1, 0, 0, 0, -1, 1])
        self.assertAllAlmostEqual(Matrix44.ortho(2, 1, 1, 2), [2, 0, 0, 0, 0, 4, 0, 0, 0, 0, -2, 0, 0, 0, -3, 1])
        self.assertAllAlmostEqual(Matrix44.ortho(2, 1, 1, 3), [2, 0, 0, 0, 0, 4, 0, 0, 0, 0, -1, 0, 0, 0, -2, 1])

    def test_look(self):
        self.assertIsNone(Matrix44.look([0], None, "Hello world!"))
        self.assertAllAlmostEqual(Matrix44.look([0, 0, 0], [0, 0, -1], [0, 1, 0]), Matrix44())
        self.assertAllAlmostEqual(Matrix44.look([0, 0, 0], [0, 0, -100], [0, 1, 0]), Matrix44())
        self.assertAllAlmostEqual(Matrix44.look([0, 0, 0], [0, 0, -100], [1, 0, 0]),
                                  [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        self.assertAllAlmostEqual(Matrix44.look([0, 0, 0], [0, 0, 100], [1, 0, 0]),
                                  [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1])
        self.assertAllAlmostEqual(Matrix44.look([1, 2, 3], [1, 2, 2], [0, 1, 0]),
                                  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1, -2, -3, 1])

    def test_translate(self):
        self.assertIsNone(Matrix44.translate(None))
        self.assertIsNone(Matrix44.translate([0, 1]))
        self.assertIsNone(Matrix44.translate("Hello world!"))
        self.assertEqual(Matrix44.translate([2]), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 2, 2, 1])
        self.assertEqual(Matrix44.translate([1, 2, 3]), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1])

    def test_scale(self):
        self.assertIsNone(Matrix44.scale(None))
        self.assertIsNone(Matrix44.scale([0, 1]))
        self.assertIsNone(Matrix44.scale("Hello world!"))
        self.assertEqual(Matrix44.scale(2), [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1])
        self.assertEqual(Matrix44.scale([1, 2, 3]), [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1])

    def test_rotate_x(self):
        self.assertIsNone(Matrix44.rotate_x(math.nan))
        self.assertEqual(Matrix44.rotate_x(0), Matrix44())
        self.assertAllAlmostEqual(Matrix44.rotate_x(1), Matrix44())
        self.assertAllAlmostEqual(Matrix44.rotate_x(0.5), [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1])

    def test_rotate_y(self):
        self.assertIsNone(Matrix44.rotate_y(math.nan))
        self.assertEqual(Matrix44.rotate_y(0), Matrix44())
        self.assertAllAlmostEqual(Matrix44.rotate_y(1), Matrix44())
        self.assertAllAlmostEqual(Matrix44.rotate_y(0.5), [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1])

    def test_rotate_z(self):
        self.assertIsNone(Matrix44.rotate_z(math.nan))
        self.assertEqual(Matrix44.rotate_z(0), Matrix44())
        self.assertAllAlmostEqual(Matrix44.rotate_z(1), Matrix44())
        self.assertAllAlmostEqual(Matrix44.rotate_z(0.5), [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    def test_rotate(self):
        self.assertEqual(Matrix44.rotate(None), Matrix44())
        self.assertEqual(Matrix44.rotate("Hello world!"), Matrix44())
        self.assertEqual(Matrix44.rotate([0.1, 0.2]), Matrix44())
        self.assertEqual(Matrix44.rotate(0), Matrix44())
        self.assertAllAlmostEqual(Matrix44.rotate(1), Matrix44())
        self.assertAllAlmostEqual(Matrix44.rotate(0.5), Matrix44())
        self.assertAllAlmostEqual(Matrix44.rotate([0.5, 0.5, 0.5]), Matrix44())
        self.assertAllAlmostEqual(Matrix44.rotate([0.5, 0, 0]), [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1])
        self.assertAllAlmostEqual(Matrix44.rotate([0, 0.5, 0]), [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1])
        self.assertAllAlmostEqual(Matrix44.rotate([0, 0, 0.5]), [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    def test_shear_x(self):
        self.assertIsNone(Matrix44.shear_x(None))
        self.assertIsNone(Matrix44.shear_x([0, 1, 2]))
        self.assertEqual(Matrix44.shear_x(0), Matrix44())
        self.assertEqual(Matrix44.shear_x([0, 0]), Matrix44())
        self.assertEqual(Matrix44.shear_x([1, 0]), [1, 0, 0, 0,  1, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1])
        self.assertEqual(Matrix44.shear_x([0, 1]), [1, 0, 0, 0,  0, 1, 0, 0,  1, 0, 1, 0,  0, 0, 0, 1])
        self.assertEqual(Matrix44.shear_x([1, 1]), [1, 0, 0, 0,  1, 1, 0, 0,  1, 0, 1, 0,  0, 0, 0, 1])
        self.assertEqual(Matrix44.shear_x(1),      [1, 0, 0, 0,  1, 1, 0, 0,  1, 0, 1, 0,  0, 0, 0, 1])
        self.assertEqual(Matrix44.shear_x([0.5, 0.75]), [1, 0, 0, 0,  0.5, 1, 0, 0,  0.75, 0, 1, 0,  0, 0, 0, 1])

    def test_shear_y(self):
        self.assertIsNone(Matrix44.shear_y(None))
        self.assertIsNone(Matrix44.shear_y([0, 1, 2]))
        self.assertEqual(Matrix44.shear_y(0), Matrix44())
        self.assertEqual(Matrix44.shear_y([0, 0]), Matrix44())
        self.assertEqual(Matrix44.shear_y([1, 0]), [1, 1, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1])
        self.assertEqual(Matrix44.shear_y([0, 1]), [1, 0, 0, 0,  0, 1, 0, 0,  0, 1, 1, 0,  0, 0, 0, 1])
        self.assertEqual(Matrix44.shear_y([1, 1]), [1, 1, 0, 0,  0, 1, 0, 0,  0, 1, 1, 0,  0, 0, 0, 1])
        self.assertEqual(Matrix44.shear_y(1),      [1, 1, 0, 0,  0, 1, 0, 0,  0, 1, 1, 0,  0, 0, 0, 1])
        self.assertEqual(Matrix44.shear_y([0.5, 0.75]), [1, 0.5, 0, 0,  0, 1, 0, 0,  0, 0.75, 1, 0,  0, 0, 0, 1])

    def test_shear_z(self):
        self.assertIsNone(Matrix44.shear_z(None))
        self.assertIsNone(Matrix44.shear_z([0, 1, 2]))
        self.assertEqual(Matrix44.shear_z(0), Matrix44())
        self.assertEqual(Matrix44.shear_z([0, 0]), Matrix44())
        self.assertEqual(Matrix44.shear_z([1, 0]), [1, 0, 1, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1])
        self.assertEqual(Matrix44.shear_z([0, 1]), [1, 0, 0, 0,  0, 1, 1, 0,  0, 0, 1, 0,  0, 0, 0, 1])
        self.assertEqual(Matrix44.shear_z([1, 1]), [1, 0, 1, 0,  0, 1, 1, 0,  0, 0, 1, 0,  0, 0, 0, 1])
        self.assertEqual(Matrix44.shear_z(1),      [1, 0, 1, 0,  0, 1, 1, 0,  0, 0, 1, 0,  0, 0, 0, 1])
        self.assertEqual(Matrix44.shear_z([0.5, 0.75]), [1, 0, 0.5, 0,  0, 1, 0.75, 0,  0, 0, 1, 0,  0, 0, 0, 1])

    def test_mmul(self):
        self.assertEqual(Matrix44() @ Matrix44(), Matrix44())
        self.assertEqual(Matrix44.translate([1, 2, 3]) @ Matrix44(), Matrix44.translate([1, 2, 3]))
        self.assertEqual(Matrix44() @ Matrix44.translate([1, 2, 3]), Matrix44.translate([1, 2, 3]))
        self.assertEqual(Matrix44.translate([1, 2, 3]) @ Matrix44.translate([1, 2, 3]), Matrix44.translate([2, 4, 6]))
        self.assertEqual(Matrix44.translate([1, 2, 3]) @ Matrix44.scale(2), [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 2, 3, 1])
        self.assertEqual(Matrix44.scale(2) @ Matrix44.translate([1, 2, 3]), [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 2, 4, 6, 1])

    def test_immul(self):
        m = m1 = Matrix44()
        m @= m1
        self.assertIs(m, m1)
        self.assertEqual(m, Matrix44())
        m = Matrix44.translate([1, 2, 3])
        m @= Matrix44()
        self.assertEqual(m, Matrix44.translate([1, 2, 3]))
        m = Matrix44()
        m @= Matrix44.translate([1, 2, 3])
        self.assertEqual(m, Matrix44.translate([1, 2, 3]))
        m = Matrix44.translate([1, 2, 3])
        m @= Matrix44.translate([1, 2, 3])
        self.assertEqual(m, Matrix44.translate([2, 4, 6]))
        m = Matrix44.translate([1, 2, 3])
        m @= Matrix44.scale(2)
        self.assertEqual(m, [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 2, 3, 1])
        m = Matrix44.scale(2)
        m @= Matrix44.translate([1, 2, 3])
        self.assertEqual(m, [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 2, 4, 6, 1])
        m @= [1, 1, 1]
        self.assertIsNot(m, m1)
        self.assertEqual(m, [4, 6, 8])

    def test_vmul(self):
        self.assertEqual(Matrix44() @ None, None)
        self.assertEqual(Matrix44() @ [], None)
        self.assertEqual(Matrix44() @ [1], None)
        self.assertEqual(Matrix44() @ [1, 2], None)
        self.assertEqual(Matrix44() @ [1, 2, 3], [1, 2, 3])
        self.assertEqual(Matrix44() @ [1, 2, 3, 1], [1, 2, 3, 1])
        self.assertEqual(Matrix44.translate([1, 2, 3]) @ [0, 0, 0], [1, 2, 3])
        self.assertEqual(Matrix44.translate([1, 2, 3]) @ [-1, -2, -3, 1], [0, 0, 0, 1])
        self.assertEqual(Matrix44.scale(2) @ [-1, -2, -3, 1], [-2, -4, -6, 1])

    def test_inverse(self):
        a = Matrix44.project(1, 1, 1, 3)
        b = Matrix44.scale([5, 6, -3])
        c = Matrix44.look([1, 2, 3], [1, 2, 2], [-1, 0, 0])
        self.assertAllAlmostEqual(a.inverse() @ a, Matrix44())
        self.assertAllAlmostEqual(b.inverse() @ b, Matrix44())
        self.assertAllAlmostEqual(c.inverse() @ c, Matrix44())
        self.assertAllAlmostEqual(b.inverse() @ (a.inverse() @ (a @ (b @ c))), c)
        self.assertAllAlmostEqual((((b.inverse() @ a.inverse()) @ a) @ b) @ c, c)

    def test_transpose(self):
        self.assertEqual(Matrix44().transpose(), Matrix44())
        self.assertEqual(Matrix44(range(16)).transpose(), [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15])

    def test_matrix33(self):
        self.assertEqual(Matrix44().matrix33(), Matrix33())
        self.assertEqual(Matrix44(range(16)).matrix33(), [0, 1, 2, 4, 5, 6, 8, 9, 10])

    def test_inverse_transpose_matrix33(self):
        m = Matrix44.project(2, 1 / 6, 1, 3) @ \
            Matrix44.look([1, 2, 3], [1, 2, 2], [-1, 0, 0]) @ \
            Matrix44.scale([5, 6, -3])
        self.assertAllAlmostEqual(m.inverse_transpose_matrix33(), m.inverse().transpose().matrix33())

    def test_matrix33_cofactor(self):
        m = Matrix44.look([1, 3, 3], [1, 2, 3], [-1, 0, 0]) @ \
            Matrix44.scale([5, 6, -3])
        self.assertAllAlmostEqual(m.matrix33_cofactor(), m.matrix33().cofactor())
        self.assertAllAlmostEqual((m.matrix33_cofactor() @ [1, 0, 0]).normalize(), [0, 1, 0])
        self.assertAllAlmostEqual((m.matrix33_cofactor() @ [0, 1, 0]).normalize(), [0, 0, -1])
        self.assertAllAlmostEqual((m.matrix33_cofactor() @ [0, 0, 1]).normalize(), [-1, 0, 0])

    def test_repr(self):
        self.assertEqual(repr(Matrix44()), """|   1.000   0.000   0.000   0.000 |
|   0.000   1.000   0.000   0.000 |
|   0.000   0.000   1.000   0.000 |
|   0.000   0.000   0.000   1.000 |""")
