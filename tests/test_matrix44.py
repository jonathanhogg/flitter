"""
Tests of the model.Matrix44 class
"""

import math
import unittest

from flitter.model import Matrix44, Matrix33


def all_isclose(xs, ys, rel_tol=1e-9, abs_tol=0):
    for x, y in zip(xs, ys):
        if not math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol):
            return False
    return True


class TestMatrix44(unittest.TestCase):
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

    def test_project(self):
        self.assertTrue(all_isclose(Matrix44.project(1, 1, 1, 2), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -3, -1, 0, 0, -4, 0]))
        self.assertTrue(all_isclose(Matrix44.project(1, 0.5, 1, 2), [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, -3, -1, 0, 0, -4, 0]))
        self.assertTrue(all_isclose(Matrix44.project(1, 0.5, 1, 3), [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, -2, -1, 0, 0, -3, 0]))
        self.assertTrue(all_isclose(Matrix44.project(0.5773502691896256, 0.2886751345948128, 1, 3),
                                    [1.7320508075688776, 0, 0, 0, 0, 3.4641016151377553, 0, 0, 0, 0, -2, -1, 0, 0, -3, 0]))

    def test_ortho(self):
        self.assertTrue(all_isclose(Matrix44.ortho(1, 2, 0, 2), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 1]))
        self.assertTrue(all_isclose(Matrix44.ortho(2, 2, 0, 2), [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, -1, 0, 0, 0, -1, 1]))
        self.assertTrue(all_isclose(Matrix44.ortho(2, 1, 0, 2), [2, 0, 0, 0, 0, 4, 0, 0, 0, 0, -1, 0, 0, 0, -1, 1]))
        self.assertTrue(all_isclose(Matrix44.ortho(2, 1, 1, 2), [2, 0, 0, 0, 0, 4, 0, 0, 0, 0, -2, 0, 0, 0, -3, 1]))
        self.assertTrue(all_isclose(Matrix44.ortho(2, 1, 1, 3), [2, 0, 0, 0, 0, 4, 0, 0, 0, 0, -1, 0, 0, 0, -2, 1]))

    def test_look(self):
        self.assertIsNone(Matrix44.look([0], None, "Hello world!"))
        self.assertTrue(all_isclose(Matrix44.look([0, 0, 0], [0, 0, -1], [0, 1, 0]), Matrix44()))
        self.assertTrue(all_isclose(Matrix44.look([0, 0, 0], [0, 0, -100], [0, 1, 0]), Matrix44()))
        self.assertTrue(all_isclose(Matrix44.look([0, 0, 0], [0, 0, -100], [1, 0, 0]),
                                    [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]))
        self.assertTrue(all_isclose(Matrix44.look([0, 0, 0], [0, 0, 100], [1, 0, 0]),
                                    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1]))
        self.assertTrue(all_isclose(Matrix44.look([1, 2, 3], [1, 2, 2], [0, 1, 0]),
                                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1, -2, -3, 1]))

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
        self.assertTrue(all_isclose(Matrix44.rotate_x(1), Matrix44(), abs_tol=1e-12))
        self.assertTrue(all_isclose(Matrix44.rotate_x(0.5), [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1], abs_tol=1e-12))

    def test_rotate_y(self):
        self.assertIsNone(Matrix44.rotate_y(math.nan))
        self.assertEqual(Matrix44.rotate_y(0), Matrix44())
        self.assertTrue(all_isclose(Matrix44.rotate_y(1), Matrix44(), abs_tol=1e-12))
        self.assertTrue(all_isclose(Matrix44.rotate_y(0.5), [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1], abs_tol=1e-12))

    def test_rotate_z(self):
        self.assertIsNone(Matrix44.rotate_z(math.nan))
        self.assertEqual(Matrix44.rotate_z(0), Matrix44())
        self.assertTrue(all_isclose(Matrix44.rotate_z(1), Matrix44(), abs_tol=1e-12))
        self.assertTrue(all_isclose(Matrix44.rotate_z(0.5), [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], abs_tol=1e-12))

    def test_rotate(self):
        self.assertEqual(Matrix44.rotate(None), Matrix44())
        self.assertEqual(Matrix44.rotate("Hello world!"), Matrix44())
        self.assertEqual(Matrix44.rotate([0.1, 0.2]), Matrix44())
        self.assertEqual(Matrix44.rotate(0), Matrix44())
        self.assertTrue(all_isclose(Matrix44.rotate(1), Matrix44(), abs_tol=1e-12))
        self.assertTrue(all_isclose(Matrix44.rotate(0.5), Matrix44(), abs_tol=1e-12))
        self.assertTrue(all_isclose(Matrix44.rotate([0.5, 0.5, 0.5]), Matrix44(), abs_tol=1e-12))
        self.assertTrue(all_isclose(Matrix44.rotate([0.5, 0, 0]), [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1], abs_tol=1e-12))
        self.assertTrue(all_isclose(Matrix44.rotate([0, 0.5, 0]), [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1], abs_tol=1e-12))
        self.assertTrue(all_isclose(Matrix44.rotate([0, 0, 0.5]), [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], abs_tol=1e-12))

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
        self.assertTrue(all_isclose(a.inverse() @ a, Matrix44()))
        self.assertTrue(all_isclose(b.inverse() @ b, Matrix44()))
        self.assertTrue(all_isclose(c.inverse() @ c, Matrix44()))
        self.assertTrue(all_isclose(b.inverse() @ (a.inverse() @ (a @ (b @ c))), c))
        self.assertTrue(all_isclose((((b.inverse() @ a.inverse()) @ a) @ b) @ c, c))

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
        self.assertTrue(all_isclose(m.inverse_transpose_matrix33(), m.inverse().transpose().matrix33()))

    def test_repr(self):
        self.assertEqual(repr(Matrix44()), """|   1.000   0.000   0.000   0.000 |
|   0.000   1.000   0.000   0.000 |
|   0.000   0.000   1.000   0.000 |
|   0.000   0.000   0.000   1.000 |""")
