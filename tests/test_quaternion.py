"""
Tests of the model.Quaternion class
"""

import math

import numpy as np

from flitter.model import Vector, Quaternion, Matrix44

from . import utils


class TestQuaternion(utils.TestCase):
    def test_construct(self):
        q = Quaternion()
        self.assertEqual(len(q), 4)
        self.assertEqual(q, [1, 0, 0, 0])
        q = Quaternion(0)
        self.assertEqual(len(q), 4)
        self.assertEqual(q, [0, 0, 0, 0])
        q = Quaternion(1)
        self.assertEqual(len(q), 4)
        self.assertEqual(q, [1, 0, 0, 0])
        q = Quaternion(2)
        self.assertEqual(len(q), 4)
        self.assertEqual(q, [2, 0, 0, 0])
        q = Quaternion([1, 2, 3, 4])
        self.assertEqual(len(q), 4)
        self.assertEqual(q, [1, 2, 3, 4])
        self.assertEqual(Quaternion(range(1, 5)), [1, 2, 3, 4])
        self.assertEqual(Quaternion(np.arange(1, 5)), [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            q = Quaternion("Hello world!")
        with self.assertRaises(ValueError):
            q = Quaternion([1, 2, 3])
        with self.assertRaises(ValueError):
            q = Quaternion([1, 2, 3, "four"])
        with self.assertRaises(ValueError):
            q = Quaternion(range(5))
        with self.assertRaises(ValueError):
            q = Quaternion(np.arange(5))

    def test_coerce(self):
        q1 = Quaternion()
        q2 = Quaternion.coerce(q1)
        self.assertIs(q1, q2)
        q1 = [1, 2, 3, 4]
        q2 = Quaternion.coerce(q1)
        self.assertEqual(q1, q2)
        self.assertIsNot(q1, q2)
        q1 = Vector([1, 2, 3, 4])
        q2 = Quaternion.coerce(q1)
        self.assertEqual(q1, q2)
        self.assertIsNot(q1, q2)

    def test_copy(self):
        q1 = Quaternion([1, 2, 3, 4])
        q2 = q1.copy()
        self.assertIsInstance(q2, Quaternion)
        self.assertEqual(q1, q2)
        self.assertIsNot(q1, q2)

    def test_euler(self):
        with self.assertRaises(ValueError):
            Quaternion.euler(None, 0.25)
        with self.assertRaises(ValueError):
            Quaternion.euler(['hello', 'cruel', 'world'], 0.25)
        with self.assertRaises(ValueError):
            Quaternion.euler([1, 0], 0.25)
        self.assertEqual(Quaternion.euler([1, 0, 0], 0), [1, 0, 0, 0])
        self.assertEqual(Quaternion.euler([0, 1, 0], 0), [1, 0, 0, 0])
        self.assertEqual(Quaternion.euler([0, 0, 1], 0), [1, 0, 0, 0])
        c, s = math.cos(math.pi*0.25), math.sin(math.pi*0.25)
        self.assertAllAlmostEqual(Quaternion.euler([1, 0, 0], 0.25), [c, s, 0, 0])
        self.assertAllAlmostEqual(Quaternion.euler([0, 1, 0], 0.25), [c, 0, s, 0])
        self.assertAllAlmostEqual(Quaternion.euler([0, 0, 1], 0.25), [c, 0, 0, s])
        self.assertAllAlmostEqual(Quaternion.euler([1, 0, 0], 0.5), [0, 1, 0, 0])
        self.assertAllAlmostEqual(Quaternion.euler([0, 1, 0], 0.5), [0, 0, 1, 0])
        self.assertAllAlmostEqual(Quaternion.euler([0, 0, 1], 0.5), [0, 0, 0, 1])
        c, s = math.cos(math.pi/3), math.sin(math.pi/3)
        self.assertAllAlmostEqual(Quaternion.euler([1, 0, 0], 1/3), [c, s, 0, 0])
        self.assertAllAlmostEqual(Quaternion.euler([0, 1, 0], 1/3), [c, 0, s, 0])
        self.assertAllAlmostEqual(Quaternion.euler([0, 0, 1], 1/3), [c, 0, 0, s])
        self.assertAllAlmostEqual(Quaternion.euler([1, 0, 0], 1), [-1, 0, 0, 0])
        self.assertAllAlmostEqual(Quaternion.euler([0, 1, 0], 1), [-1, 0, 0, 0])
        self.assertAllAlmostEqual(Quaternion.euler([0, 0, 1], 1), [-1, 0, 0, 0])
        self.assertAllAlmostEqual(Quaternion.euler([1, 1, 1], 1/3), [0.5, 0.5, 0.5, 0.5])

    def test_between(self):
        with self.assertRaises(ValueError):
            Quaternion.between(None, [0, 1, 0])
        with self.assertRaises(ValueError):
            Quaternion.between(['hello', 'cruel', 'world'], [0, 1, 0])
        with self.assertRaises(ValueError):
            Quaternion.between([1, 0], [0, 1, 0])
        with self.assertRaises(ValueError):
            Quaternion.between([1, 0, 0], None)
        with self.assertRaises(ValueError):
            Quaternion.between([1, 0, 0], ['hello', 'cruel', 'world'])
        with self.assertRaises(ValueError):
            Quaternion.between([1, 0, 0], [1, 0])
        self.assertAllAlmostEqual(Quaternion.between([1, 0, 0], [0, 1, 0]), Quaternion.euler([0, 0, 1], 0.25))
        self.assertAllAlmostEqual(Quaternion.between([0, 0, 1], [1, 0, 0]), Quaternion.euler([0, 1, 0], 0.25))
        self.assertAllAlmostEqual(Quaternion.between([0, 1, 0], [0, 0, 1]), Quaternion.euler([1, 0, 0], 0.25))
        self.assertAllAlmostEqual(Quaternion.between([1, 0, 0], [1, 1, 0]), Quaternion.euler([0, 0, 1], 0.125))
        self.assertAllAlmostEqual(Quaternion.between([0, 0, 1], [1, 0, 1]), Quaternion.euler([0, 1, 0], 0.125))
        self.assertAllAlmostEqual(Quaternion.between([0, 1, 0], [0, 1, 1]), Quaternion.euler([1, 0, 0], 0.125))
        self.assertAllAlmostEqual(Quaternion.between([1, 0, 0], [-1, 0, 0]), Quaternion.euler([0, 1, 0], 0.5))
        self.assertAllAlmostEqual(Quaternion.between([0, 1, 0], [0, -1, 0]), Quaternion.euler([-1, 0, 0], 0.5))
        self.assertAllAlmostEqual(Quaternion.between([0, 0, 1], [0, 0, -1]), Quaternion.euler([0, -1, 0], 0.5))

    def test_multiply(self):
        qx = Quaternion.euler([1, 0, 0], 0.125)
        qy = Quaternion.euler([0, 1, 0], 0.125)
        qz = Quaternion.euler([0, 0, 1], 0.125)
        self.assertAllAlmostEqual(qx @ qx, Quaternion.euler([1, 0, 0], 0.25))
        self.assertAllAlmostEqual(qy @ qy, Quaternion.euler([0, 1, 0], 0.25))
        self.assertAllAlmostEqual(qz @ qz, Quaternion.euler([0, 0, 1], 0.25))
        self.assertAllAlmostEqual(qx @ Quaternion.euler([1, 0, 0], -0.125), Quaternion())
        self.assertAllAlmostEqual(qy @ Quaternion.euler([0, 1, 0], -0.125), Quaternion())
        self.assertAllAlmostEqual(qz @ Quaternion.euler([0, 0, 1], -0.125), Quaternion())
        self.assertNotAllAlmostEqual(qx @ qy, qy @ qx)
        self.assertNotAllAlmostEqual(qx @ qz, qz @ qx)
        self.assertNotAllAlmostEqual(qy @ qz, qz @ qy)
        q = Quaternion.euler([1, 1, 1], 1/3)
        self.assertAllAlmostEqual(q @ q @ q, Quaternion(-1))
        self.assertAllAlmostEqual(q @ q @ q @ q @ q @ q, Quaternion(1))

    def test_inverse(self):
        self.assertAllAlmostEqual(Quaternion.euler([1, 0, 0], 0.125).inverse(), Quaternion.euler([1, 0, 0], -0.125))
        self.assertAllAlmostEqual(Quaternion.euler([0, 1, 0], 0.125).inverse(), Quaternion.euler([0, 1, 0], -0.125))
        self.assertAllAlmostEqual(Quaternion.euler([0, 0, 1], 0.125).inverse(), Quaternion.euler([0, 0, 1], -0.125))
        q = Quaternion.euler([1, 1, 1], 1/3)
        self.assertAllAlmostEqual(q @ q.inverse(), Quaternion())
        self.assertAllAlmostEqual(q.inverse() @ q, Quaternion())
        self.assertAllAlmostEqual(q @ q.inverse() @ q, q)

    def test_normalize(self):
        self.assertAllAlmostEqual(Quaternion([0.5, 0.5, 0.5, 0.5]).normalize(), [0.5, 0.5, 0.5, 0.5])
        self.assertAllAlmostEqual(Quaternion([1, 1, 1, 1]).normalize(), [0.5, 0.5, 0.5, 0.5])
        self.assertAllAlmostEqual(Quaternion(10).normalize(), Quaternion(1))

    def test_conjugate(self):
        qx = Quaternion.euler([1, 0, 0], 0.25)
        self.assertAllAlmostEqual(qx @ [1, 0, 0], [1, 0, 0])
        self.assertAllAlmostEqual(qx @ [0, 1, 0], [0, 0, 1])
        self.assertAllAlmostEqual(qx @ [0, 0, 1], [0, -1, 0])
        qy = Quaternion.euler([0, 1, 0], 0.25)
        self.assertAllAlmostEqual(qy @ [1, 0, 0], [0, 0, -1])
        self.assertAllAlmostEqual(qy @ [0, 1, 0], [0, 1, 0])
        self.assertAllAlmostEqual(qy @ [0, 0, 1], [1, 0, 0])
        qz = Quaternion.euler([0, 0, 1], 0.25)
        self.assertAllAlmostEqual(qz @ [1, 0, 0], [0, 1, 0])
        self.assertAllAlmostEqual(qz @ [0, 1, 0], [-1, 0, 0])
        self.assertAllAlmostEqual(qz @ [0, 0, 1], [0, 0, 1])
        with self.assertRaises(ValueError):
            qx @ None
        with self.assertRaises(ValueError):
            qx @ "Hello"
        with self.assertRaises(ValueError):
            qx @ [1, 0]
        with self.assertRaises(ValueError):
            qx @ [1, 0, 0, 0]

    def test_exponent(self):
        q = Quaternion.euler([1, 1, 1], 1/3)
        i = Quaternion()
        self.assertAllAlmostEqual(q.exponent(-1), q.inverse())
        self.assertAllAlmostEqual(q.exponent(0), i)
        self.assertAllAlmostEqual(q.exponent(1), q)
        self.assertAllAlmostEqual(q.exponent(2), q @ q)
        self.assertAllAlmostEqual(i.exponent(-1), i)
        self.assertAllAlmostEqual(i.exponent(-0.5), i)
        self.assertAllAlmostEqual(i.exponent(0), i)
        self.assertAllAlmostEqual(i.exponent(0.5), i)
        self.assertAllAlmostEqual(i.exponent(1), i)
        self.assertAllAlmostEqual(i.exponent(2), i)

    def test_slerp(self):
        qx = Quaternion.euler([1, 0, 0], 0.25)
        qy = Quaternion.euler([0, 1, 0], 0.25)
        self.assertAllAlmostEqual(qx.slerp(qy, -1), qx @ qy.inverse() @ qx)
        self.assertAllAlmostEqual(qx.slerp(qy, 0), qx)
        self.assertAllAlmostEqual(qx.slerp(qy, 1), qy)
        self.assertAllAlmostEqual(qx.slerp(qy, 2), qy @ qx.inverse() @ qy)
        self.assertAllAlmostEqual(qx.slerp(qx, 0), qx)
        self.assertAllAlmostEqual(qx.slerp(qx, 1), qx)

    def test_matrix44(self):
        self.assertAllAlmostEqual(Quaternion.euler([1, 0, 0], 0.125).matrix44(), Matrix44.rotate_x(0.125))
        self.assertAllAlmostEqual(Quaternion.euler([0, 1, 0], 0.125).matrix44(), Matrix44.rotate_y(0.125))
        self.assertAllAlmostEqual(Quaternion.euler([0, 0, 1], 0.125).matrix44(), Matrix44.rotate_z(0.125))

    def test_repr(self):
        self.assertEqual(repr(Quaternion(0)), "0﹢0𝒊﹢0𝒋﹢0𝒌")
        self.assertEqual(repr(Quaternion(1)), "1﹢0𝒊﹢0𝒋﹢0𝒌")
        self.assertEqual(repr(Quaternion(-1)), "-1﹢0𝒊﹢0𝒋﹢0𝒌")
        self.assertEqual(repr(Quaternion([1, 2, 3, 4])), "1﹢2𝒊﹢3𝒋﹢4𝒌")
        self.assertEqual(repr(Quaternion([-1, -2, -3, -4])), "-1﹣2𝒊﹣3𝒋﹣4𝒌")
        self.assertEqual(repr(Quaternion([1e-10, 2e-10, 3e-10, 4e-10])), "0﹢0𝒊﹢0𝒋﹢0𝒌")
        self.assertEqual(repr(Quaternion([-1e-10, -2e-10, -3e-10, -4e-10])), "0﹢0𝒊﹢0𝒋﹢0𝒌")
