"""
Tests of flitter 3D models
"""

import math

from flitter.render.window.models import Model

from . import utils


class TestPrimitives(utils.TestCase):
    def test_box(self):
        box = Model.box()
        mesh = box.get_trimesh()
        self.assertEqual(mesh.bounds.tolist(), [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        self.assertEqual(len(mesh.vertices), 24)
        self.assertEqual(len(mesh.faces), 12)
        self.assertEqual(mesh.area, 6)
        self.assertEqual(mesh.volume, 1)

    def test_sphere(self):
        for segments in (4, 64, 1024):
            with self.subTest(segments=segments):
                sphere = Model.sphere(segments)
                mesh = sphere.get_trimesh()
                self.assertEqual(mesh.bounds.tolist(), [[-1, -1, -1], [1, 1, 1]])
                nrows = segments // 4
                self.assertEqual(len(mesh.vertices), 4*(nrows+1)*(nrows+2))
                self.assertEqual(len(mesh.faces), 8*nrows*nrows)
                if segments == 4:
                    self.assertAlmostEqual(mesh.area, 4*math.sqrt(3))
                    self.assertAlmostEqual(mesh.volume, 4/3)
                else:
                    self.assertAlmostEqual(mesh.area, 4*math.pi, places=int(math.log10(segments)))
                    self.assertAlmostEqual(mesh.volume, 4/3*math.pi, places=int(math.log10(segments)))

    def test_cylinder(self):
        for segments in (4, 64, 1024):
            with self.subTest(segments=segments):
                cylinder = Model.cylinder(segments)
                mesh = cylinder.get_trimesh()
                self.assertEqual(mesh.bounds.tolist(), [[-1, -1, -0.5], [1, 1, 0.5]])
                self.assertEqual(len(mesh.vertices), 6*(segments+1))
                self.assertEqual(len(mesh.faces), 4*segments)
                if segments == 4:
                    self.assertAlmostEqual(mesh.area, 4*(1+math.sqrt(2)))
                    self.assertAlmostEqual(mesh.volume, 2)
                else:
                    self.assertAlmostEqual(mesh.area, 4*math.pi, places=int(math.log10(segments)))
                    self.assertAlmostEqual(mesh.volume, math.pi, places=int(math.log10(segments)))

    def test_cone(self):
        for segments in (4, 64, 1024):
            with self.subTest(segments=segments):
                cone = Model.cone(segments)
                mesh = cone.get_trimesh()
                self.assertEqual(mesh.bounds.tolist(), [[-1, -1, -0.5], [1, 1, 0.5]])
                self.assertEqual(len(mesh.vertices), 4*(segments+1))
                self.assertEqual(len(mesh.faces), 2*segments)
                if segments == 4:
                    self.assertAlmostEqual(mesh.area, 2*(1+math.sqrt(3)))
                    self.assertAlmostEqual(mesh.volume, 2/3)
                else:
                    self.assertAlmostEqual(mesh.area, (1+math.sqrt(2))*math.pi, places=int(math.log10(segments)))
                    self.assertAlmostEqual(mesh.volume, math.pi/3, places=int(math.log10(segments)))
