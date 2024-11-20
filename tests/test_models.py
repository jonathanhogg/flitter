"""
Tests of flitter 3D models
"""

import math

import trimesh

from flitter.model import Matrix44
from flitter.render.window.models import Model

from . import utils


DefaultSegments = 64


class TestPrimitives(utils.TestCase):
    def tearDown(self):
        Model.flush_caches(0, 0)

    def test_box(self):
        model = Model.box()
        mesh = model.get_trimesh()
        self.assertEqual(model.name, '!box')
        self.assertEqual(mesh.bounds.tolist(), [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        self.assertEqual(len(mesh.vertices), 24)
        self.assertEqual(len(mesh.faces), 12)
        self.assertEqual(mesh.area, 6)
        self.assertEqual(mesh.volume, 1)

    def test_sphere(self):
        for segments in (4, DefaultSegments, 1024):
            with self.subTest(segments=segments):
                model = Model.sphere(segments)
                self.assertEqual(model.name, f'!sphere-{segments}' if segments != DefaultSegments else '!sphere')
                mesh = model.get_trimesh()
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
        for segments in (4, DefaultSegments, 1024):
            with self.subTest(segments=segments):
                model = Model.cylinder(segments)
                self.assertEqual(model.name, f'!cylinder-{segments}' if segments != DefaultSegments else '!cylinder')
                mesh = model.get_trimesh()
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
        for segments in (4, DefaultSegments, 1024):
            with self.subTest(segments=segments):
                model = Model.cone(segments)
                self.assertEqual(model.name, f'!cone-{segments}' if segments != DefaultSegments else '!cone')
                mesh = model.get_trimesh()
                self.assertEqual(mesh.bounds.tolist(), [[-1, -1, -0.5], [1, 1, 0.5]])
                self.assertEqual(len(mesh.vertices), 4*(segments+1))
                self.assertEqual(len(mesh.faces), 2*segments)
                if segments == 4:
                    self.assertAlmostEqual(mesh.area, 2*(1+math.sqrt(3)))
                    self.assertAlmostEqual(mesh.volume, 2/3)
                else:
                    self.assertAlmostEqual(mesh.area, (1+math.sqrt(2))*math.pi, places=int(math.log10(segments)))
                    self.assertAlmostEqual(mesh.volume, math.pi/3, places=int(math.log10(segments)))


class TestManifoldPrimitives(utils.TestCase):
    def tearDown(self):
        Model.flush_caches(0, 0)

    def test_box(self):
        mesh = Model.box().get_manifold().to_mesh()
        mesh = trimesh.Trimesh(vertices=mesh.vert_properties, faces=mesh.tri_verts, process=False)
        self.assertEqual(mesh.bounds.tolist(), [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        self.assertEqual(len(mesh.vertices), 8)
        self.assertEqual(len(mesh.faces), 12)
        self.assertEqual(mesh.area, 6)
        self.assertEqual(mesh.volume, 1)

    def test_sphere(self):
        for segments in (4, DefaultSegments, 1024):
            with self.subTest(segments=segments):
                mesh = Model.sphere(segments).get_manifold().to_mesh()
                mesh = trimesh.Trimesh(vertices=mesh.vert_properties, faces=mesh.tri_verts, process=False)
                self.assertEqual(mesh.bounds.tolist(), [[-1, -1, -1], [1, 1, 1]])
                nrows = segments // 2
                self.assertEqual(len(mesh.vertices), 2+nrows*nrows)
                self.assertEqual(len(mesh.faces), 2*nrows*nrows)
                if segments == 4:
                    self.assertAlmostEqual(mesh.area, 4*math.sqrt(3))
                    self.assertAlmostEqual(mesh.volume, 4/3)
                else:
                    self.assertAlmostEqual(mesh.area, 4*math.pi, places=int(math.log10(segments)))
                    self.assertAlmostEqual(mesh.volume, 4/3*math.pi, places=int(math.log10(segments)))

    def test_cylinder(self):
        for segments in (4, DefaultSegments, 1024):
            with self.subTest(segments=segments):
                mesh = Model.cylinder(segments).get_manifold().to_mesh()
                mesh = trimesh.Trimesh(vertices=mesh.vert_properties, faces=mesh.tri_verts, process=False)
                self.assertEqual(mesh.bounds.tolist(), [[-1, -1, -0.5], [1, 1, 0.5]])
                self.assertEqual(len(mesh.vertices), 2*segments)
                self.assertEqual(len(mesh.faces), 4*(segments-1))
                if segments == 4:
                    self.assertAlmostEqual(mesh.area, 4*(1+math.sqrt(2)))
                    self.assertAlmostEqual(mesh.volume, 2)
                else:
                    self.assertAlmostEqual(mesh.area, 4*math.pi, places=int(math.log10(segments)))
                    self.assertAlmostEqual(mesh.volume, math.pi, places=int(math.log10(segments)))

    def test_cone(self):
        for segments in (4, DefaultSegments, 1024):
            with self.subTest(segments=segments):
                mesh = Model.cone(segments).get_manifold().to_mesh()
                mesh = trimesh.Trimesh(vertices=mesh.vert_properties, faces=mesh.tri_verts, process=False)
                self.assertEqual(mesh.bounds.tolist(), [[-1, -1, -0.5], [1, 1, 0.5]])
                self.assertEqual(len(mesh.vertices), segments+1)
                self.assertEqual(len(mesh.faces), 2*(segments-1))
                if segments == 4:
                    self.assertAlmostEqual(mesh.area, 2*(1+math.sqrt(3)))
                    self.assertAlmostEqual(mesh.volume, 2/3)
                else:
                    self.assertAlmostEqual(mesh.area, (1+math.sqrt(2))*math.pi, places=int(math.log10(segments)))
                    self.assertAlmostEqual(mesh.volume, math.pi/3, places=int(math.log10(segments)))


class TestTransform(utils.TestCase):
    def tearDown(self):
        Model.flush_caches(0, 0)

    def assertTransform(self, base_model, matrix):
        transformed_model = base_model.transform(matrix)
        self.assertEqual(transformed_model.name, f'{base_model.name}@{hex(matrix.hash(False))[2:]}')
        base_mesh = base_model.get_trimesh()
        transformed_mesh = transformed_model.get_trimesh()
        self.assertEqual(len(base_mesh.vertices), len(transformed_mesh.vertices))
        self.assertEqual(base_mesh.faces.tolist(), transformed_mesh.faces.tolist())
        for vertex1, vertex2 in zip(base_mesh.vertices.tolist(), transformed_mesh.vertices.tolist()):
            self.assertAllAlmostEqual(vertex2, matrix @ vertex1, places=6)
        base_mesh = base_model.get_manifold().to_mesh()
        transformed_mesh = transformed_model.get_manifold().to_mesh()
        self.assertEqual(len(base_mesh.vert_properties), len(transformed_mesh.vert_properties))
        self.assertEqual(base_mesh.tri_verts.tolist(), transformed_mesh.tri_verts.tolist())
        for vertex1, vertex2 in zip(base_mesh.vert_properties.tolist(), transformed_mesh.vert_properties.tolist()):
            self.assertAllAlmostEqual(vertex2, matrix @ vertex1, places=6)

    def test_translate(self):
        self.assertTransform(Model.cylinder(), Matrix44.translate((1, 2, 3)))

    def test_rotate(self):
        self.assertTransform(Model.cylinder(), Matrix44.rotate(1/3))

    def test_scale(self):
        self.assertTransform(Model.cylinder(), Matrix44.scale((2, 3, 4)))


class TestCache(utils.TestCase):
    def tearDown(self):
        Model.flush_caches(0, 0)

    def test_cached_trimesh(self):
        model = Model.box()
        bounds = model.get_bounds()
        mesh = model.get_trimesh()
        manifold = model.get_manifold()
        self.assertIs(model.get_bounds(), bounds)
        self.assertIs(model.get_trimesh(), mesh)
        self.assertIs(model.get_manifold(), manifold)

    def test_invalidate(self):
        model = Model.box()
        bounds = model.get_bounds()
        mesh = model.get_trimesh()
        manifold = model.get_manifold()
        self.assertIs(model.get_bounds(), bounds)
        self.assertIs(model.get_trimesh(), mesh)
        self.assertIs(model.get_manifold(), manifold)
        model.invalidate()
        self.assertIsNot(model.get_bounds(), bounds)
        self.assertIsNot(model.get_trimesh(), mesh)
        self.assertIsNot(model.get_manifold(), manifold)

    def test_invalidate_dependency(self):
        dependency = Model.box()
        model = dependency.transform(Matrix44())
        bounds = model.get_bounds()
        mesh = model.get_trimesh()
        manifold = model.get_manifold()
        self.assertIs(model.get_bounds(), bounds)
        self.assertIs(model.get_trimesh(), mesh)
        self.assertIs(model.get_manifold(), manifold)
        dependency.invalidate()
        self.assertIsNot(model.get_bounds(), bounds)
        self.assertIsNot(model.get_trimesh(), mesh)
        self.assertIsNot(model.get_manifold(), manifold)
