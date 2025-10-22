"""
Tests of flitter 3D models
"""

import math
import unittest.mock

import numpy as np
import trimesh

from flitter.model import Vector, Matrix44
from flitter.render.window.models import Model

from . import utils


DefaultSegments = 64


class TestPrimitives(utils.TestCase):
    def tearDown(self):
        Model.flush_caches(0, 0)

    def test_box(self):
        model = Model.box()
        self.assertFalse(model.is_manifold())
        mesh = model.get_trimesh()
        self.assertEqual(model.name, '!box')
        self.assertEqual(mesh.bounds.tolist(), [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        self.assertEqual(len(mesh.vertices), 24)
        self.assertEqual(len(mesh.faces), 12)
        self.assertEqual(mesh.area, 6)
        self.assertEqual(mesh.volume, 1)
        model = Model.box('repeat')
        self.assertEqual(model.name, '!box-repeat')

    def test_sphere(self):
        for segments in (4, DefaultSegments, 1024):
            with self.subTest(segments=segments):
                model = Model.sphere(segments)
                self.assertFalse(model.is_manifold())
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
        self.assertEqual(Model.sphere(0).name, '!sphere-4')
        self.assertEqual(Model.sphere(5).name, '!sphere-8')
        self.assertIs(Model.sphere(), Model.sphere(DefaultSegments))

    def test_cylinder(self):
        for segments in (4, DefaultSegments, 1024):
            with self.subTest(segments=segments):
                model = Model.cylinder(segments)
                self.assertFalse(model.is_manifold())
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
        self.assertEqual(Model.cylinder(0).name, '!cylinder-2')
        self.assertIs(Model.cylinder(), Model.cylinder(DefaultSegments))

    def test_cone(self):
        for segments in (4, DefaultSegments, 1024):
            with self.subTest(segments=segments):
                model = Model.cone(segments)
                self.assertFalse(model.is_manifold())
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
        self.assertEqual(Model.cone(0).name, '!cone-2')
        self.assertIs(Model.cone(), Model.cone(DefaultSegments))


class TestBasicFunctionality(utils.TestCase):
    def tearDown(self):
        Model.flush_caches(0, 0)

    def test_equality(self):
        self.assertTrue(Model.box() == Model.box())
        self.assertFalse(Model.box() == Model.sphere())

    def test_str(self):
        self.assertEqual(str(Model.sphere(128)), '!sphere-128')

    def test_repr(self):
        self.assertEqual(repr(Model.sphere(128)), '<Sphere(0x935c42374d93e335)>')

    def test_by_id(self):
        self.assertIsNone(Model.by_id(0x935c42374d93e335))
        model = Model.sphere(128)
        self.assertIs(Model.by_id(0x935c42374d93e335), model)


class MyModel(Model):
    @staticmethod
    def get():
        model = Model.by_id(123)
        if model is None:
            model = MyModel(123)
        return model

    @property
    def name(self):
        return '!mymodel'

    def is_manifold(self):
        return False

    def check_for_changes(self):
        pass

    def build_arrays(self):
        return np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype='f4'), np.array([[0, 1, 2], [2, 3, 0]], dtype='i4')


class TestSubclassing(utils.TestCase):
    def tearDown(self):
        Model.flush_caches(0, 0)

    def test_subclass_instantiation(self):
        model = MyModel.get()
        self.assertIsInstance(model, MyModel)
        self.assertEqual(model.name, '!mymodel')
        self.assertEqual(model.id, 123)
        self.assertIs(MyModel.get(), model)
        mesh = model.get_trimesh()
        self.assertIs(model.get_trimesh(), mesh)
        self.assertAllAlmostEqual(mesh.vertices, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        self.assertAllAlmostEqual(mesh.faces, [[0, 1, 2], [2, 3, 0]])
        self.assertEqual(model.snap_edges().name, 'snap_edges(!mymodel)')

    def test_bad_manifold(self):
        model = MyModel.get()
        with unittest.mock.patch('flitter.render.window.models.logger') as mock_logger:
            manifold = model.get_manifold()
            mock_logger.error.assert_called_with("Mesh is not a volume: {}", model.name)
        self.assertIsNone(manifold)


class TestVector(utils.TestCase):
    def test_bad(self):
        self.assertIsNone(Model.vector(None, None))
        self.assertIsNone(Model.vector(None, 1))

    def test_basic(self):
        vertices = [0, 0, 0,  1, 0, 0,  0, 1, 0,  0, 0, 1,  1, 0, 1,  0, 1, 1]
        faces = [0, 1, 2,  3, 4, 5,  0, 1, 4,  4, 3, 0,  1, 2, 5,  5, 4, 1,  2, 0, 3,  3, 5, 2]
        model = Model.vector(vertices, faces)
        self.assertIs(model, Model.vector(vertices, faces))
        self.assertIsNot(model, Model.vector(vertices, reversed(faces)))
        mesh = model.get_trimesh()
        self.assertEqual(mesh.bounds.tolist(), [[0, 0, 0], [1, 1, 1]])
        self.assertAllAlmostEqual(mesh.vertices, np.array(vertices).reshape((-1, 3)))
        self.assertAllAlmostEqual(mesh.faces, np.array(faces).reshape((-1, 3)))
        self.assertAlmostEqual(mesh.area, 3+math.sqrt(2))
        self.assertAlmostEqual(mesh.volume, 0.5)

    def test_convex(self):
        vertices = [-1, -1, -1,  1, -1, -1,  -1, 1, -1,  1, 1, -1,  -1, -1, 1,  1, -1, 1,  -1, 1, 1,  1, 1, 1]
        model = Model.vector(vertices)
        self.assertIs(model, Model.vector(vertices))
        mesh = model.get_trimesh()
        self.assertEqual(mesh.bounds.tolist(), [[-1, -1, -1], [1, 1, 1]])
        self.assertAlmostEqual(mesh.area, 24)
        self.assertAlmostEqual(mesh.volume, 8)

    def test_invalid(self):
        with unittest.mock.patch('flitter.render.window.models.logger') as mock_logger:
            model = Model.vector([0, 0, 0, 0, 0, 1, 0, 1], [0, 1, 2])
            self.assertIsNone(model.get_trimesh())
            mock_logger.error.assert_called_with("Bad vertices vector length: {}", model.name)
        with unittest.mock.patch('flitter.render.window.models.logger') as mock_logger:
            model = Model.vector([0, 0, 0, 0, 0, 1, 0, 1, 0], [0, 1])
            self.assertIsNone(model.get_trimesh())
            mock_logger.error.assert_called_with("Bad faces vector length: {}", model.name)
        with unittest.mock.patch('flitter.render.window.models.logger') as mock_logger:
            model = Model.vector([0, 0, 0, 0, 0, 1, 0, 1, 0], [0, 1, 3])
            self.assertIsNone(model.get_trimesh())
            mock_logger.error.assert_called_with("Bad vertex index: {}", model.name)


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
                self.assertEqual(len(mesh.vertices), 2*segments+2)
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
                mesh = Model.cone(segments).get_manifold().to_mesh()
                mesh = trimesh.Trimesh(vertices=mesh.vert_properties, faces=mesh.tri_verts, process=False)
                self.assertEqual(mesh.bounds.tolist(), [[-1, -1, -0.5], [1, 1, 0.5]])
                self.assertEqual(len(mesh.vertices), segments+2)
                self.assertEqual(len(mesh.faces), 2*segments)
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
        box = Model.box()
        transformed_box = box.transform(Matrix44.scale(2))
        self.assertIsNot(transformed_box, box)
        self.assertIn(transformed_box, box.dependents)
        bounds = transformed_box.get_bounds()
        mesh = transformed_box.get_trimesh()
        manifold = transformed_box.get_manifold()
        self.assertIs(transformed_box.get_bounds(), bounds)
        self.assertIs(transformed_box.get_trimesh(), mesh)
        self.assertIs(transformed_box.get_manifold(), manifold)
        box.invalidate()
        self.assertIsNot(transformed_box.get_bounds(), bounds)
        self.assertIsNot(transformed_box.get_trimesh(), mesh)
        self.assertIsNot(transformed_box.get_manifold(), manifold)


class TestStructuring(utils.TestCase):
    def setUp(self):
        self.M = Matrix44.translate(1)
        self.M_hash = hex(self.M.hash(False))[2:]
        self.M2 = Matrix44.translate(2)
        self.M2_hash = hex(self.M2.hash(False))[2:]
        self.P = Vector((0, 0, 0))
        self.N = Vector((1, 0, 0))
        self.PN_hash = hex(self.P.hash(False) ^ self.N.hash(False))[2:]

    def tearDown(self):
        Model.flush_caches(0, 0)

    def test_flatten(self):
        self.assertEqual(Model.box().flatten().name, 'flatten(!box)')
        self.assertFalse(Model.box().flatten().is_manifold())
        self.assertIs(Model.box().flatten().get_manifold(), Model.box().get_manifold())
        self.assertFalse(Model.union(Model.box(), Model.sphere()).flatten().is_manifold())
        self.assertEqual(Model.box().flatten().flatten().name, 'flatten(!box)')
        self.assertEqual(Model.box().flatten().invert().name, 'invert(flatten(!box))')
        self.assertEqual(Model.box().flatten().repair().name, 'repair(flatten(!box))')
        self.assertEqual(Model.box().flatten().snap_edges().name, 'flatten(!box)')
        self.assertEqual(Model.box().flatten().transform(self.M).name, f'flatten(!box)@{self.M_hash}')
        self.assertEqual(Model.box().flatten().uv_remap('sphere').name, 'uv_remap(flatten(!box), sphere)')
        self.assertEqual(Model.box().flatten().trim(self.P, self.N).name, f'trim(flatten(!box), {self.PN_hash})')

    def test_invert(self):
        self.assertEqual(Model.box().invert().name, 'invert(!box)')
        self.assertFalse(Model.box().invert().is_manifold())
        self.assertIs(Model.box().invert().get_manifold(), Model.box().get_manifold())
        self.assertFalse(Model.union(Model.box(), Model.sphere()).invert().is_manifold())
        self.assertEqual(Model.box().invert().flatten().name, 'flatten(invert(!box))')
        self.assertEqual(Model.box().invert().invert().name, '!box')
        self.assertEqual(Model.box().invert().repair().name, 'invert(repair(!box))')
        self.assertEqual(Model.box().invert().snap_edges().name, 'invert(snap_edges(!box))')
        self.assertEqual(Model.box().invert().transform(self.M).name, f'invert(!box@{self.M_hash})')
        self.assertEqual(Model.box().invert().uv_remap('sphere').name, 'uv_remap(invert(!box), sphere)')
        self.assertEqual(Model.box().invert().trim(self.P, self.N).name, f'trim(invert(!box), {self.PN_hash})')

    def test_repair(self):
        self.assertEqual(Model.box().repair().name, 'repair(!box)')
        self.assertFalse(Model.box().repair().is_manifold())
        self.assertIsNot(Model.box().repair().get_manifold(), Model.box().get_manifold())
        self.assertEqual(Model.box().repair().flatten().name, 'flatten(repair(!box))')
        self.assertEqual(Model.box().repair().invert().name, 'invert(repair(!box))')
        self.assertEqual(Model.box().repair().repair().name, 'repair(!box)')
        self.assertEqual(Model.box().repair().snap_edges().name, 'snap_edges(repair(!box))')
        self.assertEqual(Model.box().repair().transform(self.M).name, f'repair(!box)@{self.M_hash}')
        self.assertEqual(Model.box().repair().uv_remap('sphere').name, 'uv_remap(repair(!box), sphere)')
        self.assertEqual(Model.box().repair().trim(self.P, self.N).name, f'trim(repair(!box), {self.PN_hash})')

    def test_snap_edges(self):
        self.assertEqual(Model.box().snap_edges(0).name, 'flatten(!box)')
        self.assertFalse(Model.box().snap_edges().is_manifold())
        self.assertIsNot(Model.box().snap_edges().get_manifold(), Model.box().get_manifold())
        self.assertFalse(Model.union(Model.box(), Model.sphere()).snap_edges().is_manifold())
        self.assertEqual(Model.box().snap_edges().name, 'snap_edges(!box)')
        self.assertEqual(Model.box().snap_edges(0.05).name, 'snap_edges(!box)')
        self.assertEqual(Model.box().snap_edges(0.25).name, 'snap_edges(!box, 0.25)')
        self.assertEqual(Model.box().snap_edges().flatten().name, 'flatten(!box)')
        self.assertEqual(Model.box().snap_edges().invert().name, 'invert(snap_edges(!box))')
        self.assertEqual(Model.box().snap_edges().repair().name, 'snap_edges(repair(!box))')
        self.assertEqual(Model.box().snap_edges().snap_edges().name, 'snap_edges(!box)')
        self.assertEqual(Model.box().snap_edges().transform(self.M).name, f'snap_edges(!box@{self.M_hash})')
        self.assertEqual(Model.box().snap_edges().uv_remap('sphere').name, 'uv_remap(snap_edges(!box), sphere)')
        self.assertEqual(Model.box().snap_edges().trim(self.P, self.N).name, f'trim(!box, {self.PN_hash})')

    def test_transform(self):
        self.assertEqual(Model.box().transform(Matrix44()).name, '!box')
        self.assertFalse(Model.box().transform(self.M).is_manifold())
        self.assertTrue(Model.union(Model.box(), Model.sphere()).transform(self.M).is_manifold())
        self.assertEqual(Model.box().transform(self.M).name, f'!box@{self.M_hash}')
        self.assertEqual(Model.box().transform(self.M).flatten().name, f'flatten(!box@{self.M_hash})')
        self.assertEqual(Model.box().transform(self.M).invert().name, f'invert(!box@{self.M_hash})')
        self.assertEqual(Model.box().transform(self.M).repair().name, f'repair(!box)@{self.M_hash}')
        self.assertEqual(Model.box().transform(self.M).snap_edges().name, f'snap_edges(!box@{self.M_hash})')
        self.assertEqual(Model.box().transform(self.M).transform(self.M).name, f'!box@{self.M2_hash}')
        self.assertEqual(Model.box().transform(self.M).uv_remap('sphere').name, f'uv_remap(!box@{self.M_hash}, sphere)')
        self.assertEqual(Model.box().transform(self.M).trim(self.P, self.N).name, f'trim(!box@{self.M_hash}, {self.PN_hash})')

    def test_uvremap(self):
        self.assertEqual(Model.box().uv_remap('sphere').name, 'uv_remap(!box, sphere)')
        self.assertFalse(Model.box().uv_remap('sphere').is_manifold())
        self.assertIs(Model.box().uv_remap('sphere').get_manifold(), Model.box().get_manifold())
        self.assertFalse(Model.union(Model.box(), Model.sphere()).uv_remap('sphere').is_manifold())
        self.assertEqual(Model.box().uv_remap('sphere').flatten().name, 'flatten(uv_remap(!box, sphere))')
        self.assertEqual(Model.box().uv_remap('sphere').invert().name, 'invert(uv_remap(!box, sphere))')
        self.assertEqual(Model.box().uv_remap('sphere').repair().name, 'uv_remap(repair(!box), sphere)')
        self.assertEqual(Model.box().uv_remap('sphere').snap_edges().name, 'snap_edges(uv_remap(!box, sphere))')
        self.assertEqual(Model.box().uv_remap('sphere').transform(self.M).name, f'uv_remap(!box, sphere)@{self.M_hash}')
        self.assertEqual(Model.box().uv_remap('test').uv_remap('sphere').name, 'uv_remap(!box, sphere)')
        self.assertEqual(Model.box().uv_remap('sphere').trim(self.P, self.N).name, f'trim(uv_remap(!box, sphere), {self.PN_hash})')

    def test_trim(self):
        self.assertEqual(Model.box().trim(self.P, self.N).name, f'trim(!box, {self.PN_hash})')
        self.assertTrue(Model.box().trim(self.P, self.N).is_manifold())
        self.assertEqual(Model.box().trim(self.P, self.N).flatten().name, f'flatten(trim(!box, {self.PN_hash}))')
        self.assertEqual(Model.box().trim(self.P, self.N).invert().name, f'invert(trim(!box, {self.PN_hash}))')
        self.assertEqual(Model.box().trim(self.P, self.N).repair().name, f'trim(repair(!box), {self.PN_hash})')
        self.assertEqual(Model.box().trim(self.P, self.N).snap_edges().name, f'snap_edges(trim(!box, {self.PN_hash}))')
        MPN_hash = hex((self.M @ self.P).hash(False) ^ ((self.M.matrix33_cofactor() @ self.N).normalize()).hash(False))[2:]
        self.assertEqual(Model.box().trim(self.P, self.N).transform(self.M).name, f'trim(!box@{self.M_hash}, {MPN_hash})')
        self.assertEqual(Model.box().trim(self.P, self.N).uv_remap('sphere').name, f'uv_remap(trim(!box, {self.PN_hash}), sphere)')
        self.assertEqual(Model.box().trim(self.P, self.N).trim(self.P, self.N).name, f'trim(trim(!box, {self.PN_hash}), {self.PN_hash})')

    def test_union(self):
        self.assertIsNone(Model.union())
        self.assertEqual(Model.union(Model.box()).name, '!box')
        self.assertEqual(Model.union(Model.box(), Model.box()).name, '!box')
        self.assertEqual(Model.union(Model.box(), Model.sphere()).name, 'union(!box, !sphere)')
        self.assertEqual(Model.union(Model.box(), Model.sphere(), Model.box()).name, 'union(!box, !sphere)')
        self.assertEqual(Model.union(Model.box(), Model.union(Model.sphere(), Model.cylinder())).name, 'union(!box, !sphere, !cylinder)')
        self.assertTrue(Model.union(Model.box(), Model.sphere()).is_manifold())
        self.assertEqual(Model.union(Model.box(), Model.sphere()).flatten().name, 'flatten(union(!box, !sphere))')
        self.assertEqual(Model.union(Model.box(), Model.sphere()).invert().name, 'invert(union(!box, !sphere))')
        self.assertEqual(Model.union(Model.box(), Model.sphere()).repair().name, 'union(!box, !sphere)')
        self.assertEqual(Model.union(Model.box(), Model.sphere()).snap_edges().name, 'snap_edges(union(!box, !sphere))')
        self.assertEqual(Model.union(Model.box(), Model.sphere()).transform(self.M).name, f'union(!box@{self.M_hash}, !sphere@{self.M_hash})')
        self.assertEqual(Model.union(Model.box(), Model.sphere()).uv_remap('sphere').name, 'uv_remap(union(!box, !sphere), sphere)')
        self.assertEqual(Model.union(Model.box(), Model.sphere()).trim(self.P, self.N).name, f'trim(union(!box, !sphere), {self.PN_hash})')

    def test_intersect(self):
        self.assertIsNone(Model.intersect())
        self.assertEqual(Model.intersect(Model.box()).name, '!box')
        self.assertEqual(Model.intersect(Model.box(), Model.box()).name, '!box')
        self.assertEqual(Model.intersect(Model.box(), Model.sphere()).name, 'intersect(!box, !sphere)')
        self.assertEqual(Model.intersect(Model.box(), Model.sphere(), Model.box()).name, 'intersect(!box, !sphere)')
        self.assertEqual(Model.intersect(Model.box(), Model.intersect(Model.sphere(), Model.cylinder())).name, 'intersect(!box, !sphere, !cylinder)')
        self.assertTrue(Model.intersect(Model.box(), Model.sphere()).is_manifold())
        self.assertEqual(Model.intersect(Model.box(), Model.sphere()).flatten().name, 'flatten(intersect(!box, !sphere))')
        self.assertEqual(Model.intersect(Model.box(), Model.sphere()).invert().name, 'invert(intersect(!box, !sphere))')
        self.assertEqual(Model.intersect(Model.box(), Model.sphere()).repair().name, 'intersect(!box, !sphere)')
        self.assertEqual(Model.intersect(Model.box(), Model.sphere()).snap_edges().name, 'snap_edges(intersect(!box, !sphere))')
        self.assertEqual(Model.intersect(Model.box(), Model.sphere()).transform(self.M).name, f'intersect(!box@{self.M_hash}, !sphere@{self.M_hash})')
        self.assertEqual(Model.intersect(Model.box(), Model.sphere()).uv_remap('sphere').name, 'uv_remap(intersect(!box, !sphere), sphere)')
        self.assertEqual(Model.intersect(Model.box(), Model.sphere()).trim(self.P, self.N).name, f'intersect(trim(!box, {self.PN_hash}), !sphere)')

    def test_difference(self):
        self.assertIsNone(Model.difference())
        self.assertEqual(Model.difference(Model.box()).name, '!box')
        self.assertIsNone(Model.difference(Model.box(), Model.box()))
        self.assertEqual(Model.difference(Model.box(), Model.sphere()).name, 'difference(!box, !sphere)')
        self.assertIsNone(Model.difference(Model.box(), Model.sphere(), Model.box()))
        self.assertEqual(Model.difference(Model.box(), Model.difference(Model.sphere(), Model.cylinder())).name,
                         'difference(!box, difference(!sphere, !cylinder))')
        self.assertTrue(Model.difference(Model.box(), Model.sphere()).is_manifold())
        self.assertEqual(Model.difference(Model.box(), Model.sphere()).flatten().name, 'flatten(difference(!box, !sphere))')
        self.assertEqual(Model.difference(Model.box(), Model.sphere()).invert().name, 'invert(difference(!box, !sphere))')
        self.assertEqual(Model.difference(Model.box(), Model.sphere()).repair().name, 'difference(!box, !sphere)')
        self.assertEqual(Model.difference(Model.box(), Model.sphere()).snap_edges().name, 'snap_edges(difference(!box, !sphere))')
        self.assertEqual(Model.difference(Model.box(), Model.sphere()).transform(self.M).name, f'difference(!box@{self.M_hash}, !sphere@{self.M_hash})')
        self.assertEqual(Model.difference(Model.box(), Model.sphere()).uv_remap('sphere').name, 'uv_remap(difference(!box, !sphere), sphere)')
        self.assertEqual(Model.difference(Model.box(), Model.sphere()).trim(self.P, self.N).name, f'difference(trim(!box, {self.PN_hash}), !sphere)')


class TestUnaryOperations(utils.TestCase):
    def tearDown(self):
        Model.flush_caches(0, 0)

    def test_uv_remap_sphere(self):
        model = Model.box()
        mesh = model.uv_remap('sphere').get_trimesh()
        m = 0
        n = 0
        for (x, y, z), (uu, vv) in zip(mesh.vertices, mesh.visual.uv):
            u = math.atan2(y, x) / (2*math.pi) % 1
            r = math.sqrt(x*x + y*y)
            v = (math.atan2(z, r) / math.pi + 0.5) % 1
            if uu >= 1:
                self.assertAllAlmostEqual((uu, vv), (u + 1, v))
                n += 1
            else:
                self.assertAllAlmostEqual((uu, vv), (u, v))
                m += 1
        self.assertEqual(m, 24)
        self.assertEqual(n, 5)

    def test_flatten(self):
        model = Model.sphere()
        nrows = DefaultSegments // 4
        mesh = model.get_trimesh()
        self.assertEqual(len(mesh.vertices), 4*(nrows+1)*(nrows+2))
        self.assertEqual(len(mesh.faces), 8*nrows*nrows)
        self.assertAlmostEqual(mesh.area, 4*math.pi, places=1)
        self.assertAlmostEqual(mesh.volume, 4/3*math.pi, places=1)
        flat_model = model.flatten()
        self.assertIsNot(flat_model, model)
        mesh = flat_model.get_trimesh()
        self.assertEqual(len(mesh.vertices), 3*8*nrows*nrows)
        self.assertEqual(len(mesh.faces), 8*nrows*nrows)
        self.assertAlmostEqual(mesh.area, 4*math.pi, places=1)
        self.assertAlmostEqual(mesh.volume, 4/3*math.pi, places=1)


class TestTrim(utils.TestCase):
    def tearDown(self):
        Model.flush_caches(0, 0)

    def test_bad_arguments(self):
        self.assertIsNone(Model.box().trim(None, [1, 0, 0]))
        self.assertIsNone(Model.box().trim(0, None))

    def test_trim_sphere_to_box(self):
        model = Model.sphere()
        for x in (-1, 1):
            model = model.trim((x/2, 0, 0), (x, 0, 0))
        for y in (-1, 1):
            model = model.trim((0, y/2, 0), (0, y, 0))
        for z in (-1, 1):
            model = model.trim((0, 0, z/2), (0, 0, z))
        self.assertTrue(model.is_manifold())
        mesh = model.get_trimesh()
        self.assertEqual(mesh.bounds.tolist(), [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        self.assertEqual(len(mesh.vertices), 8)
        self.assertEqual(len(mesh.faces), 12)
        self.assertAlmostEqual(mesh.area, 6)
        self.assertAlmostEqual(mesh.volume, 1)

    def test_trim_to_nothing(self):
        model = Model.box()
        model = model.trim((0, 0, -1), (0, 0, 1))
        self.assertTrue(model.is_manifold())
        with unittest.mock.patch('flitter.render.window.models.logger') as mock_logger:
            manifold = model.get_manifold()
            mock_logger.warning.assert_called_with("Result of operation was empty mesh: {}", model.name)
        self.assertIsNone(manifold)
        self.assertIsNone(model.get_trimesh())


class TestBoolean(utils.TestCase):
    Accuracy = 0.01

    def setUp(self):
        self.nested_box_models = [Model.box().transform(Matrix44.scale(2)), Model.box()]
        self.cross_models = [Model.box().transform(Matrix44.scale([3, 1, 1])),
                             Model.box().transform(Matrix44.scale([1, 3, 1])),
                             Model.box().transform(Matrix44.scale([1, 1, 3]))]
        self.capsule_models = [Model.cylinder().transform(Matrix44.scale([1, 1, 2])),
                               Model.sphere().transform(Matrix44.translate([0, 0, 1])),
                               Model.sphere().transform(Matrix44.translate([0, 0, -1]))]

    def tearDown(self):
        Model.flush_caches(0, 0)

    def wrap_model(self, model):
        return model

    def wrap_name(self, name):
        return name

    def test_ignored_none(self):
        self.assertEqual(self.wrap_model(Model.union(Model.box(), None, Model.sphere())).name,
                         self.wrap_name('union(!box, !sphere)'))
        self.assertEqual(self.wrap_model(Model.intersect(Model.box(), None, Model.sphere())).name,
                         self.wrap_name('intersect(!box, !sphere)'))
        self.assertEqual(self.wrap_model(Model.difference(Model.box(), None, Model.sphere())).name,
                         self.wrap_name('difference(!box, !sphere)'))

    def test_is_manifold(self):
        self.assertTrue(self.wrap_model(Model.union(Model.box(), Model.sphere())).is_manifold())
        self.assertTrue(self.wrap_model(Model.intersect(Model.box(), Model.sphere())).is_manifold())
        self.assertTrue(self.wrap_model(Model.difference(Model.box(), Model.sphere())).is_manifold())

    def test_nested_box_union(self):
        model = self.wrap_model(Model.union(*self.nested_box_models))
        mesh = model.get_trimesh()
        self.assertAlmostEqual(mesh.area, 24)
        self.assertAlmostEqual(mesh.volume, 8)

    def test_nested_box_intersect(self):
        model = self.wrap_model(Model.intersect(*self.nested_box_models))
        mesh = model.get_trimesh()
        self.assertAlmostEqual(mesh.area, 6)
        self.assertAlmostEqual(mesh.volume, 1)

    def test_nested_box_difference(self):
        model = self.wrap_model(Model.difference(*self.nested_box_models))
        mesh = model.get_trimesh()
        self.assertAlmostEqual(mesh.area, 30)
        self.assertAlmostEqual(mesh.volume, 7)

    def test_nested_box_reversed_difference(self):
        model = self.wrap_model(Model.difference(*reversed(self.nested_box_models)))
        with unittest.mock.patch('flitter.render.window.models.logger') as mock_logger:
            mesh = model.get_trimesh()
            mock_logger.warning.assert_called_with("Result of operation was empty mesh: {}", model.name)
        self.assertIsNone(mesh)

    def test_cross_union(self):
        model = self.wrap_model(Model.union(*self.cross_models))
        mesh = model.get_trimesh()
        self.assertAlmostEqual(mesh.area, 30, places=None, delta=self.Accuracy)
        self.assertAlmostEqual(mesh.volume, 7, places=None, delta=self.Accuracy)

    def test_cross_intersect(self):
        model = self.wrap_model(Model.intersect(*self.cross_models))
        mesh = model.get_trimesh()
        self.assertAlmostEqual(mesh.area, 6, places=None, delta=self.Accuracy)
        self.assertAlmostEqual(mesh.volume, 1, places=None, delta=self.Accuracy)

    def test_cross_difference(self):
        model = self.wrap_model(Model.difference(*self.cross_models))
        mesh = model.get_trimesh()
        self.assertAlmostEqual(mesh.area, 12, places=None, delta=self.Accuracy)
        self.assertAlmostEqual(mesh.volume, 2, places=None, delta=self.Accuracy)

    def test_capsule_union(self):
        model = self.wrap_model(Model.union(*self.capsule_models))
        mesh = model.get_trimesh()
        self.assertAlmostEqual(mesh.area, 4*math.pi + 4*math.pi, places=None, delta=self.Accuracy*5)
        self.assertAlmostEqual(mesh.volume, 2*math.pi + 4/3*math.pi, places=None, delta=self.Accuracy*5)

    def test_capsule_intersect(self):
        model = self.wrap_model(Model.intersect(*self.capsule_models))
        with unittest.mock.patch('flitter.render.window.models.logger') as mock_logger:
            mesh = model.get_trimesh()
            mock_logger.warning.assert_called_with("Result of operation was empty mesh: {}", model.name)
        self.assertIsNone(mesh)

    def test_capsule_difference(self):
        model = self.wrap_model(Model.difference(*self.capsule_models))
        mesh = model.get_trimesh()
        self.assertAlmostEqual(mesh.area, 4*math.pi + 4*math.pi, places=None, delta=self.Accuracy*30)
        self.assertAlmostEqual(mesh.volume, 2*math.pi - 4/3*math.pi, places=None, delta=self.Accuracy*5)


class TestSdfBoolean(TestBoolean):
    Accuracy = 0.2

    def wrap_model(self, model):
        return Model.sdf(None, model, (-3, -3, -3), (3, 3, 3), 0.1)

    def wrap_name(self, name):
        return f'sdf({name}, -3;-3;-3, 3;3;3, 0.1)'


class TestSdfTrim(utils.TestCase):
    def tearDown(self):
        Model.flush_caches(0, 0)

    def test_is_manifold(self):
        self.assertTrue(Model.sphere().trim((0, 0, 0), (0, 0, 1)).is_manifold())

    def test_trim_sphere_to_box(self):
        model = Model.sphere()
        for x in (-1, 1):
            model = model.trim((x/2, 0, 0), (x, 0, 0))
        for y in (-1, 1):
            model = model.trim((0, y/2, 0), (0, y, 0))
        for z in (-1, 1):
            model = model.trim((0, 0, z/2), (0, 0, z))
        model = Model.sdf(None, model, (-1, -1, -1), (1, 1, 1), 0.05)
        mesh = model.get_trimesh()
        self.assertEqual(mesh.bounds.tolist(), [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        self.assertAlmostEqual(mesh.area, 6)
        self.assertAlmostEqual(mesh.volume, 1)

    def test_trim_to_nothing(self):
        model = Model.box()
        model = model.trim((0, 0, -1), (0, 0, 1))
        model = Model.sdf(None, model, (-1, -1, -1), (1, 1, 1), 0.05)
        with unittest.mock.patch('flitter.render.window.models.logger') as mock_logger:
            manifold = model.get_manifold()
            mock_logger.warning.assert_called_with("Result of operation was empty mesh: {}", model.name)
        self.assertIsNone(manifold)
        self.assertIsNone(model.get_trimesh())


class TestBuffers(utils.TestCase):
    def tearDown(self):
        Model.flush_caches(0, 0)

    def test_box_get_buffers(self):
        model = Model.box()
        glctx = unittest.mock.Mock()
        objects = {}
        buffers = model.get_buffers(glctx, objects)
        self.assertIn(model.id, objects)
        self.assertIs(objects[model.id], buffers)
        vertex_data, = glctx.buffer.mock_calls[0].args
        self.assertEqual(vertex_data.dtype.name, 'float32')
        self.assertEqual(vertex_data.shape, (24, 8))
        index_data, = glctx.buffer.mock_calls[1].args
        self.assertEqual(index_data.dtype.name, 'int32')
        self.assertEqual(index_data.shape, (12, 3))
        model.invalidate()
        self.assertNotIn(model.id, objects)
