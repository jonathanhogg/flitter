"""
Flitter engine functional tests
"""

import asyncio
from pathlib import Path
import sys
import sysconfig
import tempfile
import unittest

import PIL.Image
import PIL.ImageChops
import PIL.ImageStat

from flitter.engine.control import EngineController


def image_diff(ref, img):
    """Return how dissimilar two images are as a score between 0 (identical) and 1 (completely different)"""
    return PIL.ImageStat.Stat(PIL.ImageChops.difference(ref, img).convert('L')).sum[0] / (img.width * img.height * 255)


TEST_IMAGES_DIR = Path(__file__).parent.parent / f'build/test_images.{sysconfig.get_platform()}-{sys.implementation.cache_tag}'
TEST_IMAGES_DIR.mkdir(mode=0o775, parents=True, exist_ok=True)


class TestRendering(unittest.TestCase):
    """
    Some simple rendering functional tests

    Scripts will be executed for a single frame and a `SIZE` image should be
    saved to the filename `OUTPUT`.
    """

    SIZE = (200, 100)

    def setUp(self):
        self.script_path = Path(tempfile.mktemp('.fl'))
        self.input_path = Path(tempfile.mktemp('.png'))
        self.output_path = Path(tempfile.mktemp('.png'))
        self.controller = EngineController(realtime=False, offscreen=True, target_fps=1, run_time=1,
                                           defined_names={'SIZE': self.SIZE,
                                                          'INPUT': str(self.input_path),
                                                          'OUTPUT': str(self.output_path)})

    def tearDown(self):
        if self.script_path.exists():
            self.script_path.unlink()
        if self.input_path.exists():
            self.input_path.unlink()
        if self.output_path.exists():
            self.output_path.unlink()

    def test_simple_canvas(self):
        """Render a simple green rectangle filling the viewport"""
        self.script_path.write_text("""
!window size=SIZE
    !record filename=OUTPUT
        !canvas
            !path
                !rect size=SIZE
                !fill color=0;1;0
""", encoding='utf8', newline='\n')
        self.controller.load_page(self.script_path)
        asyncio.run(self.controller.run())
        img = PIL.Image.open(self.output_path)
        self.assertEqual(img.size, self.SIZE)
        self.assertEqual(img.tobytes(), b'\x00\xff\x00' * self.SIZE[0] * self.SIZE[1])

    def test_simple_canvas3d(self):
        """Render a simple green emissive box filling the viewport"""
        self.script_path.write_text("""
!window size=SIZE
    !record filename=OUTPUT
        !canvas3d orthographic=true
            !box size=SIZE*(1;1;0) emissive=0;1;0
""", encoding='utf8', newline='\n')
        self.controller.load_page(self.script_path)
        asyncio.run(self.controller.run())
        img = PIL.Image.open(self.output_path)
        self.assertEqual(img.size, self.SIZE)
        self.assertEqual(img.tobytes(), b'\x00\xff\x00' * self.SIZE[0] * self.SIZE[1])

    def test_simple_shader(self):
        """Render a simple shader that returns green for all fragments"""
        self.script_path.write_text("""
!window size=SIZE
    !record filename=OUTPUT
        !shader fragment='''#version 330
                            in vec2 coord;
                            out vec4 color;
                            void main() {
                                color = vec4(0.0, 1.0, 0.0, 1.0);
                            }'''
""", encoding='utf8', newline='\n')
        self.controller.load_page(self.script_path)
        asyncio.run(self.controller.run())
        img = PIL.Image.open(self.output_path)
        self.assertEqual(img.size, self.SIZE)
        self.assertEqual(img.tobytes(), b'\x00\xff\x00' * self.SIZE[0] * self.SIZE[1])

    def test_simple_image(self):
        """Render an image node from a simple temporary image"""
        PIL.Image.new('RGB', self.SIZE, (0, 255, 0)).save(self.input_path, 'PNG')
        self.script_path.write_text("""
!window size=SIZE
    !record filename=OUTPUT
        !image filename=INPUT
""", encoding='utf8', newline='\n')
        self.controller.load_page(self.script_path)
        asyncio.run(self.controller.run())
        img = PIL.Image.open(self.output_path)
        self.assertEqual(img.size, self.SIZE)
        self.assertEqual(img.tobytes(), b'\x00\xff\x00' * self.SIZE[0] * self.SIZE[1])


class ScriptTest(unittest.TestCase):
    def assertScriptOutputMatchesImage(self, script, suffix='.png', target_fps=1, run_time=1, **kwargs):
        output_path = TEST_IMAGES_DIR / script.with_suffix(suffix).name
        if output_path.exists():
            output_path.unlink()
        reference = PIL.Image.open(script.with_suffix('.png'))
        controller = EngineController(realtime=False, offscreen=True, defined_names={'OUTPUT': str(output_path)},
                                      target_fps=target_fps, run_time=run_time, **kwargs)
        controller.load_page(script)
        asyncio.run(controller.run())
        output = PIL.Image.open(output_path)
        self.assertEqual(reference.size, output.size, msg="Size mismatch")
        self.assertLess(image_diff(reference, output), 0.004, msg="Output differs from reference image")

    def assertAllScriptOutputsMatchesImages(self, scripts, suffix='.png', target_fps=1, run_time=1, **kwargs):
        for i, script in enumerate(scripts):
            with self.subTest(script=script, **kwargs):
                self.assertScriptOutputMatchesImage(script, suffix=suffix, target_fps=target_fps, run_time=run_time, **kwargs)


class TestDocumentationDiagrams(ScriptTest):
    """
    Recreate the documentation diagrams and check them against the pre-calculated ones.
    """

    DIAGRAMS = Path(__file__).parent.parent / 'docs/diagrams'

    def test_box_uvmap(self):
        self.assertScriptOutputMatchesImage(self.DIAGRAMS / 'box_uvmap.fl')

    def test_dummyshader(self):
        self.assertScriptOutputMatchesImage(self.DIAGRAMS / 'dummyshader.fl')

    def test_easings(self):
        self.assertScriptOutputMatchesImage(self.DIAGRAMS / 'easings.fl')

    def test_petri(self):
        self.assertScriptOutputMatchesImage(self.DIAGRAMS / 'petri.fl', target_fps=10, run_time=10)

    def test_pseudorandoms(self):
        self.assertScriptOutputMatchesImage(self.DIAGRAMS / 'pseudorandoms.fl')

    def test_spheroidbox(self):
        self.assertScriptOutputMatchesImage(self.DIAGRAMS / 'spheroidbox.fl')

    def test_torus(self):
        self.assertScriptOutputMatchesImage(self.DIAGRAMS / 'torus.fl')

    def test_waveforms(self):
        self.assertScriptOutputMatchesImage(self.DIAGRAMS / 'waveforms.fl')


class TestDocumentationTutorial(ScriptTest):
    """
    Recreate the tutorial images and check them against the pre-calculated ones.
    """

    TUTORIAL_IMAGES = Path(__file__).parent.parent / 'docs/tutorial_images'

    def test_tutorial1(self):
        self.assertScriptOutputMatchesImage(self.TUTORIAL_IMAGES / 'tutorial1.fl')

    def test_tutorial2(self):
        self.assertScriptOutputMatchesImage(self.TUTORIAL_IMAGES / 'tutorial2.fl')

    def test_tutorial3(self):
        self.assertScriptOutputMatchesImage(self.TUTORIAL_IMAGES / 'tutorial3.fl')

    def test_tutorial4(self):
        self.assertScriptOutputMatchesImage(self.TUTORIAL_IMAGES / 'tutorial4.fl')

    def test_tutorial5(self):
        self.assertScriptOutputMatchesImage(self.TUTORIAL_IMAGES / 'tutorial5.fl')

    def test_tutorial6(self):
        self.assertScriptOutputMatchesImage(self.TUTORIAL_IMAGES / 'tutorial6.fl')


class TestExamples(ScriptTest):
    """
    Recreate the examples and check them against the pre-calculated ones.
    """

    EXAMPLES = Path(__file__).parent.parent / 'examples'

    def test_bauble(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'bauble.fl')

    def test_bounce(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'bounce.fl', target_fps=10)

    def test_canvas3d(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'canvas3d.fl')

    def test_dots(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'dots.fl', target_fps=10)

    def test_hoops(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'hoops.fl', target_fps=10)

    def test_linear(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'linear.fl')

    def test_linelight(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'linelight.fl')

    def test_oklch(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'oklch.fl')

    def test_physics(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'physics.fl', target_fps=10)

    def test_sdf(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'sdf.fl')

    def test_smoke(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'smoke.fl', target_fps=10)

    def test_solidgeometry(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'solidgeometry.fl')

    def test_sphere(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'sphere.fl')

    def test_teaset(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'teaset.fl')

    def test_textures(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'textures.fl', target_fps=2)

    def test_translucency(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'translucency.fl')

    def test_video(self):
        self.assertScriptOutputMatchesImage(self.EXAMPLES / 'video.fl')


@unittest.skipIf(sys.platform != 'linux', 'OpenGL ES only available on Linux')
class TestExamplesOpenGLES(TestExamples):
    def assertScriptOutputMatchesImage(self, script, suffix='.es.png', target_fps=1, run_time=1, **kwargs):
        super().assertScriptOutputMatchesImage(script, suffix, target_fps, run_time, opengl_es=True, **kwargs)
