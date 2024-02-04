"""
Flitter functional testing
"""

import asyncio
from pathlib import Path
import subprocess
import tempfile
import unittest

import PIL.Image

from flitter.engine.control import EngineController


class TestRendering(unittest.TestCase):
    """
    Render tests
    """

    def setUp(self):
        self.scripts_path = Path(__file__).parent / 'scripts'

    def test_hoops(self):
        output_filename = Path(tempfile.mktemp('.jpg'))
        try:
            controller = EngineController(target_fps=30, realtime=False, run_time=1, offscreen=True,
                                          defined_variables={'SIZE': 100, 'OUTPUT': str(output_filename)})
            controller.load_page(self.scripts_path / 'simple.fl')
            controller.switch_to_page(0)
            asyncio.run(controller.run())
            img = PIL.Image.open(output_filename)
            self.assertEqual(img.size, (100, 100))
            self.assertEqual(img.tobytes(), b'\xfe\x00\x00' * 10000)
        finally:
            if output_filename.exists():
                output_filename.unlink()
