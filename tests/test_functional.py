"""
Flitter functional testing
"""

import subprocess
import unittest


class TestRendering(unittest.TestCase):
    """
    Render tests
    """

    def test_hoops(self):
        args = ('flitter', '--offscreen', '--lockstep', '--runtime=1', 'examples/hoops.fl')
        process = subprocess.run(args, capture_output=True)
        self.assertEqual(process.returncode, 0)
        self.assertTrue(process.stderr.find(b'SUCCESS: Loaded page 0: examples/hoops.fl\n') >= 0)
