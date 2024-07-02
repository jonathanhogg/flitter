"""
Flitter language integration tests
"""

from pathlib import Path
import unittest

from flitter.language.parser import parse
from flitter.model import Vector, Context, StateDict, DummyStateDict, null


class TestLanguage(unittest.TestCase):
    def _test_integration(self, filepath):
        self.maxDiff = None
        state = StateDict()
        null_state = DummyStateDict()
        names = {'beat': Vector(0), 'tempo': Vector(120), 'quantum': Vector(4), 'fps': Vector(60), 'OUTPUT': null}
        top = parse(filepath.read_text(encoding='utf8'))
        # Compile un-simplified AST and execute program:
        program1 = top.compile(initial_lnames=tuple(names))
        program1.set_path(filepath)
        program1.use_simplifier(False)
        root1 = program1.run(Context(names=dict(names), state=state)).root
        windows = [node for node in root1.children if node.kind == 'window']
        self.assertEqual(len(windows), 1, msg="Should be a single window node in program output")
        self.assertGreater(len(tuple(windows[0].children)), 0, "Output window node should have children")
        # Simplify AST with dynamic names, compile and execute:
        program2 = top.simplify(dynamic=set(names)).compile(initial_lnames=tuple(names))
        program2.set_path(filepath)
        self.assertNotEqual(len(program1), len(program2), msg="Program lengths should be different")
        root2 = program2.run(Context(names=dict(names), state=state)).root
        self.assertEqual(repr(root2), repr(root1), msg="Program outputs should match")
        # Simplify AST with static names and null state, compile and execute:
        program3 = top.simplify(static=names, state=null_state).compile()
        program3.set_path(filepath)
        self.assertNotEqual(len(program3), len(program1), msg="Program lengths should be different")
        root3 = program3.run(Context(state=state)).root
        self.assertEqual(repr(root3), repr(root1), msg="Program outputs should match")

    def _test_integration_all_scripts(self, dirpath):
        count = 0
        for filename in dirpath.iterdir():
            filepath = dirpath / filename
            if filepath.suffix == '.fl':
                with self.subTest(filepath=filepath):
                    self._test_integration(filepath)
                    count += 1
        self.assertGreater(count, 0)

    def test_integration_examples(self):
        self._test_integration_all_scripts(Path(__file__).parent.parent / 'examples')

    def test_integration_docs_diagrams(self):
        self._test_integration_all_scripts(Path(__file__).parent.parent / 'docs/diagrams')

    def test_integration_docs_tutorial(self):
        self._test_integration_all_scripts(Path(__file__).parent.parent / 'docs/tutorial_images')
