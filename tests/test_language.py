"""
Flitter language integration tests
"""

import math
from pathlib import Path
import unittest

from flitter import configure_logger
from flitter.language.tree import Literal, Let, Sequence, Export, Binding, Top
from flitter.language.parser import parse
from flitter.language.vm import Function
from flitter.model import Vector, Node, Context, StateDict, DummyStateDict, null


configure_logger('WARNING')


class TestPragmas(unittest.TestCase):
    def test_no_pragmas(self):
        top = parse("\n")
        self.assertEqual(top.pragmas, ())
        program = top.compile()
        self.assertEqual(program.pragmas, {})

    def test_pragmas(self):
        top = parse("""
%pragma foo 5
%pragma bar 'five'
%pragma baz :five
%pragma buz !five
""")
        self.assertEqual(repr(top), repr(Top((Binding('foo', Literal(5)),
                                              Binding('bar', Literal('five')),
                                              Binding('baz', Literal(Vector.symbol('five'))),
                                              Binding('buz', Literal(Node('five')))), Sequence((Export(None),)))))
        program = top.compile()
        self.assertEqual(program.pragmas, {'foo': Vector(5), 'bar': Vector('five'), 'baz': Vector.symbol('five'), 'buz': Vector(Node('five'))})


class TestLanguageFeatures(unittest.TestCase):
    """
    Test a series of language features by parsing the given code, evaluating it and comparing the
    output to that given. The code is evaluated in two ways:

    1. Compile the unsimplified AST and run the resulting VM program with any additional bound
       names passed in as initial lvalues
    2. Simplify the AST with any additional bound names passed in as static, then extract the
       literal values out of the resulting AST

    It is assumed that all of the examples can be fully reduced to a literal by the simplifier.
    """

    def assertCodeOutput(self, code, output, **names):
        top = parse(code.strip() + "\n")
        output = output.strip()
        run_context = Context(names={name: Vector(value) for name, value in names.items()}, state=StateDict())
        vm_output = '\n'.join(repr(node) for node in top.compile(initial_lnames=tuple(names)).run(run_context).root.children)
        self.assertEqual(vm_output, output, msg="VM output is incorrect")
        expr = top.simplify(static=names).body
        while not isinstance(expr, Literal):
            if isinstance(expr, Let):
                expr = expr.body
            elif isinstance(expr, Sequence) and len(expr.expressions) == 2 and isinstance(expr.expressions[1], Export):
                expr = expr.expressions[0]
            else:
                break
        self.assertIsInstance(expr, Literal, msg="Unexpected simplification output")
        simplifier_output = '\n'.join(repr(node) for node in expr.value)
        self.assertEqual(simplifier_output, output, msg="Simplifier output is incorrect")

    def test_literal_node(self):
        self.assertCodeOutput(
            """
!foo #bar x=5 y=:baz z='Hello world!'
            """,
            """
!foo #bar x=5 y=:baz z='Hello world!'
            """)

    def test_names(self):
        self.assertCodeOutput(
            """
!foo x=x y=y z=z #bar
            """,
            """
!foo #bar x=5 y=:baz z='Hello world!'
            """, x=5, y=Vector.symbol('baz'), z='Hello world!')

    def test_let(self):
        self.assertCodeOutput(
            """
let x=5 y=:baz z='Hello world!'
!foo #bar x=x y=y z=z
            """,
            """
!foo #bar x=5 y=:baz z='Hello world!'
            """)

    def test_multibind_loop(self):
        self.assertCodeOutput(
            """
for x;y in ..9
    !node x=x y=y
            """,
            """
!node x=0 y=1
!node x=2 y=3
!node x=4 y=5
!node x=6 y=7
!node x=8
            """)

    def test_nested_loops(self):
        self.assertCodeOutput(
            """
for i in ..n
    !group i=i
        for j in ..i
            !item numbers=(k*(j+1) for k in 1..i+1)
            """,
            """
!group i=0
!group i=1
 !item numbers=1
!group i=2
 !item numbers=1;2
 !item numbers=2;4
!group i=3
 !item numbers=1;2;3
 !item numbers=2;4;6
 !item numbers=3;6;9
            """, n=4)

    def test_if(self):
        self.assertCodeOutput(
            """
for i in ..2
    if i
        !one
            """,
            """
!one
            """)

    def test_if_else(self):
        self.assertCodeOutput(
            """
for i in ..4
    if not i
        !zero
    elif i == 2
        !two
    elif i > 2
        !more
    else
        !one
            """,
            """
!zero
!one
!two
!more
            """)

    def test_recursive_inlined_function(self):
        # Also default parameter values and out-of-order named arguments
        self.assertCodeOutput(
            """
func fib(n, x=0, y=1)
    fib(n-1, y=y+x, x=y) if n > 0 else x

!fib x=fib(10)
            """,
            """
!fib x=55
            """)

    def test_builtin_calls(self):
        self.assertCodeOutput(
            """
for i in ..n
    !color rgb=hsv(i/n;1;1)
            """,
            """
!color rgb=1;0;0
!color rgb=1;1;0
!color rgb=0;1;0
!color rgb=0;1;1
!color rgb=0;0;1
!color rgb=1;0;1
            """, n=6)

    def test_template_calls(self):
        self.assertCodeOutput(
            """
func foo(nodes, x=10, y, z='world')
    !foo x=x z='hello';z
        nodes #test y=y

@foo
@foo z='you'
    !bar
@foo y=5
    @foo z='me' x=99
        !baz
            """,
            """
!foo x=10 z='hello';'world'
!foo x=10 z='hello';'you'
 !bar #test
!foo x=10 z='hello';'world'
 !foo #test x=99 z='hello';'me' y=5
  !baz #test
            """)

    def test_nested_functions(self):
        self.assertCodeOutput(
            """
func fib(n)
    func fib'(n, x, y)
        fib'(n-1, y=y+x, x=y) if n > 0 else x
    fib'(n, 0, 1)

!fib x=fib(10)
            """,
            """
!fib x=55
            """)

    def test_anonymous_functions(self):
        """Note that this is only statically reducible because `map` is inlined and so the
           anonymous function is bound to `f` and therefore becomes a name"""
        self.assertCodeOutput(
            """
func map(f, xs)
    for x in xs
        f(x)

!doubled x=map(func(x, y=2) x*y, ..10)
            """,
            """
!doubled x=0;2;4;6;8;10;12;14;16;18
            """)


class TestExampleScripts(unittest.TestCase):
    """
    Run all Flitter scripts in `/examples`, `/docs/diagrams` and `/docs/tutorial_images` under three
    different conditions and test that the generated root nodes and exported values match:

    1. Compile the AST and run it without *any* simplification and with an empty state
    2. Simplify the AST with known names (`beat`, etc.) as dynamic and an empty state
    3. Simplify the AST with known names as static values and a special dummy state that claims to
       contain all keys but returns null for all as an empty state would

    In the third case, all state lookups will be statically evaluated to literal nulls by the
    simplifier. The idea is to exercise the simplifier as much as possible.
    """

    def assertAllAlmostEqual(self, xs, ys, msg=None):
        for x, y in zip(xs, ys):
            self.assertEqual(math.isnan(x), math.isnan(y), msg=msg)
            if not (math.isnan(x) or math.isnan(y)):
                self.assertAlmostEqual(x, y, msg=msg)

    def assertExportsMatch(self, exports1, exports2, msg=None):
        self.assertEqual(exports1.keys(), exports2.keys())
        for name in exports1:
            a = exports1[name]
            b = exports2[name]
            if a.isinstance(Function) and b.isinstance(Function):
                self.assertEqual(a[0].__name__, b[0].__name__, msg=msg)
            elif a.numeric and b.numeric:
                self.assertAllAlmostEqual(a, b, msg=msg)
            else:
                self.assertEqual(a, b, msg=msg)

    def _test_integration_script(self, filepath):
        self.maxDiff = None
        state = StateDict()
        null_state = DummyStateDict()
        names = {'beat': Vector(0), 'tempo': Vector(120), 'quantum': Vector(4), 'fps': Vector(60), 'OUTPUT': null}
        top = parse(filepath.read_text(encoding='utf8'))
        # Compile un-simplified AST and execute program:
        program1 = top.compile(initial_lnames=tuple(names))
        program1.set_path(filepath)
        program1.use_simplifier(False)
        context = program1.run(Context(names=dict(names), state=state))
        root1 = context.root
        exports1 = context.exports
        windows = [node for node in root1.children if node.kind == 'window']
        self.assertEqual(len(windows), 1, msg="Should be a single window node in program output")
        self.assertGreater(len(tuple(windows[0].children)), 0, "Output window node should have children")
        # Simplify AST with dynamic names, compile and execute:
        top2 = top.simplify(dynamic=set(names))
        self.assertEqual(repr(top2.simplify(dynamic=set(names))), repr(top2), msg="Dynamic simplification not complete in one step")
        program2 = top2.compile(initial_lnames=tuple(names))
        program2.set_path(filepath)
        self.assertNotEqual(len(program1), len(program2), msg="Dynamically-simplified program length should be different from original")
        context = program2.run(Context(names=dict(names), state=state))
        root2 = context.root
        exports2 = context.exports
        self.assertEqual(repr(root2), repr(root1), msg="Dynamically-simplified program output doesn't match original")
        self.assertExportsMatch(exports2, exports1, msg="Dynamically-simplified program exports don't match original")
        # Simplify AST with static names and null state, compile and execute:
        top3 = top.simplify(static=names, state=null_state)
        self.assertEqual(repr(top3.simplify(static=names, state=null_state)), repr(top3), msg="Static simplification not complete in one step")
        program3 = top3.compile()
        program3.set_path(filepath)
        self.assertNotEqual(len(program3), len(program1), msg="Statically-simplified program length should be different from original")
        context = program3.run(Context(state=state))
        root3 = context.root
        exports3 = context.exports
        for name in names:
            del exports3[name]
        self.assertEqual(repr(root3), repr(root1), msg="Statically-simplified program output doesn't match original")
        self.assertExportsMatch(exports3, exports1, msg="Statically-simplified program exports don't match original")

    def _test_integration_all_scripts(self, dirpath):
        count = 0
        for filename in dirpath.iterdir():
            filepath = dirpath / filename
            if filepath.suffix == '.fl':
                with self.subTest(filepath=filepath):
                    self._test_integration_script(filepath)
                    count += 1
        self.assertGreater(count, 0, msg="No scripts found")

    def test_integration_examples(self):
        self._test_integration_all_scripts(Path(__file__).parent.parent / 'examples')

    def test_integration_docs_diagrams(self):
        self._test_integration_all_scripts(Path(__file__).parent.parent / 'docs/diagrams')

    def test_integration_docs_tutorial(self):
        self._test_integration_all_scripts(Path(__file__).parent.parent / 'docs/tutorial_images')
