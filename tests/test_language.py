"""
Flitter language integration tests
"""

from pathlib import Path
import unittest

from flitter import configure_logger
from flitter.language.tree import Literal, Let, Sequence, Export, Binding, Top
from flitter.language.parser import parse, ParseError
from flitter.language.vm import Function
from flitter.model import Vector, Node, Context, StateDict, DummyStateDict, null

from . import utils


configure_logger('WARNING')


class TestPragmas(unittest.TestCase):
    def test_no_pragmas(self):
        top = parse("\n")
        self.assertEqual(top.pragmas, ())
        program = top.compile()
        self.assertEqual(program.pragmas, {})

    def test_pragma_not_at_top(self):
        with self.assertRaises(ParseError):
            parse("""
let x=10
%pragma tempo 120
""")

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


class TestParseErrors(unittest.TestCase):
    def test_missing_bindings(self):
        with self.assertRaises(ParseError):
            parse("""
let
!foo
""")

    def test_unexpected_indent(self):
        with self.assertRaises(ParseError):
            parse("""
let x=5
    !foo x=x
""")

    def test_missing_indent(self):
        with self.assertRaises(ParseError):
            parse("""
for x in ..5
!foo x=x
""")

    def bad_identifier(self):
        with self.assertRaises(ParseError):
            parse("""
let 1x=5
""")

    def mixed_position_and_named_args(self):
        with self.assertRaises(ParseError):
            parse("""
f(1, x=2, 3)
""")


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

    def assertCodeOutput(self, code, output, with_errors=None, skip_simplifier=False, **names):
        top = parse(code.strip())
        output = output.strip()
        if with_errors is None:
            with_errors = set()
        vm_context = Context(names={name: Vector(value) for name, value in names.items()}, state=StateDict())
        vm_output = '\n'.join(repr(node) for node in top.compile(initial_lnames=tuple(names)).run(vm_context).root.children)
        self.assertEqual(vm_output, output, msg="VM output is incorrect")
        self.assertEqual(vm_context.errors, with_errors)
        if skip_simplifier:
            return
        simplified_top, simplifier_context = top.simplify(static=names, return_context=True)
        self.assertEqual(simplifier_context.errors, with_errors)
        expr = simplified_top.body
        while not isinstance(expr, Literal):
            if isinstance(expr, Let):
                expr = expr.body
            elif isinstance(expr, Sequence) and len(expr.expressions) == 2:
                if isinstance(expr.expressions[1], Export):
                    expr = expr.expressions[0]
                elif isinstance(expr.expressions[1], Let) and isinstance(expr.expressions[1].body, Export):
                    expr = expr.expressions[0]
                else:
                    break
            else:
                break
        if isinstance(expr, Export):
            simplifier_output = ""
        else:
            self.assertIsInstance(expr, Literal, msg="Unexpected simplification output")
            simplifier_output = '\n'.join(repr(node) for node in expr.value)
        self.assertEqual(simplifier_output, output, msg="Simplifier output is incorrect")
        mixed_top, simplifier_context = top.simplify(dynamic=names, return_context=True)
        vm_context = Context(names={name: Vector(value) for name, value in names.items()}, state=StateDict())
        vm_output = '\n'.join(repr(node) for node in mixed_top.compile(initial_lnames=tuple(names)).run(vm_context).root.children)
        self.assertEqual(vm_output, output, msg="VM output is incorrect")
        self.assertEqual(simplifier_context.errors | vm_context.errors, with_errors)

    def test_empty(self):
        self.assertCodeOutput("", "")

    def test_literal_node(self):
        self.assertCodeOutput("!foo #bar x=5 y=:baz z='Hello world!'", "!foo #bar x=5 y=:baz z='Hello world!'")

    def test_unicode_names(self):
        self.assertCodeOutput("!café #århus 水=:𓀀", "!café #århus 水=:𓀀")

    def test_names_with_primes(self):
        self.assertCodeOutput(
            """
let y=1 y''=y+1 y'''=y''+1
!node' x'=y''' y=:y'''' z='prime'
            """,
            """
!node' x'=3 y=:y'''' z='prime'
            """)

    def test_named_values(self):
        self.assertCodeOutput(
            """
!node values=null;true;false;inf;-inf;nan
            """,
            """
!node values=1;0;inf;-inf;nan
            """
        )

    def test_let_only(self):
        self.assertCodeOutput("let x=5", "")

    def test_contextual_parser(self):
        self.assertCodeOutput(
            """
let in=import + 1
!foo let=in
            """,
            """
!foo let=6
            """, **{'import': 5})

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

    def test_where(self):
        self.assertCodeOutput(
            """
let z=2
!foo x=y+z where y=z*2 where z=5
            """,
            """
!foo x=15
            """)

    def test_multibind_loop(self):
        self.assertCodeOutput(
            """
for x;y in ..9
    !node x=x y=y z=z
            """,
            """
!node x=0 y=1 z=0
!node x=2 y=3 z=0
!node x=4 y=5 z=0
!node x=6 y=7 z=0
!node x=8 z=0
            """, z=0)

    def test_nested_loops(self):
        self.assertCodeOutput(
            """
let n=4
for i in ..n
    !group i=i
        for j in ..i
            !item numbers=(k*(j+1) for k in m..i+1)
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
            """, m=1)

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

    def test_template_builtin_calls(self):
        self.assertCodeOutput(
            """
func foo(n)
    @sum
        for i in ..n
            2**i

!foo foo=foo(10)
            """,
            """
!foo foo=1023
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

    def test_anonymous_function(self):
        """Note that this is statically reducible because `map` is inlined and so the
           anonymous function is bound to `f`, which therefore becomes a function name"""
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

    def test_anonymous_function_with_where(self):
        self.assertCodeOutput(
            """
let f = func(x) x+y where y=x*x
!foo bar=f(10)
            """,
            """
!foo bar=110
            """
        )

    def test_anonymous_function_returning_anonymous_function(self):
        self.assertCodeOutput(
            """
let f = func(x) func(y) x + y
!foo bar=f(10)(5)
            """,
            """
!foo bar=15
            """
        )

    def test_accidental_anonymous_vector(self):
        self.assertCodeOutput(
            """
let f = func(x) x;1;2
!foo bar=f(0)
            """,
            """
!foo bar=0
            """,
            with_errors={'1.0 is not callable', '2.0 is not callable'},
            skip_simplifier=True
        )

    def test_some_deliberately_obtuse_behaviour(self):
        self.assertCodeOutput(
            """
@(func(_, x) ((!foo x=x*y) for y in ..3)) x=5
            """,
            """
!foo x=0
!foo x=5
!foo x=10
            """)

    def test_bad_function_call(self):
        self.assertCodeOutput(
            """
!foo color=hsv()
            """,
            """
!foo
            """, with_errors={'Error calling hsv: hsv() takes exactly 1 positional argument (0 given)'})

    def test_contains(self):
        self.assertCodeOutput(
            """
!foo x=(x in y) y=(y in y) z=(y in x)
            """,
            """
!foo x=1 y=1 z=0
            """, x=Vector((4, 5, 6)), y=Vector.range(10))

    def test_sequence_let(self):
        self.assertCodeOutput(
            """
let foo;bar =
    !foo x=1
        !bar
        !baz
    !bar b=2

!frob
    bar
    foo
            """,
            """
!frob
 !bar b=2
 !foo x=1
  !bar
  !baz
            """)


class ScriptTest(utils.TestCase):
    """
    Run a Flitter script under three different conditions and test that the generated root nodes
    and exported values match:

    1. Compile the AST and run it without *any* simplification and with an empty state
    2. Simplify the AST with known names (`beat`, etc.) as dynamic and an empty state
    3. Simplify the AST with known names as static values and a special dummy state that claims to
       contain all keys but returns null for all as an empty state would

    In the third case, all state lookups will be statically evaluated to literal nulls by the
    simplifier. The idea is to exercise the simplifier as much as possible.
    """

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

    def assertSimplifierDoesntChangeBehaviour(self, script):
        self.maxDiff = None
        state = StateDict()
        null_state = DummyStateDict()
        names = {'beat': Vector(0), 'tempo': Vector(120), 'quantum': Vector(4), 'fps': Vector(60), 'OUTPUT': null,
                 'run_time': Vector(1), 'frame': Vector(0)}
        top = parse(script.read_text(encoding='utf8'))
        # Compile un-simplified AST and execute program:
        program1 = top.compile(initial_lnames=tuple(names))
        program1.set_path(script)
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
        program2.set_path(script)
        self.assertNotEqual(str(program1), str(program2), msg="Dynamically-simplified program should be different from original")
        context = program2.run(Context(names=dict(names), state=state))
        root2 = context.root
        exports2 = context.exports
        self.assertEqual(repr(root2), repr(root1), msg="Dynamically-simplified program output doesn't match original")
        self.assertExportsMatch(exports2, exports1, msg="Dynamically-simplified program exports don't match original")
        # Simplify AST with static names and null state, compile and execute:
        top3 = top.simplify(static=names, state=null_state)
        self.assertIs(top3.simplify(static=names, state=null_state), top3, msg="Static simplification not complete in one step")
        program3 = top3.compile()
        program3.set_path(script)
        self.assertNotEqual(len(program3), len(program1), msg="Statically-simplified program length should be different from original")
        context = program3.run(Context(state=state))
        root3 = context.root
        exports3 = context.exports
        for name in names:
            del exports3[name]
        self.assertEqual(repr(root3), repr(root1), msg="Statically-simplified program output doesn't match original")
        self.assertExportsMatch(exports3, exports1, msg="Statically-simplified program exports don't match original")


class TestDocumentationDiagrams(ScriptTest):
    """
    Run all documentation diagrams with and without simplification.
    """

    DIAGRAMS = Path(__file__).parent.parent / 'docs/diagrams'

    def test_box_uvmap(self):
        self.assertSimplifierDoesntChangeBehaviour(self.DIAGRAMS / 'box_uvmap.fl')

    def test_dummyshader(self):
        self.assertSimplifierDoesntChangeBehaviour(self.DIAGRAMS / 'dummyshader.fl')

    def test_easings(self):
        self.assertSimplifierDoesntChangeBehaviour(self.DIAGRAMS / 'easings.fl')

    def test_pseudorandoms(self):
        self.assertSimplifierDoesntChangeBehaviour(self.DIAGRAMS / 'pseudorandoms.fl')

    def test_spheroidbox(self):
        self.assertSimplifierDoesntChangeBehaviour(self.DIAGRAMS / 'spheroidbox.fl')

    def test_torus(self):
        self.assertSimplifierDoesntChangeBehaviour(self.DIAGRAMS / 'torus.fl')

    def test_waveforms(self):
        self.assertSimplifierDoesntChangeBehaviour(self.DIAGRAMS / 'waveforms.fl')

    def test_petri(self):
        self.assertSimplifierDoesntChangeBehaviour(self.DIAGRAMS / 'petri.fl')


class TestDocumentationTutorial(ScriptTest):
    """
    Run all tutorial images with and without simplification.
    """

    TUTORIAL_IMAGES = Path(__file__).parent.parent / 'docs/tutorial_images'

    def test_tutorial1(self):
        self.assertSimplifierDoesntChangeBehaviour(self.TUTORIAL_IMAGES / 'tutorial1.fl')

    def test_tutorial2(self):
        self.assertSimplifierDoesntChangeBehaviour(self.TUTORIAL_IMAGES / 'tutorial2.fl')

    def test_tutorial3(self):
        self.assertSimplifierDoesntChangeBehaviour(self.TUTORIAL_IMAGES / 'tutorial3.fl')

    def test_tutorial4(self):
        self.assertSimplifierDoesntChangeBehaviour(self.TUTORIAL_IMAGES / 'tutorial4.fl')

    def test_tutorial5(self):
        self.assertSimplifierDoesntChangeBehaviour(self.TUTORIAL_IMAGES / 'tutorial5.fl')

    def test_tutorial6(self):
        self.assertSimplifierDoesntChangeBehaviour(self.TUTORIAL_IMAGES / 'tutorial6.fl')


class TestExamples(ScriptTest):
    """
    Run all examples with and without simplification.
    """

    EXAMPLES = Path(__file__).parent.parent / 'examples'

    def test_bauble(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'bauble.fl')

    def test_bounce(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'bounce.fl')

    def test_canvas3d(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'canvas3d.fl')

    def test_dots(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'dots.fl')

    def test_hoops(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'hoops.fl')

    def test_linear(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'linear.fl')

    def test_linelight(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'linelight.fl')

    def test_oklch(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'oklch.fl')

    def test_physics(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'physics.fl')

    def test_sdf(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'sdf.fl')

    def test_smoke(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'smoke.fl')

    def test_solidgeometry(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'solidgeometry.fl')

    def test_sphere(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'sphere.fl')

    def test_teaset(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'teaset.fl')

    def test_textures(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'textures.fl')

    def test_translucency(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'translucency.fl')

    def test_video(self):
        self.assertSimplifierDoesntChangeBehaviour(self.EXAMPLES / 'video.fl')
