
# The Language

## Quick introduction

**flitter** is a declarative tree-construction language (`Node` in the model).
All values are arrays (`Vector` in the model), delimited with semicolons, and
all maths is piece-wise. Short vectors are repeated as necessary in binary
operations, i.e., `(1;2;3;4) * 2` is `2;4;6;8`. The `null` value is an empty
array and most expressions evaluate to this in event of an error. In particular,
all maths expressions involving a `null` will evaluate to `null`.

A-la Python, indentation represents block structuring. `let` statements name
constant values, everything else is largely about creating nodes to append to
the implicit *root* node of the tree. There are *no variables*. The language
is sort of pure-functional, if you imagine that the statements are monadic on
a monad encapsulating the current tree.

The simplest program would be something like:

```flitter
-- Hello world!

let SIZE=1280;720

!window #top size=SIZE
    !canvas size=SIZE antialias=true composite=:add
        !group font="Helvetica" font_size=100 color=sine(beat/2)
            !text point=SIZE/2 text="Hello world!"
```

This contains a comment, a `let` statement and a node creation statement.
Indented statements below this represent child nodes. Any name with a `!` in
front of it creates a node of that *kind*; the bindings following this specify
attributes of the node. Nodes can also be followed by one or more `#tag`s to
add tags that can be later searched for with *queries* (see language features
below).

When the **flitter** engine is run with this file, it will evaluate the code
repeatedly (at an attempted 60fps) and render this to screen. Note that one
explicitly specifies a window to be drawn into. The engine supports multiple
windows.

```sh
flitter examples/hello.fl
```

A `!canvas` node creates a 2D drawing canvas that follows an SVG-like drawing
model. `!group` sets styles and transforms. `!text` draws text, centred at
`point` by default. `sine()` is a function that reproduces a sine wave ranging
over [0..1] with the argument expressed in waves (turns/circumferens). `beat`
is a global representing the current floating point beat of the main clock. The
default BPM is 120, so `beat/2` is effectively a number of seconds since the
clock started. `color`s are 3- or 4-vectors (RGB and RGBA) in the range [0..1],
but the return value from `sine()` here is automagically extended out to the
3-vector resulting in a varying brightness of white.

`true` and `false` are synonyms for `1` and `0`. Truthfulness is represented by
any non-empty vector that contains something other than 0 or the empty string.
Names prefixed with a `:` are *symbols*, which internally are just string
literals (so `"add"` is the same as `:add`) but are easier to write and read
when specifying an identifier. When a string is required, such as the `text`
attribute of `!text` in the example above, each element of the vector will be
cast to a string if necessary and then concatenated together, e.g.,
`"Hello ";name;"!"`.

So the end result of this should be the text "Hello world!" pulsing white in
the middle of the window. You can edit and re-save the code while the engine is
running and it will reload the code on-the-fly - this is usually fast enough
not to be noticeable.

The engine-supplied global values are:

- `beat` - the current main clock beat (a monotonically-increasing floating
    point value)
- `quantum` - the beats per quantum (usually 4)
- `delta` - the difference between the current value of `beat` and the value
    at the last display frame
- `clock` - the time in seconds since the "start" of the main clock, this
    will adjust when the tempo or quantum is changed to keep the value of
    `beat` constant, so it's generally not a particularly useful value
- `fps` - the current target frame-rate of the engine (the `--fps` option)
- `performance` - a value in the range [0.5 .. 2.0] that increases fractionally
    if the engine has time to spare and decreases fractionally if it is missing
    the target frame rate; this value can be multiplied into variable loads
    in the code – e.g., number of things on screen – to maintain frame rate
- `realtime` - a `true`/`false` value indicating whether the engine is running
    in realtime mode (the default) or not (with the `--lockstep` option)

## Values

All values are vectors of either floating point numbers, Unicode strings, nodes
or functions (or a mix thereof). The vector implementation is optimised for
vectors of numbers, particularly short vectors. There are no dedicated integer
values in **flitter** and so one should be careful of relying on integer numbers
outside of the safe integer range of a double-precision floating point
(-2^53..2^53).

Mathematical operators operate only on pure number vectors. Using them on
anything else will return the empty vector (`null`). Unicode strings can be
concatenated by vector composition, e.g., `"Hello";" world!"` - the result will
be a 2-vector, but vectors are implicitly concatenated anywhere that strings
are used in the language. Numbers will be turned into strings using
general-purpose formatting and so `"Number ";1` is also a valid string.

Binary mathematical operators on mixed-length vectors will return a vector with
the same length as the longer of the two vectors. The shorter vector will be
repeated as necessary. This means that:

```flitter
(1;2;3;4;5;6;7;8;9) + 1       == (2;3;4;5;6;7;8;9;10)
(1;2;3;4;5;6;7;8;9) + (1;2;3) == (2;4;6;5;7;9;8;10;12)
```

Note that the vector composition operator `;` has a very low precedence and so
composed vectors will often have to be wrapped in parentheses when used with
operators:

```flitter
x;y+1 == x;(y+1)
(x;y)+1 == (x+1);(y+1)
```

Most built-in functions will do something sensible with an n-vector, e.g.,
`sin(0.1;0.2)` will return a vector equivalent to `sin(0.1);sin(0.2)`, but
will be substantially faster for long vectors. Some functions operate with both
multiple arguments *and* n-vectors, e.g.:

```flitter
hypot(3, 4) == 5
hypot(3;4) == 5
hypot(3;4, 4;3) == (5;5)
```

As functions are themselves first-order objects in the language, they may be
composed into vectors. The language takes the orthogonal approach that calling
a function vector is identical to composing the result of calling each function
with the arguments, i.e., `(sin;cos)(x) == (sin(x);cos(x))`. This is arguably
obtuse behaviour.

### Symbols

Names can be turned into short Unicode strings with a preceding `:` character.
E.g., `:foo` is equivalent to `"foo"`. These are particularly useful for
short strings that are used as enumerations, e.g.:

```flitter
!window
    !canvas composite=:add
        …
```

The `composite` attribute of `!canvas` takes a string representing the name of
the blend function to use when drawing. They are also useful when constructing
state key vectors or seed vectors (see **State** and **Pseudo-random sources** below)

## Sequences

The result of evaluating a sequence of expressions - such as might be found in
the body of a loop, conditional or function - is the vector composition of the
result of each expression.

## Ranges

```flitter
start..stop|step
```

A range creates a vector beginning at `start` and incrementing (or
decrementing if negative) by `step` until the value is equal to or passes
`stop` - the last value is *not* included in the range, i.e., it is a
half-open range.

`start` and `|stop` may be omitted, in which case the vector will begin at 0
and increment by 1. Therefore:

- `..10` evaluates to the vector `0;1;2;3;4;5;6;7;8;9`
- `1..21|5` evaluates to the vector `1;6;11;16`
- `1..0|-0.1` evaluates to the vector `1;0.9;0.8;0.7;0.6;0.5;0.4;0.3;0.2;0.1`

Ranges are *not* lazy like they are in Python, so `0..1000000` will create a
vector with 1 million items.

## Nodes

The purpose of any **flitter** program is to construct a render tree.
Individual literal nodes are represented by an exclamation mark followed by
a name, e.g., `!window`. Nodes can be tagged with additional arbitrary names
that aid in readability and in searching (see Queries below). A hash character
followed by a name will tag the preceding node (or nodes) with that name, e.g.,
`#top`. A name followed by an equals character will set that attribute on the
preceding node. The resulting value in both of these instances is the
tagged/attributed node.

For example:

```flitter
!window #top size=1920;1080 vsync=true title="Hello world!"
```

constructs a literal `!window` node, adds the `#top` tag to it, then sets the
`size`, `vsync` and `title` attributes. Note that everything here is an
expression/operator and it is equivalent to:

```flitter
(((((!window) #top) size= 1920;1080) vsync= true) title= "Hello world!")
```

`!window`, `1920;1080`, `true` and `"Hello world!"` are literal vector values,
`#top` is a unary postfix operator, and `size=`, `vsync=` and `title=` are
binary operators.

Setting an attribute to `null` will *remove* that attribute from the node if it
is already present, or do nothing otherwise.

Block indentation below a node is a binary operation that evaluates the
indented expressions as a sequence and then appends each node in the resulting
vector to the node (or nodes) above.

For example:

```flitter
!window #top size=1280;720 vsync=true title="Hello world!"
    !canvas
        !rect point=0;0 size=1280;720
        !fill color=1;0;0
```

constructs the `!rect` and `!fill` nodes and composes them into a vector, then
constructs a `!canvas` node and appends these to it. The resulting tree is
appended to a `!window` node, which is the final value of this expression.
Running this program as-is will result in a red window with the title "Hello
world!".

### Attribute name scoping

Existing attributes of a node are brought into scope as names when evaluating
the value of an attribute-set expression, e.g.:

```flitter
!rect point=200;100 size=(500;500)-point
```

is equivalent to:

```flitter
!rect point=200;100 size=300;400
```

Attribute names introduced in this way are added into a special "0-th" scope
outside of built-ins and engine-supplied names. Thus, if an attribute name
matches a name already in scope, then that existing name "wins". So:

```flitter
let point=400;200

!rect point=200;100 size=(500;500)-point
```

results in a 100x300 rectangle.

This rule is important for allowing the language partial-evaluator to determine
bindings. As nodes are themselves values, the above can be legally written as:

```flitter
let point=400;200
    rectangle=(!rect point=200;100)

rectangle size=(500;500)-point
```

Without the special rule for attribute name binding, the partial-evaluator
would not (in the general case) be able to statically determine whether the use
of `point` in the `size=` attribute refers to the name introduced with the `let`
or an attribute of the `rectangle` value.

### Vector node operations

As nodes are values, and thus vectors, tag unary-postfix operations and
attribute-set binary operations are able to operate on a vector of nodes
simultaneously. For example:

```flitter
let points=10;20;15;25;20;30;25;35

!path
    !move_to point=0;0
    ((!line_to point=x;y) for x;y in points) #segment
```

is a strange, but perfectly legal, way of doing the equivalent (and more
readable):

```flitter
let points=10;20;15;25;20;30;25;35

!path
    !move_to point=0;0
    for x;y in points:
        !line_to #segment point=x;y
```

When evaluating attribute-set operations on a vector of nodes, the attribute
value is evaluated for each node, bringing that node's attribute names into
scope on each iteration. For example:

```flitter
let nodes=((!foo a=a) for a in ..10)

nodes b=a*2
```

is equivalent to:

```flitter
!foo a=0 b=0
!foo a=1 b=2
!foo a=2 b=4
!foo a=3 b=6
!foo a=4 b=8
!foo a=5 b=10
!foo a=6 b=12
!foo a=7 b=14
!foo a=8 b=16
!foo a=9 b=18
```

Generally, this sort of behaviour is obtuse, but the semantics are important
when using queries (see **Queries** below).

## Let expressions

Values may be bound to names with the `let` keyword. It is followed by one or
more `name=expression`s. The expressions are immediately evaluated and the
resulting values are added into the scope of the expressions below.

Lets may be used at the top-level in a **flitter** script or anywhere within
a block-structured sequence, i.e., within append, function, conditional and
loop bodies. Each of these sequences represents a new let scope and names
that are re-bound will hide the same name in an outer scope. The outer scope
bindings will be in place while evaluating the values of inner scope lets.

For example:

```flitter
let x=10

if x > 5
    let x=x*2
    !foo x=x

!bar x=x
```

will evaluate to the two top-level nodes:

```flitter
!foo x=20
!bar x=10
```

There is also an inline version of let known as `where`. This allows names to
be bound within a non-sequence expression, e.g.:

```flitter
!foo x=(x*x where x=10)
```

Note that `where` has higher precedence than `;` vector composition and so
`x;x*x where x=10` is equivalent to `x;(x*x where x=10)` and thus the binding
is only in scope for the `x*x` expression.

A `let` expression may bind multiple names at once, which apply immediately
in the order they are given, and supports using indentation for continuance.
For example:

```flitter
let x=1 y=2
    z=x*y
```

A let binding may also specify multiple names separated with a semicolon to
do an unpacked vector binding. For example:

```flitter
let x;y=SIZE/2
```

This will pick off the first two items from the vector result of evaluating the
expression and bind them in order to the names `x` and `y`. If the vector is
longer than the number of names given then additional items are ignored. If the
vector is shorter then the unmatched names will be bound to `null`.

Names introduced with a `let` can redefine engine-supplied values, like `beat`,
and built-ins, like `sin`.

## Conditionals

```flitter
if test
    expression
    …
《elif test
    expression
    …》
《else
    expression
    …》
```

*test* is any expression and it will be considered *true* if it evaluates to a
non-empty vector containing something other than 0s or empty strings. So `0` is
false, as is `null`, `""` and `0;0;0`. The result of evaluating the matching
indented expressions is the result value of the `if`. In the absence of an
`else` clause the result of an `if`/`elif` with no true tests is `null`.

There is an in-line expression version of `if` that borrows its syntax from
Python:

```flitter
!fill color=(1 if x>10 else 0)
```

If the `else` is omitted then the expression will evaluate to `null` if the
condition is not true. Importantly, this means that using a bare `if` in an
attribute setting operation will result in the attribute being *unset* if the
condition is false, i.e., in:

```flitter
!window size=100;100
    !canvas
        !group color=1;0;0
            for x in ..10
                !path
                    !rect point=x*10+2.5;0 size=5;100
                    !fill color=((0;1;0) if x >= 5)
```

the `color` attribute will *not* be set on the second five `!fill` nodes and so
they will inherit the color from the enclosing `!group`. Thus, this will draw
five thick green lines followed by five red.

In-line conditionals do not have an `elif` equivalent, group multiple
conditional expressions as necessary to achieve this, e.g.:

```flitter
!foo x=(x if x < 10 else (x*2 if x < 20 else x*3))
```

## Loops

```flitter
for name《;name…》 in expression
    expression
    …
```

For loops iterate over vectors binding the values to the name(s). The result of
evaluating the expressions within the loop body are concatenated into a single
vector that represents the result value of the loop. Normally this would be a
vector of nodes to be appended to some enclosing node.

When multiple names are given, each iteration will take multiple values from the
source vector. If there are not enough values left in the source vector to bind
all names in the last iteration, then the names lacking values will be bound to
`null`.

Iterating with multiple names is particularly useful combined with the `zip()`
function, which merges multiple vectors together:

```flitter
!group #clockface
    let theta=(..12)/12
    for x;y in zip(cos(theta), sin(theta)) * 100
        !ellipse point=x;y radius=10
    !fill color=1
```

Note that `cos()` and `sin()` here are taking a 12-vector and returning another
12-vector, `zip()` combines these into a 24-vector and the multiplication by
100 is applied to every element of this. Also of note here is that the flitter
versions of `cos()` and `sin()` take values in *turns* not in radians. The
dozen `!ellipse` nodes created by the loop are combined into a 12-vector, this
has the final `!fill` node concatenated with it and then all of these nodes are
appended to the `!group`.

There is actually a more convenient `polar(theta)` function that does the same
thing as `zip(cos(theta), sin(theta))`. Arguably, it would be even neater to
implement a clock face using `!transform rotate=` instead.

Again, loops may also be used in-line in non-sequence expressions with syntax
borrowed from Python:

```flitter
!line points=((x;x*5) for x in ..5)
```

This will evaluate to:

```flitter
!line points=0;0;1;5;2;10;3;15;4;20
```

## Functions

```flitter
func name(parameter《=default》《, parameter…》)
    expression
    …
```

`func` will create a new function and bind it to `name`. Default values may be
given for the parameters and will be used if the function is later called with
an insufficient number of matching arguments, otherwise any parameters lacking
matching arguments will be bound to `null`. The result of evaluating all
body expressions will be returned as a single vector to the caller.

Functions may refer to names bound in the enclosing scope(s) to the definition.
These will be captured at definition time. Thus rebinding a name later in
the same scope will be ignored: E.g.:

```flitter
let x=10

func add_x(y)
    x+y

let x=20

!foo z=add_x(5)
```

will evaluate to `!foo z=15` *not* `!foo z=25`.

A function definition is itself an implicit `let` that binds the function
definition to the function name in the definition scope. Functions are values
in the **flitter** language and may be manipulated as such.

## Template function calls

This is something of a hack, but the special `@` operator allows calling a
function using similar syntax to constructing a node. The name following `@`
should be the name of the function to be called, any "attributes" placed after
this are passed as arguments to the respectively-named function parameters. Any
indented expressions are evaluated and the resulting vector passed as the first
argument to the function. Function parameters that are not bound by a
pseudo-attribute will have their default value if one was specified in the
function definition or `null` otherwise.

For example:

```flitter
func shrink(nodes, percent=0)
    !transform scale=1-percent/100
        nodes

!window size=1280;720
    !canvas translate=640;360
        @shrink
            !path
                !ellipse radius=100
                !fill color=0;1;0
        @shrink percent=25
            !path
                !ellipse radius=100
                !fill color=1;0;0
```

This (rather pointless) example draws a 100px radius circle in green and then
draws another circle 25% smaller in red on top. Both `!path` nodes will be
wrapped with `!transform` nodes, the first with `scale=1` and the second with
`scale=0.75`.

## Queries

Queries allow the node tree so-far to be searched and manipulated. They use
a CSS-selector-like syntax that is best explained by example:

- `{*}` matches *all* nodes in the tree
- `{window}` matches any `!window` node
- `{#spot}` matches any node with the `#spot` tag
- `{shader#blur}` matches `!shader` nodes with the `#blur` tag
- `{ellipse|rect}` matches any `!ellipse` or `!rect` node
- `{group path}` matches `!path` nodes anywhere within a `!group` node
- `{path>line_to}` matches `!line_to` nodes *immediately* below a `!path` node
- `{group.}` returns all `!group` nodes reachable from the root, but *not* any
further `!group` nodes *within* those, i.e., it stops recursive descent on a
match

The result of a query expression is a vector of matching nodes. Note that nodes
are appended to the main tree from the top-level after the complete evaluation
of each expression, including any indented append operations. Thus a query
evaluated within a nested expression cannot match any node that makes up part
of that expression.

A query expression may be combined with tag, attribute-set or node-append
expressions to amend the current tree. For example:

```flitter
let SIZE=1280;720

!window size=SIZE
    !canvas size=SIZE color=1 font_size=100
        !text point=SIZE/2 text="Hello world!"

if beat > 10
    {canvas} color=1;0;0
```

In this example, the text will turn red after 10 beats (5 seconds by default).

Appending nodes to a query will append those nodes to all matches:

```flitter
…

if beat > 10
    {canvas}
        !text point=SIZE*(0.5;0.75) text="(RED)" color=1;0;0
```

Appending queries to a node will re-parent the matching nodes into this new
position in the tree. For example, the following code combines two queries
to wrap all nodes within the canvas in a new group that turns them upside down
and changes the default color to red:

```flitter
…

if beat > 10
    {canvas}
        !group color=1;0;0 translate=SIZE rotate=0.5
            {canvas>*}
```

As discussed in **Nodes** above, attributes from the matching nodes are brought
into scope as names when evaluating attribute-set operations. So a queries can
be used to amend an existing attribute value, e.g.:

```flitter
…

{window} size=size*2
```

will double the `size` attribute of the window, as long as no other binding
for `size` exists in the current scope.

## Pseudo-random sources

**flitter** provides three useful sources of pseudo-randomness: `uniform()`,
`normal()` and `beta()`. These built-in functions return special "infinite"
vectors that may only be indexed. These infinite vectors provide a reproducible
stream of numbers matching the *Uniform(0,1)*, *Normal(0,1)* and *Beta(2,2)*
distributions.

The single argument to both functions is a vector that acts as the
pseudo-random seed. Floating-point numbers within this seed vector are truncated
to whole numbers before the seed is calculated. This is deliberate to allow new
seeds to be generated at intervals, e.g.: `uniform(:foo;beat)` will create a new
stream of pseudo-random numbers for each beat of the main clock. Multiplying or
dividing this seed value then allows for different intervals, e.g., four times a
beat: `uniform(:foo;beat*4)`.

Similarly, the index value to the infinite-vector is truncated to pick a
specific number in the stream. For example:

```flitter
let SIZE=1280;720

!window size=SIZE
    !canvas
        for i in ..10
            let x=uniform(:x;i)[beat] y=uniform(:y;i)[beat]
                r=2*beta(:r;i)[beat] h=uniform(:h;i)[beat]
            !path
                !ellipse point=(x;y)*SIZE radius=r*50
                !fill color=hsv(h;1;1)
```

This will create 10 circles distributed uniformly around the window with
different radii clustered around 50 pixels and different uniformly picked hues.
Every beat, the circles will be drawn in different places and with different
sizes and hues. Note the use of symbols (`:x`, `:y`, etc.) combined with the
index of the circle to create unique seeds. This code will draw exactly the
same sequence of circles *every* time it is run as the pseudo-random functions
are stable on their seed argument. There is no mechanism for generating true
random numbers in **flitter**.

The pseudo-random streams can be indexed with a range vector to generate a
vector of numbers, e.g.: `uniform(:some_seed)[..100]` will generate a 100-vector
of uniformly distributed numbers. The streams are arbitrarily long and are
unit-cost to call, so indexing the billionth number takes the same amount of
time as the 0th. Unlike normal vectors, the streams also extend into negative
indices (which actually wrap around to the end of the 64-bit unsigned integer
index range).

Pseudo-random streams may be bound to a name list in a `let` expression to
pick off the first few values, e.g.:

```flitter
let x;y;z = uniform(:position)
```

They are considered to be `true` in conditional expressions (`if`, `and`, etc.).
In all other aspects, pseudo-random streams behave like the `null` vector, e.g.,
attempts to use them in mathematical expressions will evaluate to `null`.

## Counters

Arbitrary counters can be managed with the `counter()` function. This takes
either two or three arguments and **has state side-effects**.

In the three-argument form, `counter(counter_id, clock, speed)`, the function
will create or update a counter with a specific `clock` value and `speed`,
storing the current state of the counter in the state mapping with the given
`counter_id` key. The function returns the current count.

Counters begin at zero and increment upwards by `speed` every unit increase of
`clock`. The counter speed can be changed at any point and the counter will
return the count at the previous speed before switching to counting with the
new speed. The stored state only changes when the speed changes. `clock`
*should* be a monotonically increasing number, such as `beat`. `speed` is
allowed to be zero to stop the counter or negative to count downwards.

Either `clock` or `speed` can be given as an n-vector to create a
multi-dimensional counter. This can be useful, for instance, to move an object
through space with a velocity vector:

```flitter
let velocity = 0.2;-1.5;3.0
    position = counter(:thing_position, beat, velocity)
```

The two argument form of the function omits the speed and returns the count
matching the value of `clock` at the last counter speed. If the counter has not
already been initialised, then the speed will default to 1.

## State

Any interactive component of **flitter**, such as a MIDI controller,
communicates with a running **flitter** program via a *state* mapping. This can
be queried with the `$` operator like so:

```flitter
let SIZE=1280;720

!window size=SIZE
    !canvas size=SIZE translate=SIZE/2
        !ellipse radius=$:circle_radius
        !fill color=1;0;0

!controller driver=:xtouch_mini
    !rotary id=1 state=:circle_radius lower=0 upper=300 initial=100
```

This shows an X-Touch mini MIDI surface being configured with one encoder as a
knob that changes the radius of a red filled-circle drawn in the middle of the
window. The `state=:circ_radius` attribute of the `!rotary` node specifies the
state key to write values to and the current value is retrieved with
`$:circle_radius`.

A key is any vector of numbers and/or strings (/symbols). As the `;` operator
binds with very low precedence, a non-singleton key needs to be surrounded with
brackets, e.g., `$(:circle;:radius)`.

The state system is also used by the [physics engine](./physics.md) to
communicate particle properties back to the program and by counters (see above)
for storing a starting clock value and current speed. Be careful when choosing
state keys to avoid collisions between these different uses.

## Pragmas

There are three supported pragmas that may be placed at the top-level of a
source file:

```flitter
%pragma tempo 110
%pragma quantum 3
%pragma fps 30
```

These respectively set the initial tempo and/or quantum of the main clock (the
defaults are 120 and 4, respectively), and the *current* target frame rate of
the engine (default is 60 or the value specified with the `--fps` command-line
option).

## Partial-evaluation

When a source file is loaded it is parsed into an abstract syntax tree and then
that is *partially-evaluated*. This attempts to evaluate all static expressions.
The partial-evaluator is quite sophisticated and is able to construct static
parts of node trees, unroll loops with constant source vectors, evaluate
conditionals with constant tests, call functions with constant arguments
(including creating pseudo-random streams), replace `let` names with literal
values, evaluate mathematical expressions (including some rearranging where
necessary to achieve this) and generally reduces as much of the evaluation tree
as possible to constant values.

Unbound names (which includes all of the globals listed above, like `beat`) and
queries (`{...}`) are always dynamic, as will then obviously be any expressions
that include these.

After partial-evaluation has simplified the tree (and note that thanks to loop
unrolling, "simpler" doesn't necessarily mean "smaller"), the tree is compiled
into instructions for a stack-based virtual machine. These instructions are
interpreted to run the program.

The partial-evaluator and compiler can run again incorporating the state if
that has been stable for a period of time (configurable with the `--evalstate`
command-line option). If the state then changes (i.e., a pad or encoder is
touched) the engine will return to the original compiled program. Counters
update the state whenever their speed changes, and so doing this continously
will defeat state-based partial-evaluation. The physics engine updates state on
every iteration and so does the same.

### A note on multi-processing

I love Python, but the global interpreter lock basically makes any kind of
serious multi-threading impossible. **flitter** supports a limited form of
multi-processing instead. Adding the `--multiprocess` option on the command line
will cause a separate renderer child process to be executed for each window,
laser or DMX bus.

The main process handles evaluating the script to produce an output tree. This
tree is then fed to each renderer and all renderers are waited on before moving
on to the next iteration. This works well if the script and the renderers are
expensive to run and the tree is small. As the tree grows large, the cost of
pickling and un-pickling it across the process boundaries becomes a bottleneck,
so your mileage may vary.
