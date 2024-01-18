
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
- `clock` - the current frame time (derived from Python's `perf_counter()`
    usually, though increasing by exactly `1/fps` per frame in non-realtime
    mode)
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

### SI Prefixes

**flitter** supports adding an SI prefix to the end of a number. This is
confusing terminology, but an SI prefix is a prefix to a unit suffix.
**flitter** does *not* support units, so you just end up with the SI prefix
as a suffix. (Confused yet?)

The allowed SI prefixes are:

- `T` - x 10e12
- `G` - x 10e9
- `M` - x 10e6
- `k` - x 10e3
- `m` - x 10e-3
- `u` or `µ` - x 10e-6
- `n` - x 10e-9
- `p` - x 10e-12

So you can suffix any number with one of these, e.g., `10m` is the same as
`0.01`. They are primarily useful for avoiding difficult-to-read long sequences
of zeros – e.g. when specifying the brightness of point and spot lights:

```flitter
!light position=0 color=1M
```

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
state key vectors or seed vectors (see [State](#state) and [Pseudo-random
sources](#pseudo-random-sources) below)

## Operators

**flitter** supports the usual range of operators, with a lean towards the
syntax and semantics of Python. The binary mathematical operators are:

- *x* `+` *y* - addition
- *x* `-` *y* - subtraction
- *x* `/` *y* - division
- *x* `*` *y* - multiplication
- *x* `//` *y* - floor division (i.e, divide and round down)
- *x* `%` *y* - modulo (pairs with floor division)
- *x* `**` *y* - raise *x* to the power *y*

All of these operations return a vector with same length vector as the longer
of *x* and *y*. The shorter vector is repeated as necessary. Note that, as well
as this meaning `(1;2;3;4) + 1` is equal to `2;3;4;5`, this also means that
`(1;2;3;4) + (1;-1)` is equal to `2;1;4;3`.

The unary mathematical operators are:

- `-` *x* - negate
- `+` *x* - identity

All of the mathematical operators return `null` if either *x* or *y* is an
empty or non-numeric vector.

Logical operators:

- *x* `or` *y* - shortcut or: returns *x* if it evaluates as true or *y*
  otherwise
- *x* `and` *y* - shortcut and: returns *x* if it evaluates as false or
  *y* otherwise
- *x* `xor` *y* - exclusive or: returns *y* if *x* evaluates as false,
  *x* if *y* evaluates as false, otherwise returns `false`
- `not` *x* - logical inverse: returns `false` if *x* evaluates as true,
  `true` otherwise

## Function calling

Functions are called in the normal way, with the name of the function followed
by zero or more arguments, separated by `,`, within `()` brackets, e.g.,
`cos(x)` or `zip(xs, ys)`.

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

### Built-in functions

- `abs(x)` - return absolute value of *x* (ignoring sign)
- `accumulate(xs)` - return a vector with the same length as *xs* formed by
adding each value to an accumulator starting at 0
- `acos(x)` - arc-cosine (in *turns* not radians) of *x*
- `angle(x, y)` - return the angle (in *turns*) of the cartesian
vector *x,y*
- `asin(x)` - arc-sine (in *turns*) of *x*
- `beta(seed)` - see [Psuedo-random sources](#pseudo-random-sources) below
- `bounce(x)` - return a repeating bouncing wave (akin to a perfectly bouncing
ball) in the range *[0,1]* with one wave per unit of *x*, with the *0* point
when `x%1 == 0` and the *1* point when `x%1 == 0.5`
- `ceil(x)` - return mathematical ceiling of *x*
- `colortemp(t)` - return a 3-vector of *R*, *G* and *B* **linear sRGB** values
for an approximation of a Planckian (blackbody) radiator at temperature *t*
scaled so that `colortemp(6503.5)` (the sRGB whitepoint correlated colour
temperature) is close to `1;1;1`; the approximation only holds within the
range *[1667,25000]* and, strictly speaking, values below 1900°K are outside
the sRGB gamut
- `cos(x)` - return cosine of *x* (in *turns*)
- `counter(...)` - see [Counters](#counters) below
- `csv(filename, row)` - return a vector of values obtained by reading a
specific *row* (indexed from *0*) from the CSV file with the given *filename*;
this function intelligently caches and will convert numeric-looking columns in
the row into numeric values
- `exp(x)` - return *e* raised to the power of *x*
- `floor(x)` - return mathematical floor of *x*
- `hsl(h;s;l)` - return a 3-vector of *R*, *G* and *B* in the range *[0,1]*
from a 3-vector of hue, saturation and lightness (also in the range *[0,1]*)
- `hsv(h;s;v)` - return a 3-vector of *R*, *G* and *B* in the range *[0,1]*
from a 3-vector of hue, saturation and value (also in the range *[0,1]*)
- `hypot(x, [...])` - return the square root of the sum of the square of each
value in `x` with one argument, with multiple arguments return a vector formed
by calculating the same for the 1st, 2nd, etc., element of each of the
argument vectors
- `impulse(x, [y=0.25])` - return a repeating impulse wave in the range *[0,1]*
with one wave per unit of *x*, with the *0* point when `x%1 == 0` and the *1*
point when `x%1 == y`
- `len(xs)` - return the length of vector *xs*
- `linear(x)` - a linear "easing" function in the range *[0, 1]* with values of
*x* less than *0* returning *0* and values greater than *1* returning *1*
- `log(x)` - return natural log of *x*
- `log2(x)` - return log 2 of *x*
- `log10(x)` - return log 10 of *x*
- `map(x, y, z)` - maps a value of *x* in the range *[0,1]* into the range
*[y,z]*; equivalent to `y*x + (1-y)*z` (including in n-vector semantics)
- `max(x, [...])` - return the maximum value in the vector *x* with one
argument, or the largest of the arguments in vector sort order
- `maxindex(x, [...])` - return the index of the maximum value in the vector
*x* with one argument, or the index of the largest argument in vector sort
order (with the 1st argument being index *0*)
- `min(x, [...])` - return the minimum value in the vector *x* with one
argument, or the smallest of the arguments in vector sort order
- `minindex(x, [...])` - return the index of the minimum value in the vector
*x* with one argument, or the index of the smallest argument in vector sort
order (with the 1st argument being index *0*)
- `noise(...)` - see [Noise functions](#noise-functions) below
- `normal(seed)` - see [Psuedo-random sources](#pseudo-random-sources) below
- `normalize(x)` - return `x / hypot(x)`
- `octnoise(...)` - see [Noise functions](#noise-functions) below
- `polar(th)` - equivalent to `zip(cos(th), sin(th))`
- `quad(x)` - a quadratic "easing" function in the range *[0, 1]* with values of
*x* less than *0* returning *0* and values greater than *1* returning *1*
- `read(filename)` - returns a single string value containing the entire text
of *filename* (this function intelligently caches)
- `round(x)` - return mathematical round-towards-zero of *x*, with 0.5 rounding
up
- `sawtooth(x)` - return a repeating sawtooth wave in the range *[0,1)* with one
wave per unit of *x*, with the *0* point at `x%1 == 0` and linearly rising
towards 1
- `sharkfin(x)` - return a repeating sharkfin wave in the range *[0,1]* with
one wave per unit of *x*, with the *0* point when `x%1 == 0` and the *1* point
when `x%1 == 0.5`
- `shuffle(source, xs)` - return the shuffled elements of *xs* using the
psuedo-random *source* (which should be the result of calling `uniform(...)`)
- `sin(x)` - return sine of *x* (in *turns*)
- `sine(x)` - return a repeating sine wave in the range *[0,1]* with one wave
per unit of *x*, with the *0* point when `x%1 == 0` and the *1* point when
`x%1 == 0.5`
- `snap(x)` - a square-root "easing" function in the range *[0, 1]* with values
of *x* less than *0* returning *0* and values greater than *1* returning *1*
(conceptually a quadratic easing function with *x* and *y* axes flipped)
- `split(text)` - return a vector formed by splitting the string *text* at
newlines (not included)
- `sqrt(x)` - return the square root of *x*
- `square(x)` - equivalent to `x * x`
- `sum(xs)` - return a single numeric value obtained by summing each element of
vector *xs*
- `triangle(x)` - return a repeating triangle wave in the range *[0,1]* with
one wave per unit of *x*, with the *0* point when `x%1 == 0` and the *1* point
when `x%1 == 0.5`
- `uniform(seed)` - see [Psuedo-random sources](#pseudo-random-sources) below
- `zip(xs, [...])` - return a vector formed by interleaving values from each
argument vector; for *m* arguments the resulting vector will be *n * m*
elements long, where *n* is the length of the longest vector; short arguments
will repeat, so `zip(1;2;3;4, 0) == (1;0;2;0;3;0;4;0)`

## Sequences

The result of evaluating an indented sequence of expressions separated by
line breaks - such as might be found in the body of a loop, conditional or
function - is the vector composition of the result of each expression.

## Ranges

```flitter
start..stop|step
```

A range creates a vector beginning at `start` and incrementing (or
decrementing if negative) by `step` until the value is equal to or passes
`stop` - the last value is *not* included in the range, i.e., it is a
half-open range.

`start` and `|step` may be omitted, in which case the vector will begin at 0
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
that aid in readability and in searching (see [Queries](#queries) below). A
hash character followed by a name will tag the preceding node (or nodes) with
that name, e.g., `#top`. A name followed by an equals character will set that
attribute on the preceding node. The resulting value in both of these instances
is the tagged/attributed node.

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
when using queries (see [Queries](#queries) below).

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
evaluating the expressions within the loop body are composed into a single
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
has the final `!fill` node composed with it and then all of these nodes are
appended to the `!group` node.

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

## User-defined functions

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
the same scope will be ignored, e.g.:

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

Function calls may include out-of-order named arguments, e.g.:

```flitter
func multiply_add(x, y=1, z)
    x*y + z

!foo w=multiply_add(2, z=3)
```

will bind the arguments to parameters with `x` taking the value `2`, `y` taking
the value `1` and `z` taking the value `3`. Named arguments should be given
*after* any positional arguments and should not repeat positional arguments.

## Template function calls

The special `@` operator provides syntactic sugar for calling a function using
syntax similar to constructing a node. The name following `@` should be the
name of the function to be called, any "attributes" placed after this are
passed as named arguments. Any indented expressions are evaluated and the
resulting vector passed as the *first* argument to the function.

As normal, function parameters that are not bound will take their default value
if one is given in the function definition or `null` otherwise.

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
`scale=0.75`. The `!transform` nodes returned by the two template function
calls are appended to the `!canvas` node.

## Queries

Queries allow the node tree so-far to be searched and manipulated. They use
a CSS-selector-like syntax that is best explained by example:

- `{*}` matches *all* nodes in the tree
- `{window}` matches all `!window` nodes
- `{window!}` matches only the *first* `!window` node in the tree
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
of that expression. Query searches are *depth-first*.

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

## Noise functions

Often, more useful than a random source is a noise function. These produce
smoothly changing output values across one or more input dimensions. **flitter**
contains an implementation of [OpenSimplex 2S](https://github.com/KdotJPG/OpenSimplex2)
noise in 1, 2 and 3 dimensions.

The basic noise function is:

`noise(` *seed* `,` *x* *[* `,` *y* *[* `,` *z* *] ] ]* `)`

where:

- *seed* is a seed value, as per [Psuedo-random sources](#pseudo-random-sources)
above
- *x* is the first dimension
- *y* is an optional second dimension
- *z* is an optional third dimension

The function returns a value in the range *(-1,1)*. The function can be thought
of as creating a wiggly line in 1 dimension, a landscape with hills and dips in
2 dimensions or a volume of space filled with clouds in 3 dimensions.

A slice through 3D noise will look like the output of a 2D noise function and,
similarly, a slice through 2D noise will look like the output of a 1D noise
function. Often a useful thing to do is to use a value derived from the beat
counter as one of the inputs, yielding a 1D or 2D noise function that will
smoothly change over time for the same input space.

The function is entirely deterministic - always producing the same output
for the same inputs. The *seed* value can be used to create multiple
independent noise sources.

### Multi-value vector inputs

If one of the *x*, *y* or *z* arguments is a vector longer than 1, then the
function will return a multi-value output. The return value is equivalent to
the code:

```flitter
((noise(seed, ix, iy, iz) for iz in z) for iy in y) for ix in x
```

However, calling `noise()` with an *n*-vector will be significantly faster than
*n* separate calls.

### Multi-octave noise

It is often useful to layer a number of noise functions on top of each other
with different scales and weights to produce a more complex surface - this is
particularly useful when attempting to produce organic looking results.

For example:

```flitter
let n = 4
    scale = 1;2;4;8
    weight = 1;0.5;0.25;0.125
    total = sum(weight)
    z = (noise(:seed;i, x*scale[i], y*scale[i])*weight[i] for i in ..n) / total
```

Here the scale of the inputs doubles with each iteration and the weight halves.

**flitter** provides a function that will do this calculation significantly faster
than the equivalent code:

`octnoise(` *seed* `,` *n* `,` *k* `,` *x* *[* `,` *y* *[* `,` *z* *] ] ]* `)`

where:

- *seed* is a seed value
- *n* is the number of octaves
- *k* is a weight constant
- *x* is the first dimension
- *y* is an optional second dimension
- *z* is an optional third dimension

The individual weight for each iteration is computed as *k<sup>-i</sup>* and the
scaling factor for the inputs as *2<sup>i</sup>*. A unique seed value for each
iteration is derived from *seed*.

So the equivalent call to the code above would be:

```flitter
let z = octnoise(:seed, 4, 0.5, x, y)
```

Again, this function will accept *n*-vectors as inputs.

## Counters

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
communicate particle properties back to the program and by counters (see
[Counters](#counters) above) for storing a starting clock value and current
speed. Be careful when choosing state keys to avoid collisions between these
different uses.

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
(including creating pseudo-random streams), inline functions that contain only
local names, replace `let` names with literal values, evaluate mathematical
expressions (including some rearranging where necessary to achieve this) and
generally reduces as much of the evaluation tree as possible to constant values.

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
