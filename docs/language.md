
# The Language

## Quick introduction

**Flitter** is a declarative tree-construction language. All values are vectors,
delimited with semicolons, and all maths is element-wise. Short vectors are
repeated as necessary in binary operations, i.e., `(1;2;3;4) * 2` is `2;4;6;8`.
The `null` value is an empty vector and most expressions evaluate to this in
the event of an error. In particular, all maths expressions involving a `null`
will evaluate to `null`.

A-la Python (or Haskell), indentation represents block structuring. `let`
expressions name constant values, everything else is largely about creating
nodes to append to the implicit *root* node of the tree. There are *no
variables* – the language is pure-functional.

The simplest program would be something like:

```flitter
-- Hello world!

let SIZE=1280;720

!window #top size=SIZE
    !canvas size=SIZE antialias=true composite=:add
        !group font="Helvetica" font_size=100 color=sine(beat/2)
            !text point=SIZE/2 text="Hello world!"
```

This contains a comment, a `let` expression and a node creation expression.
Indented expressions below this represent child nodes. Any name with a `!` in
front of it creates a node of that *kind*; the bindings following this specify
attributes of the node. Nodes can also be followed by one or more `#tag`s to
add tags for readability and logging.

When the **Flitter** engine is run with this file, it will evaluate the code
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
is a global representing the current floating point beat of the beat counter.
The default BPM is 120, so `beat/2` is effectively a number of seconds since
the counter started. `color`s are 3- or 4-vectors (RGB and RGBA) in the range
[0..1], but the return value from `sine()` here is automagically extended out
to a 3-vector resulting in a varying brightness of white.

`true` and `false` are synonyms for `1` and `0`. Truthfulness is represented by
any non-empty vector that contains something other than 0 or the empty string.
Names prefixed with a `:` are *symbols*, which should be considered as opaque
values (see [Symbols](#symbols) below for details). When a string is required,
such as the `text` attribute of `!text` in the example above or the `composite`
attribute of `!canvas`, each element of the vector will be converted to a
string as necessary and then these concatenated together, e.g.,
`"Hello ";name;"!"`. Symbols return to being strings (without the leading `:`)
in this conversion.

So the end result of this should be the text "Hello world!" pulsing white in
the middle of the window. You can edit and re-save the code while the engine is
running and it will reload the code on-the-fly - this is usually fast enough
not to be noticeable.

The engine-supplied global values are:

- `beat` - the current beat counter beat (a monotonically-increasing floating
    point value)
- `quantum` - the beats per quantum (usually 4)
- `delta` - the difference between the current value of `beat` and the value
    at the last display frame
- `time` - the time that the current frame began execution (either the value
    of the high-resolution Python `perf_counter()` timer or a counter beginning
    at `0` and incrementing by exactly `1/fps` on each frame in non-realtime
    mode)
- `clock` - the current UTC time as a seconds-since-UNIX-epoch value
- `frame` - the current frame number, counting from 0 and increasing by one
    for each rendered frame (this will increase by exactly 1 on each program
    execution when running in non-realtime mode)
- `fps` - the current target frame-rate of the engine (the `--fps` option)
- `performance` - a value in the range [0.5 .. 2.0] that increases fractionally
    if the engine has time to spare and decreases fractionally if it is missing
    the target frame rate; this value can be multiplied into variable loads
    in the code – e.g., number of things on screen – to maintain frame rate
- `realtime` - a `true`/`false` value indicating whether the engine is running
    in realtime mode (the default) or not (with the `--lockstep` option)
- `run_time` - the number of seconds that the engine will run for before
    exiting, controlled with the `--runtime` options.

## Unicode Support

All **Flitter** source files must be UTF-8 encoded. All names, including
[symbols](#symbols) and [node](#nodes) kinds and attributes, must begin with a
Unicode alphabetic character or an underscore and then may contain any number
and combination of underscores and Unicode alphanumeric characters, suffixed by
zero or more single quote [primes](https://en.wikipedia.org/wiki/Prime_(symbol))
(`'`). Therefore names may contain the full range of non-Latin characters
including diacritics.

All **Flitter** language keywords and supported render nodes and attributes use
only Latin characters and are generally English words or abbreviations (using US
spelling). Therefore, **Flitter** programs can be written using only ASCII
characters, but feel free to use hieroglyphs if you wish.

## Comments

Everything following a pair of sequential dash characters (`--`) up to the end
of that line is a comment. There is no multi-line comment sequence in
**Flitter**, use `--` at the beginning of each line.

## Values

All values are vectors of either floating point numbers, Unicode strings, nodes
or functions (or a mix thereof). The vector implementation is optimised for
vectors of numbers, particularly short vectors. There are no dedicated integer
values in **Flitter** and so one should be careful of relying on integer numbers
outside of the safe integer range of a double-precision floating point
($-2^{53}$ .. $2^{53}$). Numbers may use exponential `e` notation and may
contain underscores `_` to separate groups of digits for readability.

Mathematical operators operate only on pure number vectors. Using them on
anything else will return the empty vector (`null`). Unicode strings can be
concatenated by vector composition, e.g., `"Hello";" world!"` - the result will
be a 2-vector, but vectors are implicitly concatenated anywhere that strings
are used in the language. Numbers will be turned into strings using
general-purpose formatting and so `"Number ";1` is also a valid string.

### Named values

`null`
: The empty vector

`true`
: The value `1`

`false`
: The value `0`

`inf`
: The IEEE-754 floating-point infinity value

`nan`
: The IEEE-754 floating-point "not a number" value

### SI Prefixes

**Flitter** supports adding an SI prefix to the end of a number. This is
confusing terminology, but an SI prefix is a prefix to a units suffix.
**Flitter** does *not* support units, so you just end up with the SI prefix
as a suffix. (Confused yet?)

The allowed SI prefixes are:

- `T` – $\times 10^{12}$
- `G` – $\times 10^{9}$
- `M` – $\times 10^{6}$
- `k` – $\times 10^{3}$
- `m` – $\times 10^{-3}$
- `u` – $\times 10^{-6}$ (also `µ`)
- `n` – $\times 10^{-9}$
- `p` – $\times 10^{-12}$

You can suffix any number with one of these letters, e.g., `10m` is the same as
`0.01`. They are primarily useful for avoiding difficult-to-read long sequences
of zeros, for example, when specifying the brightness of point and spot lights:

```flitter
!light position=0 color=1M
```

### Unicode Strings

All **Flitter** source code files must be UTF-8 encoded and all Unicode
characters are permitted in string values. In-line strings may be denoted with
straight single (`'`) or double (`"`) quotes. Strings may be broken across
multiple lines by enclosing them in triple-single (`'''`) or triple-double
(`"""`) quotes. The usual range of backslash escape sequences are supported
within strings, including Unicode hexadecimal escapes with `\u`.

Anywhere in the rendering engine where strings values are expected, vectors will
be converted into a single string value. This will be done by concatenating each
element of the vector after converting any non-string values into strings.
Numeric values will be converted into their "general" representation (integer
numbers will not have a decimal component, very large numbers will use `e`
notation) with a maximum of 9 decimal places. Functions will be converted into
the function name. Nodes will be converted into their kind, without the node
literal exclamation character `!`, tags, attributes or children. Symbols will be
converted into their name, without the `:` character.

### Symbols

Symbols are names that can be used as values. They are used in various places
in the rendering engine for specifying enumerated values. For example, the
`composite` attribute of `!canvas` specifies the blend function to use when
drawing:

```flitter
!window
    !canvas composite=:add
        …
```

They are also commonly used when constructing state key vectors or seed vectors
(see [State](#state) below and [Pseudo-random
sources](builtins.md#pseudo-random-sources)). Although strings may be used for
the same purpose, symbols are more readable in the code and are specifically
optimised in the engine for faster execution.

Symbols should be considered to be opaque values. They are actually
deterministically converted to *very* large negative numbers in the parser
(below $-10^{292}$). Whenever a name value is expected by the engine, numbers
will be looked-up in the symbol table to see if they match a known symbol. If
so, the number will be converted into the matching name. This should be treated
as an implementation detail and not relied upon in code. In particular, because
they are really just numbers, symbols *can* be used in mathematical operations.
They shouldn't be, and are deliberately massive to hopefully cause bad things to
happen if they are.

While a clash between a symbol's number and an actual number being used in a
**Flitter** program is possible, it is very unlikely and wouldn't cause any
problems unless that number is converted into a string – in which case, the
number will become the symbol name.

:::{warning}
While symbols and string values are interchangeable in situations where a string
is accepted (such as the `composite` attribute in the example above), they are
not interchangeable as [state mapping](#state) keys.

If this documentation refers to a specific state key as using a symbol, then
that symbol **must** be used when retrieving that state value. Similarly, if a
state key or prefix given in the program as a `state` attribute contains a
symbol, then that symbol must be used to retrieve the state value. If a string
has been used, then a symbol *cannot* be used to retrieve the value.
:::

### Time codes

In addition to normal literal numbers, as described above, **Flitter** supports
literal *time codes*, which are given as a sequence of hours, minutes and
seconds separated with colon characters (`:`), with the hours being optional
and the seconds having an optional decimal fraction. For example:

```flitter
!video filename='test.mp4' position=02:05.3
```

Time codes are converted by the parser into a single-item numeric vector
representing the total number of seconds (*125.3* in the example above).
The hours component may be an arbitrarily large integer value, the minutes
and seconds must be in the range *[0,60)*. Time codes may not be combined with
exponents or SI prefixes, and do not support `_` separators.

### Nodes

The purpose of any **Flitter** program is to construct a render tree.
Individual literal nodes are represented by an exclamation mark followed by
a name, e.g., `!window`. Nodes can be tagged with additional arbitrary names
that aid in readability. A hash character followed by a name will tag the
preceding node (or nodes) with that name, e.g., `#top`. A name followed by an
equals character will set that attribute on the preceding node. The resulting
value in both of these instances is the tagged/attributed node.

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
indented expressions as a sequence and then appends each node in the result
vector to the node above.

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

#### Vector Node Operations

As nodes are values, and thus vectors, tag unary-postfix operations,
attribute-set binary operations and append operations are able to operate on a
vector of nodes simultaneously. For example:

```flitter
let points=10;20;15;25;20;30;25;35

!path
    !move_to point=0;0
    ((!line_to point=x;y) for x;y in points) #segment
```

This is a strange, but perfectly legal, way of doing the equivalent (and more
readable):

```flitter
let points=10;20;15;25;20;30;25;35

!path
    !move_to point=0;0
    for x;y in points
        !line_to #segment point=x;y
```

One could also write:

```flitter
(!group translate=x*5;x*10) for x in ..10
    !ellipse radius=5
```

to construct a 10-item vector of `!group` nodes and then simultaneously
append an `!ellipse` node to each one.

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

Ranges are *not* lazy as they are in Python, so `..1000000` will create a
vector with 1 million items.

## Indexing

Specific elements, or ranges of elements, may be extracted from a multi-element
vector using indexing. Indexing uses the familiar syntax:

*src*`[`*index*`]`

As *all* values in **Flitter** are vectors, *index* may
itself be a multi-element vector.

The rules for indexing an *n*-element vector, *src*, are:

- If either *index* or *src* is `null`, the operation will evaluate to `null`
- *index* **must** be an entirely numeric vector or it will be treated as `null`
- *index* is considered element-at-a-time
- Non-integer indices are *floor*-ed to the next integer value down
- Indices are used, modulo the length of *src*, to select an item from *src*
- All selected elements are composed together into a new vector

[Ranges](#ranges) are a convenient way to create indices for slicing a vector,
e.g., `xs[..5]` will extract the first 5 elements of `xs` (assuming `xs` has
5 or more elements). Indices do not need to be contiguous or in any specific
order. It is perfectly valid to use `x[1;6;0]` to extract the 2nd, 7th and 1st
items, in that order.

As indices are used modulo the length of the source vector, indices past the
end of the vector will wrap around to the beginning and negative indices will
select items backwards from the end. This means that `xs[-1]` can be used to
pick out the last element of `xs`. It also means that all index values of a
single element vector will return that element, and that a value can be easily
expanded out to a long vector with a range. For example, the following will set
`xs` to the number `5` repeated 100 times:

```
let xs = 5[..100]
```

Indexing of Unicode string vectors extracts the *n*-th element of the vector,
*not* the *n*-th character of the string. Therefore `("Hello ";"world!")[0]` is
`"Hello "`. See the [text functions](builtins.md#text-functions) for a mechanism
for extracting individual characters or ranges of characters.

## Operators

**Flitter** supports the usual range of operators, with a lean towards the
syntax and semantics of Python.

### Mathematical operators

The binary vector mathematical operators are:

- *x* `+` *y* - addition
- *x* `-` *y* - subtraction
- *x* `/` *y* - division
- *x* `*` *y* - multiplication
- *x* `//` *y* - floor division (i.e, divide and round down)
- *x* `%` *y* - modulo (pairs with floor division)
- *x* `**` *y* - raise *x* to the power *y*

All of these operations return a vector with the same length as the longer of
*x* and *y*. The shorter vector is repeated as necessary. Note that, as well as
this meaning `(1;2;3;4) + 1` is equal to `2;3;4;5`, it also means that
`(1;2;3;4) + (1;-1)` is equal to `2;1;4;3`. The operators are left-associative,
with `**` having the highest precedence, then `/`, `*`, `//` and `%` at the next
level, and finally `+` and `-`.

Note that the vector composition operator `;` has a *very* low precedence and so
composed vectors will generally have to be wrapped in parentheses when used with
operators:

```flitter
x;y + 1 -- is equivalent to: x;(y+1)
(x;y)+1 -- is equivalent to: (x+1);(y+1)
```

The `%` modulo operator follows the Python convention rather than C style. It
is best understood as the remainder of a `//` floor division:

```flitter
x % y == x - x // y * y
```

This means that `%` will provide a seamless repeating sequence around zero:

```flitter
(-10..10) // 5 == (-2;-2;-2;-2;-2;-1;-1;-1;-1;-1; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1)
(-10..10) % 5  == ( 0; 1; 2; 3; 4; 0; 1; 2; 3; 4; 0; 1; 2; 3; 4; 0; 1; 2; 3; 4)
```

The unary mathematical operators are:

- `-` *x* - negate
- `+` *x* - identity

They sit between `**` and `/`, `*`, `//` and `%` in precedence and so:

```flitter
-x ** y == -(x ** y)
-x * y == (-x) * y
```

All of the mathematical operators return `null` if either *x* or *y* is an
empty or non-numeric vector. Therefore, unary `+x` will return `x` iff `x` is
numeric and `null` otherwise.

### Comparison operators

The binary vector comparison operators are:

- *x* `==` *y* - equality
- *x* `!=` *y* - inequality
- *x* `<` *y* - less than
- *x* `>` *y* - greater than
- *x* `<=` *y* - less than or equal
- *x* `>=` *y* - greater than or equal

Complete vectors are compared and a single truth value returned. The vectors
*x* and *y* are equal iff they are of the same length and each element of *x*
is identical to the corresponding element of *y*. Logical *true* is indicated
by a result of `1` and *false* with a result of `0`. The built-in names `true`
and `false` evaluate to the same values.

The lesser/greater comparison operators compare vectors an element at a time
until an element of *x* is less/greater than the corresponding element in *y*.
If one vector runs out of elements before the other then the shorter one is
lesser. If an element pair cannot be compared because they are not of the same
type, e.g., an attempt to compare a Unicode string with a number, then the
result of the comparison is the `null` vector (which will be interpreted as
`false` in logical expressions).

Additionally, a vector can be tested to see if it contains a sub-vector with
the *contains* operator:

- *x* `in` *y* - returns `true` if any sub-vector of `y` equals `x`

This operator will always return `true` if *x* is the null vector, regardless
of what *y* is.

### Logical operators

Flitter also supports the usual short-cutting logical operators:

- *x* `or` *y* - short-circuiting *or*: returns *x* if *x* is true, *y* otherwise
- *x* `and` *y* - short-circuiting *and*: returns *x* if *x* is false, *y*
  otherwise

In addition, it provides a non-shortcutting exclusive-or operator:

- *x* `xor` *y* - *exclusive or*: returns *y* if *x* is false, *x* if *y* is
  false, `false` otherwise

And the usual logical inverse operator:

- `not` *x* - logical inverse: returns `false` if *x* is true, `true` otherwise

For the purposes of all of these operators, a vector is considered to *false*
if it is zero-length (i.e., `null`) or each element is either `0` or an empty
Unicode string. All other vectors are considered to be *true*.

## Let Expressions

Values may be bound to names with the `let` keyword. It is followed by one or
more `name=expression`s. The expressions are evaluated from left to right, with
each name being bound to the resulting value and added into the scope of the
current sequence.

Lets may be used at the top-level in a **Flitter** script or anywhere within
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

A `let` expression may bind multiple names at once, which apply immediately
in the order they are given, and supports using indentation for continuance.
For example:

```flitter
let x=1 y=2
    z=x*y
```

A let binding may also specify multiple names separated with semicolons to
do an unpacked vector binding. For example:

```flitter
let x;y=SIZE/2
```

This will pick off the first two items from the vector result of evaluating the
expression and bind them in order to the names `x` and `y`. The above code is
roughly the same as:

```flitter
let src=SIZE/2
    x=src[0]
    y=src[1]
```

Therefore, unpacked binding follows the rules for [indexing](#indexing). If the
source vector is longer than the number of names given then additional items are
ignored. If the vector is shorter, then the the additional names will be bound
to items wrapped around from the start again. If the vector is `null` then all
names will be bound to `null`.

A let binding may also bind one or more names to the result of evaluating an
indented "body" sequence of expressions. For example:

```flitter
let foo=
    !foo bar=12
        !baz
```

A let binding of this form may only have one name (or semicolon-separated list
of names for an unpacked vector binding) followed by an `=`, a newline and then
an indented sequence of expressions. This *sequence let* may contain any
multi-line sequence expressions, the same as the body of a function or a loop.
This is particularly useful for binding nested node structures to a name.

If a semicolon-separated name list is provided in a sequence let, then the
names will be bound to values following the unpacking logic described above,
with the vector formed from the indented body being the source vector.

Names introduced with a `let` can redefine engine-supplied values, like `beat`,
and built-ins, like `sin`.

:::{note}
**Flitter** supports [primes](https://en.wikipedia.org/wiki/Prime_(symbol)) in
names. This is a common mathematical convenience for a new unique name that
references another. For example:

```flitter
let x' = x + 1
```

All identifiers may contain any number of single quote characters at the end.
:::

## Where

There is also an inline version of `let` known as `where`. This allows names to
be bound within a non-sequence expression, e.g.:

```flitter
!foo x=(x*x where x=10)
```

It is good practice, although not always necessary, to surround `where`
expressions with parentheses to make the scope clear. However, note that `where`
has higher precedence than `;` vector composition and so `(x;x*x where x=10)` is
equivalent to `x;(x*x where x=10)`.

## Sequences

The result of evaluating an indented sequence of expressions separated by
line breaks – such as might be found in the body of a loop, conditional or
function – is the vector composition of the result of each expression. These
need not necessarily be node expressions, it is perfectly normal for functions
to operator on, and return, numeric or string vectors. For example:

```flitter
func fib(n, x=0, y=1)
    x
    if n > 1
        fib(n-1, y, x+y)
```

This recursive function returns an `n`-item (at least one) vector of Fibonacci
numbers beginning with `x;y`.

```flitter
debug(fib(10))
```

will output the following on the console:

```flitter
0;1;1;2;3;5;8;13;21;34
```

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
non-empty vector containing something other than zeroes or empty strings. So `0`
is false, as is `null`, `""` and `0;0;0`. The result of evaluating the matching
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

## For Loops

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
source vector. Unlike [unpacked vector binding](#let-expressions), if there are
not enough values left in the source vector to bind all names in the last
iteration, then the names lacking matching values will be bound to `null`, i.e.,
a `for` loop cannot iterate past the end of the source vector.

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

Some syntactic sugar is provided to make directly nested loops more readable.
For example:

```flitter
for x in ..N
    for y in ..N
        for z in ..N
            !sphere position=x;y;z size=0.5
```

may be written as the more concise:

```flitter
for x in ..N, y in ..N, z in ..N
    !sphere position=x;y;z size=0.5
```

Again, loops may also be used in-line in non-sequence expressions with syntax
borrowed from Python:

```flitter
!line points=((x;x*5) for x in ..5)
```

This will evaluate to:

```flitter
!line points=0;0;1;5;2;10;3;15;4;20
```

There is no special syntax for nested in-line loops as these can already be
simply written as:

```flitter
let grid = (x;y) for x in ..N for y in ..N
```

It is worth noting that – following the reversed notation – these loops operate
in the *reverse* order to nested non-inline loops, in that this example is
equivalent to:

```flitter
let grid = ((x;y) for x in ..N) for y in ..N
```

and therefore the *last* loop is outermost.

In-line `for` loops can be combined with in-line `if` expressions to filter
elements from the result vector. As the result of a false inline `if` lacking
an `else` is `null`, and a `null` composed into a vector is ignored, one can
use the form:

```flitter
let xs = x*2 if f(x) > 3 for x in ..100
```

## Function calling

Functions are called in the normal way, with the name of the function followed
by zero or more comma-separated arguments within parentheses, e.g., `cos(x)` or
`zip(xs, ys)`.

Most built-in functions will do something sensible with an n-vector, e.g.,
`sin(0.1;0.2)` will return a vector equivalent to `sin(0.1);sin(0.2)`, but
will be substantially faster for long vectors. Some functions operate with both
multiple arguments *and* n-vectors, e.g.:

```flitter
hypot(3;4) == 5
hypot(3, 4) == 5
hypot(3;30, 4;40) == (5;50)
```

As functions are themselves first-order objects in the language, they may be
composed into vectors. The language takes the orthogonal approach that calling
a function vector is identical to composing the result of calling each function
with the arguments, i.e., `(sin;cos)(x) == (sin(x);cos(x))`. This is arguably
obtuse behaviour.

The **Flitter** [built-in functions](builtins.md) are documented separately.

## Function Definitions

```flitter
func name(parameter《=default》《, parameter…》)
    expression
    …
```

`func` will create a new function and bind it to `name`. Default values may be
given for the parameters and will be used if the function is later called with
an insufficient number of matching arguments, otherwise any parameters lacking
matching arguments will be bound to `null`. The result of evaluating the body
sequence will be returned as a vector to the caller.

Functions may be declared anywhere in a program including within another
function definition. Functions may refer to names defined outside of the
function definition. The values of these names will be captured at definition
time and so rebinding a name later in the same scope will be ignored. For
example:

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
in the **Flitter** language and may be manipulated as such. Functions may also
recursively call themselves. That is, the function name is in scope within the
body of the function.

Function calls to user-defined functions may include out-of-order named
arguments, e.g.:

```flitter
func multiply_add(x, y=1, z)
    x*y + z

!foo w=multiply_add(2, z=3)
```

will bind the arguments to parameters with `x` taking the value `2`, `y` taking
its default value `1` and `z` taking the value `3`. Named arguments must be
given *after* any positional arguments and should not repeat positional
arguments (they will be ignored if they do).

Functions that that do not reference any non-local dynamic names within the
body (including recursive functions) are eligible for inlining by the
simplifier at each call site. The simplifier is able to bind the parameters
to the argument expressions and continue simplifying the body on that basis.
Therefore, it is often more performant to pass dynamic values (such as `beat`)
into the function as parameters than allow them to be captured from the
environment.

## Anonymous functions

**Flitter** also allows anonymous functions to be defined and used as values
with the syntax:

```flitter
func (parameter《=default》《, parameter…》) body
```

The body of an anonymous function may only contain *in-line* expressions. In
the grammar, anonymous functions have higher precedence than `;` composition
but lower than inline `for` and `where` expressions. Therefore, the following
defines a function containing a for loop, as might be expected:

```flitter
let f = func(x) x*y for y in ..10
```

However, the following binds `f` to a 3-vector consisting of an anonymous
function, and the numbers *1* and *2*:

```flitter
let f = func(x) x;1;2
```

Note that calling this `f` is *not* in itself an error, as a call to a vector
is valid - as explained in [Function calling](#function-calling) above. The
anonymous function will be called and the attempted calls to the two numbers
will log evaluation errors and otherwise be ignored. So `f(0)` will evaluate
to just `0`.

Any node expression (other than a simple `!kind`) must be parenthesised. The
following superficially sensible code is incorrect:

```flitter
let f = func(c) !light color=c
```

As it is parsed as:

```flitter
let f = func(c) !light
let color = c
```

An anonymous function may return another anonymous function, e.g.:

```flitter
let f = func(x) func(y) x + y
```

As with regular functions, any captured names are bound at the point of
definition and so the value of `x` passed into the first function call will be
bound into the returned anonymous function

## Template function calls

The special `@` operator provides syntactic sugar for calling a function using
syntax similar to constructing a node. The name following `@` should be the
name of the function to be called, any "attributes" placed after this are
passed as named arguments. Any indented expressions are evaluated and the
resulting vector passed as the *first* positional argument to the function,
otherwise the first argument will be `null`.

A template function call like this:

```flitter
@fname name1=value1 name2=value2 …
    sequence
```

roughly translates to:

```flitter
let nodes=
    sequence
fname(nodes, name1=value1, name2=value2, …)
```

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

As template function calls convert into regular function calls, templates may
be used recursively.

### Other ways to use template function calls

Although template function calls are designed to allow custom node-like
operations to be defined, they can be used in other ways. A particularly useful
application is debugging node construction by adding a `@debug` call. As the
`debug()` function logs and returns its first argument, it can be inserted at
any point in a node tree to see what has been constructed without altering the
tree. For example:

```flitter
!window
    !canvas3d viewpoint=1
        !light color=1 direction=-1
        @debug
            for i in ..10
                !sphere position=beta(:position;i)[..3]-0.5 size=0.1
```

Similarly, template function calls can be used in functions to wrap a sequence
of expressions. For example:

```flitter
func onoise(seed, n, k, x, y)
    let weights=k**i for i in ..n
        total=sum(weights)
    @sum
        for i in ..n
            let scale=2**i
            noise(seed, x*scale, y*scale) * weights[i] / total
```

This function is (almost) equivalent to the built-in function `octnoise()` and
makes use of `@sum` to add up all of the calls to `noise()`. Although this
could have been done with `sum()` and an in-line for loop, this code is more
readable.

## State

All mutable state in **Flitter**, such as the outputs of MIDI controllers, the
values of [counters](counters.md) or the properties of [particle physics
systems](physics.md), is contained within the *state mapping*.

State look-ups are done with the `$` prefix operator, which should be followed
by a state key. A key is any vector of numbers, symbols and/or strings. As the
`;` operator binds with lower precedence than `$`, a composed key must be
surrounded with parentheses, e.g., `$(:circle;:radius)`.

For example:

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
window. The `state=:circle_radius` attribute of the `!rotary` node specifies the
state key to write values to and the current value is retrieved with
`$:circle_radius`.

## Pragmas

One or more pragmas may be placed at the top of a source file before any other
expressions. Pragmas take a name and a single literal number or string.

There are three currently supported pragmas:

```flitter
%pragma tempo 110
%pragma quantum 3
%pragma fps 30
```

These respectively set the initial tempo and/or quantum of the main clock (the
defaults are 120 and 4, respectively), and the *current* target frame rate of
the engine (default is 60 or the value specified with the `--fps` command-line
option).

## Imports

Top-level definitions (`let`s and `func`s) may be imported from one **Flitter**
program file into another. This allows common definitions to be collected into
*modules* that can be used elsewhere. For example:

```flitter
import SIZE;thing from 'common.fl'

!window size=SIZE
    @thing
```

The `import` `from` expression will execute the module with the supplied file
name (interpreted as a string file name relative to the current program's
source file) and include one or more names, separated with semicolons, into the
current program as if they had been defined with a `let` or `func` at this
point. Names not defined in the module will be bound to `null`.

The module is executed in a new context and so top-level node expressions
will not be added to the current tree, any [pragmas](#pragmas) will be ignored
and all [state](#state) look-ups will evaluate to `null`. All [built-in
names](builtins.md) will be defined in a module, but not runtime names (such as
`beat`) or command-line names (supplied with `--define`). To use any of these
values in an imported function they must be passed in as arguments. However,
called functions *do* have access to the current [state](#state).

During execution of an imported module, any calls to [file
built-ins](builtins.md#file-functions) – such as `read()` – will resolve
filenames relative to the path of the module. However, if a function in a module
is called from the main program, and that function calls any file built-ins,
then these will resolve filenames relative to the path of the main program, not
the module.

Modules may import from other modules, provided that a cycle is not created: if
a module attempts to import a module further up the current import chain, then
an error will be output to the console and all of the names for the failing
import will be bound to `null`.

## Parsing

**Flitter** uses an LALR(1) parser with a contextual lexer. This means that, in
general, there are *no reserved keywords*. The following is perfectly legal
code that will parse and run:

```flitter
let let=..10

for in in let
    !let let=in
```

This is convenient in allowing things like `from` to be used as a keyword in an
`import` `from` expression and also as an attribute name in a `!distance from=`
physics node. However, it would be best not to rely on this flexibility too
much as it will fail in many contexts. For instance, this very similar code to
the above will not parse:

```flitter
let let=..10

for in in let
    !let for=in
```

This is because `for` cannot be used as an attribute name here as it is also
a valid parser match at this point for the `for` keyword of an in-line `for`
expression.

## Simplification

When a source file is loaded it is parsed into an abstract syntax tree and then
that is *partially-evaluated* by the simplifier. This attempts to evaluate all
static expressions. The simplifier is quite sophisticated and is able to
construct static parts of node trees, unroll loops with constant source vectors,
evaluate conditionals with constant tests, call functions with constant
arguments (including creating pseudo-random streams), inline functions that
contain only local names, replace `let` names with literal values, evaluate
mathematical expressions (including some rearranging where necessary to achieve
this) and generally reduce as much of the expression tree as possible to
constant values.

The simplifier is also able to evaluate imported modules and pull static names
directly out of them - including inline-able functions. As imports are always
evaluated statically (see [Imports](#imports)), this means module imports can
usually be entirely collapsed at compile-time. The engine will maintain a set of
file dependencies when this happens and will re-compile the running program if
the source of an imported module changes.

Known global names (like `beat`) are always dynamic and so any expressions that
include these will, generally, be dynamic. However, unbound names are
interpreted as null values, therefore the simplifier will simply replace these
with static `null`s, issuing a warning as it does so. It is sometimes useful to
leave unbound names in a program in order to allow behaviour to be switched at
run-time using the `--define NAME=value` command-line option. As the simplifier
is able to follow these null values through conditionals, it will not affect
performance and such uses can be considered similar to `#if` preprocessor
instructions in other languages.

After the simplifier has partially-evaluated the tree (and note that thanks to
loop unrolling and function inlining, "simpler" most often doesn't mean
"smaller"), the tree is compiled into instructions for a stack-based virtual
machine. These instructions are interpreted to run the program.

The simplifier and compiler can run again incorporating any state that has been
stable for a period of time (configurable with the [`--simplifystate`
command-line option](install.md#running-flitter)). If any of these state keys
then changes (i.e., a pad or encoder is touched) the engine will immediately
return to the original compiled program. Programs that involve large static
loops may take a noticeable amount of time to re-simplify and so it may be
necessary to turn off simplification on state in programs that involve
unpredictable state changes (`--simplifystate=0`).

The simplifier can be completely disabled with the `--nosimplify` option.

## Run-time Error Behaviour

**Flitter** is designed for live performance and so makes every attempt to
charge on in the presence of errors. Most erroneous behaviour is simply silently
ignored – such as numerical operations on non-numerical values resulting in
`null`s. Some erroneous behaviour will log warnings or errors – such as using
an unbound name or calling a built-in function with an incorrect number of
arguments – before also evaluating to `null`.

If a live code change makes a program un-parseable then the engine will log a
parser error and continue executing the previously loaded version of the file.
Unfortunately, this does not hold for GLSL shader code, which resolves to
strings that are parsed and compiled on-the-fly and so parse/compile errors in
this will cause the runtime to ignore the code completely. This will cause
`!shader` nodes to fall-back to their default pass-through behaviour and
models contained in a `!canvas3d` group with a custom shader will fall back on
the default shader.

It is generally a good idea to keep the console log from **Flitter** visible
when making live changes to be able to spot errors if they come up.
