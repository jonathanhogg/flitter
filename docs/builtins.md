# Built-in Functions

## Full list of functions

`abs(` *x* `)`
: Return absolute value of *x* (ignoring sign).

accumulate(xs)
: Return a vector with the same length as *xs* formed by adding each value to an
accumulator starting at *0*.

acos(x)
: Return arc-cosine (in *turns* not radians) of *x*.

angle(x, y)
: Return the angle (in *turns*) of the cartesian vector *x,y*.

asin(x)
: Return the arc-sine (in *turns*) of *x*.

beta(seed)
: See [Pseudo-random sources](language.md#pseudo-random-sources) section in the
language documentation.

bounce(x)
: Return a repeating bouncing wave (akin to a perfectly bouncing
ball) in the range *[0,1]* with one wave per unit of *x*, with the *0* point
when `x%1 == 0` and the *1* point when `x%1 == 0.5`.

ceil(x)
: Return mathematical ceiling of *x*.

colortemp(t)
: Return a 3-vector of *R*, *G* and *B* **linear sRGB** values for an
approximation of the irradiance of a Planckian (blackbody) radiator at
temperature *t*, scaled so that `colortemp(6503.5)` (the sRGB whitepoint
correlated colour temperature) is close to `1;1;1`; the approximation only holds
within the range *[1667,25000]* and, strictly speaking, values below 1900°K are
outside the sRGB gamut; irradiance is proportional to the 4th power of the
temperature, so the values are very small at low temperatures and become
*significantly* larger at higher temperatures.

cos(x)
: Return cosine of *x* (with *x* expressed in *turns*).

counter(...)
: See explanation in the [Counters](language.md#counters) section of the
language documentation.

csv(filename, row)
: Return a vector of values obtained by reading a
specific *row* (indexed from *0*) from the CSV file with the given *filename*;
this function intelligently caches and will convert numeric-looking columns in
the row into numeric values.

exp(x)
: Return *e* raised to the power of *x*.

floor(x)
: Return mathematical floor of *x*.

fract(x)
: Return mathematical fractional part of *x* (equivalent to `x - floor(x)`).

hsl(h;s;l)
: Return a 3-vector of *R*, *G* and *B* in the range *[0,1]* from a 3-vector of
hue, saturation and lightness (also in the range *[0,1]*).

hsv(h;s;v)
: Return a 3-vector of *R*, *G* and *B* in the range *[0,1]* from a 3-vector of
hue, saturation and value (also in the range *[0,1]*).

hypot(x, [...])
: Return the square root of the sum of the square of each value in `x` with one
argument, with multiple arguments return a vector formed by calculating the same
for the 1st, 2nd, etc., element of each of the argument vectors.

impulse(x, [y=0.25])
: Return a repeating impulse wave in the range *[0,1]* with one wave per unit of
*x*, with the *0* point when `x%1 == 0` and the *1* point when `x%1 == y`.

len(xs)
: Return the length of vector *xs*.

linear(x)
: a linear "easing" function in the range *[0, 1]* with values of *x* less than
*0* returning *0* and values greater than *1* returning *1*.

log(x)
: Return natural log of *x*.

log2(x)
: Return log 2 of *x*.

log10(x)
: Return log 10 of *x*.

map(x, y, z)
: Map a value of *x* in the range *[0,1]* into the range *[y,z]*; equivalent
to `y*x + (1-y)*z` (including in n-vector semantics).

max(x, [...])
: Return the maximum value in the vector *x* with one argument, or the largest
of the arguments in vector sort order.

maxindex(x, [...])
: Return the index of the maximum value in the vector *x* with one argument, or
the index of the largest argument in vector sort order (with the 1st argument
being index *0*).

min(x, [...])
: Return the minimum value in the vector *x* with one argument, or the smallest
of the arguments in vector sort order.

minindex(x, [...])
: Return the index of the minimum value in the vector *x* with one argument, or
the index of the smallest argument in vector sort order (with the 1st argument
being index *0*).

noise(...)
: See [Noise functions](#noise-functions) below.

normal(seed)
: See [Pseudo-random sources](language.md#pseudo-random-sources) section in the
language documentation.

normalize(x)
: Return `x / hypot(x)`.

octnoise(...)
: See [Noise functions](#noise-functions) below.

polar(th)
: Equivalent to `zip(cos(th), sin(th))`.

quad(x)
: A quadratic "easing" function in the range *[0, 1]* with values of *x* less
than *0* returning *0* and values greater than *1* returning *1*.

read(filename)
: Returns a single string value containing the entire text of *filename* (this
function intelligently caches).

round(x)
: Return mathematical round-towards-zero of *x*, with *0.5* rounding up.

sawtooth(x)
: A repeating sawtooth wave function in the range *[0,1)* with one wave per unit
of *x*, with the *0* point at `x%1 == 0` and linearly rising towards *1*.

sharkfin(x)
: A repeating sharkfin wave function in the range *[0,1]* with one wave per unit
of *x*, with the *0* point when `x%1 == 0` and the *1* point when `x%1 == 0.5`.

shuffle(source, xs)
: Return the shuffled elements of *xs* using the psuedo-random *source* (which
should be the result of calling `uniform(...)`).

sin(x)
: Return sine of *x* (with *x* expressed in *turns*).

sine(x)
: A repeating sine wave function in the range *[0,1]* with one wave per unit of
*x*, with the *0* point when `x%1 == 0` and the *1* point when `x%1 == 0.5`.

snap(x)
: A square-root "easing" function in the range *[0, 1]* with values of *x* less
than *0* returning *0* and values greater than *1* returning *1* (conceptually
a quadratic easing function with *x* and *y* axes flipped).

split(text)
: Return a vector formed by splitting the string *text* at newlines (not
included).

sqrt(x)
: Return the square root of *x*.

square(x)
: Equivalent to `x * x`.

sum(xs)
: Return a single numeric value obtained by summing each element of vector *xs*.

triangle(x)
: A repeating triangle wave function in the range *[0,1]* with one wave per unit
of *x*, with the *0* point when `x%1 == 0` and the *1* point when `x%1 == 0.5`.

uniform(seed)
: See [Pseudo-random sources](language.md#pseudo-random-sources) section in the
language documentation.

zip(xs, [...])
: Return a vector formed by interleaving values from each argument vector; for
*m* arguments the resulting vector will be *n * m* elements long, where *n* is
the length of the longest vector; short arguments will repeat, so
`zip(1;2;3;4, 0) == (1;0;2;0;3;0;4;0)`.

## Noise functions

Often, more useful than a random source is a noise function. These produce
smoothly changing output values across one or more input dimensions. **Flitter**
contains an implementation of [OpenSimplex 2S](https://github.com/KdotJPG/OpenSimplex2)
noise in 1, 2 and 3 dimensions.

The basic noise function is:

`noise(` *seed* `,` *x* *[* `,` *y* *[* `,` *z* *] ] ]* `)`

where:

- *seed* is a seed value, as per [Pseudo-random sources](language.md#pseudo-random-sources)
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
for the same inputs. The *seed* value should be used to create multiple
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

**Flitter** provides a function that will do this calculation significantly faster
than the equivalent code:

`octnoise(` *seed* `,` *n* `,` *k* `,` *x* *[* `,` *y* *[* `,` *z* *] ] ]* `)`

where:

- *seed* is a seed value
- *n* is the number of octaves
- *k* is a weight constant
- *x* is the first dimension
- *y* is an optional second dimension
- *z* is an optional third dimension

For each octave iteration (from $0$ to $n-1$), the individual weight is computed
as $k^{-i}$ and the scaling factor for the inputs as $2^i$. A unique seed value
for each iteration is derived automatically from *seed*.

The equivalent `octnoise()` call to the code above would be:

```flitter
let z = octnoise(:seed, 4, 0.5, x, y)
```

Again, this function will accept *n*-vectors as inputs.
