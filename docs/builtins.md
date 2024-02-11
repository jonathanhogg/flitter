# Built-in Functions

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
for an approximation of the irradiance of a Planckian (blackbody) radiator at
temperature *t*, scaled so that `colortemp(6503.5)` (the sRGB whitepoint
correlated colour temperature) is close to `1;1;1`; the approximation only holds
within the range *[1667,25000]* and, strictly speaking, values below 1900°K are
outside the sRGB gamut; irradiance is proportional to the 4th power of the
temperature, so lights become *significantly* brighter at higher temperatures
- `cos(x)` - return cosine of *x* (in *turns*)
- `counter(...)` - see [Counters](#counters) below
- `csv(filename, row)` - return a vector of values obtained by reading a
specific *row* (indexed from *0*) from the CSV file with the given *filename*;
this function intelligently caches and will convert numeric-looking columns in
the row into numeric values
- `exp(x)` - return *e* raised to the power of *x*
- `floor(x)` - return mathematical floor of *x*
- `fract(x)` - return mathematical fractional part of *x* (equivalent to
`x - floor(x)`)
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
