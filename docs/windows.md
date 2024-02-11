
# Windowing

`!window`s composite their children and a `!canvas` is transparent until drawn
into. The blending function for compositing is controlled with the `blend`
attribute on `!window`, which defaults to `:over` (source-over), but may be
changed to `:dest-over`, `:lighten` or `:darken`.

## OpenGL Shaders

The `!shader` node allows insertion of an OpenGL shader into the window render
graph. This takes a `fragment` GLSL text code attribute (also an optional
`vertex` text code attribute, but I've never used this) and a `size` for the
framebuffer. The entire framebuffer is passed to the vertex shader as a pair of
triangles. The default vertex shader passes through [0..1] standardised UV
coordinates to the fragment shader as an `out vec2 coord`. The fragment shader
should have a matching `in vec2 coord` declaration and an `out vec4 color`
declaration for the color to write to the framebuffer. All textures are
RGBA with pre-multiplied alpha. The following standard uniforms may be declared
and will be set to the appropriate values for you:

- `uniform vec2 size` - is the pixel-size of the framebuffer, multiply `coord`
by this to get actual framebuffer coordinates
- `uniform float beat` - the current beat
- `uniform float delta` - the difference between the current beat and the last
iteration of the evaluation/render loop
- `uniform sampler2D texture0` - will be bound to a texture representing the
output of the first child node of the shader; a shader may have multiple child
nodes and further nodes will be bound to `texture1`, etc.
- `uniform sampler2D last` - declaring this will cause the shader renderer to
save the output of the last iteration as a texture that may be used as an input
to the current iteration - this is great for implementing feedback-style
effects

Additional uniforms may be declared and will take values from matching
attributes of the `!shader` node. `float`, `vec2`, `vec3` and `vec4` uniforms
expect 1-, 2- , 3- and 4-vector values respectively. Arrays of these types
will expect a vector with an appropriate multiple of those sizes. GLSL default
values for the uniforms can be given in the code and will be used if the
attribute value is not specified (or is an invalid type or size). This makes it
easy to parameterize a shader and manipulate its output on-the-fly.

Generally one would use the built-in `read()` function to load text from a file
for the code attribute, a la:

```
let SIZE=1920;1080

!window size=SIZE
    !shader fragment=read('blur.frag') radius=5
        !canvas
            ...
```

Shader code read in this way will be reloaded and recompiled on-the-fly if that
file's modification timestamp changes. On macOS the maximum OpenGL level is
`#version 410`. Shaders can be nested to apply multiple effects.

Note that `size` is inherited from the containing node by `!shader`, `!canvas`,
`!canvas3d` and `!video`.

### Reference nodes

The output of one node in a window graph can be passed into multiple shaders
using a `!reference` node. This takes an `id` attribute that should match an
`id` attribute on the source node. For example, a bloom filter pipeline might
look like this:

```
let SIZE=1920;1080

!window size=SIZE blend=:lighten
    !canvas id=:bloom_source
        ...
    !shader fragment=read('blur.frag') radius=5
        !shader fragment=read('threshold.frag') level=0.5
            !reference id=:bloom_source
```

Each node in the graph renders its children in order before rendering itself,
so the `!canvas` node will be drawn first and then this is passed as the
`texture0` uniform to the `threshold.frag` shader, the output of which is
passed into the `blur.frag` shader and then the output of this is composited
with the original canvas image (using the lighten blend function).

### Linear and HDR color

Adding `linear=true` to a `!window` will force the entire OpenGL pipeline of
that window to use linear-sRGB color. This is arguably The Right Thing to do and
should probably be the default. Images and videos will be converted into linear
color values, all drawing and blending will be done in linear-space, and then
the final window framebuffer will be converted back into the standard screen
logarithmic-sRGB for display.

As logarithmic-sRGB optimises the limited 8-bit color channel depth for the way
that the human eye perceives brightness, you will almost certainly not want to
switch to linear color without *also* enabling deeper color channels with
the `colorbits=16` attribute on `!window`. This is inherited by all `!shader`,
`!canvas`, `!canvas3d` and `!video` nodes underneath the window and forces all
textures and framebuffers to be 16-bits per channel.

An added benefit of 16-bit channel depths is that the color format is changed
from pseudo-floats (0..255 integers scaled to 0..1) to actual half-precision
floating point numbers, which can be negative and greater than 1, allowing for
a high dynamic range pipeline. This is particularly useful for effects like
bloom filters. You may need to add a final tone-mapping pass as high brightness
values will just get clipped in the final window render.

The `hsl()` and `hsv()` color functions do not support values of `l` or `v` that
are greater than 1, so multiply the resulting RGB vector to construct high
brightness colors, e.g., `hsv(hue;0.9;1) * 100`. You'll need these bright colors
when setting the `color` of point and spot `!light` sources in `!canvas3d` as
these have an inverse-squared fall-off and so get dim very quickly.
