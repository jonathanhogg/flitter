
# Windows

The primary visual output from a **Flitter** program is normally through
windows. Windows are explicitly created rather than there being some default
output space. This allows for multiple windows to be created and controlled
simultaneously with a single program. It also allows properties of the windows
to be controlled.

In order to render anything, a *window rendering tree* is defined by placing
one or more output nodes inside the window. Each of these rendering nodes
creates (at least) one OpenGL texture as an output. Some of these nodes may
have children and the textures generated by these become input textures to those
nodes.

The window rendering nodes are:

`!window`
: An on-screen window. This is always a top-level node in a **Flitter** program.
It may have multiple child nodes.

`!offscreen`
: An off-screen window. This is always a top-level node in a **Flitter**
program.

`!shader`
: An OpenGL shader program. This may appear as a child node anywhere in the
window rendering tree. It may have multiple child nodes. A number of built-in
shaders are available – see [Filters](#filters) below.

`!image`
: Loads an image from an external file. This may appear as a child node anywhere
in the window rendering tree.

`!video`
: Loads and plays a video from an external file. This may appear as a child node
anywhere in the window rendering tree.

`!canvas`
: A [2D drawing canvas](canvas.md). This may appear as a child node anywhere in
the window rendering tree.

`!canvas3d`
: A [3D drawing canvas](canvas3d.md). This may appear as a child node anywhere
in the window rendering tree.

`!record`
: This node can record its output directly to an image or video file. This may
appear as a child node anywhere in the window rendering tree. It may have
multiple child nodes.

`!reference`
: Allows the output of a named window rendering node to be used elsewhere in the
tree.

## Common attributes

All window rendering nodes share a common structure and so they share some
common attributes:

`size=` *WIDTH*`;`*HEIGHT*
: Specifies the size of the texture that this node will render into. This value
is inherited from the parent node if not specified or, in the case of `!image`,
matched to the content.

`id=` *ID*
: Specifies a string or symbol identifier for this node that allows the output
texture from it to be referenced from elsewhere – either a `!reference` node,
a [texture map](canvas3d,md#texture-mapping) on a 3D model, or as an
[image](canvas.md#images) or [pattern](canvas.md#pattern) in a 2D drawing.

`hidden=` [ `true` | `false` ]
: This is valid on any child node (i.e., not on `!window` or `!offscreen`).
Setting this attribute to `true` will cause the parent node to ignore this node
as a child. The output will still be rendered, which means it can still be
referenced.

In addition, `!window`, `!offscreen`, `!shader`, `!video` and `!record` are all
*program nodes* that run an OpenGL shader program. These programs can be changed
with the following attributes:

`vertex=` *STRING*
: Specifies an override vertex shader as a text string containing the GLSL code.
Usually this would be read from a file with the [`read()` built-in
function](builtins.md#file-functions). If unspecified, a standard internal
shader will be used.

`fragment=` *STRING*
: Specifies an override fragment shader as a text string containing the GLSL
code. Usually this would be read from a file with the [`read()` built-in
function](builtins.md#file-functions). If unspecified, a standard internal
shader will be used.

The shader program for `!video` has a specific function related to the rendering
of video frames and so changing this shader is not advised unless you know what
you are doing. However, the other nodes follow a standard scheme that is
described below for [`!shader`](#shader).

:::{warning}
Although the `!canvas3d` node supports `vertex` and `fragment` shader
attributes, these actually override the model instance shader program for the
default [render group](canvas3d.md#render-groups). This is a much more
specialised program and writing a new one is a more complicated endeavour.
:::

## `!window` and `!offscreen`

The `!window` and `!offscreen` nodes are largely identical except for the
latter not opening on-screen. `!offscreen` nodes are primarily intended to
collect window rendering nodes that are to be used as references rather than
through direct rendering into a window. However, all `!window` nodes can be made
to behave as `!offscreen` nodes with the `--offscreen` [command-line
option](install,md#running-flitter). This can be useful for running tests
or for saving output to files without opening a window.

`!window` and `!offscreen` nodes support the following specific attributes:

`linear=` [ `true` | `false` ]
: This specifies the whether linear or logarithmic color-handling is desired.
This only affects the behaviour of [`!canvas` 2D drawing](canvas.md#canvases)
nodes. All other color processing in the pipeline assumes linear color-handling.

`colorbits=` [ `8` | `16` | `32` ]
: This specifies the default bit depth of output texture color channels for the
window rendering tree. If not specified, it defaults to `16` bits. The color
depth of `!window` and `!offscreen` frame-buffers cannot be controlled and is
OS-defined.

The default shader program behaviour for `!window` and `!offscreen` is the same
as that for [`!shader` below](#shader).

### `!key` and `!pointer`

`!window` nodes support a basic input system similar to
[controllers](controllers.md) that allows keyboard and pointer input to be
connected to the [state system](language.md#state). This is controlled by
adding one or more `!key` nodes as children of the `!window` node and/or a
`!pointer` node.

`!key` nodes support the following attributes:

`state=` *PREFIX*
: The prefix for state keys related to this key.

`name=` *NAME*
: The *name* of the key as a string or symbol. The key names are those defined
by [GLFW](https://www.glfw.org/docs/latest/group__keys.html) (without the
leading `GLFW_KEY_` prefix).

A `!key` node must be given for each key that the program is interested in. The
following entries will be created in the state mapping for each key:

*PREFIX*
: A value of `true` if the key is currently pressed, `false` if it is released
or `null` if this is unknown.

*PREFIX* `;pushed`
: This is the same value as the *PREFIX* key.

*PREFIX* `;pushed;:beat`
: The beat counter value at the moment that the key was *last* pressed, or
`null` if this event has not yet occurred.

*PREFIX* `;released`
: This is the logical negation of the the *PREFIX* key value, i.e., `true` if
the key is currently *released*, `false` if it is *pressed* or `null` if this is
unknown.

*PREFIX* `;released;:beat`
: The beat counter value at the moment that the key was *last* released, or
`null` if this event has not yet occurred.

A `!pointer` node supports just the `state` attribute and creates the following
entries in the state mapping:

*PREFIX*
: The current pointer position as a 2-item vector normalized to the $[0,1]$
range, where $0$ is the left/top of the window and $1$ is the right/bottom. If
the pointer is not within the bounds of the window then this state key will be
`null`.

*PREFIX* `;` (`0` | `1` | … )
: The status of the pointer button(s), numbered from `0` upwards – which of
these is "left" or "right" is OS dependent. The state value will be `true` if
the pointer button is currently pressed, `false` if it is released or `null` if
the state is not currently known (for instance the window has just opened and
no pointer events have been processed).

## `!shader`

The `!shader` node allows insertion of an arbitrary OpenGL shader program into
the window render tree. `!shader` nodes (and all other program nodes) support
the following attributes:

`passes=` *PASSES*
: This specifies how many times the shader should be executed in succession for
each frame. Defaults to `1` if not specified. Specifying a number greater than
`1` should be accompanied with use of the `pass` and `last` uniforms described
below.

`colorbits=` [ `8` | `16` | `32` ]
: This overrides the default color channel bit depth for this node's output
texture.

Shader programs can access the texture backing of all child nodes declared
within the `!shader` node. These textures are sampled with samplers controlled
by the following attributes:

`border=` *R*`;`*G*`;`*B*`;`*A*
: Specifies a color that will be returned for any coordinates outside of the
texture.

`repeat=` *RX*`;`*RY*
: Specifies whether to wrap around (and therefore repeat) the texture in the
X and Y axes, if *RX* or *RY* is `true`, or to return the color at the edge of
the image, if *RY* or *RX* is `false`.

If neither `border` nor `repeat` is specified then the default is to return
transparent.

The default shader program (and that also used for `!window`, `!offscreen`
nodes) is a single-pass shader that composites together the output textures of
all child nodes. It can be controlled with the following attributes:

`composite=` [ `:over` | `:dest_over` | `:lighten` | `:darken` | `:add` | `:difference` | `:multiply` ]
: Specifies the blend function to use, default `:over`.

`gamma=` *ALPHA*
: Specifies a gamma curve correction to be applied after compositing, default
`1` (i.e., no correction). Values less than 1 will lighten the output image and
values greater than 1 will darken it.

`alpha=` *ALPHA*
: Specifies a final alpha value to be applied to the entire shader output,
default `1`.

If more specialised behaviour is required then a custom shader program can be
specified. The rendering approach is as follows:

- The $[-1,1]$ screen-space vertices of a single quad covering the whole output
frame-buffer is passed to the vertex shader as an `in vec2`
- The default vertex shader passes standardized $[0,1]$ UV coordinates to the
fragment shader via an `out vec2`
- The fragment shader is expected to declare an `out vec4` fragment color that
will be written to the node's texture

The vertex and fragment shaders have a number of available uniforms that
will be automatically set if declared:

`uniform int passes`
: Will be set to the number of times that this shader will be executed as
specified by the `passes` attribute or `1` if not specified.

`uniform int pass`
: Will be set to the pass number of this execution, counting from `0`.

`uniform vec2 size`
: Will be set to the the pixel-size of the node's output frame-buffer, the
`coord` UV coordinates can be multiplied by this to get actual pixel
coordinates in the fragment shader.

`uniform float beat`
: The current beat counter.

`uniform float quantum`
: The current quantum.

`uniform float tempo`
: The current tempo.

`uniform float delta`
: The difference between the current beat counter and the its value on the last
frame.

`uniform float clock`
: The current frame time in seconds.

`uniform int fps`
: The target frame-rate.

`uniform float performance`
: A value in the range $[0.5,2]$ that indicates how well the engine is managing
to hit the target frame-rate.

`uniform float alpha`
: The value of the `alpha` attribute on the node or a default value of `1`.
This is *not* automatically applied to the output of custom fragment shaders
and must be implemented in the code if desired.

`uniform float gamma`
: The value of the `gamma` attribute on the node or a default value of `1`.
This is *not* automatically applied to the output of custom fragment shaders
and must be implemented in the code if desired.

`uniform sampler2D` [ `texture0` | `texture1` | … ]
: These will be bound to the texture of each child node in turn.

`uniform sampler2D last`
: If specified, this sampler allows access to the final output of the shader
from the previous frame, if this is the first (or only) pass, or the output of
the previous pass otherwise.

`uniform sampler2D first`
: If specified, this sampler allows access to the output of the *first pass* of
the shader (`pass` equal to `0`). In the first pass this sampler is undefined,
for the second pass it will be identical to `last`. Its utility comes in
shaders with more than 2 passes, where it allows later passes to refer to the
results of an initial processing step.

In addition to these, the shader program may declare arbitrary numeric uniforms
that can be set using attributes with matching names on the shader node.
`float`, `vec2`, `vec3`, `vec4`, `mat3` and `mat4` uniforms expect 1-, 2- , 3-,
4-, 9- and 16-item numeric vectors, respectively; arrays of these types expect
vectors with an appropriate multiple of these sizes. `int`s, `double`s and
`bool`s, plus their `vec` and `mat` variants are also supported. If an
attribute matching the uniform is not provided (or is of an incorrect type)
then the uniform will be set to 0 (or the type-appropriate variant of this).

### Shader templating

**Flitter** uses [Mako templates](https://www.makotemplates.org) internally to
allow shaders to adapt to different uses. Shader code will be evaluated using
this before being compiled. All of the uniform names described above (except
`pass`), including any custom attributes on the `!shader` node, are available
as values to the template logic. In addition, the following special names are
defined:

`HEADER`
: Will be set to either the standard OpenGL 3.3 version header or an OpenGL ES
3.0 version header, depending on whether the engine is runnning in ES mode or
not. Shader code that is compatible with both of these versions (which includes
all of the internal **Flitter** GLSL code) can include `${HEADER}` as the first
line to automatically adapt.

`child_textures`
: Will be set to a list of strings representing the uniform name for each of
the sub-nodes of this shader, e.g., `texture0`. If the shader has no children
then this list will be empty. A shader can use template `<% for %/>` loops to
declare the correct sampler uniforms and process multiple input textures.

Note that if a shader is templated on a value, then it will be automatically
recompiled if that value changes (or, to be precise, the changed value causes
a change in the evaluated source text). This will cause severe rendering
performance problems if if occurs frequently. So, while templating on the value
of `passes` or perhaps `size` would be a reasonable thing to do, templating
on the value of `beat` would be a bad idea.

The current value of `pass` is *not* provided to the template engine – the same
shader program is always used for every pass of a multi-pass shader. Shader
programs must use the `pass` uniform and switch behaviour in code for different
passes.

## `!image`

An `!image` node loads the contents of an external image file into a texture
for use in the window rendering tree. This might be as an input to a `!shader`,
for displaying static slideshow images in a `!window`, or in an `!offscreen` to
use as a referenced texture for [3D model texture
mapping](canvas3d.md#texture-mapping).

There are two supported attributes:

`filename=` *PATH*
: Specifies the path of the file to load with respect to the location of the
running **Flitter** program.

`size=` *WIDTH*`;`*HEIGHT*
: If specified, then the image will be resized to this size with bilinear
interpolation.

Unlike the rest of the window rendering nodes, `!image` does not inherit its
size from its parent. If `size` is not specified, then the output texture size
will match the pixel dimensions of the loaded image.

The `!image` output texture is only changed if the underlying file changes, or
the `filename` or `size` attributes are changed. This makes it a very cheap
node to render compared to, say, creating a `!canvas` node and drawing an image
into it. `!image` can load all image file types [supported by the **Pillow**
image library](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html).

## `!video`

A `!video` node loads and renders frames from an external video file into a
texture. The texture will inherit the `size` attribute as normal, or this can
be specified explicitly. By default, the video will be stretched to fill this
size, but this can be controlled with the `aspect` attribute described below.

The following attributes are supported:

`filename=` *PATH*
: Specifies the path of the video to open with respect to the location of the
running **Flitter** program.

`position=` *TIMESTAMP*
: Specifies the time-stamp, in seconds from the start of the video, of the
frame to be output.

`loop=` [ `true` | `false` ]
: Specifies whether the time-stamps beyond the end of the video will loop around
to the beginning. This also enables negative time-stamps, which will loop around
to the end. If set to `false` (the default), time-stamps outside of the video
range will be clamped to the first or last frames of the video.

`back_and_forth=` [ `true` | `false` ]
: If `loop=true` and `back_and_forth=true` then timestamps beyond the end of
the video will work backwards through the video to the start and then forwards
again. This is useful for textural videos to avoid a discontinuity but may
cause performance problems depending on the video encoding (see note below).

`trim=` *START*`;`*END*
: Specifies an amount of time in seconds to trim off the beginning and end of
the video. This affects how `position` is mapped to the source video and the
operation of `loop`.

`interpolate=` [ `true` | `false` ]
: Specifies whether to mix two successive frames if `position` references a
value between the frame time-stamps. This can be useful for generating
slow-motion output if the video does not contain much movement. The default is
`false`.

`aspect=` [ `:fit` | `:fill` ]
: If the source video is a different aspect ratio to that of the node `size`,
then this specifies that the video aspect ratio should be respected and the
video either scaled to *fit* in the frame (with borders on the top/bottom or
left/right sides) or scaled to *fill* the entire frame (with the video cropped
at the top/bottom or left/right sides). If not specified, then the default is
to stretch the video to fill the frame.

`fill=` [ `true` | `false` ]
: If the source video is a different aspect ratio to that of the node `size`,
then this specifies that the video aspect ratio should be respected and the
video scaled and cropped so that the entire frame is filled. Defaults to
`false` and ignored if `fit=true`.

`thread=` [ `true` | `false` ]
: Specifies whether to use multi-threaded video decoding. This has higher
overall performance, but introduces a small delay on the first frame. This delay
may be unacceptable if the video needs to start immediately on load.

`gamma=` *ALPHA*
: Specifies a gamma curve correction to be applied to the video frames, default
`1` (i.e., no correction). Values less than 1 will lighten the output image and
values greater than 1 will darken it.

`alpha=` *ALPHA*
: Specifies a final alpha value to be applied to the video frames, default `1`.

A video is played by setting the `position` attribute to a time-varying value
in code such as the `time` global name. There is no requirement for this value
to vary at real-time, it can slow to a stop or run faster. `position` may skip
forward or backwards by a large step, which will cause a frame seek to the new
location.

:::{warning}
`position` may run *backwards*. However there is an important caveat: if the
video makes extensive use of
[P-frames](https://en.wikipedia.org/wiki/Video_compression_picture_types) then
this will cause a slight judder at each I-frame boundary. The video player will
need to seek back to the previous I-frame and then decode forwards to the
desired frame. The in-between frames are cached, but the same seek and decode
forwards will have to be done each time an I-frame is hit.
:::

`!video` uses the [**PyAV** library](https://pyav.org), which is a wrapper
around [**ffmpeg**](https://ffmpeg.org). It thus supports a wide range of
video file types (including animated GIF files).

## `!record`

The `!record` node expects a single child node which will be written to an
image or video file and then passed through untouched as the output texture
of the `!record` node. If a number of children need to be composited together
for output, then place a [`!shader`](#shader) node between them and `!record`.

`!record` supports the following attributes:

`filename=` *PATH*
: Specifies the path of the image or video file to write to, with respect to the
location of the running **Flitter** program. Whether the output is an image or
a video depends on the extension of the filename. If `filename` is `null`, then
the `!record` node will do nothing – this is a simple way to delay output until
when a particular condition holds.

`quality=` *Q*
: Specifies a quality setting for image formats that support it (such as JPEG).

`codec=` *CODEC*
: For generic video container outputs, this specifies the video codec to use.
Defaults to `:h264`.

`crf=` *CRF*
: For video codecs that support it, this provides a "constant rate factor" that
defines how much the codec should prioritise size over quality. Smaller values
mean better quality and larger values mean a smaller size. A value around `25`
is generally an acceptable compromise for the `:h264` codec. For `:h265`, this
can often be pushed up to a higher value for smaller files while still keeping
a decent quality encoding.

`preset=` *PRESET*
: Specifies a video codec preset if supported. This bunches up different codec
settings. Common presets have names like `:fast` or `:slow` and the
[**ffmpeg**](https://ffmpeg.org) documentation should be referred to for
details.

`limit=` *SECONDS*
: Specifies a maximum number of seconds of video output to write before closing
the file. Otherwise, the video output will continue for as long as `filename` is
valid and the program is running.

Filenames with an `.mp4`, `.mov`, `.m4v`, `.mkv`, `.webm` or `.ogg` extension
are assumed to be video outputs with the appropriate container type. In
addition, if the extension is `.gif` and `codec=:gif` is *also* supplied, then
an animated GIF file will be written with the video output path. Otherwise a
static GIF image will be written.

A particular image file will be written once per run of a **Flitter** program,
i.e., once an image has been written to a particular file, the `!record` node
will do nothing. However, the `filename` attribute can be changed to record a
new image. In this way, a constantly changing filename can be used to write
individual animation frames as images. For example, this program will write a
new JPEG snapshot into the `output` folder on every frame:

```flitter
!window …
    !record filename='output/frame';frame;'.jpg' quality=90
        …
```

:::{note}
If the **Flitter** engine is running in realtime mode, `!record` can will
attempt to record a "live" video. The frame rate of the output video will be
variable and will depend on how fast the encoder can run.

The video encoding runs on a background thread with a 1-second frame buffer.
With a fast hardware accelerated encoder (such as `:hevc_videotoolbox` on a
Mac), this can be used to record a decent video of a live performance. However,
if the encoder is unable to keep up with the running frame rate of the engine,
then live frames will be dropped and the output video will contain stutters.

To record clean videos with a fixed frame rate, the engine should be run in
non-realtime mode with the `--lockstep` command-line option.
:::

## `!reference`

The output texture of one node in a window rendering tree can be used in
multiple places in the tree with a `!reference` node. The node takes a single
attribute:

`id=` *ID*
: Specifies the value of the `id` attribute of matching node.

For example, a bloom-filter pipeline might look like this:

```flitter
let SIZE=1920;1080

!window size=SIZE composite=:lighten
    !canvas3d id=:bloom_source
        …
    !shader fragment=read('blur.frag') radius=5
        !shader fragment=read('threshold.frag') level=0.5
            !reference id=:bloom_source
```

This allows the `!canvas3d` output to appear at two places in the window
rendering tree: as a direct child of the window, and as an input to the
`threshold.frag` shader program.

References *within* a single `!window` tree are updated in render order: the
children of a node are rendered in-order before rendering the node. So a
`!reference` node that refers to a node that has already been rendered will
use the *current* rendered image of that node. A reference to a node that occurs
later (including any parent node of the `!reference`) is valid only *after* the
first frame and will return the rendered image from the previous frame.

A `!reference` to a node *outside* of the enclosing `!window` (or `!offscreen`)
is also valid only after the first frame and will return the rendered texture
from *either the current or previous frame*. No guarantees are made about the
render order of top-level nodes.

## Filters

In addition to the nodes above, a set of useful filters are provided, each of
which is implemented as an OpenGL shader program. Each of these nodes, in common
with the default [`!shader` program](#shader), accepts one or more child nodes
which will be composited together with the blend function controlled with
the `composite` attribute (default `:over`). All of the filters also support
the standard shader `gamma` and `alpha` attributes.

### `!transform`

Composites its input nodes and then scales, rotates and translates its output.
This is similar to the `!translate` node in `!canvas` and `!canvas3d` with the
exception that the order of operations is fixed: scale, rotate, translate. The
origin for all of these operations is the *centre* of the image.

`scale=` *SX*`;`*SY*
: Specifies an amount to scale the image on the X and Y axes, default `1`.
Negative scales will flip the image on the X and/or Y axis.

`rotate=` *TURNS*
: Specifies a clockwise rotation in *turns*, default `0`.

`translate=` *X*`;`*Y*
: Specifies an amount to translate the image on the X and Y axes specified in
*pixels*, with the Y axis pointing up, default `0`.

Areas "outside" the transformed image will be transparent by default. This can
be controlled with the `border` and `repeat` attributes described above for
[`!shader`](#shader).

### `!vignette`

A *very* simple vignette filter that (alpha-) fades out the edges of its output.
It is controlled with the single attribute:

`inset=` *(0,0.5)*
: Specifies the inset as a proportion of the height and width at which the
fade-out will occur, default `0.25`.

### `!adjust`

The `!adjust` node allows for basic lightness adjustments and takes the
following attributes:

`exposure=` *STOPS*
: Specifies an exposure adjustment in stops. This defaults to `0`. An exposure
adjustment of `1` will double the color value of each pixel, an adjustment of
`-1` will half the value of each pixel.

`contrast=` *MULTIPLIER*
: Specifies a contrast adjustment as a multiplier. This defaults to `1`. A
contrast adjustment above 1 multiplies the value of each pixel around the
midpoint, i.e., channels above 0.5 will become brighter and channels below 0.5
will become darker. A contrast adjustment below 1 will compress the dynamic
range around 0.5.

`brightness=` *LEVEL*
: Specifies an exposure adjustment in stops. This defaults to `0`. An exposure
adjustment of `1` will double the color value of each pixel, an adjustment of
`-1` will half the value of each pixel.

These adjustments may be combined (e.g., adjusting contrast and brightness
together). Note that color values may become greater than 1 with these
adjustments, but will be clamped to zero to avoid negative values.

### `!blur`

The `!blur` node applies a blur using a 2-pass (horizontal and vertical), 1D,
normalized Gaussian filter. It is controlled with the attributes:

`radius=` *PIXELS*
: Specifies the size of the blur to be applied as a number of pixels in each
direction. A value of `0` will result in no blur being applied, which is the
default if `radius` is not specified.

`sigma=` *SIGMA*
: Specifies the sigma value for the Gaussian filter as a multiple of the
`radius`. This defaults to `0.3` and controls how quickly the blur will fall
off.

If visible square edges form around bright spots then it may be necessary to
increase the value of `radius` and decrease the value of `sigma`. However, note
that the larger the value of `radius`, the greater the GPU computational
resource required to compute the blur.

### `!bloom`

A `!bloom` filter creates a soft glow around bright parts of the image to
recreate the bloom effect commonly produced by camera lenses. It works by
applying a lightness adjustment to darken the entire image, then applies a
Gaussian blur and finally composites this together with the original image
with a *lighten* blend function.

The filter supports the same attributes as [`!adjust`](#adjust) – except with
`exposure` defaulting to `-1` – and [`!blur`](#blur). The `radius` attribute
must be specified and greater than zero for a bloom to be applied.

The default settings of the `!bloom` node assume that the input will contain
high dynamic range values, i.e., pixels with channel values much larger than 1.
This is common when lighting 3D scenes, but unusual in 2D drawings. For the
latter it may be better to set `exposure=0` and use `contrast` values greater
than *1* instead.

### `!edges`

The `!edges` node applies a simple edge-detection filter by blurring the input
and then blending this with the original image with a *difference* blend
function. It supports the same attributes as [`!blur`](#blur) and, again,
`radius` must be greater than zero or the output will be blank.

### `!feedback`

The `!feedback` node simulates the effect of the analogue video feedback loop
formed by pointing a camera at a screen and mixing this with this input signal.
The following attributes control this mixing and the transformation applied to
the feedback signal:

`timebase=` *BEATS*
: Specifies a number of ticks of the beat counter that controls the application
of the other attributes. This defaults to `1` beat.

`mixer=` *AMOUNT*
: Specifies the mix between the feedback signal and the input signal over
`timebase` beats.

`translate=` *X*`;`*Y*
: Specifies how far the decaying feedback signal will move in pixels per
`timebase` beats using the canvas coordinate system (i.e., origin in the top
left). Defaults to `0`.

`scale=` *SX*`;`*SY*
: Specifies how much the decaying feedback signal will be scaled as a multiple
over `timebase` beats. Defaults to `1`.

`rotate=` *TURNS*
: Specifies how much the decaying feedback signal will be rotated clockwise as
a number of full turns per `timebase` beats. Defaults to `0`.

:::{note}
The `!blur`, `!bloom`, `!edges` and `!feedback` shader programs use samplers
that default to sampling the edge color for pixels beyond the edge of the image
as this produces the best results for those programs. This can be controlled
with the `border` and `repeat` attributes as described above for
[the `!shader` node](#shader).
:::

### `!noise`

The `!noise` node is primarily an image generator that generates 2D slices
through [OpenSimplex 2S](https://github.com/KdotJPG/OpenSimplex2) 3D ("improved
XY") noise. It is controlled with the following attributes:

`seed=` *SEED*
: `!noise` generates reproducible output with the same input values. Supply a
unique vector with the `seed` attribute to generate different outputs, defaults
to the null vector if not supplied.

`components=` *1..4*
: Specify how many distinct noise planes to create, default `1`. Each will
be assigned to one channel of the output image (in the order R, G, B, A).

`octaves=` *OCTAVES*
: Specify how many octaves of noise to generate, default `1`.

`roughness=` *ROUGHNESS*
: Each additional octave of noise has its input coordinate space and output
value scaled by `roughness`, default `0.5`.

`scale=` *SX*`;`*SY*`;`*SZ*
: Specifies a scaling vector to be applied to the X, Y and Z coordinates
passed into the noise function, default `1`.

`origin=` *X*`;`*Y*
: Specifies an offset for the *pre-scaled* X and Y input values, default `0`.
The pre-scaled X and Y coordinates are in pixels from the top left.

`z=` *Z*
: Specifies a *pre-scaled* Z coordinate for the plane to be calculated,
default `0`.

If one or more child textures are defined within the `!noise` node then they
will be composited together and the resulting R, G, and B values passed into
the noise function as X, Y and Z offsets, controlled with the attribute:

`tscale=` *TX*`;`*TY*`;`*TZ*
: Specifies a scaling factor for the RGB values read from the input image into
offsets that will be added to the pre-`scale`d noise coordinates, default `1`.
