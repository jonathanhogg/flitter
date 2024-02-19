
# 3D Canvases

3D rendering in **Flitter** is achieved by adding a `!canvas3d` node into a
[window rendering tree](windows.md). For example:

```flitter
!window size=1920;1080
    !canvas3d
        …
```

`!canvas3d` differs from `!canvas` not just in number of dimensions, but in the
entire rendering approach. While the contents of a regular 2D canvas are mostly
interpreted as individual drawing instructions, the contents of a 3D canvas
build a scene containing models, lights and one or more cameras.

The `!canvas3d` node has only one attribute that is unique to it.

`camera_id=` *ID*
: This specifies the camera to use for the output of this node in the window
rendering tree. If not specified, then the default canvas camera is used.

Beyond this single attribute, the `!canvas3d` node combines the functionality of
[transforms](#transforms), [render groups](#render-groups) and
[cameras](#cameras). All of the attributes listed below for those nodes is also
supported on the `!canvas3d` node.

## Transforms

The `!transform` node applies changes the local transformation matrix that
defines the coordinate system for all of the enclosed nodes. The supported
attributes are:

`translate=`*X*`;`*Y*`;`*Z*
: Moves the origin to *X*`;`*Y*`;`*Z* in the current coordinate system.

`scale=`*sX*`;`*sY*`;`*sZ*
: Scales the coordinate system so that each unit of $x$, $y$ and $z$ become
$sX \cdot x$, $sY \cdot x$ and $sZ \cdot z$ in the current coordinate system.
If given as a single item vector then scale all axes by that amount.

`rotate=`*tX*`;`*tY*`;`*tZ*
: Adds three rotation steps into the local transformation matrix, rotating by
$tZ$ turns around the $z$-axis, then $tY$ turns around the $y$-axis and then
$tX$ turns around the $x$-axis. Rotates equally around all axes if given as a
single item vector.

`rotate_x`=*tX*
: Add a rotation step around the $x$-axis alone.

`rotate_y`=*tY*
: Add a rotation step around the $y$-axis alone.

`rotate_z`=*tZ*
: Add a rotation step around the $z$-axis alone.

`shear_x`=*kY*;*kZ*
: Add a shear transformation of the $x$-axis in terms of the $y$ and $z$ axes.

`shear_y`=*kX*;*kZ*
: Add a shear transformation of the $y$-axis in terms of the $y$ and $z$ axes.

`shear_z`=*kX*;*kY*
: Add a shear transformation of the $z$-axis in terms of the $y$ and $z$ axes.

`matrix=`*M*
: Multiply the current local transformation matrix by the matrix *M* given as
a 16-item vector in column-major-order.

The transform attributes honour the order that they are applied to a
`!transform` node. So the matrix is updated left-to-right in the following:

```flitter
!transform translate=10;20;30 rotate_x=0.25 scale=1;2;1
    …
```

and is equivalent to:

```flitter
!transform translate=10;20;30
    !transform rotate_x=0.25
        !transform scale=1;2;1
            …
```

From the perspective of the world coordinate system, this transform results in
the contained models being scaled by double in their $y$-axis, then rotated by
0.25 turns around their $x$-axis and then offset by 10;20;30.

The transform attributes are also valid on `!canvas3d` and `!group` nodes,
applying to everything in the scene or to the contents of that render group.

## Render Groups

A render `!group` bundles up part of a scene that will be rendered together with
a specific shader program and set of lights. A default render group is created
at the same time as the canvas and is configured using these attributes on
the `!canvas3d` node. However, additional render groups can be created inside
the canvas, including within nested `!transform` nodes, in which case the local
transform matrix will apply within the group, or even within another group.

The supported attributes are:

`max_lights=`*N*
: Set the maximum number of lights that the shader program will support.
Additional lights beyond this number will be ignored when rendering this group.
The default is 50. The default shader supports up to a few hundred lights.
Changing this attribute will cause the program to be recompiled.

`composite=` [ `:over` | `:dest_over` | `:lighten` | `:darken` | `:add` |
`:difference` | `:multiply` ]
: Control the OpenGL blend mode used when rendering models that overlap each
other. The default is `:over`.

`depth_test=` [ `true` | `false` ]
: Turn off OpenGL depth-testing for this render group if set to `true`, the
default is `false`. This also disables the **Flitter** 3D engine model [instance
ordering](#instance-ordering) and will result in instances being dispatched for
rendering in arbitrary order. Generally this is only useful when combined with a
blending function like `:add` or `:lighten`.

`cull_face=` [ `true` | `false` ]
: Turn off OpenGL backface-culling for this render group if set to `true`, the
default is `false`. All faces of models will be fed into the shader program in
an arbitrary order.

`vertex=`*TEXT*
: Supply an override vertex shader as a text string containing the GLSL code.
Usually this would be read from a file with the [`read()` built-in
function](builtins.md#file-functions). If unspecified, the standard internal
[PBR shader](#pbr-shading) will be used.

`fragment=`*TEXT*
: Supply an override fragment shader as a text string containing the GLSL code.
Usually this would be read from a file with the [`read()` built-in
function](builtins.md#file-functions). If unspecified, the standard internal
[PBR shader](#pbr-shading) will be used.

Supplying your own shader program is beyond the scope of this document, but
there is [an example of doing this available in the **flitter-examples**
repo](https://github.com/jonathanhogg/flitter-examples/blob/main/genuary2024/day29.fl).

If any lights are defined *inside* the render group, those lights will apply
*only* to the models inside that group, along with any lights defined in the
enclosing groups or at the top level of the `!canvas3d` node (which is itself
a render group).

Pulling models and lights into a `!group` is a useful way to limit lighting
effects within the scene. As there is no support for shadow-casting in
**Flitter**, this can be particularly useful to limit the effect of enclosed
lights that shouldn't affect the rest of the scene.

## Cameras

A default camera is created at the same time as the canvas and is configured
by specifying camera attributes on the `!canvas3d` node. Additional cameras
can be defined anywhere inside the hierarchy of `!transform` and `!group` nodes.
Cameras defined in this way count as objects within the scene and so will
respect any local transformation matrix in effect.

Any attributes not specified on a `!camera` node will take the same values as
the default camera from the `!canvas3d` node. The defaults given below apply
to any attribute not specified there either.

The supported attributes are:

`id=` *ID*
: This is a string (or [symbol](language.md#symbols)) identifier that can be
used to select this camera as the primary output by specifying the same *ID*
as the `camera_id` attribute of `!canvas3d`.

`size=` *WIDTH*`;`*HEIGHT*
: This specifies the pixel dimensions to render this camera view at. The value
will default to the size of the parent node of `!canvas3d` in the window
rendering tree.

`secondary=` [ `true` | `false` ]
: If set to `true`, this camera will be rendered regardless of whether it is
the current primary camera and the resulting output will be made available
as a texture for referencing, either with a `!reference` node within the
window rendering tree, with a `texture_id` attribute on an `!image, or even
as a [texture mapped on a model](#texture-mapping).

`position=` *X*`;`*Y*`;`*Z* | `viewpoint=` *X*`;`*Y*`;`*Z*
: The position of this camera, respecting any local transformation matrix.
It is common to use `viewpoint` when specifying this attribute on a `!canvas3d`
node to make clear that this is referring to the position of the camera.

`focus=` *X*`;`*Y*`;`*Z*
: A point that the camera is pointing towards, respecting any local
transformation matrix. The direction of this from the position of the camera
provides camera view direction – its *z*-axis.

`up=` *X*`;`*Y*`;`*Z*
: A vector giving the *y*-axis of the camera. This is a direction not
an absolute position, respects any local transformation matrix, and will be
normalized. If this direction is not at right angles to the camera view
direction, then it will be corrected first. The camera *x*-axis is derived from
this and the view direction.

:::{note}
`position`, `focus` and `up` are all converted into the world coordinate system
before being stored as camera properties. This means that inherited defaults
will be in the world coordinate system.

This means that one can specify a `focus` for the default camera on `!canvas3d`
and then place another moving camera in the scene with a constantly changing
local transformation matrix, and have this camera continue to point towards the
default `focus` point regardless of its local position.
:::

`orthographic=` [ `true` | `false` ]
: Specifies whether this camera uses an orthographic (non-perspective)
projection. The default is `false`.

`fov=` *FOV*
: Specifies the field-of-view for perspective cameras (`orthographic=false`) in
*turns*, i.e., 90° is `0.25` – effectively the zoom vs wide-angle setting of
the camera.

`fov_ref=` [ `:horizontal` | `:vertical` | `:diagonal` | `:narrow` | `:wide` ]
: Specifies the reference length for field-of-view. The default is
`:horizontal`, meaning that `fov` specifies the horizontal field-of-view.
The special values `:narrow` and `:wide` allow one to refer to the narrower or
wider of the horizontal and vertical camera view dimensions. This is useful
if the camera `size` might change between portrait and landscape aspects and a
minimum or maximum field-of-view is desired.

`width=` *WIDTH*
: Specifies the *width* of the camera view in the *world* coordinate system. An
orthographic camera is effectively a rectangle that projects in the camera
view direction. The aspect ratio of that rectangle is taken from the `size`
attribute, this attribute gives the actual width of the rectangle.

`near=` *NEAR*
: Specifies the near clip plane of the camera. Anything on the near side of
a plane (orthogonal to the camera view direction) at this distance from the
camera in the *world* coordinate system will not be rendered.

`far=` *FAR*
: Specifies the far clip plane of the camera. Anything on the far side of a
plane (orthogonal to the camera view direction) at this distance from the camera
in the *world* coordinate system will not be rendered.

:::{note}
For a perspective camera, it is important that the values of `near` and `far`
are not too small or too big, respectively. Otherwise the OpenGL coordinate
calculations can suffer from precision problems that can affect the rendering.
It is best to keep these numbers to just either side of the expected scene
dimensions.
:::

`monochrome=` [ `true` | `false` ]
: If set to `true` then the output of the camera will be grayscale. The RGB
values will be converted into a single luminance value that will then be used
for each fo the RGB channels of the output. The default is `false`.

`tint=` *R*`;`*G*`;`*B*
: A tint value that will be multiplied into all of the pixel RGB values. This
can be used to do simple whitepoint correction or to achieve particular effects.
This can be combined with `monochrome` and will happen *after* the RGB values
have been turned into grayscale values – as such it can be used to produce
effects like sepia toned monochrome output. The default is `1;1;1`, i.e., no
tinting.

`colorbits=` [ `8` | `16` | `32` ]
: The bit depth of the output color channels. The canvas default is taken from
its parent(s) in the window rendering tree. If not specified on any ancestor
node, it defaults to `16` bits.

:::{warning}
It is generally a bad idea to use 8-bit color depth with the 3D renderer. As all
rendering calculations are done in *linear* color space and the conversion to
monitor sRGB *logarithmic* space is done as a final window rendering step,
starting with only 8-bits of color resolution will end up producing visible
banding in dark areas of the window as this range gets expanded out by the sRGB
transfer function. Working in 16-bits provides plenty of precision.

The second advantage to working in 16-bits is that the color space expands out
to half-floats instead of 256-level $[0,1]$ pseudo-floats. This means that
channel values greater than 1 won't clip. As the lighting calculations often
produce high brightness values – particularly in specular reflections or near
point and spot lights – clipping severely reduces your room to correct for
this with a tone-mapping pass and/or bloom filter.
:::

`samples=` [ `0` | `1` | `2` | `4` | `8` ]
: Controls the amount of Multi-sample Anti-Aliasing (MSAA) to do. The default
is `0` (none). Generally `4` works well with most GPUs and is really good
value-for-money.

`fog_color=` *R*`;`*G*`;`*B*
: The color to use for fog. Fog is only enabled if `fog_max` is greater than
`fog_min`. If so, the camera frame-buffer will be filled with this color,
instead of transparent pixels, before rendering starts and rendered fragments
will be mixed with the fog color depending on their distance from the camera.
The default color is black, `0;0;0`.

`fog_min=` *MIN*
: The minimum distance before fog begins to apply. Defaults to `0`.

`fog_max=` *MAX*
: The maximum distance at which point all fragments will be rendered as
`fog_color`. Defaults to `0`, which means fog is *not applied*.

`fog_curve=` *EXPONENT*
: The fog calculation takes the relative distance between `fog_min` and
`fog_max` in the range $[0,1]$ and raises it to this power before using that
as the constant for mixing the fragment color with `fog_color`. The default is
`1`, i.e., linear fog. You may find that `2` gives a more authentic result.
