
# 3D Rendering

## Overview

3D rendering in **Flitter** uses a forward renderer with a [physically-based
rendering](https://en.wikipedia.org/wiki/Physically_based_rendering) (PBR)
model. Shadow-casting is **not** supported. Transparency **is** supported using
ordered rendering.

## 3D Canvases

A 3D canvas is added into the [window rendering tree](windows.md) with a
`!canvas3d` node. For example:

```flitter
!window size=1920;1080
    !canvas3d
        …
```

`!canvas3d` differs from `!canvas` not just in number of dimensions, but in the
entire rendering approach. While the contents of a regular 2D canvas are mostly
interpreted as individual drawing instructions, the contents of a 3D canvas
build a scene containing models, lights and one or more cameras. This scene is
then passed to one or more 3D shader programs to render.

The `!canvas3d` node has only one attribute that is unique to it:

`camera_id=` *ID*
: This specifies the camera to use for the output of this node in the window
rendering tree. If not specified, then the default canvas camera is used.

Beyond this single attribute, the `!canvas3d` node combines the functionality of
[transforms](#transforms), [render groups](#render-groups), [cameras](#cameras)
and [materials](#materials). All of the attributes listed below for those nodes
are also supported on the `!canvas3d` node.

## Transforms

The `!transform` node applies changes to the local transformation matrix that
defines the coordinate system for all of the enclosed nodes. The supported
attributes are:

`translate=`*X*`;`*Y*`;`*Z*
: Moves the origin to *X*`;`*Y*`;`*Z* in the current coordinate system.

`scale=`*sX*`;`*sY*`;`*sZ*
: Scales the coordinate system so that each unit of $x$, $y$ and $z$ become
${sX} \cdot x$, ${sY} \cdot y$ and ${sZ} \cdot z$ in the current coordinate system.
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

If any lights are defined *inside* the render group, those lights will apply
*only* to the models inside that group, along with any lights defined in the
enclosing groups or at the top level of the `!canvas3d` node (which is itself
a render group).

Pulling models and lights into a `!group` is a useful way to limit lighting
effects within the scene. As there is no support for shadow-casting in
**Flitter**, this can be particularly useful to limit the effect of enclosed
lights that shouldn't affect the rest of the scene.

The supported attributes are:

`max_lights=`*N*
: Set the maximum number of lights that the shader program will support.
Additional lights beyond this number will be ignored when rendering this group.
The default is 50. The default shader supports up to a few hundred lights.
Changing this attribute will cause the program to be recompiled.

`composite=` [ `:over` | `:dest_over` | `:lighten` | `:darken` | `:add` | `:difference` | `:multiply` ]
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
PBR shader will be used.

`fragment=`*TEXT*
: Supply an override fragment shader as a text string containing the GLSL code.
Usually this would be read from a file with the [`read()` built-in
function](builtins.md#file-functions). If unspecified, the standard internal
PBR shader will be used.

:::{note}
Supplying your own shader program is beyond the scope of this document, but
there is [an example of doing this available in the **flitter-examples**
repo](https://github.com/jonathanhogg/flitter-examples/blob/main/genuary2024/day29.fl).

Also worth noting is that, while **Flitter** internally keeps to OpenGL version
3.3, you should be OK to use a higher version number in your shader `#version`
specifier if your platform supports it. On macOS, the highest OpenGL version
supported is 4.1.
:::

### Instance Ordering

Within a render group, all instances of specific models are dispatched to the
GPU in one call, with per-instance data providing the specific transformation
matrix and [material properties](#materials). Instances that have no
transparency are sorted from front-to-back before being dispatched so that early
depth-testing can discard fragments of occluded instances. This is done for each
model in turn.

After non-transparent instances have been dispatched, all instances with
transparency of all models are collected together and rendered in back-to-front
depth order. If a sequence of successive instances are of the same model, then
they will be dispatched in a single call.

Instance ordering is based on the model transformation matrix and assumes that
models have their centre placed at the model origin and are unit-sized. This
will generally work for the built-in [primitive models](#primitive-models), but
may fail to correctly sort for [external models](#external-models) and so
a mix of different models with transparency may draw in the wrong order.

Depth-buffer *writing* is turned **off** when rendering instances with
transparency. This means that all transparent objects will be rendered fully
even if they intersect with one another, overlap in non-trivial ways or the
depth-ordering calculation results in an incorrect order. However, the
depth-buffer is still honoured for deciding whether a fragment is to be rendered
and so instances with transparency occluded by a non-transparent instance will
be hidden correctly.

## Cameras

A default camera is created at the same time as the canvas and is configured
by specifying camera attributes on the `!canvas3d` node. Additional cameras
can be defined anywhere inside the scene tree where a [model](#models) can
be placed. Cameras defined in this way count as objects within the scene and so
will respect any local transformation matrix in effect.

Any attributes not specified on a `!camera` node will take the same values as
the `!canvas3d` node default camera. The defaults given below apply to any
attribute not specified there either.

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
window rendering tree, with a `texture_id` attribute on an `!image`, or even
as a [texture mapped on a model](#texture-mapping). This defaults to `false`
and is **not** inherited from the default camera.

`position=` *X*`;`*Y*`;`*Z* | `viewpoint=` *X*`;`*Y*`;`*Z*
: The position of this camera, respecting any local transformation matrix.
It is common to use `viewpoint` when specifying this attribute on a `!canvas3d`
node to make clear that this is referring to the position of the camera.

`focus=` *X*`;`*Y*`;`*Z*
: A point that the camera is aimed towards, respecting any local transformation
matrix. The direction of this from the camera `position` provides the camera
view direction – its *z*-axis.

`up=` *X*`;`*Y*`;`*Z*
: A vector giving the *y*-axis of the camera. This is a direction not
an absolute position. It respects any local rotations. If this direction is not
at right angles to the camera view direction, then it will be corrected while
maintaining the plane formed by the two.

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
It is a bad idea to use 8-bit color depth with the 3D renderer. All rendering
calculations are done in *linear* color space and the conversion to
monitor sRGB *logarithmic* space is done as a final window rendering step.
Starting with only 8-bits of color resolution will produce visible banding in
darker areas of the window, as this range gets expanded by the logarithmic
conversion while the bright range is compressed. Working in 16-bits provides
ample precision to ensure that the final sRGB output is smooth.

The second advantage to working in 16-bits is that the color space expands out
to half-floats instead of 256-level $[0,1]$ pseudo-floats. This means that
channel values greater than 1 won't clip. As the lighting calculations often
produce high brightness values – particularly in specular reflections or near
point and spot lights – clipping severely reduces your room to correct for this
with a tone-mapping filter or deliberately exploit it with a bloom filter.
:::

`samples=` [ `0` | `2` | `4` | `8` | … ]
: Controls the amount of
[multi-sampling](https://www.khronos.org/opengl/wiki/Multisampling) to do. The
default is `0` (none). Generally `4` works well with most GPUs and is really
good value-for-money – particularly if you have a lot of small/fine models.

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

## Lights

Lights are all specified with the `!light` node, which supports the following
attributes:

`color=` *R*`;`*G*`;`*B*
: Specifies the light color and brightness. These values may, and often will
need to be, significantly greater than `1`. In fact, lights may also be
negative as there's nothing in the maths that stops this. While not being
physically realistic, this can be used for some interesting effects. A light
with `color` equal to `0` will be completely ignored, and this is the default
value if the attribute is missing.

`position=` *X*`;`*Y*`;`*Z*
: Specifies the location in space of the light, respecting any local
transformation matrix.

`direction=` *X*`;`*Y*`;`*Z* ( | `focus=` *X*`;`*Y*`;`*Z* )
: Specifies the direction that this light shines, either as a direction vector
or as an absolute position (`focus`). `focus` can **only** be used if `position`
has also been specified.

`outer=` `0`…`0.5`
: Specifies the angle of the *cone* of a spotlight beam, in *turns*. Defaults to
`0.25`, i.e., 90°, which means the light will shine out 45° all around from
the central direction line.

`inner=` `0`…`outer`
: Specifies the portion of a spotlight beam angle that is at full brightness, in
*turns*. Outside of this angle, the light will dim towards 0 at `outer`.
Defaults to `0`.

`falloff=` *A*`;`*B*`;`*C*`;`*D*
: Supplies the coefficients for the light fall-off equation:
  ```{math}
  {light} = {\textbf{color} \over {A} + {B}\cdot{d} + {C}\cdot{d^2} + {D}\cdot{d^3}}
  ```
  Defaults to `0;0;1;0`, i.e., inverse-squared fall-off with distance. If
  models can pass close to lights and the subsequent very bright spots need to
  be avoided, then it can be useful to introduce a constant component to this
  with something like `1;0;1;0`.

### Light Types

**Flitter** supports four kinds of lights, based on which of the above
attributes have been specified:

**Ambient**
: If only `color` is given then this light is an ambient light that will fall
equally on all models in the render group in all directions. As such this
normally has only small values for `color`.

**Directional**
: If `direction` is given in addition to `color` this this light is a
directional light that shines everywhere with equal brightness in one direction.
As for ambient lights, the values of `color` will normally be smaller than `1`.
This light type is typically used for large, very distant light sources, like
sunlight.

**Point**
: If `position` is given in addition to `color` then this light is a point light
that shines outwards in all directions from `position`. The light brightness
will fall-off with distance according to `falloff`. Due to fall-off, it is
common for the `color` values to be very large.

**Spot**
: If both `position` *and* `direction` (or `focus`) are specified then this
light is a spotlight that shines from `position` in `direction`. The beam will
spread outwards in a cone with angle `outer` and will fall-off according to
`falloff`. As for point lights, it is common for the `color` values to be very
large.

While lights are specified as objects in the scene, they are **not** rendered
themselves and only affect models in the scene. If a visible representation of
a light is required then one would normally need to place a model at the same
location as the light and give it an `emissive` [material](#materials) color.
As long as this model is convex, and the light is positioned within it, the
light will not affect the model.

**Flitter** does **not** support shadow-casting and lights will illuminate all
models in the render group regardless of occlusion.

## Materials

Materials specify the surface properties of models. A "current" material is
maintained alongside the local transformation matrix. This material is
changed by specifying properties using the attributes below on `!canvas3d` or
`!group` nodes, on specific `!material` nodes, or directly on [models](#models).

The `!material` node only makes changes to the current material and this then
applies to any models defined as children of that node. `!material` nodes may
be intermixed in the tree with `!transform` nodes, i.e., a model could be placed
within a `!material` node inside a `!transform` node, or within a `!transform`
node inside `!material` node. This allows for significant flexibility in
combining multiple models that share material or location properties.

**Flitter** uses physically-based rendering and so the material properties are
defined in terms of that standard workflow.

The supported material attributes are:

`metal=` [ `true` | `false` | `0`…`1` ]
: Specifies whether the material is a metal or a dielectric. Logically, this
value is a boolean. However, values between `0` and `1` **are** supported to
represent a mix of these two properties. This is primarily useful when a
metal [texture map](#texture-mapping) is used, to allow for smooth edge
conditions between areas of metal and some other material (corrosion for
example).

`color=` *R*`;`*G*`;`*B*
: Specifies the albedo color of dielectrics or the base reflectivity of metals.
It defaults to `0;0;0`. These values would not normally be greater than `1`, but
there is no limit defined and so models can reflect more light than falls on
them for special-effect purposes.

`ior=` *IOR*
: Specifies the index-of-refraction of the material surface. This alters how
light is scattered off the surface. The default is `1.5`, which is a reasonable
value for most solid materials.

`roughness=` `0`…`1`
: Specifies the roughness of the material, where `0` is a perfectly shiny
surface and `1` is completely matt.

`occlusion=` `0`…`1`
: Specifies an ambient occlusion level for the material. Ambient lights will be
multiplied by this value when being applied. This is really only useful when
this property is [texture mapped](#texture-mapping), where it allows for parts
of a model to be occluded.

`emissive=` *R*`;`*G*`;`*B*
: Specifies an amount of color that this object emits. This does *not* make the
model into a [light](#lights), it only affects how the surface of the model
is rendered. These values may be greater than `1`, and this is not an uncommon
thing to do if tone-mapping or bloom-filtering is in use.

`transparency=` `0`…`1`
: Specifies how transparent this material is, with `0` (the default) being not
transparent at all and `1` meaning fully transparent. **Flitter** does **not**
support refraction, so objects will not appear realistically "glassy". Any
transparency value greater than `0` will affect the [model render
order](#instance-ordering). Transparency applies only to diffuse light scattered
from the surface, specular reflections will be calculated as normal.

### Texture Mapping

In addition to the above single value attributes, material nodes support
specifying textures to be used for per-fragment material properties. Each
of these attributes takes a string or symbol value identifying a node elsewhere
in the [window rendering tree](windows.md) that will be used as the input
texture (including the output of [secondary cameras](#cameras)).

The simplest way to load a bunch of images to use as textures is to place
`!image` nodes in an `!offscreen`. See the [textures
example](https://github.com/jonathanhogg/flitter/blob/main/examples/textures.fl)
in the main repo for how to do this.

`texture_id=` *ID*
: Specifies the *ID* of a node to use for the material `color` property.

`metal_texture_id` *ID*
: Specifies the *ID* of a node to use for the material `metal` property.

`roughness_texture_id` *ID*
: Specifies the *ID* of a node to use for the material `roughness` property.

`occlusion_texture_id` *ID*
: Specifies the *ID* of a node to use for the material `occlusion` property.

`emissive_texture_id` *ID*
: Specifies the *ID* of a node to use for the material `emissive` property.

`transparency_texture_id` *ID*
: Specifies the *ID* of a node to use for the material `transparency` property.

For color properties, the color is read directly from the texture. For non-color
properties, the texture color is converted into a luminance value in the range
$[0,1]$ and this is used for the property value.

All textures support an alpha channel. If this is less than $1$, the value read
from the texture will be mixed with the respective single-value property from
the `!material` node (or the default). This allows, for example, a generic
"corrosion" or "dirt" texture with alpha transparency to be applied over
multiple instances of a model with different base `color`s.

## Models

Actual renderable objects are placed in the scene with model nodes. There are
a number of built-in primitive models and the ability to load an arbitrary
triangular mesh from a file. All model nodes share a set of common attributes:

`position=` *X*`;`*Y*`;`*Z*
: Specifies a local position for the origin of the model. This will be `0;0;0`
if not specified. This allows models to be easily positioned using an enclosing
`!transform` node instead.

`size=` *sX*`;`*sY*`;`*sZ*
: Specifies a "size" for the model, which is just a scaling factor. Makes most
sense when the model is unit-sized. The default is `1;1;1`.

`rotation=` *tX*`;`*tY*`;`*tZ*
: Specifies rotation (in *turns*) around the respective axes. The default is
`0;0;0`.

These three attributes are equivalent to:

```flitter
!transform translate=X;Y;Z rotate=tX;tY;tZ scale=sX;sY;sZ
    …
```

However, the `position`, `size` and `rotation` attributes may be specified in
any order and the resulting local transformation matrix will be calculated
with the transforms applied in this specific order.

An alternative to `position`/`size`/`rotation` placement is the attributes:

`start=` *X0*`;`*Y0*`;`*Z0*
: Specifies the position of the point $(0, 0, -0.5)$ of the model.

`end=` *X1*`;`*Y1*`;`*Z1*
: Specifies the position of the point $(0, 0, 0.5)$ of the model.

`radius=` *R*
: Specifies an $x$ and $y$-axis scale.

These attributes are specifically designed to be used with unit-radius and
unit-length models that have their origin in their centre of mass, which
conveniently matches the `!cylinder` and `!cone` [primitives](#primitive-models)
below. They encompass a position, rotation and $z$-axis scaling with `start`
and `end`, and then a scaling in the other two axes with `radius`.

In addition to these transformation attributes, all models may have
[material](#materials) attributes that provide material properties specific to
the model.

All model date is aggressively cached but automatically rebuilt if required.
This includes automatically reloading external models if the file modification
time-stamp changes. Multiple instances of the *same* model are actually collated
and [dispatched simultaneously](#instance-ordering) to the GPU.

### Primitive Models

These models are all generated on-the-fly and all of them have their origin at
their centre of mass.

`!box`
: This model represents a unit-edge cube, i.e, the corners are at
$(±0.5, ±0.5, ±0.5)$. This model has UV coordinates for [texture
mapping](#texture-mapping) that map each face to a $1/6$ vertical slice of the
UV space.

`!sphere`
: This model represents a unit-radius sphere (strictly speaking, the surface is
made up of triangular faces with corners at `1`). UV coordinates use the
Equirectangular projection.

`!cylinder`
: This is a unit-radius and unit-height cylinder with its axis of rotational
symmetry in the $z$ direction. UV coordinates are similar to an Equirectangular
projection, with the bottom circle mapped to the lower $1/4$ of the UV space,
the top to the top $1/4$, and the middle $1/2$ wrapped around the sides of the
cylinder.

`!cone`
: This is a unit-radius and unit-height cone with its axis of rotational
symmetry in the $z$ direction. UV coordinates are similar to `!cylinder` except
that the upper $3/4$ of the UV space are wrapped around the sides of the cone.
``

The nodes `!sphere`, `!cylinder` and `!cone` all support an additional
attribute:

`segments=` *N*
: This specifies the number of segments to be generated in the model. For a
cylinder this will be the number of rectangles making up the sides and the
number of triangles making up the top and bottom circles. Cones are very
similar except that the sides are made up of triangles as well. For spheres,
the value gives the number of stripes of longitude and the number of stripes of
latitude will be half this. The default in all primitives is `64`.

:::{warning}
The number of model vertices and faces scales linearly or quadratically (in the
case of spheres) with the number of `segments` and so this should be no greater
than that necessary to eliminate obvious visual artefacts.

That said, when rendering large numbers of a particular kind of primitive, it
is best to keep them all at the same value for `segments`, as this allows the
engine to dispatch them simultaneously.
:::

### External Models

External mesh models are loaded with the `!model` node, which takes the
single additional attribute:

`filename=` *FILENAME*
: Specifies the model file to load, relative to the program path. The model
will be automatically reloaded if this file changes.

Meshes are loaded using the [**trimesh**](https://trimesh.org) library and so
**Flitter** supports all of the file-types supported by that, which includes
OBJ and STL files. No material properties are loaded, just the triangular mesh,
so you will need to re-specify the material properties using a `!material`
node or on the `!model` node itself.

### Smooth and Flat Model Shading

The primitive models are all designed with seams and vertex normals so that
they render in a sane way: flat sides are uniformly flat and curved sides have
interpolated normals that ensure they render smoothly.

You can *probably* assume that any external model you load is designed sensibly,
but there are a couple of model shading controls that can be used to force
specific shading behaviour. These are controlled with the following attributes
on a model node:

`flat=` [ `true` | `false` ]
: Setting `flat=true` will cause all faces to be disconnected so that each face
shades as a separate flat surface.

:::{note}
Flat shading will create duplicate vertices at all edges. Basically, the model
will have the same number of faces, but three distinct vertices per face.
:::

For finer-grained control over shading, there is an edge snapping algorithm
that will take a smooth-shaded model, find sharp edges and split them into
seams. This algorithm can be controlled with these attributes:

`snap_edges=` `0`…`0.5`
: This specifies the minimum edge angle (in *turns*) at which to snap. It
represents the difference between the normals of the adjoining faces, so an
angle of `0` would mean that the two faces are in the same plane, `0.25` would
mean that they are at right angles to one another. Specifying `0.5` will disable
the algorithm completely, `0` will cause all edges to be snapped (which is
equivalent to specifying `flat=true`).

`minimum_area=` `0`…`1`
: This specifies a minimum area for a face below which it will be ignored by the
algorithm. This is given as a ratio of face area to total model area. If not
specified, then all faces will be considered.

## Constructive Solid Geometry

**Flitter** supports [Constructive Solid
Geometry](https://en.wikipedia.org/wiki/Constructive_solid_geometry) (CSG) using
features of the **trimesh** and **manifold3d** packages (plus a handful of other
utility libraries). This is managed by creating a tree of operation, transform
and model nodes.

The basic CSG operation nodes are:

`!union`
: Combines all child nodes together into a single model.

`!intersect`
: Computes the intersection of all child nodes.

`!difference`
: Computes the first child node with all following child nodes subtracted from
it.

These take no operation-specific attributes.

Additionally, there is a `!slice` node that cuts a model with a plane specified
with the attributes:

`origin=` *X*`;`*Y*`;`*Z*
: The origin of the cutting plane.

`normal=` *nX*`;`*nY*`;`*nZ*
: The normal of the cutting plane (surface "up" direction).

Everything on the up side of the cutting plane will be cut, and then the engine
will attempt to fill the holes left in the mesh – this may not succeed for
complex shapes. The `!slice` node may have multiple child nodes, in which case
the result will be equivalent to a slice of the `!union` of the child nodes.

A model construction tree may contain `!transform` nodes at any point. These
differ from normal transformations in that they apply the transforms to the
actual model vertices to construct new models that can then be operated on.
This also applies to the usual `position`, `size` and `rotation` attributes on
sub-models, which will be automatically converted into equivalent transform
nodes in the model tree.

:::{note}
A model construction tree **cannot** contain [lights](#lights) or
[cameras](#cameras), and these nodes will be ignored if encountered.
:::

The top node of a tree of model construction operations represents a new model.
It therefore supports all of the standard [model](#models) and
[material](#materials) attributes. The model will be cached so that each
unique sequence of operations is only carried out once. Using the same tree in
multiple places will result in multiple instances of this model as normal. Any
change to the tree, including changes to slice planes or any transforms will
result in a new model being generated.

:::{warning}
Animating a transform or slice operation inside a model construction tree will
cause the model to be reconstructed repeatedly. If the construction operations
are non-trivial to carry out, then this will slow the engine down significantly.

If the animation loops, then you can take advantage of caching by "stepping"
the animated values so that they loop through a fixed, repeating sequence of
values. For instance using the following code will cause a new model to be
created on every frame, as the maths varies slightly every time.

```flitter
!difference
    !sphere size=2
    !cylinder size=(r;r;4) where r=0.5+sine(beat/10)
```

Using this code will create a maximum of 50 versions of the model and repeat
them:

```flitter
!difference
    !sphere size=2
    !cylinder size=(r;r;4) where r=0.5+sine(beat/10)*50//1/50
```
:::

The CSG operations require all models to be "watertight" for them to work. This
means that there are no disconnected edges in the model and no holes. This
might seem pretty straightforward but none of the **Flitter** primitive models
satisfy these constraints. They are all designed to have sane normals and UV
coordinates, which requires them to have split seams and duplicated vertices.

**Flitter** will check models before attempting to operate on them. If any of
the constituent models of a CSG operation are not watertight, it will attempt
to "fix" them with the following – increasingly intrusive – steps:

- First the model will be processed to merge all duplicate vertices and remove
any duplicate faces
- If the model is still not watertight, then an attempt will be made to cap any
holes
- If this fails then a convex hull will be computed from the model and this used
instead.

Note that the last step, computing a convex hull, effectively shrink-wraps the
model. This will work fine if the original model was already convex, but if not
this will paper over any concave sections. A warning will be written to the
console if this step is taken.

The result of any CSG operation will also be a watertight mesh, this means that
all adjacent faces will have shared vertices with normals computed as an average
of the face normals. For a smooth object, this will render correctly. However,
a model with any sharp edges will show strange shading distortions at these
edges. For this reason, constructed models automatically have [edge
snapping](#model-shading) applied with the snap angle set to 0.05 turns (about
18°). This can be controlled by adding an explicit `snap_edges` attribute to the
top node in the model construction tree.
