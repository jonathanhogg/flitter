
# 2D Drawing

:::{note}
This is currently just a placeholder to capture the full horror of everything I
have to document for `!canvas`.
:::

## Overview

## Canvases

`colorbits=`

`linear=`

## Groups

## Layers

`size=`

`origin=`

`alpha=`

## Transforms

`translate=`

`rotate=`

`scale=`

## Paints

`color=`

`stroke_width=`

`stroke_join=` [ `:miter` | `:round` | `:bevel` ]

`stroke_cap=` [ `:butt` | `:round` | `:square` ]

`composite=`

`antialias=` [ `true` | `false` ]

`dither=` [ `true` | `false` ]

## Fonts

`font_family=`

`font_weight=` [ `:black` | `:bold` | `:extra_black` | `:extra_bold` | `:extra_light` | `:invisible` | `:light` | `:medium` | `:normal` | `:semi_bold` | `:thin` ]

`font_width=` [ `:condensed` | `:expanded` | `:extra_condensed` | `:extra_expanded` | `:normal` | `:semi_condensed` | `:semi_expanded` | `:ultra_condensed` | `:ultra_expanded`]

`font_slant=` [ `:italic` | `:oblique` | `:upright` ]

`font_size=`

## Paths

`fill_type=` [ `:even_odd` | `:inverse_even_odd` | `:inverse_winding` | `:winding` ]

### `!move_to`

### `!line_to`

### `!curve_to`

### `!arc_to`

### `!close`

### `!arc`

### `!rect`

### `!ellipse`

### `!line`

### `!stroke`

### `!fill`

### `!clip`

### `!mask`

## Text

## Images

`filename=`

`texture_id=`

## Shaders

### `!color`

`color=`

### `!gradient`

`start=`

`end=`

`radius=`

`rotate=`

#### `!stop`

`color=`

`offset=`

### `!noise`

`frequency=`

`octaves=`

`seed=`

`size=`

`type=` [ `:fractal` | `:turbulence` ]

### `!pattern`

`filename=`

`texture_id=`

`translate=`

`rotate=`

`scale=`

### `!blend`

`ratio=`

`mode=`

## Image Filters

### `!blur`

`radius=`

### `!shadow`

`radius=`

`offset=`

`shadow_only=` [ `true` | `false` ]

### `!offset`

`offset=`

### `!dilate`

`radius=`

### `!erode`

`radius=`

### `!paint`

See [Paints](#paints)

### `!color_matrix`

`matrix=`

`red=` , `green=` , `blue=` , `alpha=`

`scale=` , `offset=`

`brightness=` , `contrast=` , `saturation=`

### `!blend`

`coefficients=`

`ratio=`

`mode=`

### `!source`

## Path Effects

### `!dash`

`intervals=`

`offset=`

### `!round_corners`

`radius=`

### `!jitter`

`length=`

`deviation=`

`seed=`

### `!path_matrix`

`matrix=`

`scale=` , `rotate=`, `translate=`

### `!sum`

## Blend modes

`:clear`

`:source`

`:dest`

`:over`

`:dest_over`

`:in`

`:dest_in`

`:out`

`:dest_out`

`:atop`

`:dest_atop`

`:xor`

`:add`

`:modulate`

`:screen`

`:overlay`

`:darken`

`:lighten`

`:color_dodge`

`:color_burn`

`:hard_light`

`:soft_light`

`:difference`

`:exclusion`

`:multiply`

`:hue`

`:saturation`

`:color`

`:luminosity`
