"""
Flitter drawing canvas based on cairo
"""

import enum

import cairo


TWOPI = 6.283185307179586


class Composite(enum.IntEnum):
    CLEAR = cairo.Operator.CLEAR
    SOURCE = cairo.Operator.SOURCE
    OVER = cairo.Operator.OVER
    IN = cairo.Operator.IN
    OUT = cairo.Operator.OUT
    ATOP = cairo.Operator.ATOP
    DEST = cairo.Operator.DEST
    DEST_OVER = cairo.Operator.DEST_OVER
    DEST_IN = cairo.Operator.DEST_IN
    XOR = cairo.Operator.XOR
    ADD = cairo.Operator.ADD
    SATURATE = cairo.Operator.SATURATE
    MULTIPLY = cairo.Operator.MULTIPLY
    SCREEN = cairo.Operator.SCREEN
    OVERLAY = cairo.Operator.OVERLAY
    DARKEN = cairo.Operator.DARKEN
    LIGHTEN = cairo.Operator.LIGHTEN
    COLOR_DODGE = cairo.Operator.COLOR_DODGE
    COLOR_BURN = cairo.Operator.COLOR_BURN
    HARD_LIGHT = cairo.Operator.HARD_LIGHT
    SOFT_LIGHT = cairo.Operator.SOFT_LIGHT
    DIFFERENCE = cairo.Operator.DIFFERENCE
    EXCLUSION = cairo.Operator.EXCLUSION
    HSL_HUE = cairo.Operator.HSL_HUE
    HSL_SATURATION = cairo.Operator.HSL_SATURATION
    HSL_COLOR = cairo.Operator.HSL_COLOR
    HSL_LUMINOSITY = cairo.Operator.HSL_LUMINOSITY


class Antialias(enum.IntEnum):
    DEFAULT = cairo.Antialias.DEFAULT
    NONE = cairo.Antialias.NONE
    GRAY = cairo.Antialias.GRAY
    SUBPIXEL = cairo.Antialias.SUBPIXEL
    FAST = cairo.Antialias.FAST
    GOOD = cairo.Antialias.GOOD
    BEST = cairo.Antialias.BEST


class LineJoin(enum.IntEnum):
    MITER = cairo.LineJoin.MITER
    ROUND = cairo.LineJoin.ROUND
    BEVEL = cairo.LineJoin.BEVEL


def set_styles(node, ctx):
    rgb = node.get('color', 3, float)
    if rgb is not None:
        ctx.set_source_rgb(*rgb)
    else:
        rgba = node.get('color', 4, float)
        if rgba is not None:
            ctx.set_source_rgba(*rgba)
    translate = node.get('translate', 2, float)
    if translate is not None:
        ctx.translate(*translate)
    rotate = node.get('rotate', 1, float)
    if rotate is not None:
        ctx.rotate(rotate * TWOPI)
    scale = node.get('scale', 2, float)
    if scale is not None and scale[0] != 0 and scale[1] != 0:
        ctx.scale(*scale)
    line_width = node.get('line_width', 1, float)
    if line_width is not None:
        ctx.set_line_width(line_width)
    line_width = node.get('line_join', 1, str)
    if line_width is not None and line_width.upper() in LineJoin.__members__:
        ctx.set_line_join(LineJoin.__members__[line_width.upper()])
    composite = node.get('composite', 1, str)
    if composite is not None and composite.upper() in Composite.__members__:
        ctx.set_operator(Composite.__members__[composite.upper()])
    antialias = node.get('antialias', 1, str)
    if antialias is not None and antialias.upper() in Antialias.__members__:
        ctx.set_antialias(Antialias.__members__[antialias.upper()])
    font_size = node.get('font_size', 1, float)
    if font_size is not None:
        ctx.set_font_size(font_size)
    font_face = node.get('font_face', 1, str)
    if font_face is not None:
        ctx.select_font_face(font_face)


def draw(node, ctx):
    match node.kind:
        case "canvas":
            set_styles(node, ctx)
            for child in node.children:
                draw(child, ctx)

        case "group":
            alpha = node.get('alpha', 1, float, 1)
            if alpha == 0:
                return
            ctx.save()
            if alpha < 1:
                ctx.push_group()
            set_styles(node, ctx)
            for child in node.children:
                draw(child, ctx)
            if alpha < 1:
                ctx.pop_group_to_source()
                ctx.paint_with_alpha(alpha)
            ctx.restore()

        case "text":
            point = node.get('point', 2, float)
            text = node.get('text', 1, str)
            if point is not None and text is not None:
                ctx.move_to(*point)
                ctx.show_text(text)

        case "path":
            ctx.new_path()
            for child in node.children:
                draw(child, ctx)

        case "move_to":
            point = node.get('point', 2, float)
            if point is not None:
                ctx.move_to(*point)

        case "line_to":
            point = node.get('point', 2, float)
            if point is not None:
                ctx.line_to(*point)

        case "curve_to":
            point = node.get('point', 2, float)
            c1 = node.get('c1', 2, float)
            c2 = node.get('c2', 2, float)
            if point is not None and c1 is not None and c2 is not None:
                ctx.curve_to(*c1, *c2, *point)

        case "arc":
            point = node.get('point', 2, float)
            radius = node.get('radius', 2, float)
            if point is not None and radius is not None:
                start = node.get('start', 1, float, 0)
                end = node.get('end', 1, float, 1)
                ctx.save()
                ctx.translate(*point)
                ctx.scale(*radius)
                ctx.arc(0., 0., 1., start * TWOPI, end * TWOPI)
                ctx.restore()

        case "close":
            ctx.close_path()

        case "rect":
            size = node.get('size', 2, float)
            if size is not None:
                point = node.get('point', 2, float, (0, 0))
                ctx.rectangle(*point, *size)

        case "ellipse":
            radius = node.get('radius', 2, float)
            if radius is not None:
                point = node.get('point', 2, float, (0, 0))
                ctx.save()
                ctx.translate(*point)
                ctx.scale(*radius)
                ctx.move_to(1, 0)
                ctx.arc(0, 0, 1, 0, TWOPI)
                ctx.restore()

        case "fill":
            ctx.fill_preserve()

        case "stroke":
            ctx.stroke_preserve()

        case "gradient":
            start = node.get('start', 2, float)
            end = node.get('end', 2, float)
            if start is not None and end is not None:
                gradient = cairo.LinearGradient(*start, *end)
                for child in node.children:
                    match child.kind:
                        case "stop":
                            offset = child.get('offset', 1, float)
                            if offset is not None:
                                rgb = child.get('color', 3, float)
                                if rgb is not None:
                                    gradient.add_color_stop_rgb(offset, *rgb)
                                else:
                                    rgba = child.get('color', 4, float)
                                    if rgba is not None:
                                        gradient.add_color_stop_rgba(offset, *rgba)
                ctx.set_source(gradient)
