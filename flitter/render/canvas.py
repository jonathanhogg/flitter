"""
Flitter drawing canvas based on Skia
"""

import enum

import skia
import numpy as np
from PIL import Image


TWOPI = 6.283185307179586


class Composite(enum.IntEnum):
    CLEAR = skia.BlendMode.kClear
    SOURCE = skia.BlendMode.kSrc
    DEST = skia.BlendMode.kDst
    OVER = skia.BlendMode.kSrcOver
    DEST_OVER = skia.BlendMode.kDstOver
    IN = skia.BlendMode.kSrcIn
    DEST_IN = skia.BlendMode.kDstIn
    OUT = skia.BlendMode.kSrcOut
    DEST_OUT = skia.BlendMode.kDstOut
    ATOP = skia.BlendMode.kSrcATop
    DEST_ATOP = skia.BlendMode.kDstATop
    XOR = skia.BlendMode.kXor
    ADD = skia.BlendMode.kPlus
    MODULATE = skia.BlendMode.kModulate
    SCREEN = skia.BlendMode.kScreen
    OVERLAY = skia.BlendMode.kOverlay
    DARKEN = skia.BlendMode.kDarken
    LIGHTEN = skia.BlendMode.kLighten
    COLOR_DODGE = skia.BlendMode.kColorDodge
    COLOR_BURN = skia.BlendMode.kColorBurn
    HARD_LIGHT = skia.BlendMode.kHardLight
    SOFT_LIGHT = skia.BlendMode.kSoftLight
    DIFFERENCE = skia.BlendMode.kDifference
    EXCLUSION = skia.BlendMode.kExclusion
    MULTIPLY = skia.BlendMode.kMultiply
    HSL_HUE = skia.BlendMode.kHue
    HSL_SATURATION = skia.BlendMode.kSaturation
    HSL_COLOR = skia.BlendMode.kColor
    HSL_LUMINOSITY = skia.BlendMode.kLuminosity


class LineJoin(enum.IntEnum):
    MITER = skia.Paint.Join.kMiter_Join
    ROUND = skia.Paint.Join.kRound_Join
    BEVEL = skia.Paint.Join.kBevel_Join


def set_styles(node, ctx, paint, font):
    rgb = node.get('color', 3, float)
    if rgb is not None:
        paint.setColor4f(skia.Color4f(*rgb, 1))
    else:
        rgba = node.get('color', 4, float)
        if rgba is not None:
            ctx.setColor4f(skia.Color4f(*rgba))
    translate = node.get('translate', 2, float)
    if translate is not None:
        ctx.translate(*translate)
    rotate = node.get('rotate', 1, float)
    if rotate is not None:
        ctx.rotate(rotate * 360)
    scale = node.get('scale', 2, float)
    if scale is not None and scale[0] != 0 and scale[1] != 0:
        ctx.scale(*scale)
    line_width = node.get('line_width', 1, float)
    if line_width is not None:
        paint.setStrokeWidth(line_width)
    line_width = node.get('line_join', 1, str)
    if line_width is not None and line_width.upper() in LineJoin.__members__:
        paint.setStrokeJoin(skia.Paint.Join(LineJoin.__members__[line_width.upper()]))
    composite = node.get('composite', 1, str)
    if composite is not None and composite.upper() in Composite.__members__:
        paint.setBlendMode(skia.BlendMode(Composite.__members__[composite.upper()]))
    antialias = node.get('antialias', 1, str)
    if antialias is not None:
        paint.setAntiAlias(antialias.upper() != 'NONE')
    font_size = node.get('font_size', 1, float)
    if font_size is not None:
        font.setSize(font_size)
    font_face = node.get('font_face', 1, str)
    if font_face is not None:
        font_face = font_face.lower()
        if font_face.endswith(' bold'):
            ctx.setTypeface(skia.Typeface(font_face[:5], skia.FontStyle.Bold()))
        else:
            ctx.setTypeface(skia.Typeface(font_face))


def draw(node, ctx, paint=None, font=None, path=None):
    if paint is None:
        paint = skia.Paint()
    if font is None:
        font = skia.Font()
    if path is None:
        path = skia.Path()
    match node.kind:
        case "canvas":
            set_styles(node, ctx, paint, font)
            for child in node.children:
                draw(child, ctx, paint, font, path)

        case "group":
            alpha = node.get('alpha', 1, float, 1)
            if alpha == 0:
                return
            if alpha < 1:
                ctx.saveLayerAlpha(None, int(alpha * 255))
            else:
                ctx.save()
            paint, font = skia.Paint(paint), skia.Font(font.getTypeface(), font.getSize())
            set_styles(node, ctx, paint, font)
            for child in node.children:
                draw(child, ctx, paint, font)
            ctx.restore()

        case "text":
            point = node.get('point', 2, float)
            text = node.get('text', 1, str)
            center = node.get('center', 1, bool, True)
            if point is not None and text is not None:
                blob = skia.TextBlob(text, font)
                if center:
                    rect = blob.bound()
                    ctx.drawTextBlob(blob, point[0]-rect.width()/2, point[1]+rect.height()/2, paint)
                else:
                    ctx.drawTextBlob(blob, *point, paint)

        case "path":
            path = skia.Path()
            for child in node.children:
                draw(child, ctx, paint, font, path)

        case "move_to":
            point = node.get('point', 2, float)
            if point is not None:
                path.moveTo(*point)

        case "line_to":
            point = node.get('point', 2, float)
            if point is not None:
                path.lineTo(*point)

        case "curve_to":
            point = node.get('point', 2, float)
            c1 = node.get('c1', 2, float)
            c2 = node.get('c2', 2, float)
            if point is not None and c1 is not None:
                if c2 is not None:
                    path.cubicTo(*c1, *c2, *point)
                else:
                    path.quadTo(*c1, *point)

        case "arc":
            point = node.get('point', 2, float)
            radius = node.get('radius', 2, float)
            if point is not None and radius is not None:
                start = node.get('start', 1, float, 0)
                end = node.get('end', 1, float, 1)
                path.arcTo(skia.Rect(point[0]-radius[0]/2, point[1]-radius[1]/2, point[0]+radius[0]/2, point[1]+radius[1]/2), start*360, end*360)

        case "close":
            path.close()

        case "rect":
            size = node.get('size', 2, float)
            if size is not None:
                point = node.get('point', 2, float, (0, 0))
                path.addRect(*point, point[0]+size[0], point[1]+size[1])

        case "ellipse":
            radius = node.get('radius', 2, float)
            if radius is not None:
                point = node.get('point', 2, float, (0, 0))
                path.addOval(skia.Rect(point[0]-radius[0]/2, point[1]-radius[1]/2, point[0]+radius[0]/2, point[1]+radius[1]/2))

        case "fill":
            paint = skia.Paint(paint)
            paint.setStyle(skia.Paint.Style.kFill_Style)
            ctx.drawPath(path, paint)

        case "stroke":
            paint = skia.Paint(paint)
            paint.setStyle(skia.Paint.Style.kStroke_Style)
            ctx.drawPath(path, paint)

#         case "gradient":
#             start = node.get('start', 2, float)
#             end = node.get('end', 2, float)
#             if start is not None and end is not None:
#                 gradient = cairo.LinearGradient(*start, *end)
#                 for child in node.children:
#                     match child.kind:
#                         case "stop":
#                             offset = child.get('offset', 1, float)
#                             if offset is not None:
#                                 rgb = child.get('color', 3, float)
#                                 if rgb is not None:
#                                     gradient.add_color_stop_rgb(offset, *rgb)
#                                 else:
#                                     rgba = child.get('color', 4, float)
#                                     if rgba is not None:
#                                         gradient.add_color_stop_rgba(offset, *rgba)
#                 ctx.set_source(gradient)
#
#         case "image":
#             filename = node.get('filename', 1, str)
#             if filename is not None:
#                 image = load_image(filename)
#                 if image is not None:
#                     width, height = node.get('size', 2, float, (image.get_width(), image.get_height()))
#                     x, y = node.get('point', 2, float, (0, 0))
#                     matrix = cairo.Matrix()
#                     matrix.scale(image.get_width() / width, image.get_height() / height)
#                     matrix.translate(-x, -y)
#                     pattern = cairo.SurfacePattern(image)
#                     pattern.set_matrix(matrix)
#                     ctx.save()
#                     ctx.set_source(pattern)
#                     ctx.rectangle(x, y, width, height)
#                     ctx.fill()
#                     ctx.restore()
#
#
# _ImageCache = {}
#
# def load_image(filename):
#     if filename in _ImageCache:
#         return _ImageCache[filename]
#     try:
#         image = Image.open(filename).convert(mode='RGBA')
#         data = np.array(image)[..., [2,1,0,3]].copy().data
#         surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, image.width, image.height, image.width * 4)
#     except Exception as exc:
#         print(exc)
#         image = None
#     _ImageCache[filename] = surface
#     return surface
