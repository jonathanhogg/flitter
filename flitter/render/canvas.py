"""
Flitter drawing canvas based on Skia
"""

# pylama:ignore=W0703,R0912,R0914,R0915,C0103

import enum
import logging
from pathlib import Path

import skia


Log = logging.getLogger(__name__)

_ImageCache = {}


def load_image(filename):
    path = Path(filename)
    if path.exists():
        current_mtime = path.stat().st_mtime
        if filename in _ImageCache:
            mtime, image = _ImageCache[filename]
            if mtime == current_mtime:
                return image
        try:
            image = skia.Image.open(filename)
            Log.info("Read image file %s", filename)
        except Exception:
            Log.exception("Unexpected error opening file %s", filename)
            image = None
        _ImageCache[filename] = current_mtime, image
        return image
    return None


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
    HUE = skia.BlendMode.kHue
    SATURATION = skia.BlendMode.kSaturation
    COLOR = skia.BlendMode.kColor
    LUMINOSITY = skia.BlendMode.kLuminosity


class LineJoin(enum.IntEnum):
    MITER = skia.Paint.Join.kMiter_Join
    ROUND = skia.Paint.Join.kRound_Join
    BEVEL = skia.Paint.Join.kBevel_Join


class LineCap(enum.IntEnum):
    BUTT = skia.Paint.Cap.kButt_Cap
    ROUND = skia.Paint.Cap.kRound_Cap
    SQUARE = skia.Paint.Cap.kSquare_Cap


class FontWeight(enum.IntEnum):
    BLACK = skia.FontStyle.Weight.kBlack_Weight
    BOLD = skia.FontStyle.Weight.kBold_Weight
    EXTRABLACK = skia.FontStyle.Weight.kExtraBlack_Weight
    EXTRABOLD = skia.FontStyle.Weight.kExtraBold_Weight
    EXTRALIGHT = skia.FontStyle.Weight.kExtraLight_Weight
    INVISIBLE = skia.FontStyle.Weight.kInvisible_Weight
    LIGHT = skia.FontStyle.Weight.kLight_Weight
    MEDIUM = skia.FontStyle.Weight.kMedium_Weight
    NORMAL = skia.FontStyle.Weight.kNormal_Weight
    SEMIBOLD = skia.FontStyle.Weight.kSemiBold_Weight
    THIN = skia.FontStyle.Weight.kThin_Weight


class FontWidth(enum.IntEnum):
    CONDENSED = skia.FontStyle.Width.kCondensed_Width
    EXPANDED = skia.FontStyle.Width.kExpanded_Width
    EXTRACONDENSED = skia.FontStyle.Width.kExtraCondensed_Width
    EXTRAEXPANDED = skia.FontStyle.Width.kExtraExpanded_Width
    NORMAL = skia.FontStyle.Width.kNormal_Width
    SEMICONDENSED = skia.FontStyle.Width.kSemiCondensed_Width
    SEMIEXPANDED = skia.FontStyle.Width.kSemiExpanded_Width
    ULTRACONDENSED = skia.FontStyle.Width.kUltraCondensed_Width
    ULTRAEXPANDED = skia.FontStyle.Width.kUltraExpanded_Width


class FontSlant(enum.IntEnum):
    ITALIC = skia.FontStyle.Slant.kItalic_Slant
    OBLIQUE = skia.FontStyle.Slant.kOblique_Slant
    UPRIGHT = skia.FontStyle.Slant.kUpright_Slant


def set_styles(node, ctx=None, paint=None, font=None):
    if ctx is not None:
        translate = node.get('translate', 2, float)
        if translate is not None:
            ctx.translate(*translate)
        rotate = node.get('rotate', 1, float)
        if rotate is not None:
            ctx.rotate(rotate * 360)
        scale = node.get('scale', 2, float)
        if scale is not None:
            ctx.scale(*scale)
    if paint is not None:
        rgb = node.get('color', 3, float)
        if rgb is not None:
            paint.setColor4f(skia.Color4f(*rgb, 1))
        else:
            rgba = node.get('color', 4, float)
            if rgba is not None:
                paint.setColor4f(skia.Color4f(*rgba))
        line_width = node.get('line_width', 1, float)
        if line_width is not None:
            paint.setStrokeWidth(line_width)
        line_join = node.get('line_join', 1, str)
        if line_join is not None and line_join.upper() in LineJoin.__members__:
            paint.setStrokeJoin(skia.Paint.Join(LineJoin.__members__[line_join.upper()]))
        line_cap = node.get('line_cap', 1, str)
        if line_cap is not None and line_cap.upper() in LineCap.__members__:
            paint.setStrokeCap(skia.Paint.Cap(LineCap.__members__[line_cap.upper()]))
        composite = node.get('composite', 1, str)
        if composite is not None and composite.upper() in Composite.__members__:
            paint.setBlendMode(skia.BlendMode(Composite.__members__[composite.upper()]))
        antialias = node.get('antialias', 1, bool)
        if antialias is not None:
            paint.setAntiAlias(antialias)
    if font is not None:
        font_size = node.get('font_size', 1, float)
        if font_size is not None:
            font.setSize(font_size)
        if node.keys() & {'font_family', 'font_weight', 'font_width', 'font_slant'}:
            typeface = font.getTypeface()
            font_style = typeface.fontStyle()
            font_family = node.get('font_family', 1, str, typeface.getFamilyName())
            font_weight = node.get('font_weight', 1, str)
            if font_weight is not None and font_weight.upper() in FontWeight.__members__:
                weight = skia.FontStyle.Weight(FontWeight.__members__[font_weight.upper()])
            else:
                weight = font_style.weight()
            font_width = node.get('font_width', 1, str)
            if font_width is not None and font_width.upper() in FontWidth.__members__:
                width = skia.FontStyle.Width(FontWidth.__members__[font_width.upper()])
            else:
                width = font_style.width()
            font_slant = node.get('font_slant', 1, str)
            if font_slant is not None and font_slant.upper() in FontSlant.__members__:
                slant = skia.FontStyle.Slant(FontSlant.__members__[font_slant.upper()])
            else:
                slant = font_style.slant()
            font.setTypeface(skia.Typeface(font_family, skia.FontStyle(weight, width, slant)))


def draw(node, ctx, paint, font, path):
    match node.kind:
        case "group":
            ctx.save()
            paint, font = skia.Paint(paint), skia.Font(font.getTypeface(), font.getSize())
            set_styles(node, ctx, paint, font)
            for child in node.children:
                draw(child, ctx, paint, font, path)
            ctx.restore()

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
                sweep = node.get('sweep', 1, float)
                if sweep is None:
                    end = node.get('end', 1, float)
                    if end is not None:
                        sweep = end - start
                if sweep is not None:
                    path.arcTo(skia.Rect(point[0]-radius[0], point[1]-radius[1], point[0]+radius[0], point[1]+radius[1]), start*360, sweep*360, False)

        case "rect":
            size = node.get('size', 2, float)
            if size is not None:
                point = node.get('point', 2, float, (0, 0))
                path.addRect(*point, point[0]+size[0], point[1]+size[1])

        case "ellipse":
            radius = node.get('radius', 2, float)
            if radius is not None:
                point = node.get('point', 2, float, (0, 0))
                path.addOval(skia.Rect(point[0]-radius[0], point[1]-radius[1], point[0]+radius[0], point[1]+radius[1]))

        case "fill":
            paint = skia.Paint(paint)
            set_styles(node, paint=paint)
            paint.setStyle(skia.Paint.Style.kFill_Style)
            ctx.drawPath(path, paint)

        case "stroke":
            paint = skia.Paint(paint)
            set_styles(node, paint=paint)
            paint.setStyle(skia.Paint.Style.kStroke_Style)
            ctx.drawPath(path, paint)

        case "close":
            path.close()

        case "text":
            set_styles(node, paint=paint, font=font)
            point = node.get('point', 2, float)
            text = node.get('text', 1, str)
            center = node.get('center', 1, bool, True)
            if point is not None and text is not None:
                bounds = skia.Rect(0, 0, 0, 0)
                font.measureText(text, bounds=bounds)
                if center:
                    ctx.drawString(text, point[0]-bounds.x()-bounds.width()/2, point[1]-bounds.y()-bounds.height()/2, font, paint)
                else:
                    ctx.drawString(text, *point, font, paint)

        case "gradient":
            colors = []
            positions = []
            for child in node.children:
                match child.kind:
                    case "stop":
                        offset = child.get('offset', 1, float)
                        rgb = child.get('color', 3, float)
                        if rgb is not None:
                            positions.append(offset)
                            colors.append(skia.Color4f(*rgb, 1))
                        else:
                            rgba = child.get('color', 4, float)
                            if rgba is not None:
                                positions.append(offset)
                                colors.append(skia.Color4f(*rgba))
                            else:
                                positions.append(offset)
                                colors.append(paint.getColor())
            nstops = len(positions)
            if nstops:
                for i in range(nstops):
                    if positions[i] is None:
                        positions[i] = i / (nstops - 1)
                start = node.get('start', 2, float)
                end = node.get('end', 2, float)
                if start is not None and end is not None:
                    points = [skia.Point(*start), skia.Point(*end)]
                    paint.setShader(skia.GradientShader.MakeLinear(points, colors, positions))
                else:
                    radius = node.get('radius', 2, float)
                    if radius is not None:
                        rotate = node.get('rotate', 1, float, 0)
                        point = skia.Point(node.get('point', 2, float, (0, 0)))
                        matrix = skia.Matrix.Scale(*radius).postRotate(rotate * 360)
                        paint.setShader(skia.GradientShader.MakeRadial(point, 1, colors, positions, localMatrix=matrix))

        case "image":
            filename = node.get('filename', 1, str)
            if filename is not None:
                image = load_image(filename)
                if image is not None:
                    dst = skia.Rect.MakeXYWH(*node.get('point', 2, float, (0, 0)), *node.get('size', 2, float, (image.width(), image.height())))
                    ctx.drawImageRect(image, dst, paint)

        case "layer":
            size = node.get('size', 2, float)
            if size is not None:
                alpha = node.get('alpha', 1, float, 1)
                origin = node.get('origin', 2, float, (0, 0))
                rect = skia.Rect.MakeXYWH(*origin, *size)
                ctx.saveLayerAlpha(rect, int(alpha * 255))
                ctx.clipRect(rect)
                paint, font = skia.Paint(paint), skia.Font(font.getTypeface(), font.getSize())
                set_styles(node, ctx, paint, font)
                for child in node.children:
                    draw(child, ctx, paint, font, path)
                ctx.restore()

        case "canvas":
            set_styles(node, ctx, paint, font)
            for child in node.children:
                draw(child, ctx, paint, font, path)
