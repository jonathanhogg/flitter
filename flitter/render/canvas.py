"""
Flitter drawing canvas based on Skia
"""

# pylama:ignore=W0703,R0912,R0914,R0915,C0103,R0913,R0911,C901,W0632

import enum
import logging
import math
from pathlib import Path
import time

import skia


Log = logging.getLogger(__name__)

_ImageCache = {}
_RecordStats = logging.getLogger().level >= logging.INFO
_Counts = {}
_Durations = {}


def dump_stats():
    if _Counts:
        total_duration = sum(_Durations.values())
        Log.info("Total time spent canvas rendering: %.0fs, comprised of...", total_duration)
        for key, count in _Counts.items():
            duration = _Durations[key]
            Log.info("%15s  - %8d  x %6.1fµs  = %6.1fs  (%4.1f%%)", key, count, 1e6*duration/count, duration, 100*duration/total_duration)


if _RecordStats:
    import atexit
    atexit.register(dump_stats)


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
    EXTRA_BLACK = skia.FontStyle.Weight.kExtraBlack_Weight
    EXTRA_BOLD = skia.FontStyle.Weight.kExtraBold_Weight
    EXTRA_LIGHT = skia.FontStyle.Weight.kExtraLight_Weight
    INVISIBLE = skia.FontStyle.Weight.kInvisible_Weight
    LIGHT = skia.FontStyle.Weight.kLight_Weight
    MEDIUM = skia.FontStyle.Weight.kMedium_Weight
    NORMAL = skia.FontStyle.Weight.kNormal_Weight
    SEMI_BOLD = skia.FontStyle.Weight.kSemiBold_Weight
    THIN = skia.FontStyle.Weight.kThin_Weight


class FontWidth(enum.IntEnum):
    CONDENSED = skia.FontStyle.Width.kCondensed_Width
    EXPANDED = skia.FontStyle.Width.kExpanded_Width
    EXTRA_CONDENSED = skia.FontStyle.Width.kExtraCondensed_Width
    EXTRA_EXPANDED = skia.FontStyle.Width.kExtraExpanded_Width
    NORMAL = skia.FontStyle.Width.kNormal_Width
    SEMI_CONDENSED = skia.FontStyle.Width.kSemiCondensed_Width
    SEMI_EXPANDED = skia.FontStyle.Width.kSemiExpanded_Width
    ULTRA_CONDENSED = skia.FontStyle.Width.kUltraCondensed_Width
    ULTRA_EXPANDED = skia.FontStyle.Width.kUltraExpanded_Width


class FontSlant(enum.IntEnum):
    ITALIC = skia.FontStyle.Slant.kItalic_Slant
    OBLIQUE = skia.FontStyle.Slant.kOblique_Slant
    UPRIGHT = skia.FontStyle.Slant.kUpright_Slant


class FilterQuality(enum.IntEnum):
    NONE = skia.FilterQuality.kNone_FilterQuality
    LOW = skia.FilterQuality.kLow_FilterQuality
    MEDIUM = skia.FilterQuality.kMedium_FilterQuality
    HIGH = skia.FilterQuality.kHigh_FilterQuality


def get_color(node, default=None):
    color = node.get('color', 0, float)
    if color is not None and len(color) in (3, 4):
        return skia.Color4f(*color)
    return default


def turn_angle(x0, y0, x1, y1, x2, y2):
    xa, ya, xb, yb = x1 - x0, y1 - y0, x2 - x1, y2 - y1
    la, lb = math.sqrt(xa*xa + ya*ya), math.sqrt(xb*xb + yb*yb)
    if la == 0 or lb == 0:
        return 0
    return math.acos(min(max(0, (xa*xb + ya*yb) / (la*lb)), 1)) / (2*math.pi)


def set_styles(node, ctx=None, paint=None, font=None):
    for key, value in node.items():
        match key:
            case 'translate':
                translate = value.match(2, float)
                if translate is not None and ctx is not None:
                    ctx.translate(*translate)
            case 'rotate':
                rotate = value.match(1, float)
                if rotate is not None and ctx is not None:
                    ctx.rotate(rotate * 360)
            case 'scale':
                scale = value.match(2, float)
                if scale is not None and ctx is not None:
                    ctx.scale(*scale)
            case 'color':
                color = value.match(3, float) or value.match(4, float)
                if color is not None and paint is not None:
                    paint.setColor4f(skia.Color4f(*color))
            case 'stroke_width':
                stroke_width = value.match(1, float)
                if stroke_width is not None and paint is not None:
                    paint.setStrokeWidth(stroke_width)
            case 'stroke_join':
                stroke_join = value.match(1, str)
                if stroke_join is not None and stroke_join.upper() in LineJoin.__members__ and paint is not None:
                    paint.setStrokeJoin(skia.Paint.Join(LineJoin.__members__[stroke_join.upper()]))
            case 'stroke_cap':
                stroke_cap = value.match(1, str)
                if stroke_cap is not None and stroke_cap.upper() in LineCap.__members__ and paint is not None:
                    paint.setStrokeCap(skia.Paint.Cap(LineCap.__members__[stroke_cap.upper()]))
            case 'composite':
                composite = value.match(1, str)
                if composite is not None and composite.upper() in Composite.__members__ and paint is not None:
                    paint.setBlendMode(skia.BlendMode(Composite.__members__[composite.upper()]))
            case 'antialias':
                antialias = value.match(1, bool)
                if antialias is not None and paint is not None:
                    paint.setAntiAlias(antialias)
            case 'dither':
                dither = value.match(1, bool)
                if dither is not None and paint is not None:
                    paint.setDither(dither)
            case 'quality':
                quality = value.match(1, str)
                if quality is not None and quality.upper() in FilterQuality.__members__ and paint is not None:
                    paint.setFilterQuality(skia.FilterQuality(FilterQuality.__members__[quality.upper()]))
            case ('font_size' | 'font_family' | 'font_weight' | 'font_width' | 'font_slant') if font is not None:
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
                font_size = value.match(1, float)
                if font_size is not None:
                    font.setSize(font_size)
                font = None


def make_shader(node, paint):
    shaders = []
    for child in node.children:
        shader = make_shader(child, paint)
        if shader is not None:
            shaders.append(shader)

    match node.kind:
        case "color":
            color = get_color(node, paint.getColor4f())
            return skia.Shaders.Color(color)

        case "gradient":
            colors = []
            positions = []
            for child in node.children:
                match child.kind:
                    case "stop":
                        positions.append(child.get('offset', 1, float))
                        colors.append(get_color(child, paint.getColor()))
            nstops = len(positions)
            if nstops:
                for i in range(nstops):
                    if positions[i] is None:
                        positions[i] = i / (nstops - 1)
                start = node.get('start', 2, float)
                end = node.get('end', 2, float)
                if start is not None and end is not None:
                    points = [skia.Point(*start), skia.Point(*end)]
                    return skia.GradientShader.MakeLinear(points, colors, positions)
                radius = node.get('radius', 2, float)
                if radius is not None:
                    rotate = node.get('rotate', 1, float, 0)
                    matrix = skia.Matrix.Scale(*radius).postRotate(rotate * 360).postTranslate(*node.get('point', 2, float, (0, 0)))
                    return skia.GradientShader.MakeRadial(skia.Point(0, 0), 1, colors, positions, localMatrix=matrix)

        case "noise":
            frequency = node.get('frequency', 2, float)
            if frequency is not None:
                octaves = node.get('octaves', 1, int, 8)
                seed = node.get('seed', 1, float, 0)
                size = skia.ISize(*node.get('size', 2, int, (0, 0)))
                match node.get('type', 1, str, "improved"):
                    case "fractal":
                        return skia.PerlinNoiseShader.MakeFractalNoise(*frequency, octaves, seed, size)
                    case "turbulence":
                        return skia.PerlinNoiseShader.MakeTurbulence(*frequency, octaves, seed, size)
                    case _:
                        return skia.PerlinNoiseShader.MakeImprovedNoise(*frequency, octaves, seed)

        case "blend":
            if len(shaders) == 2:
                ratio = node.get('ratio', 1, float)
                if ratio is not None:
                    return skia.Shaders.Lerp(ratio, *shaders)
                mode = node.get('mode', 1, str, "").upper()
                if mode in Composite.__members__:
                    return skia.Shaders.Blend(skia.BlendMode(Composite.__members__[mode]), *shaders)

    return None


def make_image_filter(node, paint):
    sub_filters = []
    for child in node.children:
        image_filter = make_image_filter(child, paint)
        if image_filter is not False:
            sub_filters.append(image_filter)

    match node.kind:
        case "source":
            return None

        case "blend" if len(sub_filters) == 2:
            background, foreground = sub_filters
            ratio = node.get('ratio', 1, float)
            if ratio is not None:
                coefficients = (0, 1-ratio, ratio, 0)
            else:
                coefficients = node.get('coefficients', 4, float)
            if coefficients is not None:
                return skia.ImageFilters.Arithmetic(*coefficients, True, background, foreground)
            mode = node.get('mode', 1, str, "").upper()
            if mode in Composite.__members__:
                return skia.ImageFilters.Xfermode(skia.BlendMode(Composite.__members__[mode]), background, foreground)

        case "blur" if len(sub_filters) <= 1:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            radius = node.get('radius', 2, float)
            if radius is not None:
                return skia.ImageFilters.Blur(*radius, skia.TileMode.kClamp, input=input_filter)

        case "shadow" if len(sub_filters) <= 1:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            radius = node.get('radius', 2, float)
            if radius is not None:
                offset = node.get('offset', 2, float, (0, 0))
                color = get_color(node, paint.getColor4f())
                shadow_only = node.get('shadow_only', 1, bool, False)
                if shadow_only:
                    return skia.ImageFilters.DropShadowOnly(*offset, *radius, color.toColor(), input=input_filter)
                return skia.ImageFilters.DropShadow(*offset, *radius, color.toColor(), input=input_filter)

        case "offset" if len(sub_filters) <= 1:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            offset = node.get('offset', 2, float)
            if offset is not None:
                return skia.ImageFilters.Offset(*offset, input=input_filter)

        case "dilate" if len(sub_filters) <= 1:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            radius = node.get('radius', 2, float)
            if radius is not None:
                return skia.ImageFilters.Dilate(*radius, input=input_filter)

        case "erode" if len(sub_filters) <= 1:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            radius = node.get('radius', 2, float)
            if radius is not None:
                return skia.ImageFilters.Erode(*radius, input=input_filter)

        case "paint" if len(sub_filters) == 0:
            paint = skia.Paint(paint)
            set_styles(node, paint=paint)
            return skia.ImageFilters.Paint(paint)

        case "color_matrix" if len(sub_filters) <= 1:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            matrix = node.get('matrix', 20, float)
            if matrix is None and node.keys() & {'red', 'green', 'blue', 'alpha'}:
                red = node.get('red', 5, float, [1, 0, 0, 0, 0])
                green = node.get('green', 5, float, [0, 1, 0, 0, 0])
                blue = node.get('blue', 5, float, [0, 0, 1, 0, 0])
                alpha = node.get('alpha', 5, float, [0, 0, 0, 1, 0])
                matrix = red + green + blue + alpha
            if matrix is None and node.keys() & {'scale', 'offset'}:
                scale = node.get('scale', 3, float, [1, 1, 1])
                offset = node.get('offset', 3, float, [0, 0, 0])
                matrix = [scale[0], 0, 0, 0, offset[0],
                          0, scale[1], 0, 0, offset[1],
                          0, 0, scale[2], 0, offset[2],
                          0, 0, 0, 1, 0]
            if matrix is None and node.keys() & {'brightness', 'contrast'}:
                brightness = node.get('brightness', 1, float, 0)
                contrast = node.get('contrast', 1, float, 0) + 1
                offset = brightness + (1 - contrast) / 2
                matrix = [contrast, 0, 0, 0, offset,
                          0, contrast, 0, 0, offset,
                          0, 0, contrast, 0, offset,
                          0, 0, 0, 1, 0]
            if matrix is not None:
                color_filter = skia.ColorFilters.Matrix(matrix)
                return skia.ImageFilters.ColorFilter(color_filter, input=input_filter)

    return False


def make_path_effect(node):
    sub_path_effects = []
    for child in node.children:
        path_effect = make_path_effect(child)
        if path_effect is not False:
            sub_path_effects.append(path_effect)

    match node.kind:
        case "dash":
            intervals = node.get('intervals', 0, float)
            if intervals:
                offset = node.get('offset', 1, float, 0)
                return skia.DashPathEffect.Make(intervals, offset)

        case "round_corners":
            radius = node.get('radius', 1, float)
            if radius:
                return skia.CornerPathEffect.Make(radius)

        case "randomize":
            length = node.get('length', 1, float)
            deviation = node.get('deviation', 1, float)
            seed = node.get('seed', 1, int, 0)
            if length and deviation:
                return skia.DiscretePathEffect.Make(length, deviation, seed)

        case "path_matrix":
            matrix = node.get('matrix', 9, float)
            if matrix is None and node.keys() & {'scale', 'rotate', 'translate'}:
                scale = node.get('scale', 2, float)
                matrix = skia.Matrix.Scale(*scale) if scale else skia.Matrix.I()
                rotate = node.get('rotate', 1, float)
                if rotate:
                    matrix = skia.Matrix.Concat(skia.Matrix.RotateDeg(rotate * 360), matrix)
                translate = node.get('translate', 2, float)
                if translate:
                    matrix = skia.Matrix.Concat(skia.Matrix.Translate(*translate), matrix)
            if matrix is not None:
                return skia.MatrixPathEffect.Make(skia.Matrix.MakeAll(*matrix))

        case "sum" if len(sub_path_effects) == 2:
            return skia.PathEffect.MakeSum(*sub_path_effects)

        case "compose" if len(sub_path_effects) == 2:
            return skia.PathEffect.MakeCompose(*sub_path_effects)

    return None


def draw(node, ctx, paint=None, font=None, path=None):
    start = time.time()
    match node.kind:
        case "group":
            ctx.save()
            path, paint, font = skia.Path(), skia.Paint(paint), font.makeWithSize(font.getSize())
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

        case "line":
            points = node.get('points', 0, float)
            if points:
                curve = node.get('curve', 1, float, 0)
                n = len(points) // 2
                last_mid_x = last_mid_y = last_x = last_y = None
                for i in range(n):
                    x, y = points[i*2], points[i*2+1]
                    if i == 0:
                        path.moveTo(x, y)
                    elif curve <= 0:
                        path.lineTo(x, y)
                    else:
                        mid_x, mid_y = (last_x + x) / 2, (last_y + y) / 2
                        if i == 1:
                            path.lineTo(mid_x, mid_y)
                        elif curve >= 0.5 or turn_angle(last_mid_x, last_mid_y, last_x, last_y, mid_x, mid_y) <= curve:
                            path.quadTo(last_x, last_y, mid_x, mid_y)
                        else:
                            path.lineTo(last_x, last_y)
                            path.lineTo(mid_x, mid_y)
                        if i == n-1:
                            path.lineTo(x, y)
                        last_mid_x, last_mid_y = mid_x, mid_y
                    last_x, last_y = x, y

        case "close":
            path.close()

        case "clip":
            ctx.clipPath(path, skia.ClipOp.kIntersect, paint.isAntiAlias())

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

        case "text":
            if 'text' in node:
                text = node['text'].as_string()
                point = node.get('point', 2, float, (0, 0))
                paint, font = skia.Paint(paint), font.makeWithSize(font.getSize())
                set_styles(node, paint=paint, font=font)
                stroke = node.get('stroke', 1, bool, False)
                paint.setStyle(skia.Paint.Style.kStroke_Style if stroke else skia.Paint.Style.kFill_Style)
                if node.get('center', 1, bool, True):
                    bounds = skia.Rect(0, 0, 0, 0)
                    font.measureText(text, bounds=bounds)
                    ctx.drawString(text, point[0]-bounds.x()-bounds.width()/2, point[1]-bounds.y()-bounds.height()/2, font, paint)
                else:
                    ctx.drawString(text, *point, font, paint)

        case "image":
            if 'filename' in node:
                image = load_image(node['filename'].as_string())
                if image is not None:
                    width, height = image.width(), image.height()
                    point = node.get('point', 2, float, (0, 0))
                    fill = node.get('fill', 2, float)
                    fit = node.get('fit', 2, float)
                    size = node.get('size', 2, float, (width, height))
                    if fill is not None:
                        aspect = fill[0] / fill[1]
                        if width/height > aspect:
                            w = height * aspect
                            src = skia.Rect.MakeXYWH((width-w)/2, 0, w, height)
                        else:
                            h = width / aspect
                            src = skia.Rect.MakeXYWH(0, (height-h)/2, width, h)
                        dst = skia.Rect.MakeXYWH(*point, *fill)
                    elif fit is not None:
                        aspect = width / height
                        src = skia.Rect.MakeXYWH(0, 0, width, height)
                        x, y = point
                        if fit[0]/fit[1] > aspect:
                            w = fit[1] * aspect
                            dst = skia.Rect.MakeXYWH(x+(fit[0]-w)/2, y, w, fit[1])
                        else:
                            h = fit[0] / aspect
                            dst = skia.Rect.MakeXYWH(x, y+(fit[1]-h)/2, fit[0], h)
                    else:
                        src = skia.Rect.MakeXYWH(0, 0, width, height)
                        dst = skia.Rect.MakeXYWH(*point, *size)
                    ctx.drawImageRect(image, src, dst, paint)

        case "layer":
            size = node.get('size', 2, float)
            if size is not None:
                alpha = node.get('alpha', 1, float, 1)
                origin = node.get('origin', 2, float, (0, 0))
                rect = skia.Rect.MakeXYWH(*origin, *size)
                ctx.clipRect(rect)
                ctx.saveLayerAlpha(rect, int(alpha * 255))
                path, paint, font = skia.Path(), skia.Paint(paint), font.makeWithSize(font.getSize())
                set_styles(node, ctx, paint, font)
                for child in node.children:
                    draw(child, ctx, paint, font, path)
                ctx.restore()

        case "canvas":
            ctx.save()
            paint, font, path = skia.Paint(AntiAlias=True), skia.Font(skia.Typeface(), 14), skia.Path()
            set_styles(node, ctx, paint, font)
            for child in node.children:
                draw(child, ctx, paint, font, path)
            ctx.restore()

        case _:
            shader = make_shader(node, paint)
            if shader:
                paint.setShader(shader)
            else:
                image_filter = make_image_filter(node, paint)
                if image_filter:
                    paint.setImageFilter(image_filter)
                else:
                    path_effect = make_path_effect(node)
                    if path_effect:
                        paint.setPathEffect(path_effect)

    if _RecordStats:
        duration = time.time() - start
        kind = node.kind
        _Counts[kind] = _Counts.get(kind, 0) + 1
        _Durations[kind] = _Durations.get(kind, 0) + duration
        if kind != 'canvas':
            kind = node.parent.kind
            _Durations[kind] = _Durations.get(kind, 0) - duration
