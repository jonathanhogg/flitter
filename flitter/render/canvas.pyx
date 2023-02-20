# cython: language_level=3, profile=True

"""
Flitter drawing canvas based on Skia
"""

import functools
import logging
from pathlib import Path
import time

from libc.math cimport acos, sqrt

import cython
import skia

from ..model cimport Vector, Node


DEF TwoPI = 6.283185307179586

Log = logging.getLogger(__name__)

cdef dict _ImageCache = {}
cdef bint _RecordStats = logging.getLogger().level >= logging.INFO
cdef dict _Counts = {}
cdef dict _Durations = {}


cdef dict Composite = {
    "clear": skia.BlendMode.kClear,
    "source": skia.BlendMode.kSrc,
    "dest": skia.BlendMode.kDst,
    "over": skia.BlendMode.kSrcOver,
    "dest_over": skia.BlendMode.kDstOver,
    "in": skia.BlendMode.kSrcIn,
    "dest_in": skia.BlendMode.kDstIn,
    "out": skia.BlendMode.kSrcOut,
    "dest_out": skia.BlendMode.kDstOut,
    "atop": skia.BlendMode.kSrcATop,
    "dest_atop": skia.BlendMode.kDstATop,
    "xor": skia.BlendMode.kXor,
    "add": skia.BlendMode.kPlus,
    "modulate": skia.BlendMode.kModulate,
    "screen": skia.BlendMode.kScreen,
    "overlay": skia.BlendMode.kOverlay,
    "darken": skia.BlendMode.kDarken,
    "lighten": skia.BlendMode.kLighten,
    "color_dodge": skia.BlendMode.kColorDodge,
    "color_burn": skia.BlendMode.kColorBurn,
    "hard_light": skia.BlendMode.kHardLight,
    "soft_light": skia.BlendMode.kSoftLight,
    "difference": skia.BlendMode.kDifference,
    "exclusion": skia.BlendMode.kExclusion,
    "multiply": skia.BlendMode.kMultiply,
    "hue": skia.BlendMode.kHue,
    "saturation": skia.BlendMode.kSaturation,
    "color": skia.BlendMode.kColor,
    "luminosity": skia.BlendMode.kLuminosity,
}

cdef dict StrokeJoin = {
    "miter": skia.Paint.Join.kMiter_Join,
    "round": skia.Paint.Join.kRound_Join,
    "bevel": skia.Paint.Join.kBevel_Join,
}

cdef dict StrokeCap = {
    "butt": skia.Paint.Cap.kButt_Cap,
    "round": skia.Paint.Cap.kRound_Cap,
    "square": skia.Paint.Cap.kSquare_Cap,
}

cdef dict FontWeight = {
    "black": skia.FontStyle.Weight.kBlack_Weight,
    "bold": skia.FontStyle.Weight.kBold_Weight,
    "extra_black": skia.FontStyle.Weight.kExtraBlack_Weight,
    "extra_bold": skia.FontStyle.Weight.kExtraBold_Weight,
    "extra_light": skia.FontStyle.Weight.kExtraLight_Weight,
    "invisible": skia.FontStyle.Weight.kInvisible_Weight,
    "light": skia.FontStyle.Weight.kLight_Weight,
    "medium": skia.FontStyle.Weight.kMedium_Weight,
    "normal": skia.FontStyle.Weight.kNormal_Weight,
    "semi_bold": skia.FontStyle.Weight.kSemiBold_Weight,
    "thin": skia.FontStyle.Weight.kThin_Weight,
}

cdef dict FontWidth = {
    "condensed": skia.FontStyle.Width.kCondensed_Width,
    "expanded": skia.FontStyle.Width.kExpanded_Width,
    "extra_condensed": skia.FontStyle.Width.kExtraCondensed_Width,
    "extra_expanded": skia.FontStyle.Width.kExtraExpanded_Width,
    "normal": skia.FontStyle.Width.kNormal_Width,
    "semi_condensed": skia.FontStyle.Width.kSemiCondensed_Width,
    "semi_expanded": skia.FontStyle.Width.kSemiExpanded_Width,
    "ultra_condensed": skia.FontStyle.Width.kUltraCondensed_Width,
    "ultra_expanded": skia.FontStyle.Width.kUltraExpanded_Width,
}

cdef dict FontSlant = {
    "italic": skia.FontStyle.Slant.kItalic_Slant,
    "oblique": skia.FontStyle.Slant.kOblique_Slant,
    "upright": skia.FontStyle.Slant.kUpright_Slant,
}

cdef dict FilterQuality = {
    "none": skia.FilterQuality.kNone_FilterQuality,
    "low": skia.FilterQuality.kLow_FilterQuality,
    "medium": skia.FilterQuality.kMedium_FilterQuality,
    "high": skia.FilterQuality.kHigh_FilterQuality,
}


def dump_stats():
    if _Durations:
        total_duration = sum(_Durations.values())
        Log.info("Total time spent canvas rendering: %.0fs, comprised of...", total_duration)
        for duration, key in sorted(((duration, key) for (key, duration) in _Durations.items()), reverse=True):
            count = _Counts[key]
            Log.info("%15s  - %8d  x %6.1fµs  = %6.1fs  (%4.1f%%)", key, count, 1e6*duration/count, duration, 100*duration/total_duration)

if _RecordStats:
    import atexit
    atexit.register(dump_stats)


cpdef object load_image(str filename):
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


@cython.cdivision(True)
cdef double turn_angle(double x0, double y0, double x1, double y1, double x2, double y2):
    cdef double xa=x1-x0, ya=y1-y0, xb=x2-x1, yb=y2-y1
    cdef double la=sqrt(xa*xa + ya*ya), lb=sqrt(xb*xb + yb*yb)
    if la == 0 or lb == 0:
        return 0
    return acos(min(max(0, (xa*xb + ya*yb) / (la*lb)), 1)) / TwoPI


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object line_path(object path, Vector points, double curve):
    cdef int i=0, n=points.length-2
    assert points.numbers != NULL
    cdef double last_mid_x, last_mid_y, last_x, last_y, x, y
    lineTo = path.lineTo
    quadTo = path.quadTo
    while i <= n:
        x, y = points.numbers[i], points.numbers[i+1]
        if i == 0:
            path.moveTo(x, y)
        elif curve <= 0:
            lineTo(x, y)
        else:
            mid_x, mid_y = (last_x + x) / 2, (last_y + y) / 2
            if i == 2:
                lineTo(mid_x, mid_y)
            elif curve >= 0.5 or turn_angle(last_mid_x, last_mid_y, last_x, last_y, mid_x, mid_y) <= curve:
                quadTo(last_x, last_y, mid_x, mid_y)
            else:
                lineTo(last_x, last_y)
                lineTo(mid_x, mid_y)
            if i == n:
                lineTo(x, y)
            last_mid_x, last_mid_y = mid_x, mid_y
        last_x, last_y = x, y
        i += 2


cdef object get_color(Node node, default=None):
    cdef Vector v = node._attributes.get('color')
    cdef double w
    if v is not None and v.numbers is not NULL:
        if v.length == 1:
            w = v.numbers[0]
            return skia.Color4f(w, w, w)
        if v.length == 3:
            return skia.Color4f(v.numbers[0], v.numbers[1], v.numbers[2])
        if v.length == 4:
            return skia.Color4f(v.numbers[0], v.numbers[1], v.numbers[2], v.numbers[3])
    return default


cdef object update_context(Node node, ctx):
    cdef str key
    cdef Vector value
    for key, value in node._attributes.items():
        if key == 'translate':
            translate = value.match(2, float)
            if translate is not None:
                ctx.translate(*translate)
        elif key == 'rotate':
            rotate = value.match(1, float)
            if rotate is not None:
                ctx.rotate(rotate * 360)
        elif key == 'scale':
            scale = value.match(2, float)
            if scale is not None:
                ctx.scale(*scale)


cdef object update_paint(Node node, start_paint):
    cdef str key
    cdef Vector value
    paint = start_paint
    for key, value in node._attributes.items():
        if key == 'color':
            color = get_color(node)
            if color is not None:
                if paint is start_paint:
                    paint = skia.Paint(paint)
                paint.setColor4f(color)
                if paint.getShader():
                    paint.setShader(skia.Shaders.Color(color))
        elif key == 'stroke_width':
            stroke_width = value.match(1, float)
            if stroke_width is not None:
                if paint is start_paint:
                    paint = skia.Paint(paint)
                paint.setStrokeWidth(stroke_width)
        elif key == 'stroke_join':
            stroke_join = StrokeJoin.get(value.match(1, str))
            if stroke_join is not None:
                if paint is start_paint:
                    paint = skia.Paint(paint)
                paint.setStrokeJoin(stroke_join)
        elif key == 'stroke_cap':
            stroke_cap = StrokeCap.get(value.match(1, str))
            if stroke_cap is not None:
                if paint is start_paint:
                    paint = skia.Paint(paint)
                paint.setStrokeCap(stroke_cap)
        elif key == 'composite':
            composite = Composite.get(value.match(1, str))
            if composite is not None:
                if paint is start_paint:
                    paint = skia.Paint(paint)
                paint.setBlendMode(composite)
        elif key == 'antialias':
            antialias = value.match(1, bool)
            if antialias is not None:
                if paint is start_paint:
                    paint = skia.Paint(paint)
                paint.setAntiAlias(antialias)
        elif key == 'dither':
            dither = value.match(1, bool)
            if dither is not None:
                if paint is start_paint:
                    paint = skia.Paint(paint)
                paint.setDither(dither)
        elif key == 'quality':
            quality = FilterQuality.get(value.match(1, str))
            if quality is not None:
                if paint is start_paint:
                    paint = skia.Paint(paint)
                paint.setFilterQuality(quality)
    return paint


cdef object update_font(Node node, font):
    if 'font_size' in node._attributes or 'font_family' in node._attributes or 'font_weight' in node._attributes or \
       'font_width' in node._attributes or 'font_slant' in node._attributes:
        typeface = font.getTypeface()
        font_style = typeface.fontStyle()
        font_family = node.get('font_family', 1, str, typeface.getFamilyName())
        font_weight = FontWeight.get(node.get('font_weight', 1, str))
        weight = skia.FontStyle.Weight(font_weight) if font_weight is not None else font_style.weight()
        font_width = FontWidth.get(node.get('font_width', 1, str))
        width = skia.FontStyle.Width(font_width) if font_width is not None else font_style.width()
        font_slant = FontSlant.get(node.get('font_slant', 1, str))
        slant = skia.FontStyle.Slant(font_slant) if font_slant is not None else font_style.slant()
        font_size = node.get('font_size', 1, float, font.getSize())
        return skia.Font(skia.Typeface(font_family, skia.FontStyle(weight, width, slant)), font_size)
    return font


cdef object make_shader(Node node, paint):
    cdef list colors, positions, shaders = []
    cdef int i, nstops
    cdef str noise_type

    cdef Node child = node.first_child
    while child is not None:
        shader = make_shader(child, paint)
        if shader is not None:
            shaders.append(shader)
        child = child.next_sibling

    cdef str kind = node.kind
    if kind == "color":
        color = get_color(node, paint.getColor4f())
        return skia.Shaders.Color(color)

    elif kind == "gradient":
        colors = []
        positions = []
        child = node.first_child
        while child is not None:
            if child.kind == 'stop':
                positions.append(child.get('offset', 1, float))
                colors.append(get_color(child, paint.getColor()))
            child = child.next_sibling
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

    elif kind == "noise":
        frequency = node.get('frequency', 2, float)
        if frequency is not None:
            octaves = node.get('octaves', 1, int, 8)
            seed = node.get('seed', 1, float, 0)
            size = skia.ISize(*node.get('size', 2, int, (0, 0)))
            noise_type = node.get('type', 1, str, "improved")
            if noise_type == "fractal":
                return skia.PerlinNoiseShader.MakeFractalNoise(*frequency, octaves, seed, size)
            elif noise_type == "turbulence":
                return skia.PerlinNoiseShader.MakeTurbulence(*frequency, octaves, seed, size)
            else:
                return skia.PerlinNoiseShader.MakeImprovedNoise(*frequency, octaves, seed)

    elif kind == "blend":
        if len(shaders) == 2:
            ratio = node.get('ratio', 1, float)
            if ratio is not None:
                return skia.Shaders.Lerp(ratio, *shaders)
            mode = Composite.get(node.get('mode', 1, str))
            if mode is not None:
                return skia.Shaders.Blend(skia.BlendMode(mode, *shaders))

    return None


cdef object make_image_filter(Node node, paint):
    cdef list sub_filters = []
    cdef Node child = node.first_child
    while child is not None:
        image_filter = make_image_filter(child, paint)
        if image_filter is not False:
            sub_filters.append(image_filter)
        child = child.next_sibling

    cdef str kind = node.kind
    if kind == "source":
        return None

    elif kind == "blend":
        if len(sub_filters) == 2:
            coefficients = node.get('coefficients', 4, float)
            if coefficients is None:
                ratio = node.get('ratio', 1, float)
                if ratio is not None:
                    coefficients = (0, 1-ratio, ratio, 0)
            if coefficients is not None:
                return skia.ImageFilters.Arithmetic(*coefficients, True, *sub_filters)
            mode = Composite.get(node.get('mode', 1, str))
            if mode is not None:
                return skia.ImageFilters.Xfermode(mode, *sub_filters)

    elif kind == "blur":
        if len(sub_filters) <= 1:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            radius = node.get('radius', 2, float)
            if radius is not None:
                return skia.ImageFilters.Blur(*radius, skia.TileMode.kClamp, input=input_filter)

    elif kind == "shadow":
        if len(sub_filters) <= 1:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            radius = node.get('radius', 2, float)
            if radius is not None:
                offset = node.get('offset', 2, float, (0, 0))
                color = get_color(node, paint.getColor4f())
                shadow_only = node.get('shadow_only', 1, bool, False)
                if shadow_only:
                    return skia.ImageFilters.DropShadowOnly(*offset, *radius, color.toColor(), input=input_filter)
                return skia.ImageFilters.DropShadow(*offset, *radius, color.toColor(), input=input_filter)

    elif kind == "offset":
        if len(sub_filters) <= 1:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            offset = node.get('offset', 2, float)
            if offset is not None:
                return skia.ImageFilters.Offset(*offset, input=input_filter)

    elif kind == "dilate":
        if len(sub_filters) <= 1:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            radius = node.get('radius', 2, float)
            if radius is not None:
                return skia.ImageFilters.Dilate(*radius, input=input_filter)

    elif kind == "erode":
        if len(sub_filters) <= 1:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            radius = node.get('radius', 2, float)
            if radius is not None:
                return skia.ImageFilters.Erode(*radius, input=input_filter)

    elif kind == "paint":
        if len(sub_filters) == 0:
            paint = update_paint(node, paint)
            return skia.ImageFilters.Paint(paint)

    elif kind == "color_matrix":
        if len(sub_filters) <= 1:
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
            if matrix is None and node.keys() & {'brightness', 'contrast', 'saturation'}:
                brightness = node.get('brightness', 1, float, 0)
                contrast = node.get('contrast', 1, float, 1)
                saturation = node.get('saturation', 1, float, 1)
                offset = brightness + (1 - contrast) / 2
                s0 = contrast * (1 - saturation)
                s1 = contrast * saturation
                rs0, gs0, bs0 = s0*0.2126, s0*0.7152, s0*0.0722
                matrix = [rs0+s1, gs0, bs0, 0, offset,
                          rs0, gs0+s1, bs0, 0, offset,
                          rs0, gs0, bs0+s1, 0, offset,
                          0, 0, 0, 1, 0]
            if matrix is not None:
                color_filter = skia.ColorFilters.Matrix(matrix)
                return skia.ImageFilters.ColorFilter(color_filter, input=input_filter)

    return False


cdef object make_path_effect(Node node):
    cdef list sub_path_effects = []
    cdef Node child = node.first_child
    while child is not None:
        path_effect = make_path_effect(child)
        if path_effect is not False:
            sub_path_effects.append(path_effect)
        child = child.next_sibling

    cdef str kind = node.kind
    if kind == "dash":
        if len(sub_path_effects) <= 1:
            intervals = node.get('intervals', 0, float)
            if intervals:
                offset = node.get('offset', 1, float, 0)
                path_effect = skia.DashPathEffect.Make(intervals, offset)
                return path_effect if not sub_path_effects else skia.PathEffect.MakeCompose(path_effect, sub_path_effects[0])

    elif kind == "round_corners":
        if len(sub_path_effects) <= 1:
            radius = node.get('radius', 1, float)
            if radius:
                path_effect = skia.CornerPathEffect.Make(radius)
                return path_effect if not sub_path_effects else skia.PathEffect.MakeCompose(path_effect, sub_path_effects[0])

    elif kind == "jitter":
        if len(sub_path_effects) <= 1:
            length = node.get('length', 1, float)
            deviation = node.get('deviation', 1, float)
            seed = node.get('seed', 1, int, 0)
            if length and deviation:
                path_effect = skia.DiscretePathEffect.Make(length, deviation, seed)
                return path_effect if not sub_path_effects else skia.PathEffect.MakeCompose(path_effect, sub_path_effects[0])

    elif kind == "path_matrix":
        if len(sub_path_effects) <= 1:
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
                path_effect = skia.MatrixPathEffect.Make(skia.Matrix.MakeAll(*matrix))
                return path_effect if not sub_path_effects else skia.PathEffect.MakeCompose(path_effect, sub_path_effects[0])

    elif kind == "sum":
        if len(sub_path_effects) >= 2:
            return functools.reduce(skia.PathEffect.MakeSum, sub_path_effects)

    return None


cpdef object draw(Node node, ctx, paint=None, font=None, path=None):
    start = time.perf_counter()

    cdef Vector points
    cdef Node child
    cdef str kind = node.kind
    if kind == "group":
        ctx.save()
        path = skia.Path()
        update_context(node, ctx)
        group_paint = update_paint(node, paint)
        font = update_font(node, font)
        child = node.first_child
        while child is not None:
            group_paint = draw(child, ctx, group_paint, font, path)
            child = child.next_sibling
        ctx.restore()

    elif kind == "transform":
        ctx.save()
        update_context(node, ctx)
        child = node.first_child
        while child is not None:
            paint = draw(child, ctx, paint, font, path)
            child = child.next_sibling
        ctx.restore()

    elif kind == "paint":
        paint_paint = update_paint(node, paint)
        child = node.first_child
        while child is not None:
            paint_paint = draw(child, ctx, paint_paint, font, path)
            child = child.next_sibling

    elif kind == "font":
        font = update_font(node, font)
        child = node.first_child
        while child is not None:
            paint = draw(child, ctx, paint, font, path)
            child = child.next_sibling

    elif kind == "path":
        path = skia.Path()
        child = node.first_child
        while child is not None:
            paint = draw(child, ctx, paint, font, path)
            child = child.next_sibling

    elif kind == "move_to":
        point = node.get('point', 2, float)
        if point is not None:
            path.moveTo(*point)

    elif kind == "line_to":
        point = node.get('point', 2, float)
        if point is not None:
            path.lineTo(*point)

    elif kind == "curve_to":
        point = node.get('point', 2, float)
        c1 = node.get('c1', 2, float)
        c2 = node.get('c2', 2, float)
        if point is not None and c1 is not None:
            if c2 is not None:
                path.cubicTo(*c1, *c2, *point)
            else:
                path.quadTo(*c1, *point)

    elif kind == "arc":
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

    elif kind == "rect":
        size = node.get('size', 2, float)
        if size is not None:
            point = node.get('point', 2, float, (0, 0))
            path.addRect(*point, point[0]+size[0], point[1]+size[1])

    elif kind == "ellipse":
        radius = node.get('radius', 2, float)
        if radius is not None:
            point = node.get('point', 2, float, (0, 0))
            path.addOval(skia.Rect(point[0]-radius[0], point[1]-radius[1], point[0]+radius[0], point[1]+radius[1]))

    elif kind == "line":
        points = node._attributes.get('points')
        if points is not None and points.numbers is not NULL:
            curve = node.get('curve', 1, float, 0)
            line_path(path, points, curve)
            if node.get('close', 1, bool, False):
                path.close()

    elif kind == "close":
        path.close()

    elif kind == "clip":
        ctx.clipPath(path, skia.ClipOp.kIntersect, paint.isAntiAlias())

    elif kind == "mask":
        ctx.clipPath(path, skia.ClipOp.kDifference, paint.isAntiAlias())

    elif kind == "fill":
        fill_paint = update_paint(node, paint)
        fill_paint.setStyle(skia.Paint.Style.kFill_Style)
        ctx.drawPath(path, fill_paint)

    elif kind == "stroke":
        stroke_paint = update_paint(node, paint)
        stroke_paint.setStyle(skia.Paint.Style.kStroke_Style)
        ctx.drawPath(path, stroke_paint)

    elif kind == "text":
        text = node.get('text', 1, str)
        if text is not None:
            point = node.get('point', 2, float, (0, 0))
            text_paint = update_paint(node, paint)
            font = update_font(node, font)
            stroke = node.get('stroke', 1, bool, False)
            text_paint.setStyle(skia.Paint.Style.kStroke_Style if stroke else skia.Paint.Style.kFill_Style)
            if node.get('center', 1, bool, True):
                bounds = skia.Rect(0, 0, 0, 0)
                font.measureText(text, bounds=bounds)
                ctx.drawString(text, point[0]-bounds.x()-bounds.width()/2, point[1]-bounds.y()-bounds.height()/2, font, text_paint)
            else:
                ctx.drawString(text, *point, font, text_paint)

    elif kind == "image":
        filename = node.get('filename', 1, str)
        if filename:
            image = load_image(filename)
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

    elif kind == "layer":
        size = node.get('size', 2, float)
        if size is not None:
            alpha = node.get('alpha', 1, float, 1)
            origin = node.get('origin', 2, float, (0, 0))
            rect = skia.Rect.MakeXYWH(*origin, *size)
            ctx.clipRect(rect)
            ctx.saveLayerAlpha(rect, int(alpha * 255))
            path = skia.Path()
            update_context(node, ctx)
            layer_paint = update_paint(node, paint)
            font = update_font(node, font)
            child = node.first_child
            while child is not None:
                layer_paint = draw(child, ctx, layer_paint, font, path)
                child = child.next_sibling
            ctx.restore()

    elif kind == "canvas":
        ctx.save()
        update_context(node, ctx)
        paint = update_paint(node, skia.Paint(AntiAlias=True) if paint is None else paint)
        font = update_font(node, skia.Font(skia.Typeface(), 14) if font is None else font)
        path = skia.Path()
        child = node.first_child
        while child is not None:
            paint = draw(child, ctx, paint, font, path)
            child = child.next_sibling
        ctx.restore()

    else:
        shader = make_shader(node, paint)
        if shader:
            paint = skia.Paint(paint)
            paint.setShader(shader)
        else:
            image_filter = make_image_filter(node, paint)
            if image_filter:
                paint = skia.Paint(paint)
                paint.setImageFilter(image_filter)
            else:
                path_effect = make_path_effect(node)
                if path_effect:
                    paint = skia.Paint(paint)
                    paint.setPathEffect(path_effect)

    if _RecordStats:
        duration = time.perf_counter() - start
        _Counts[kind] = _Counts.get(kind, 0) + 1
        _Durations[kind] = _Durations.get(kind, 0) + duration
        if kind != 'canvas':
            parent_kind = node.parent.kind
            _Durations[parent_kind] = _Durations.get(parent_kind, 0) - duration

    return paint
