# cython: language_level=3, profile=True

"""
Flitter drawing canvas based on Skia
"""

import array
import functools
from pathlib import Path
import time

from cpython cimport array
import cython
from libc.math cimport acos, sqrt
from loguru import logger
import skia

from ..model cimport Vector, Node


cdef double TwoPI = 6.283185307179586
cdef dict _ImageCache = {}


cdef dict Composite = {
    'clear': skia.BlendMode.kClear,
    'source': skia.BlendMode.kSrc,
    'dest': skia.BlendMode.kDst,
    'over': skia.BlendMode.kSrcOver,
    'dest_over': skia.BlendMode.kDstOver,
    'in': skia.BlendMode.kSrcIn,
    'dest_in': skia.BlendMode.kDstIn,
    'out': skia.BlendMode.kSrcOut,
    'dest_out': skia.BlendMode.kDstOut,
    'atop': skia.BlendMode.kSrcATop,
    'dest_atop': skia.BlendMode.kDstATop,
    'xor': skia.BlendMode.kXor,
    'add': skia.BlendMode.kPlus,
    'modulate': skia.BlendMode.kModulate,
    'screen': skia.BlendMode.kScreen,
    'overlay': skia.BlendMode.kOverlay,
    'darken': skia.BlendMode.kDarken,
    'lighten': skia.BlendMode.kLighten,
    'color_dodge': skia.BlendMode.kColorDodge,
    'color_burn': skia.BlendMode.kColorBurn,
    'hard_light': skia.BlendMode.kHardLight,
    'soft_light': skia.BlendMode.kSoftLight,
    'difference': skia.BlendMode.kDifference,
    'exclusion': skia.BlendMode.kExclusion,
    'multiply': skia.BlendMode.kMultiply,
    'hue': skia.BlendMode.kHue,
    'saturation': skia.BlendMode.kSaturation,
    'color': skia.BlendMode.kColor,
    'luminosity': skia.BlendMode.kLuminosity,
}

cdef dict StrokeJoin = {
    'miter': skia.Paint.Join.kMiter_Join,
    'round': skia.Paint.Join.kRound_Join,
    'bevel': skia.Paint.Join.kBevel_Join,
}

cdef dict StrokeCap = {
    'butt': skia.Paint.Cap.kButt_Cap,
    'round': skia.Paint.Cap.kRound_Cap,
    'square': skia.Paint.Cap.kSquare_Cap,
}

cdef dict FontWeight = {
    'black': skia.FontStyle.Weight.kBlack_Weight,
    'bold': skia.FontStyle.Weight.kBold_Weight,
    'extra_black': skia.FontStyle.Weight.kExtraBlack_Weight,
    'extra_bold': skia.FontStyle.Weight.kExtraBold_Weight,
    'extra_light': skia.FontStyle.Weight.kExtraLight_Weight,
    'invisible': skia.FontStyle.Weight.kInvisible_Weight,
    'light': skia.FontStyle.Weight.kLight_Weight,
    'medium': skia.FontStyle.Weight.kMedium_Weight,
    'normal': skia.FontStyle.Weight.kNormal_Weight,
    'semi_bold': skia.FontStyle.Weight.kSemiBold_Weight,
    'thin': skia.FontStyle.Weight.kThin_Weight,
}

cdef dict FontWidth = {
    'condensed': skia.FontStyle.Width.kCondensed_Width,
    'expanded': skia.FontStyle.Width.kExpanded_Width,
    'extra_condensed': skia.FontStyle.Width.kExtraCondensed_Width,
    'extra_expanded': skia.FontStyle.Width.kExtraExpanded_Width,
    'normal': skia.FontStyle.Width.kNormal_Width,
    'semi_condensed': skia.FontStyle.Width.kSemiCondensed_Width,
    'semi_expanded': skia.FontStyle.Width.kSemiExpanded_Width,
    'ultra_condensed': skia.FontStyle.Width.kUltraCondensed_Width,
    'ultra_expanded': skia.FontStyle.Width.kUltraExpanded_Width,
}

cdef dict FontSlant = {
    'italic': skia.FontStyle.Slant.kItalic_Slant,
    'oblique': skia.FontStyle.Slant.kOblique_Slant,
    'upright': skia.FontStyle.Slant.kUpright_Slant,
}

cdef dict FilterQuality = {
    'none': skia.FilterQuality.kNone_FilterQuality,
    'low': skia.FilterQuality.kLow_FilterQuality,
    'medium': skia.FilterQuality.kMedium_FilterQuality,
    'high': skia.FilterQuality.kHigh_FilterQuality,
}


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
            logger.info("Read image file {}", filename)
        except Exception:
            logger.exception("Unexpected error opening file {}", filename)
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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef object line_path(object path, Vector points, double curve, bint close):
    cdef int nverbs = 0, npoints = 0
    cdef array.array points_array = array.array('f')
    array.resize(points_array, len(points)*2)
    cdef float[:] ppoints = points_array
    cdef array.array verbs_array = array.array('B')
    array.resize(verbs_array, len(points) + 4)
    cdef unsigned char[:] pverbs = verbs_array
    cdef int i=0, n=points.length-2
    cdef double last_mid_x, last_mid_y, last_x, last_y, x, y
    while i <= n:
        x, y = points.numbers[i], points.numbers[i+1]
        if i == 0:
            pverbs[nverbs] = 0
            nverbs += 1
            ppoints[npoints] = x
            ppoints[npoints+1] = y
            npoints += 2
        elif curve <= 0:
            pverbs[nverbs] = 1
            nverbs += 1
            ppoints[npoints] = x
            ppoints[npoints+1] = y
            npoints += 2
        else:
            mid_x, mid_y = (last_x + x) / 2, (last_y + y) / 2
            if i == 2:
                pverbs[nverbs] = 1
                nverbs += 1
                ppoints[npoints] = mid_x
                ppoints[npoints+1] = mid_y
                npoints += 2
            elif curve >= 0.5 or turn_angle(last_mid_x, last_mid_y, last_x, last_y, mid_x, mid_y) <= curve:
                pverbs[nverbs] = 2
                nverbs += 1
                ppoints[npoints] = last_x
                ppoints[npoints+1] = last_y
                npoints += 2
                pverbs[nverbs] = 1
                ppoints[npoints] = mid_x
                ppoints[npoints+1] = mid_y
                npoints += 2
            else:
                pverbs[nverbs] = 1
                nverbs += 1
                ppoints[npoints] = last_x
                ppoints[npoints+1] = last_y
                npoints += 2
                pverbs[nverbs] = 1
                ppoints[npoints] = mid_x
                ppoints[npoints+1] = mid_y
                npoints += 2
            if i == n:
                pverbs[nverbs] = 1
                nverbs += 1
                ppoints[npoints] = x
                ppoints[npoints+1] = y
                npoints += 2
            last_mid_x, last_mid_y = mid_x, mid_y
        last_x, last_y = x, y
        i += 2
    if close:
        pverbs[nverbs] = 5
        nverbs += 1
    cdef bytearray data = bytearray()
    cdef array.array header_array = array.array('I')
    array.resize(header_array, 4)
    cdef unsigned int[:] header = header_array
    header[0] = 5
    header[1] = npoints / 2
    header[2] = 0
    header[3] = nverbs
    data.extend(header_array)
    array.resize(points_array, npoints)
    data.extend(points_array)
    if nverbs % 4 != 0:
        nverbs += 4 - nverbs % 4
    array.resize(verbs_array, nverbs)
    data.extend(verbs_array)
    cdef int processed
    if path.isEmpty():
        processed = path.readFromMemory(data)
    else:
        new_path = skia.Path()
        processed = new_path.readFromMemory(data)
        path.addPath(new_path)
    if processed != len(data):
        raise ValueError("Bad path data")


cdef object get_color(Node node, default=None):
    cdef Vector v = node._attributes.get('color')
    cdef unsigned int r, g, b, a, color
    if v is not None and v.numbers != NULL:
        if v.length == 1:
            r = min(max(0, <int>(v.numbers[0] * 255)), 255)
            return <unsigned int>0xff000000 + r * 0x010101
        if v.length == 3:
            r = min(max(0, <unsigned int>(v.numbers[0] * 255)), 255)
            g = min(max(0, <unsigned int>(v.numbers[1] * 255)), 255)
            b = min(max(0, <unsigned int>(v.numbers[2] * 255)), 255)
            return <unsigned int>0xff000000 + (r << 16) + (g << 8) + b
        if v.length == 4:
            a = min(max(0, <unsigned int>(v.numbers[3] * 255)), 255)
            r = min(max(0, <unsigned int>(v.numbers[0] * a)), 255)
            g = min(max(0, <unsigned int>(v.numbers[1] * a)), 255)
            b = min(max(0, <unsigned int>(v.numbers[2] * a)), 255)
            return (a << 24) + (r << 16) + (g << 8) + b
    return default


cdef object update_context(Node node, ctx):
    cdef str key
    cdef Vector value
    for key, value in node._attributes.items():
        if key == 'translate':
            if (translate := value.match(2, float)) is not None:
                ctx.translate(*translate)
        elif key == 'rotate':
            if (rotate := value.match(1, float)) is not None:
                ctx.rotate(rotate * 360)
        elif key == 'scale':
            if (scale := value.match(2, float)) is not None:
                ctx.scale(*scale)


cdef object update_paint(Node node, start_paint):
    cdef str key
    cdef Vector value
    paint = start_paint
    for key, value in node._attributes.items():
        if key == 'color':
            if (color := get_color(node)) is not None:
                if paint is start_paint:
                    paint = skia.Paint(paint)
                paint.setColor(color)
                if paint.getShader():
                    paint.setShader(skia.Shaders.Color(color))
        elif key == 'stroke_width':
            if (stroke_width := value.match(1, float)) is not None:
                if paint is start_paint:
                    paint = skia.Paint(paint)
                paint.setStrokeWidth(stroke_width)
        elif key == 'stroke_join':
            if (stroke_join := StrokeJoin.get(value.match(1, str))) is not None:
                if paint is start_paint:
                    paint = skia.Paint(paint)
                paint.setStrokeJoin(stroke_join)
        elif key == 'stroke_cap':
            if (stroke_cap := StrokeCap.get(value.match(1, str))) is not None:
                if paint is start_paint:
                    paint = skia.Paint(paint)
                paint.setStrokeCap(stroke_cap)
        elif key == 'composite':
            if (composite := Composite.get(value.match(1, str))) is not None:
                if paint is start_paint:
                    paint = skia.Paint(paint)
                paint.setBlendMode(composite)
        elif key == 'antialias':
            if (antialias := value.match(1, bool)) is not None:
                if paint is start_paint:
                    paint = skia.Paint(paint)
                paint.setAntiAlias(antialias)
        elif key == 'dither':
            if (dither := value.match(1, bool)) is not None:
                if paint is start_paint:
                    paint = skia.Paint(paint)
                paint.setDither(dither)
        elif key == 'quality':
            if (quality := FilterQuality.get(value.match(1, str))) is not None:
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
    if kind == 'color':
        color = get_color(node, paint.getColor())
        return skia.Shaders.Color(color)

    elif kind == 'gradient':
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
            if (start := node.get('start', 2, float)) is not None and (end := node.get('end', 2, float)) is not None:
                points = [skia.Point(*start), skia.Point(*end)]
                return skia.GradientShader.MakeLinear(points, colors, positions)
            if (radius := node.get('radius', 2, float)) is not None:
                rotate = node.get('rotate', 1, float, 0)
                matrix = skia.Matrix.Scale(*radius).postRotate(rotate * 360).postTranslate(*node.get('point', 2, float, (0, 0)))
                return skia.GradientShader.MakeRadial(skia.Point(0, 0), 1, colors, positions, localMatrix=matrix)

    elif kind == 'noise':
        if (frequency := node.get('frequency', 2, float)) is not None:
            octaves = node.get('octaves', 1, int, 8)
            seed = node.get('seed', 1, float, 0)
            size = skia.ISize(*node.get('size', 2, int, (0, 0)))
            noise_type = node.get('type', 1, str, 'improved')
            if noise_type == 'fractal':
                return skia.PerlinNoiseShader.MakeFractalNoise(*frequency, octaves, seed, size)
            elif noise_type == 'turbulence':
                return skia.PerlinNoiseShader.MakeTurbulence(*frequency, octaves, seed, size)
            else:
                return skia.PerlinNoiseShader.MakeImprovedNoise(*frequency, octaves, seed)

    elif kind == 'blend':
        if len(shaders) == 2:
            if (ratio := node.get('ratio', 1, float)) is not None:
                return skia.Shaders.Lerp(ratio, *shaders)
            if (mode := Composite.get(node.get('mode', 1, str))) is not None:
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
    if kind == 'source':
        return None

    elif kind == 'blend':
        if len(sub_filters) == 2:
            if (coefficients := node.get('coefficients', 4, float)) is not None:
                return skia.ImageFilters.Arithmetic(*coefficients, True, *sub_filters)
            if (ratio := node.get('ratio', 1, float)) is not None:
                return skia.ImageFilters.Arithmetic(0, 1-ratio, ratio, 0, True, *sub_filters)
            if (mode := Composite.get(node.get('mode', 1, str))) is not None:
                return skia.ImageFilters.Xfermode(mode, *sub_filters)

    elif kind == 'blur':
        if len(sub_filters) <= 1 and (radius := node.get('radius', 2, float)) is not None:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            return skia.ImageFilters.Blur(*radius, skia.TileMode.kClamp, input=input_filter)

    elif kind == 'shadow':
        if len(sub_filters) <= 1 and (radius := node.get('radius', 2, float)) is not None:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            offset = node.get('offset', 2, float, (0, 0))
            color = get_color(node, paint.getColor())
            if node.get('shadow_only', 1, bool, False):
                return skia.ImageFilters.DropShadowOnly(*offset, *radius, color, input=input_filter)
            return skia.ImageFilters.DropShadow(*offset, *radius, color, input=input_filter)

    elif kind == 'offset':
        if len(sub_filters) <= 1 and (offset := node.get('offset', 2, float)) is not None:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            return skia.ImageFilters.Offset(*offset, input=input_filter)

    elif kind == 'dilate':
        if len(sub_filters) <= 1 and (radius := node.get('radius', 2, float)) is not None:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            return skia.ImageFilters.Dilate(*radius, input=input_filter)

    elif kind == 'erode':
        if len(sub_filters) <= 1 and (radius := node.get('radius', 2, float)) is not None:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            return skia.ImageFilters.Erode(*radius, input=input_filter)

    elif kind == 'paint':
        if len(sub_filters) == 0:
            paint = update_paint(node, paint)
            return skia.ImageFilters.Paint(paint)

    elif kind == 'color_matrix':
        if len(sub_filters) <= 1:
            input_filter = sub_filters[0] if len(sub_filters) == 1 else None
            matrix = node.get('matrix', 20, float)
            if matrix is None:
                if node.keys() & {'red', 'green', 'blue', 'alpha'}:
                    red = node.get('red', 5, float, [1, 0, 0, 0, 0])
                    green = node.get('green', 5, float, [0, 1, 0, 0, 0])
                    blue = node.get('blue', 5, float, [0, 0, 1, 0, 0])
                    alpha = node.get('alpha', 5, float, [0, 0, 0, 1, 0])
                    matrix = red + green + blue + alpha
                elif node.keys() & {'scale', 'offset'}:
                    scale = node.get('scale', 3, float, [1, 1, 1])
                    offset = node.get('offset', 3, float, [0, 0, 0])
                    matrix = [scale[0], 0, 0, 0, offset[0],
                              0, scale[1], 0, 0, offset[1],
                              0, 0, scale[2], 0, offset[2],
                              0, 0, 0, 1, 0]
                elif node.keys() & {'brightness', 'contrast', 'saturation'}:
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
    if kind == 'dash':
        if len(sub_path_effects) <= 1 and (intervals := node.get('intervals', 0, float)):
            offset = node.get('offset', 1, float, 0)
            path_effect = skia.DashPathEffect.Make(intervals, offset)
            return path_effect if not sub_path_effects else skia.PathEffect.MakeCompose(path_effect, sub_path_effects[0])

    elif kind == 'round_corners':
        if len(sub_path_effects) <= 1 and (radius := node.get('radius', 1, float)):
            path_effect = skia.CornerPathEffect.Make(radius)
            return path_effect if not sub_path_effects else skia.PathEffect.MakeCompose(path_effect, sub_path_effects[0])

    elif kind == 'jitter':
        if len(sub_path_effects) <= 1 and (length := node.get('length', 1, float)) and (deviation := node.get('deviation', 1, float)):
            seed = node.get('seed', 1, int, 0)
            path_effect = skia.DiscretePathEffect.Make(length, deviation, seed)
            return path_effect if not sub_path_effects else skia.PathEffect.MakeCompose(path_effect, sub_path_effects[0])

    elif kind == 'path_matrix':
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

    elif kind == 'sum':
        if len(sub_path_effects) >= 2:
            return functools.reduce(skia.PathEffect.MakeSum, sub_path_effects)

    return None


cpdef object draw(Node node, ctx, paint=None, font=None, path=None, dict stats=None):
    cdef double start_time = time.perf_counter()
    cdef Vector points
    cdef Node child
    cdef str kind = node.kind
    cdef double x, y, rx, ry

    if kind == 'group':
        ctx.save()
        path = skia.Path()
        update_context(node, ctx)
        group_paint = update_paint(node, paint)
        font = update_font(node, font)
        child = node.first_child
        while child is not None:
            group_paint = draw(child, ctx, group_paint, font, path, stats)
            child = child.next_sibling
        ctx.restore()

    elif kind == 'transform':
        ctx.save()
        update_context(node, ctx)
        child = node.first_child
        while child is not None:
            paint = draw(child, ctx, paint, font, path, stats)
            child = child.next_sibling
        ctx.restore()

    elif kind == 'paint':
        paint_paint = update_paint(node, paint)
        child = node.first_child
        while child is not None:
            paint_paint = draw(child, ctx, paint_paint, font, path, stats)
            child = child.next_sibling

    elif kind == 'font':
        font = update_font(node, font)
        child = node.first_child
        while child is not None:
            paint = draw(child, ctx, paint, font, path, stats)
            child = child.next_sibling

    elif kind == 'path':
        path = skia.Path()
        child = node.first_child
        while child is not None:
            paint = draw(child, ctx, paint, font, path, stats)
            child = child.next_sibling

    elif kind == 'move_to':
        if (point := node.get('point', 2, float)) is not None:
            path.moveTo(*point)

    elif kind == 'line_to':
        if (point := node.get('point', 2, float)) is not None:
            path.lineTo(*point)

    elif kind == 'curve_to':
        if (point := node.get('point', 2, float)) is not None and (c1 := node.get('c1', 2, float)) is not None:
            if c2 := node.get('c2', 2, float) is not None:
                path.cubicTo(*c1, *c2, *point)
            else:
                path.quadTo(*c1, *point)

    elif kind == 'arc_to':
        if (point := node.get('point', 2, float)) is not None and (radius := node.get('radius', 2, float)) is not None:
            x, y = point
            rx, ry = radius
            rotate = node.get('rotation', 1, float, 0)
            large = node.get('large', 1, bool, False)
            ccw = node.get('ccw', 1, bool, False)
            path.arcTo(rx, ry, rotate, large, ccw, x, y)

    elif kind == 'arc':
        if (point := node.get('point', 2, float, (0, 0))) is not None and (radius := node.get('radius', 2, float)) is not None:
            x, y = point
            rx, ry = radius
            start = node.get('start', 1, float, 0)
            if (sweep := node.get('sweep', 1, float)) is None:
                end = node.get('end', 1, float)
                if end is not None:
                    sweep = end - start
            if sweep is not None:
                move_to = node.get('move_to', 1, bool, False)
                path.arcTo(skia.Rect(x-rx, y-ry, x+rx, y+ry), start*360, sweep*360, move_to)

    elif kind == 'rect':
        if (size := node.get('size', 2, float)) is not None:
            rx, ry = size
            x, y = node.get('point', 2, float, (0, 0))
            path.addRect(x, y, x+rx, y+ry)

    elif kind == 'ellipse':
        if (radius := node.get('radius', 2, float)) is not None:
            rx, ry = radius
            x, y = node.get('point', 2, float, (0, 0))
            path.addOval(skia.Rect(x-rx, y-ry, x+rx, y+ry))

    elif kind == 'line':
        if (points := node._attributes.get('points')) is not None and points.numbers is not NULL:
            curve = node.get('curve', 1, float, 0)
            close = node.get('close', 1, bool, False)
            line_path(path, points, curve, close)

    elif kind == 'close':
        path.close()

    elif kind == 'clip':
        ctx.clipPath(path, skia.ClipOp.kIntersect, paint.isAntiAlias())

    elif kind == 'mask':
        ctx.clipPath(path, skia.ClipOp.kDifference, paint.isAntiAlias())

    elif kind == 'fill':
        fill_paint = update_paint(node, paint)
        fill_paint.setStyle(skia.Paint.Style.kFill_Style)
        ctx.drawPath(path, fill_paint)

    elif kind == 'stroke':
        stroke_paint = update_paint(node, paint)
        stroke_paint.setStyle(skia.Paint.Style.kStroke_Style)
        ctx.drawPath(path, stroke_paint)

    elif kind == 'text':
        if (text := node.get('text', 1, str)) is not None:
            x, y = node.get('point', 2, float, (0, 0))
            text_paint = update_paint(node, paint)
            font = update_font(node, font)
            stroke = node.get('stroke', 1, bool, False)
            text_paint.setStyle(skia.Paint.Style.kStroke_Style if stroke else skia.Paint.Style.kFill_Style)
            if node.get('center', 1, bool, True):
                bounds = skia.Rect(0, 0, 0, 0)
                font.measureText(text, bounds=bounds)
                rx = bounds.x() + bounds.width()/2
                ry = bounds.y() + bounds.height()/2
                ctx.drawString(text, x-rx, y-ry, font, text_paint)
            else:
                ctx.drawString(text, x, y, font, text_paint)

    elif kind == 'image':
        if (filename := node.get('filename', 1, str)) and (image := load_image(filename)) is not None:
            width, height = image.width(), image.height()
            point = node.get('point', 2, float, (0, 0))
            if (fill := node.get('fill', 2, float)) is not None:
                aspect = fill[0] / fill[1]
                if width/height > aspect:
                    w = height * aspect
                    src = skia.Rect.MakeXYWH((width-w)/2, 0, w, height)
                else:
                    h = width / aspect
                    src = skia.Rect.MakeXYWH(0, (height-h)/2, width, h)
                dst = skia.Rect.MakeXYWH(*point, *fill)
            elif (fit := node.get('fit', 2, float)) is not None:
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
                size = node.get('size', 2, float, (width, height))
                src = skia.Rect.MakeXYWH(0, 0, width, height)
                dst = skia.Rect.MakeXYWH(*point, *size)
            ctx.drawImageRect(image, src, dst, paint)

    elif kind == 'layer':
        if (size := node.get('size', 2, float)) is not None:
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
                layer_paint = draw(child, ctx, layer_paint, font, path, stats)
                child = child.next_sibling
            ctx.restore()

    elif kind == 'canvas':
        ctx.save()
        update_context(node, ctx)
        paint = update_paint(node, skia.Paint(AntiAlias=True) if paint is None else paint)
        font = update_font(node, skia.Font(skia.Typeface(), 14) if font is None else font)
        path = skia.Path()
        child = node.first_child
        while child is not None:
            paint = draw(child, ctx, paint, font, path, stats)
            child = child.next_sibling
        ctx.restore()

    elif shader := make_shader(node, paint):
        paint = skia.Paint(paint)
        paint.setShader(shader)

    elif image_filter := make_image_filter(node, paint):
        paint = skia.Paint(paint)
        paint.setImageFilter(image_filter)

    elif path_effect := make_path_effect(node):
        paint = skia.Paint(paint)
        paint.setPathEffect(path_effect)

    cdef double duration, total
    cdef int count
    cdef str parent_kind
    if stats is not None:
        duration = time.perf_counter() - start_time
        count, total = stats.get(kind, (0, 0))
        stats[kind] = count+1, total+duration
        if kind != 'canvas':
            parent_kind = node.parent.kind
            count, total = stats.get(parent_kind, (0, 0))
            stats[parent_kind] = count, total-duration

    return paint
