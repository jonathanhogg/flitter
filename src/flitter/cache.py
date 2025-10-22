"""
General file cache
"""

import fractions
from os import stat
from pathlib import Path
from queue import Queue, Full
import threading

from loguru import logger

from .clock import system_clock


DEFAULT_CLEAN_TIME = 30
MAX_CACHE_VIDEO_FRAMES = 100


class CachePath:
    def __init__(self, path, absolute):
        self._path = path
        self._absolute = absolute
        self._cache = {}
        self._mtime = None
        self.check_unmodified()

    @property
    def suffix(self):
        return self._path.suffix

    @property
    def parent(self):
        return self._path.parent

    def exists(self):
        return self._path.exists()

    def cleanup(self):
        full_collect = False
        while self._cache:
            key, value = self._cache.popitem()
            if key[0] == 'video':
                container, decoder, frames = value
                if decoder is not None:
                    decoder.close()
                if container is not None:
                    container.close()
                    logger.debug("Closing video: {}", self._path)
                full_collect = True
            elif key[0] == 'video_output':
                writer, queue, start = value
                if queue is not None:
                    queue.put(None)
                    writer.join()
        return full_collect

    def check_unmodified(self):
        self._touched = system_clock()
        try:
            mtime = stat(self._absolute).st_mtime
        except FileNotFoundError:
            mtime = None
        if mtime == self._mtime:
            return True
        self._mtime = mtime
        self.cleanup()
        return False

    def read_text(self, encoding=None, errors=None):
        key = 'text', encoding, errors
        if self.check_unmodified() and (text := self._cache.get(key, False)) is not False:
            return text
        if self._mtime is None:
            logger.warning("File not found: {}", self._path)
            text = None
        else:
            try:
                text = self._path.read_text(encoding, errors)
            except Exception as exc:
                logger.opt(exception=exc).warning("Error reading text: {}", self._path)
                text = None
            else:
                logger.debug("Read text: {}", self._path)
        self._cache[key] = text
        return text

    def read_bytes(self):
        key = 'bytes',
        if self.check_unmodified() and (data := self._cache.get(key, False)) is not False:
            return data
        if self._mtime is None:
            logger.warning("File not found: {}", self._path)
            data = None
        else:
            try:
                data = self._path.read_bytes()
            except Exception as exc:
                logger.opt(exception=exc).warning("Error reading bytes: {}", self._path)
                data = None
            else:
                logger.debug("Read bytes: {}", self._path)
        self._cache[key] = data
        return data

    def read_flitter_top(self):
        key = 'flitter_top'
        current_top = self._cache.get(key, False)
        if self.check_unmodified() and current_top is not False:
            return current_top
        top = None
        if self._mtime is None:
            if current_top is False:
                logger.warning("Program not found: {}", self._path)
            elif current_top is not None:
                logger.error("Program disappeared: {}", self._path)
        else:
            from .language.parser import parse, ParseError
            try:
                parse_time = -system_clock()
                source = self._path.read_text(encoding='utf8')
                top = parse(source)
                parse_time += system_clock()
                logger.debug("Parsed '{}' in {:.1f}ms", self._path, parse_time*1000)
            except ParseError as exc:
                logger.error("Error parsing {} at line {} column {}:\n{}",
                             self._path.name, exc.line, exc.column, exc.context)
            except Exception as exc:
                logger.opt(exception=exc).error("Error reading program: {}", self._path)
        self._cache[key] = top
        return top

    def read_flitter_program(self, static=None, dynamic=None, simplify=True):
        key = 'flitter_program', tuple(sorted(static.items())) if static else None, tuple(dynamic) if dynamic else (), simplify
        current_program = self._cache.get(key, False)
        if self.check_unmodified() and current_program is not False:
            for path in current_program.top.dependencies if current_program is not None else ():
                if not path.check_unmodified():
                    break
            else:
                return current_program
        if self._mtime is None:
            if current_program is False:
                logger.warning("Program not found: {}", self._path)
                program = current_program = None
            elif current_program is not None:
                logger.error("Program disappeared: {}", self._path)
                program = current_program
        else:
            if current_program is False:
                current_program = None
            from .language.parser import parse, ParseError
            try:
                parse_time = -system_clock()
                source = self._path.read_text(encoding='utf8')
                initial_top = parse(source)
                now = system_clock()
                parse_time += now
                simplify_time = -now
                top, context = initial_top.simplify(static=static, dynamic=dynamic, path=self, return_context=True) if simplify else (initial_top, None)
                now = system_clock()
                simplify_time += now
                compile_time = -now
                program = top.compile(initial_lnames=tuple(dynamic) if dynamic else (), initial_errors=context.errors if simplify else None)
                program.set_top(top)
                program.set_path(self)
                program.use_simplifier(simplify)
                compile_time += system_clock()
                logger.debug("Compiled '{}' to {} instructions in {:.1f}/{:.1f}/{:.1f}ms",
                             self._path, len(program), parse_time*1000, simplify_time*1000, compile_time*1000)
            except ParseError as exc:
                if current_program is None:
                    logger.error("Error parsing {} at line {} column {}:\n{}",
                                 self._path.name, exc.line, exc.column, exc.context)
                else:
                    logger.warning("Unable to re-parse {}, error at line {} column {}:\n{}",
                                   self._path.name, exc.line, exc.column, exc.context)
                program = current_program
            except Exception as exc:
                logger.opt(exception=exc).error("Error reading program: {}", self._path)
                program = current_program
        self._cache[key] = program
        return program

    def read_csv_vector(self, row_number):
        key = 'csv_vector',
        import csv
        from .model import Vector, null
        if self.check_unmodified() and (cached := self._cache.get(key, False)) is not False:
            reader, rows = cached
        elif self._mtime is None:
            logger.warning("File not found: {}", self._path)
            reader = None
            rows = []
        else:
            try:
                reader = csv.reader(self._path.open(newline='', encoding='utf8'))
                logger.debug("Opened CSV file: {}", self._path)
            except Exception as exc:
                logger.opt(exception=exc).warning("Error reading CSV: {}", self._path)
                reader = None
            rows = []
        while reader is not None and row_number >= len(rows):
            try:
                row = next(reader)
                values = []
                for value in row:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    values.append(value)
                while values and values[-1] == '':
                    values.pop()
                rows.append(Vector.coerce(values))
            except StopIteration:
                logger.debug("Closed CSV file: {}", self._path)
                reader = None
                break
            except Exception as exc:
                logger.opt(exception=exc).warning("Error reading CSV: {}", self._path)
                reader = None
        self._cache[key] = reader, rows
        if 0 <= row_number < len(rows):
            return rows[row_number]
        return null

    def read_pil_image(self):
        key = 'pil_image',
        if self.check_unmodified() and (image := self._cache.get(key, False)) is not False:
            return image
        import PIL.Image
        if self._mtime is None:
            logger.warning("File not found: {}", self._path)
            image = None
        else:
            try:
                image = PIL.Image.open(str(self._path))
            except Exception as exc:
                logger.opt(exception=exc).warning("Error reading image: {}", self._path)
                image = None
            else:
                logger.debug("Read image file: {}", self._path)
        self._cache[key] = image
        return image

    def read_skia_image(self):
        key = 'skia_image',
        if self.check_unmodified() and (image := self._cache.get(key, False)) is not False:
            return image
        import skia
        if self._mtime is None:
            logger.warning("File not found: {}", self._path)
            image = None
        else:
            try:
                image = skia.Image.open(str(self._path))
            except Exception as exc:
                logger.opt(exception=exc).warning("Error reading image: {}", self._path)
                image = None
            else:
                logger.debug("Read image file: {}", self._path)
        self._cache[key] = image
        return image

    def read_video_frames(self, obj, position, loop=False, back_and_forth=False, trim_start=0, trim_end=0, threading=False):
        key = 'video', id(obj), threading
        container = decoder = current_frame = next_frame = None
        frames = []
        ratio = 0
        if self.check_unmodified() and (cached := self._cache.get(key, False)) is not False:
            container, decoder, frames = cached
        elif self._mtime is None:
            logger.warning("File not found: {}", self._path)
        elif position is not None:
            try:
                import av
                container = av.container.open(str(self._path))
                stream = container.streams.video[0]
                if threading:
                    stream.thread_type = 'AUTO'
                ctx = stream.codec_context
                logger.debug("Opened {}x{} {} {:.0f}fps video file: {}", ctx.width, ctx.height, ctx.name, float(stream.average_rate), self._path)
            except Exception as exc:
                logger.opt(exception=exc).warning("Error reading video: {}", self._path)
                container = None
        if container is not None:
            if position is None:
                logger.debug("Closing video: {}", self._path)
                if decoder is not None:
                    decoder.close()
                container.close()
                container = decoder = None
                frames = []
            else:
                try:
                    stream = container.streams.video[0]
                    start = stream.start_time + int(trim_start / stream.time_base)
                    duration = stream.duration - int((trim_start + trim_end) / stream.time_base)
                    pos = int(position / stream.time_base)
                    if loop:
                        pd = pos % duration
                        if back_and_forth and pos // duration % 2 == 1:
                            pd = duration - pd
                        timestamp = start + pd
                    else:
                        timestamp = start + min(max(0, pos), duration)
                    count = 0
                    while True:
                        if len(frames) >= 2 and timestamp >= frames[0].pts and (frames[-1] is None or timestamp < frames[-1].pts):
                            break
                        if frames and timestamp < frames[0].pts:
                            frames = []
                        if len(frames) >= 2 and frames[-1].key_frame and timestamp > 2*frames[-1].pts - frames[0].pts:
                            frames = []
                        if not frames:
                            if decoder is not None:
                                decoder.close()
                            logger.trace("Seek video {} to {:.2f}s", self._path, float(timestamp * stream.time_base))
                            container.seek(timestamp, stream=stream)
                            decoder = container.decode(streams=(stream.index,))
                        if decoder is not None:
                            try:
                                frame = next(decoder)
                                if frame.key_frame:
                                    for i in range(len(frames)-1, 0, -1):
                                        if frames[i].key_frame:
                                            frames = frames[i:]
                                            break
                                count += 1
                            except StopIteration:
                                logger.trace("Hit end of video {}", self._path)
                                decoder = None
                                frame = None
                            if len(frames) == MAX_CACHE_VIDEO_FRAMES:
                                frames.pop(0)
                            frames.append(frame)
                            if len(frames) == 1:
                                logger.trace("Decoding frames from {:.2f}s", float(frames[0].pts * stream.time_base))
                    if count > 1:
                        logger.trace("Decoded {} frames to find position {:.2f}s", count, float(timestamp * stream.time_base))
                    for current_frame in reversed(frames):
                        if current_frame is not None and current_frame.pts <= timestamp:
                            break
                        next_frame = current_frame
                    ratio = 0 if next_frame is None else (timestamp - current_frame.pts) / (next_frame.pts - current_frame.pts)
                except Exception as exc:
                    logger.opt(exception=exc).warning("Error reading video: {}", self._path)
                    container = decoder = None
                    frames = []
        self._cache[key] = container, decoder, frames
        return ratio, current_frame, next_frame

    def read_trimesh_model(self):
        key = 'trimesh',
        if self.check_unmodified() and (trimesh_model := self._cache.get(key, False)) is not False:
            return trimesh_model
        if self._mtime is None:
            logger.warning("File not found: {}", self._path)
            trimesh_model = None
        else:
            try:
                import trimesh
                trimesh_model = trimesh.load_mesh(self._path)
                assert isinstance(trimesh_model, trimesh.Trimesh)
            except Exception as exc:
                logger.opt(exception=exc).warning("Error reading model file: {}", self._path)
                trimesh_model = None
            else:
                logger.debug("Read {}v/{}f mesh: {}", len(trimesh_model.vertices), len(trimesh_model.faces), self._path)
        self._cache[key] = trimesh_model
        return trimesh_model

    def write_image(self, image, quality=None, alpha=False):
        import PIL.Image
        import PIL.ImageCms
        self._touched = system_clock()
        key = 'write_image', quality, alpha
        if key in self._cache and self._path.exists():
            return
        suffix = self._path.suffix.lower()
        registered_extensions = PIL.Image.registered_extensions()
        if suffix not in registered_extensions:
            logger.warning("Unrecognised image suffix: {}", self._path)
        else:
            self.cleanup()
            if self._path.exists():
                logger.warning("Existing image file will be overwritten: {}", self._path)
            encoder = registered_extensions[suffix]
            if not alpha or encoder not in ('PNG', 'TIFF', 'GIF', 'JPEG2000', 'WEBP'):
                image = image.convert('RGB')
            options = {'icc_profile': PIL.ImageCms.ImageCmsProfile(PIL.ImageCms.createProfile('sRGB')).tobytes()}
            if quality:
                options['quality'] = quality
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                image.save(self._path, **options)
            except Exception as exc:
                logger.opt(exception=exc).error("Unable to save image to: {}", self._path)
            else:
                logger.success("Saved image to file: {}", self._path)
        self._cache[key] = True

    def write_video_frame(self, frame, timestamp, codec='h264', pixfmt='yuv420p', fps=60, realtime=False,
                          options=None, limit=None, alpha=False):
        import av
        self._touched = system_clock()
        width, height = frame.width, frame.height
        options = tuple(sorted(options.items())) if options else None
        key = 'video_output', width, height, alpha, codec, pixfmt, fps, options, limit
        writer, queue, start = self._cache.get(key, (None, None, None))
        if start is None:
            self.cleanup()
            if self._path.exists():
                logger.warning("Existing video file will be overwritten: {}", self._path)
            options = dict(options) if options else {}
            options.setdefault('color_primaries', 'bt709')
            options.setdefault('color_trc', 'bt709')
            options.setdefault('colorspace', 'bt709')
            try:
                codec = codec.lower()
                if codec not in av.codecs_available:
                    raise ValueError(f"Unrecognised codec '{codec}''")
                av_codec = av.codec.Codec(codec, mode='w')
                if av_codec.type != 'video':
                    raise ValueError(f"'{codec}' not a video codec")
                if av_codec.name == 'hevc_videotoolbox' and alpha:
                    options.setdefault('alpha_quality', '1')
                self._path.parent.mkdir(parents=True, exist_ok=True)
                container = av.open(str(self._path), mode='w')
                stream = container.add_stream(codec, rate=fps, options=options)
            except Exception as exc:
                logger.opt(exception=exc).error("Unable to open video output: {}", self._path)
                self._cache[key] = False, None, timestamp
                return
            stream.width = width
            stream.height = height
            stream.pix_fmt = pixfmt
            stream.codec_context.time_base = fractions.Fraction(1, fps)
            if av_codec.name == 'libx265' or av_codec.name == 'hevc_videotoolbox':
                stream.codec_context.codec_tag = 'hvc1'
            queue = Queue(maxsize=fps)
            writer = threading.Thread(target=self._video_writer, args=(container, stream, queue))
            writer.start()
            start = timestamp
            self._cache[key] = writer, queue, start
            logger.debug("Beginning {} {} video output{}: {}", av_codec.name, stream.pix_fmt, " (with alpha)" if alpha else "", self._path)
        if queue is not None and (not realtime or not queue.full()):
            frame_time = int(round((timestamp - start) * fps))
            if limit is not None and frame_time >= int(round(limit * fps)):
                queue.put(None)
                writer.join()
                self._cache[key] = None, None, start
                return
            frame.pts = frame_time
            frame.time_base = fractions.Fraction(1, fps)
            try:
                queue.put(frame, block=not realtime)
            except Full:
                pass

    def _video_writer(self, container, stream, queue):
        try:
            while True:
                frame = queue.get()
                for packet in stream.encode(frame):
                    container.mux(packet)
                if frame is None:
                    break
        except Exception as exc:
            logger.opt(exception=exc).error("Error encoding video frame")
        container.close()
        logger.success("Flushed and closed video output file: {}", self._path)

    def __str__(self):
        return str(self._path)


class FileCache:
    def __init__(self):
        self._cache = {}
        self._root = Path('.')

    def clean(self, max_age=DEFAULT_CLEAN_TIME):
        full_collect = False
        cutoff = system_clock() - max_age
        for path in list(self._cache):
            cache_path = self._cache[path]
            if cache_path._touched < cutoff:
                full_collect |= cache_path.cleanup()
                del self._cache[path]
                logger.trace("Discarded {}", path)
        return full_collect

    def set_root(self, path):
        if isinstance(path, CachePath):
            path = path._path
        else:
            path = Path(path)
        if path.exists():
            if not path.is_dir():
                path = path.parent
            self._root = path
        else:
            self._root = Path('.')

    def get_with_root(self, path, root):
        if not isinstance(path, Path):
            path = Path(path)
        if not path.is_absolute():
            if isinstance(root, CachePath):
                root = root._path.parent
            else:
                if not isinstance(root, Path):
                    root = Path(root)
                if root.is_file():
                    root = root.parent
            path = root / path
        absolute = str(path.resolve())
        if (cached := self._cache.get(absolute)) is not None:
            return cached
        self._cache[absolute] = path = CachePath(path, absolute)
        return path

    def __getitem__(self, path):
        return self.get_with_root(path, self._root)


SharedCache = FileCache()
