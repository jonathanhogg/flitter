"""
General file cache
"""

import fractions
from pathlib import Path
from queue import Queue, Full
import threading

from loguru import logger

from .clock import system_clock


DEFAULT_CLEAN_TIME = 30
MAX_CACHE_VIDEO_FRAMES = 100


class CachePath:
    def __init__(self, path, absolute):
        self._touched = system_clock()
        self._path = path
        self._absolute = absolute
        self._cache = {}

    @property
    def suffix(self):
        return self._path.suffix

    def exists(self):
        return self._path.exists()

    def cleanup(self):
        while self._cache:
            key, value = self._cache.popitem()
            match key:
                case 'video', _:
                    mtime, container, decoder, frames = value
                    if decoder is not None:
                        decoder.close()
                    if container is not None:
                        container.close()
                        logger.debug("Closing video file: {}", self._path)
                case 'video_output':
                    writer, queue, *_ = value
                    if queue is not None:
                        queue.put(None)
                        writer.join()

    def read_text(self, encoding=None, errors=None):
        self._touched = system_clock()
        key = 'text', encoding, errors
        mtime = self._path.stat().st_mtime if self._path.is_file() else None
        if key in self._cache:
            cache_mtime, text = self._cache[key]
            if mtime == cache_mtime:
                return text
        if mtime is None:
            logger.warning("File not found: {}", self._path)
            text = None
        else:
            try:
                text = self._path.read_text(encoding, errors)
            except Exception as exc:
                logger.opt(exception=exc).warning("Error reading text file: {}", self._path)
                text = None
            else:
                logger.debug("Read text file: {}", self._path)
        self._cache[key] = mtime, text
        return text

    def read_flitter_program(self, variables=None, undefined=None):
        from .language.parser import parse, ParseError
        self._touched = system_clock()
        mtime = self._path.stat().st_mtime if self._path.is_file() else None
        if 'flitter' in self._cache:
            cache_mtime, program = self._cache['flitter']
            if mtime == cache_mtime:
                return program
        else:
            program = None
        if mtime is None:
            logger.warning("Program file not found: {}", self._path)
            program = None
        else:
            try:
                parse_time = -system_clock()
                source = self._path.read_text(encoding='utf8')
                initial_top = parse(source)
                now = system_clock()
                parse_time += now
                simplify_time = -now
                top = initial_top.simplify(variables=variables, undefined=undefined)
                now = system_clock()
                simplify_time += now
                compile_time = -now
                program = top.compile()
                program.set_top(top)
                program.set_path(self)
                compile_time += system_clock()
                logger.debug("Read program file: {}", self._path)
                logger.debug("Compiled to {} instructions in {:.1f}/{:.1f}/{:.1f}ms",
                             len(program), parse_time*1000, simplify_time*1000, compile_time*1000)
            except ParseError as exc:
                if program is None:
                    logger.error("Error parsing {} at line {} column {}:\n{}",
                                 self._path.name, exc.line, exc.column, exc.context)
                else:
                    logger.warning("Unable to re-parse {}, error at line {} column {}:\n{}",
                                   self._path.name, exc.line, exc.column, exc.context)
            except Exception as exc:
                logger.opt(exception=exc).error("Error reading program file: {}", self._path)
                program = None
        self._cache['flitter'] = mtime, program
        return program

    def read_csv_vector(self, row_number):
        import csv
        from .model import Vector, null
        self._touched = system_clock()
        mtime = self._path.stat().st_mtime if self._path.is_file() else None
        if 'csv' in self._cache and self._cache['csv'][0] == mtime:
            _, reader, rows = self._cache['csv']
        elif mtime is None:
            logger.warning("File not found: {}", self._path)
            reader = None
            rows = []
        else:
            try:
                reader = csv.reader(self._path.open(newline='', encoding='utf8'))
                logger.debug("Opened CSV file: {}", self._path)
            except Exception as exc:
                logger.opt(exception=exc).warning("Error reading CSV file: {}", self._path)
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
                logger.opt(exception=exc).warning("Error reading CSV file: {}", self._path)
                reader = None
        self._cache['csv'] = mtime, reader, rows
        if 0 <= row_number < len(rows):
            return rows[row_number]
        return null

    def read_image(self):
        import skia
        self._touched = system_clock()
        mtime = self._path.stat().st_mtime if self._path.is_file() else None
        if 'image' in self._cache:
            cache_mtime, image = self._cache['image']
            if mtime == cache_mtime:
                return image
        if mtime is None:
            logger.warning("File not found: {}", self._path)
            image = None
        else:
            try:
                image = skia.Image.open(str(self._path))
            except Exception as exc:
                logger.opt(exception=exc).warning("Error reading image file: {}", self._path)
                image = None
            else:
                logger.debug("Read image file: {}", self._path)
        self._cache['image'] = mtime, image
        return image

    def read_video_frames(self, obj, position, loop=False, threading=False):
        import av
        self._touched = system_clock()
        key = 'video', id(obj)
        container = decoder = current_frame = next_frame = None
        frames = []
        ratio = 0
        mtime = self._path.stat().st_mtime if self._path.is_file() else None
        if key in self._cache and self._cache[key][0] == mtime:
            _, container, decoder, frames = self._cache[key]
        elif mtime is None:
            logger.warning("File not found: {}", self._path)
        else:
            try:
                container = av.container.open(str(self._path))
                stream = container.streams.video[0]
                if threading:
                    stream.thread_type = 'AUTO'
                ctx = stream.codec_context
                logger.debug("Opened {}x{} {:.0f}fps video file: {}", ctx.width, ctx.height, float(stream.average_rate), self._path)
            except Exception as exc:
                logger.opt(exception=exc).warning("Error reading video file: {}", self._path)
                container = None
        if container is not None:
            try:
                stream = container.streams.video[0]
                if loop:
                    timestamp = stream.start_time + int(position / stream.time_base) % stream.duration
                else:
                    timestamp = stream.start_time + min(max(0, int(position / stream.time_base)), stream.duration)
                count = 0
                while True:
                    if len(frames) >= 2 and timestamp >= frames[0].pts and (frames[-1] is None or timestamp < frames[-1].pts):
                        break
                    if frames and timestamp < frames[0].pts:
                        logger.trace("Discard {} buffered video frames", len(frames))
                        frames = []
                    if len(frames) >= 2 and frames[-1].key_frame and timestamp > 2*frames[-1].pts - frames[0].pts:
                        logger.trace("Discard {} buffered video frames", len(frames))
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
                                logger.trace("Read video key frame @ {:.2f}s", float(frame.pts * stream.time_base))
                                for i in range(len(frames)-1, 0, -1):
                                    if frames[i].key_frame:
                                        frames = frames[:i]
                                        logger.trace("Discard {} buffered video frames", i)
                                        break
                            else:
                                logger.trace("Read video frame @ {:.2f}s", float(frame.pts * stream.time_base))
                            count += 1
                        except StopIteration:
                            logger.trace("Hit end of video {}", self._path)
                            decoder = None
                            frame = None
                        if len(frames) == MAX_CACHE_VIDEO_FRAMES:
                            frames.pop(0)
                            logger.trace("Discard one buffered video frame (hit maximum)")
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
                logger.opt(exception=exc).warning("Error reading video file: {}", self._path)
                container = decoder = None
                frames = []
        self._cache[key] = mtime, container, decoder, frames
        return ratio, current_frame, next_frame
        if frames:
            return ratio, *frames
        else:
            return 0, None, None

    def read_trimesh_model(self):
        import trimesh
        self._touched = system_clock()
        mtime = self._path.stat().st_mtime if self._path.is_file() else None
        if 'trimesh' in self._cache:
            cache_mtime, trimesh_model = self._cache['trimesh']
            if mtime == cache_mtime:
                return trimesh_model
        if mtime is None:
            logger.warning("File not found: {}", self._path)
            trimesh_model = None
        else:
            try:
                trimesh_model = trimesh.load(str(self._path))
            except Exception as exc:
                logger.opt(exception=exc).warning("Error reading model file: {}", self._path)
                trimesh_model = None
            else:
                logger.debug("Read {}v/{}f mesh model file: {}", len(trimesh_model.vertices), len(trimesh_model.faces), self._path)
        self._cache['trimesh'] = mtime, trimesh_model
        return trimesh_model

    def write_image(self, texture, quality=None):
        import PIL.Image
        import PIL.ImageCms
        self._touched = system_clock()
        if 'write_image' in self._cache:
            return
        suffix = self._path.suffix.lower()
        registered_extensions = PIL.Image.registered_extensions()
        if suffix not in registered_extensions:
            logger.warning("Unrecognised image suffix: {}", self._path)
        else:
            self.cleanup()
            if self._path.exists():
                logger.warning("Existing image file will be overwritten: {}", self._path)
            image = PIL.Image.frombytes('RGBA' if texture.components == 4 else 'RGB', (texture.width, texture.height), texture.read())
            encoder = registered_extensions[suffix]
            if image.mode != 'RGB' and encoder not in ('PNG', 'TIFF', 'GIF', 'JPEG2000', 'WEBP'):
                image = image.convert('RGB')
            options = {'icc_profile': PIL.ImageCms.ImageCmsProfile(PIL.ImageCms.createProfile('sRGB')).tobytes()}
            if quality:
                options['quality'] = quality
            try:
                image.save(self._path, **options)
            except Exception as exc:
                logger.opt(exception=exc).error("Unable to save image to file: {}", self._path)
            else:
                logger.success("Saved image to file: {}", self._path)
        self._cache['write_image'] = True

    def write_video_frame(self, texture, timestamp, codec='h264', pixfmt='yuv420p', fps=60, realtime=False, crf=None, bitrate=None,
                          preset=None, limit=None):
        import av
        self._touched = system_clock()
        writer = queue = start = None
        width, height = texture.width, texture.height
        has_alpha = texture.components == 4
        config = [width, height, has_alpha, codec, pixfmt, fps, crf, bitrate, preset, limit]
        if 'video_output' in self._cache:
            writer, queue, start, *cached_config = self._cache['video_output']
            if cached_config != config:
                if queue is not None:
                    logger.debug("Closing output video container as configuration has changed")
                    queue.put(None)
                    writer.join()
                container = queue = writer = start = None
        if start is None:
            self.cleanup()
            if self._path.exists():
                logger.warning("Existing video file will be overwritten: {}", self._path)
            options = {}
            options['color_primaries'] = 'bt709'
            options['color_trc'] = 'bt709'
            options['colorspace'] = 'bt709'
            if crf is not None:
                options['crf'] = str(crf)
            elif bitrate is not None:
                options['maxrate'] = str(bitrate)
                options['bufsize'] = str(bitrate * 2)
            if preset is not None:
                options['preset'] = preset
            try:
                codec = codec.lower()
                if codec not in av.codecs_available:
                    raise ValueError(f"Unrecognised codec '{codec}''")
                av_codec = av.codec.Codec(codec, mode='w')
                if av_codec.type != 'video':
                    raise ValueError(f"'{codec}' not a video codec")
                if av_codec.name == 'hevc_videotoolbox' and has_alpha:
                    options['alpha_quality'] = '1'
                container = av.open(str(self._path), mode='w')
                stream = container.add_stream(av_codec, rate=fps, options=options)
            except Exception as exc:
                logger.error("Unable to open video output {}: {}", self._path, str(exc))
                self._cache['video_output'] = False, None, timestamp, *config
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
            self._cache['video_output'] = writer, queue, start, *config
            logger.debug("Beginning {} {} video output{}: {}", av_codec.name, stream.pix_fmt, " (with alpha)" if has_alpha else "", self._path)
        if queue is not None and (not realtime or not queue.full()):
            frame_time = timestamp - start
            if limit is not None and frame_time >= limit:
                queue.put(None)
                writer.join()
                self._cache['video_output'] = None, None, start, *config
                return
            frame = av.VideoFrame(width, height, 'rgba' if has_alpha else 'rgb24')
            line_size = frame.planes[0].line_size
            if line_size != width:
                import numpy as np
                components = 4 if has_alpha else 3
                data = np.ndarray((height, width*components), dtype='uint8', buffer=texture.read())
                array = np.empty((height, line_size), dtype='uint8')
                array[:, :width*components] = data
                frame.planes[0].update(array.data)
            else:
                frame.planes[0].update(texture.read())
            frame.pts = int(round(frame_time * fps))
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
            logger.error("Error encoding video frame: {}", str(exc))
        container.close()
        logger.success("Flushed and closed video output file: {}", self._path)

    def __str__(self):
        return str(self._path)


class FileCache:
    def __init__(self):
        self._cache = {}
        self._root = Path('.')

    def clean(self, max_age=DEFAULT_CLEAN_TIME):
        cutoff = system_clock() - max_age
        for path in list(self._cache):
            cache_path = self._cache[path]
            if cache_path._touched < cutoff:
                cache_path.cleanup()
                del self._cache[path]
                logger.trace("Discarded {}", path)

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
        if isinstance(root, CachePath):
            root = root._path
        path = Path(path)
        if not path.is_absolute():
            root = Path(root)
            if not root.is_dir():
                root = root.parent
            path = root / path
        key = str(path.resolve())
        if key in self._cache:
            return self._cache[key]
        path = CachePath(path, key)
        self._cache[key] = path
        return path

    def __getitem__(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self._root / path
        key = str(path.resolve())
        if key in self._cache:
            return self._cache[key]
        path = CachePath(path, key)
        self._cache[key] = path
        return path


SharedCache = FileCache()
