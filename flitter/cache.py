"""
General file cache
"""

import csv
from pathlib import Path

import av
from loguru import logger
import skia

from .model import Vector


class CachePath:
    def __init__(self, path):
        self._path = path
        self._cache = {}

    def read_text(self, encoding=None, errors=None):
        key = 'text', encoding, errors
        mtime = self._path.stat().st_mtime if self._path.exists() else None
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

    def read_csv_vector(self, row_number):
        mtime = self._path.stat().st_mtime if self._path.exists() else None
        if 'csv' in self._cache and self._cache['csv'][0] == mtime:
            _, reader, rows = self._cache['csv']
        elif mtime is None:
            logger.warning("File not found: {}", self._path)
            reader = None
            rows = []
        else:
            try:
                reader = csv.reader(self._path.open(newline=''))
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
        mtime = self._path.stat().st_mtime if self._path.exists() else None
        if 'image' in self._cache:
            cache_mtime, image = self._cache['image']
            if mtime == cache_mtime:
                return image
        if mtime is None:
            logger.warning("File not found: {}", self._path)
            text = None
        else:
            try:
                image = skia.Image.open(str(self._path))
            except Exception as exc:
                logger.opt(exception=exc).warning("Error reading image file: {}", self._path)
                text = None
            else:
                logger.debug("Read image file: {}", self._path)
        self._cache['image'] = mtime, image
        return image

    def read_video_frames(self, position, loop=False):
        container = decoder = current_frame = next_frame = None
        frames = []
        ratio = 0
        mtime = self._path.stat().st_mtime if self._path.exists() else None
        if 'video' in self._cache and self._cache['video'][0] == mtime:
            _, container, decoder, frames = self._cache['video']
        elif mtime is None:
            logger.warning("File not found: {}", self._path)
        else:
            try:
                container = av.container.open(str(self._path))
                logger.debug("Opened video file: {}", self._path)
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
                while True:
                    if len(frames) == 2:
                        if timestamp >= frames[0].pts and (frames[1] is None or timestamp <= frames[1].pts):
                            break
                        if timestamp < frames[0].pts or (timestamp > frames[1].pts and frames[1].key_frame):
                            frames = []
                    if not frames:
                        if decoder is not None:
                            decoder.close()
                        container.seek(timestamp, stream=stream)
                        decoder = container.decode(streams=(stream.index,))
                    if decoder is not None:
                        if len(frames) == 2:
                            frames.pop(0)
                        try:
                            frame = next(decoder)
                        except StopIteration:
                            decoder = None
                            frame = None
                        frames.append(frame)
                current_frame, next_frame = frames
                ratio = (timestamp - current_frame.pts) / (next_frame.pts - current_frame.pts) if next_frame is not None else 0
            except Exception as exc:
                logger.opt(exception=exc).warning("Error reading video file: {}", self._path)
                container = decoder = None
                frames = []
        self._cache['video'] = mtime, container, decoder, frames
        return ratio, current_frame, next_frame
        if frames:
            return ratio, *frames
        else:
            return 0, None, None

    def __str__(self):
        return str(self._path)


class FileCache:
    def __init__(self):
        self._cache = {}
        self._root = Path('.')

    def set_root(self, path):
        if isinstance(path, CachePath):
            path = path._path
        else:
            path = Path(path)
        assert path.exists()
        if not path.is_dir():
            path = path.parent
        self._root = path

    def __getitem__(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self._root / path
        key = str(path.resolve())
        if key in self._cache:
            return self._cache[key]
        path = CachePath(path)
        self._cache[key] = path
        return path


SharedCache = FileCache()
