"""
Kinect scene node

pip3 install git+https://github.com/chris-castillo/freenect2-python.git
"""

import threading
import os

os.environ['LIBFREENECT2_LOGGER_LEVEL'] = "error"

from freenect2 import Device, NoDeviceError, FrameType, FrameFormat
from loguru import logger

from . import Shader, COLOR_FORMATS
from .glsl import TemplateLoader


class Kinect(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('kinect.frag')
    TEXTURE_PARAMS = {
        FrameFormat.BGRX: (4, 'f1', 'BGR1'),
        FrameFormat.Float: (1, 'f4', 'RRR1'),
        FrameFormat.Gray: (1, 'f1', 'RRR1'),
    }

    Device = None

    def __init__(self, glctx):
        super().__init__(glctx)
        self._color_texture = None
        self._depth_texture = None
        self._color_frame = None
        self._depth_frame = None
        self._frame_lock = threading.Lock()

    @property
    def child_textures(self):
        textures = {}
        if self._color_texture is not None:
            textures['color'] = self._color_texture
        if self._depth_texture is not None:
            textures['depth'] = self._depth_texture
        return textures

    def release(self):
        super().release()
        if self.Device is not None:
            logger.debug("Stopping Kinect v2 device")
            self.Device.stop()
        self._color_texture = None
        self._depth_texture = None
        self._color_frame = None
        self._depth_frame = None

    def create(self, engine, node, resized, **kwargs):
        if self._framebuffer is None or self._texture is None or resized:
            depth = COLOR_FORMATS[16]
            self._last = None
            self._texture = self.glctx.texture((self.width, self.height), 4, dtype=depth.moderngl_dtype)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
            self._framebuffer.clear()
            self._colorbits = 16
        if self.Device is None:
            try:
                self.Device = Device()
            except NoDeviceError:
                return
            logger.debug("Starting Kinect v2 device")
            self.Device.start(self._process_frame)

    def _process_frame(self, frame_type, frame):
        with self._frame_lock:
            if frame_type == FrameType.Color:
                if self._color_frame is None:
                    logger.debug("Receiving color frames")
                self._color_frame = frame
            elif frame_type == FrameType.Depth:
                if self._depth_frame is None:
                    logger.debug("Receiving depth frames")
                self._depth_frame = frame

    async def descend(self, engine, node, **kwargs):
        pass

    def render(self, node, **kwargs):
        with self._frame_lock:
            color_frame = self._color_frame
            depth_frame = self._depth_frame
        mode = {'color': 1, 'depth': 2}.get(node.get('output', 1, str, 'combined').lower(), 0)
        kwargs['mode'] = mode
        if mode == 0 and color_frame is not None and depth_frame is not None:
            depth_frame, color_frame = self.Device.registration.apply(color_frame, depth_frame)
        if color_frame is not None:
            if self._color_texture is None or self._color_texture.size != (color_frame.width, color_frame.height):
                components, dtype, swizzle = self.TEXTURE_PARAMS[color_frame.format]
                self._color_texture = self.glctx.texture((color_frame.width, color_frame.height), components, dtype=dtype)
                self._color_texture.swizzle = swizzle
                logger.debug("Create {}x{} kinect color texture", color_frame.width, color_frame.height)
            self._color_texture.write(color_frame.data)
        if depth_frame is not None:
            if self._depth_texture is None or self._depth_texture.size != (depth_frame.width, depth_frame.height):
                components, dtype, swizzle = self.TEXTURE_PARAMS[depth_frame.format]
                self._depth_texture = self.glctx.texture((depth_frame.width, depth_frame.height), components, dtype=dtype)
                self._depth_texture.swizzle = swizzle
                logger.debug("Create {}x{} kinect depth texture", depth_frame.width, depth_frame.height)
            self._depth_texture.write(depth_frame.data)
        super().render(node, **kwargs)


SCENE_NODE_CLASS = Kinect
