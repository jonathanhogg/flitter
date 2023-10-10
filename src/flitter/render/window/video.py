"""
Video scene node
"""

from . import Shader, COLOR_FORMATS
from ...cache import SharedCache
from .glconstants import GL_SRGB8
from .glsl import TemplateLoader


class Video(Shader):
    DEFAULT_VERTEX_SOURCE = TemplateLoader.get_template('video.vert')
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('video.frag')

    def __init__(self, glctx):
        super().__init__(glctx)
        self._filename = None
        self._frame0 = None
        self._frame1 = None
        self._frame0_texture = None
        self._frame1_texture = None
        self._colorbits = None

    def release(self):
        self._frame0_texture = None
        self._frame1_texture = None
        self._frame0 = None
        self._frame1 = None
        self._colorbits = None
        super().release()

    @property
    def child_textures(self):
        return {'frame0': self._frame0_texture, 'frame1': self._frame1_texture}

    def similar_to(self, node):
        return super().similar_to(node) and node.get('filename', 1, str) == self._filename

    async def update(self, engine, node, **kwargs):
        references = kwargs.setdefault('references', {})
        if node_id := node.get('id', 1, str):
            references[node_id] = self
        self.hidden = node.get('hidden', 1, bool, False)
        self._filename = node.get('filename', 1, str)
        position = node.get('position', 1, float, 0)
        loop = node.get('loop', 1, bool, False)
        threading = node.get('thread', 1, bool, False)
        if self._filename is not None:
            ratio, frame0, frame1 = SharedCache[self._filename].read_video_frames(self, position, loop, threading=threading)
        else:
            ratio, frame0, frame1 = 0, None, None
        colorbits = node.get('colorbits', 1, int, self.glctx.extra['colorbits'])
        if colorbits not in COLOR_FORMATS:
            colorbits = self.glctx.extra['colorbits']
        if self._texture is not None and (frame0 is None or (self.width, self.height) != (frame0.width, frame0.height)) \
                or colorbits != self._colorbits:
            self.release()
        if frame0 is None:
            return
        if self._texture is None:
            self.width, self.height = frame0.width, frame0.height
            depth = COLOR_FORMATS[colorbits]
            self._texture = self.glctx.texture((self.width, self.height), 4, dtype=depth.moderngl_dtype)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
            self._frame0_texture = self.glctx.texture((self.width, self.height), 3, internal_format=GL_SRGB8)
            self._frame1_texture = self.glctx.texture((self.width, self.height), 3, internal_format=GL_SRGB8)
            self._colorbits = colorbits
        if frame0 is self._frame1 or frame1 is self._frame0:
            self._frame0_texture, self._frame1_texture = self._frame1_texture, self._frame0_texture
            self._frame0, self._frame1 = self._frame1, self._frame0
        if frame0 is not self._frame0:
            rgb_frame = frame0.to_rgb()
            plane = rgb_frame.planes[0]
            data = rgb_frame.to_ndarray().data if plane.line_size > plane.width * 3 else plane
            self._frame0_texture.write(memoryview(data))
            self._frame0 = frame0
        if frame1 is None:
            self._frame1 = None
        elif frame1 is not self._frame1:
            rgb_frame = frame1.to_rgb()
            plane = rgb_frame.planes[0]
            data = rgb_frame.to_ndarray().data if plane.line_size > plane.width * 3 else plane
            self._frame1_texture.write(memoryview(data))
            self._frame1 = frame1
        interpolate = node.get('interpolate', 1, bool, False)
        self.render(node, ratio=ratio if interpolate else 0, **kwargs)


SCENE_NODE_CLASS = Video
