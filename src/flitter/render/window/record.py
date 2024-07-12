"""
Record scene node
"""

from . import ProgramNode
from ...cache import SharedCache
from .glconstants import GL_SRGB8_ALPHA8
from .glsl import TemplateLoader


class Record(ProgramNode):
    DEFAULT_VERTEX_SOURCE = TemplateLoader.get_template('video.vert')

    def __init__(self, glctx):
        super().__init__(glctx)
        self._framebuffer = None
        self._texture = None

    @property
    def texture(self):
        return self.children[0].texture if len(self.children) == 1 else None

    @property
    def framebuffer(self):
        return self._framebuffer

    def release(self):
        self._framebuffer = None
        self._texture = None
        super().release()

    def create(self, engine, node, resized, **kwargs):
        super().create(engine, node, resized, **kwargs)
        if self._framebuffer is None or self._texture is None or resized:
            self._last = None
            self._texture = self.glctx.texture((self.width, self.height), 4, dtype='f1', internal_format=GL_SRGB8_ALPHA8)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))

    def render(self, node, **kwargs):
        if filename := node.get('filename', 1, str):
            super().render(node, **kwargs)
            path = SharedCache[filename]
            ext = path.suffix.lower()
            codec = node.get('codec', 1, str, 'h264')
            keep_alpha = node.get('keep_alpha', 1, bool, False)
            if ext in ('.mp4', '.mov', '.m4v', '.mkv', '.webm', '.ogg') or (ext == '.gif' and codec == 'gif'):
                pixfmt = node.get('pixfmt', 1, str, 'rgb8' if codec == 'gif' else 'yuv420p')
                crf = node.get('crf', 1, int)
                preset = node.get('preset', 1, str)
                limit = node.get('limit', 1, float)
                path.write_video_frame(self._framebuffer, kwargs['clock'],
                                       fps=int(kwargs['fps']), realtime=kwargs['realtime'], codec=codec,
                                       pixfmt=pixfmt, crf=crf, preset=preset, limit=limit, alpha=keep_alpha)
            else:
                quality = node.get('quality', 1, int)
                path.write_image(self._framebuffer, quality=quality, alpha=keep_alpha)


SCENE_NODE_CLASS = Record
