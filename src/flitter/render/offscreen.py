"""
Flitter offscreen OpenGL rendering
"""

from loguru import logger
import moderngl

from .window import ProgramNode, DEFAULT_LINEAR, DEFAULT_COLORBITS, COLOR_FORMATS


class Offscreen(ProgramNode):
    def __init__(self, **kwargs):
        super().__init__(None)
        self._framebuffer = None
        self._texture = None
        self._colorbits = None

    def release(self):
        self._colorbits = None
        self._framebuffer = None
        self._texture = None
        if self.glctx is not None:
            self.glctx.finish()
            self.glctx.extra.clear()
            if count := self.glctx.gc():
                logger.trace("Offscreen release collected {} OpenGL objects", count)
            self.glctx.release()
            self.glctx = None
        super().release()

    def purge(self):
        self.glctx.finish()
        super().purge()
        if count := self.glctx.gc():
            logger.trace("Offscreen purge collected {} OpenGL objects", count)

    @property
    def texture(self):
        return self._texture

    @property
    def framebuffer(self):
        return self._framebuffer

    def create(self, engine, node, resized, **kwargs):
        super().create(engine, node, resized, **kwargs)
        if self.glctx is None:
            self.glctx = moderngl.create_context(self.GL_VERSION[0] * 100 + self.GL_VERSION[1] * 10, standalone=True)
            self.glctx.gc_mode = 'context_gc'
            self.glctx.extra = {}
            logger.debug("Created standalone OpenGL context for {}", self.name)
            logger.debug("OpenGL info: {GL_RENDERER} {GL_VERSION}", **self.glctx.info)
            logger.trace("{!r}", self.glctx.info)
        self.glctx.extra['linear'] = node.get('linear', 1, bool, DEFAULT_LINEAR)
        colorbits = node.get('colorbits', 1, int, DEFAULT_COLORBITS)
        if colorbits not in COLOR_FORMATS:
            colorbits = DEFAULT_COLORBITS
        self.glctx.extra['colorbits'] = colorbits
        self.glctx.extra['size'] = self.width, self.height
        if not node.get('id', 1, str):
            if self._framebuffer is not None:
                self._texture = None
                self._framebuffer = None
        elif self._framebuffer is None or self._texture is None or resized or colorbits != self._colorbits:
            depth = COLOR_FORMATS[colorbits]
            self._texture = self.glctx.texture((self.width, self.height), 4, dtype=depth.moderngl_dtype)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
            self._framebuffer.clear()
            self._colorbits = colorbits
            logger.debug("Created {}x{} {}-bit framebuffer for {}", self.width, self.height, self._colorbits, self.name)

    def render(self, node, **kwargs):
        if self._framebuffer is not None:
            super().render(node, **kwargs)


RENDERER_CLASS = Offscreen
