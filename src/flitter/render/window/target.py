"""
Flitter render targets
"""

from collections import namedtuple

from loguru import logger

from ...clock import system_clock
from .glconstants import GL_RGBA8, GL_SRGB8_ALPHA8, GL_RGBA16F, GL_RGBA32F, GL_FRAMEBUFFER_SRGB


ColorFormat = namedtuple('ColorFormat', ('moderngl_dtype', 'gl_format', 'srgb_gl_format'))

COLOR_FORMATS = {
    8: ColorFormat('f1', GL_RGBA8, GL_SRGB8_ALPHA8),
    16: ColorFormat('f2', GL_RGBA16F, False),
    32: ColorFormat('f4', GL_RGBA32F, False)
}


class RenderTarget:
    @classmethod
    def get(cls, glctx, width, height, colorbits, *, srgb=False, has_depth=False, samples=0):
        if colorbits not in COLOR_FORMATS:
            colorbits = glctx.extra['colorbits']
        pool = glctx.extra.setdefault('_RenderTarget_pool', {})
        key = width, height, colorbits, srgb, has_depth, samples
        targets = pool.setdefault(key, [])
        if targets:
            target = targets.pop()
            target._release_time = None
            return target
        return RenderTarget(glctx, width, height, colorbits, srgb, has_depth, samples)

    @classmethod
    def empty_pool(cls, glctx, age=0):
        cutoff = system_clock() - age
        pool = glctx.extra.setdefault('_RenderTarget_pool', {})
        for key, targets in pool.items():
            while targets and targets[0]._release_time < cutoff:
                target = targets.pop(0)
                logger.debug("Destroyed {}", str(target))

    def __init__(self, glctx, width, height, colorbits, srgb, has_depth, samples):
        self._glctx = glctx
        self.width = width
        self.height = height
        self.colorbits = colorbits
        self.srgb = srgb
        self.has_depth = has_depth
        self.samples = samples
        self._release_time = None
        format = COLOR_FORMATS[self.colorbits]
        gl_format = format.srgb_gl_format if self.srgb else format.gl_format
        self._image_texture = self._glctx.texture((self.width, self.height), 4, dtype=format.moderngl_dtype, internal_format=gl_format)
        self._depth_renderbuffer = self._glctx.depth_renderbuffer((self.width, self.height), samples=self.samples) if self.has_depth else None
        if self.samples:
            self._color_renderbuffer = self._glctx.renderbuffer((self.width, self.height), 4, samples=self.samples, dtype=format.moderngl_dtype)
            self._render_framebuffer = self._glctx.framebuffer(color_attachments=(self._color_renderbuffer,), depth_attachment=self._depth_renderbuffer)
            self._image_framebuffer = self._glctx.framebuffer(color_attachments=(self._image_texture,))
        else:
            self._color_renderbuffer = None
            self._render_framebuffer = self._glctx.framebuffer(color_attachments=(self._image_texture,), depth_attachment=self._depth_renderbuffer)
            self._image_framebuffer = self._render_framebuffer
        logger.debug("Created {}", str(self))

    def release(self):
        self._release_time = system_clock()
        pool = self._glctx.extra['_RenderTarget_pool']
        key = self.width, self.height, self.colorbits, self.srgb, self.has_depth, self.samples
        pool[key].append(self)

    def __str__(self):
        text = f"{self.width}x{self.height} {self.colorbits}-bit"
        if self.samples:
            text += f" {self.samples}x"
        if self.srgb:
            text += " sRGB"
        text += " render target"
        if self.has_depth:
            text += " with depth"
        return text

    @property
    def size(self):
        return self.width, self.height

    @property
    def texture(self):
        if self._release_time is not None:
            return None
        return self._image_texture

    @property
    def framebuffer(self):
        if self._release_time is not None:
            return None
        return self._image_framebuffer

    def clear(self, color=(0, 0, 0, 0)):
        self._render_framebuffer.clear(*tuple(color))

    def use(self):
        if self.srgb:
            self._glctx.enable_direct(GL_FRAMEBUFFER_SRGB)
        self._render_framebuffer.use()

    def finish(self):
        if self.srgb:
            self._glctx.disable_direct(GL_FRAMEBUFFER_SRGB)
        if self._image_framebuffer is not None:
            self._glctx.copy_framebuffer(self._image_framebuffer, self._render_framebuffer)

    def depth_write(self, enabled):
        self._render_framebuffer.depth_mask = enabled

    def __enter__(self):
        self.use()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()
