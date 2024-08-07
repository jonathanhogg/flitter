"""
Image window node
"""

from loguru import logger
import PIL.Image

from . import WindowNode
from ...cache import SharedCache
from .glconstants import GL_SRGB8_ALPHA8


class Image(WindowNode):
    def __init__(self, glctx):
        super().__init__(glctx)
        self._filename = None
        self._image = None
        self._texture = None

    def release(self):
        self._filename = None
        self._image = None
        self._texture = None

    @property
    def texture(self):
        return self._texture

    async def update(self, engine, node, default_size=(512, 512), **kwargs):
        references = kwargs.setdefault('references', {})
        if node_id := node.get('id', 1, str):
            references[node_id] = self
        self.hidden = node.get('hidden', 1, bool, False)
        self.tags = node.tags
        filename = node.get('filename', 1, str)
        if self._image is not None:
            default_size = [self._image.width, self._image.height]
        else:
            default_size = None
        size = node.get('size', 2, int, default_size)
        if filename != self._filename or (size and size != [self.width, self.height]):
            self.release()
            self._filename = filename
            self.width = None
            self.height = None
        if self._filename is not None:
            image = SharedCache[filename].read_pil_image()
            if image is not self._image:
                self._image = image
                if size and size != [image.width, image.height]:
                    image = image.resize(size, PIL.Image.Resampling.BILINEAR)
                    logger.debug("Resized {} to {}x{}", self._filename, *size)
                self.width = image.width
                self.height = image.height
                self._texture = self.glctx.texture((self.width, self.height), 4, internal_format=GL_SRGB8_ALPHA8)
                self._texture.write(image.transpose(PIL.Image.FLIP_TOP_BOTTOM).convert('RGBA').convert('RGBa').tobytes())

    def similar_to(self, node):
        return super().similar_to(node) and node.get('filename', 1, str) == self._filename
