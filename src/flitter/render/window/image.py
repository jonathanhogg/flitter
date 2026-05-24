"""
Image window node
"""

from loguru import logger
import PIL.Image

from . import WindowNode
from ...cache import SharedCache
from .target import RenderTarget


class Image(WindowNode):
    def __init__(self, glctx):
        super().__init__(glctx)
        self._filename = None
        self._flip = None
        self._image = None
        self._target = None

    def release(self):
        self.width = None
        self.height = None
        self._flip = None
        self._filename = None
        self._image = None
        if self._target is not None:
            self._target.release()
            self._target = None

    @property
    def texture(self):
        return self._target.texture if self._target is not None else None

    @property
    def array(self):
        return self._target.array if self._target is not None else None

    async def update(self, engine, node, references, **kwargs):
        self.node_id = node.get('id', 1, str)
        if self.node_id is not None:
            references[self.node_id] = self
        self.hidden = node.get('hidden', 1, bool, False)
        self.tags = node.tags
        filename = node.get('filename', 1, str)
        image = SharedCache[filename].read_pil_image() if filename is not None else None
        width, height = node.get('size', 2, int, [image.width, image.height]) if image is not None else [None, None]
        flip = node.get('flip', 1, str)
        if filename != self._filename or image is not self._image or width != self.width or height != self.height or flip != self._flip:
            self.release()
            self._filename = filename
            self._image = image
            self.width = width
            self.height = height
            self._flip = flip
            if image is not None:
                if image.width != width or image.height != height:
                    image = image.resize((width, height), PIL.Image.Resampling.BILINEAR)
                    logger.trace("Resized {} to {}x{}", self._filename, width, height)
                if flip in {'horizontal', 'both'}:
                    image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                    logger.trace("Flipped {} horizontally", self._filename)
                if flip not in {'vertical', 'both'}:
                    image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                else:
                    logger.trace("Flipped {} vertically", self._filename)
                image = image.convert('RGBA').convert('RGBa')
                self._target = RenderTarget.get(self.glctx, width, height, 8, srgb=True)
                self._target.texture.write(image.tobytes())

    async def descend(self, engine, node, references, **kwargs):
        # An image is a leaf node from the perspective of the OpenGL world
        pass

    def similar_to(self, node):
        return super().similar_to(node) and (node.get('filename', 1, str) == self._filename or node.get('id', 1, str) == self.node_id)
