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
        if self._image is not None:
            default_size = [self._image.width, self._image.height]
        else:
            default_size = None
        size = node.get('size', 2, int, default_size)
        flip = node.get('flip', 1, str)
        if flip not in {'horizontal', 'vertical', 'both'}:
            flip = None
        if filename != self._filename or (size and size != [self.width, self.height]) or flip != self._flip:
            self.release()
            self._filename = filename
            self._flip = flip
        if self._filename is not None:
            image = SharedCache[filename].read_pil_image()
            if image is not self._image:
                self._image = image
                if self._target is not None:
                    self._target.release()
                    self._target = None
                if image is not None:
                    if size and size != [image.width, image.height]:
                        image = image.resize(size, PIL.Image.Resampling.BILINEAR)
                        logger.debug("Resized {} to {}x{}", self._filename, *size)
                    self.width = image.width
                    self.height = image.height
                    if self._flip in {'horizontal', 'both'}:
                        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                    if self._flip not in {'vertical', 'both'}:
                        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                    image = image.convert('RGBA').convert('RGBa')
                    self._target = RenderTarget.get(self.glctx, self.width, self.height, 8, srgb=True)
                    self._target.texture.write(image.tobytes())

    async def descend(self, engine, node, references, **kwargs):
        # An image is a leaf node from the perspective of the OpenGL world
        pass

    def similar_to(self, node):
        return super().similar_to(node) and (node.get('filename', 1, str) == self._filename or node.get('id', 1, str) == self.node_id)
