"""
Record window node
"""

from . import ProgramNode
from ...cache import SharedCache


class Record(ProgramNode):
    def render(self, node, references, *, time=None, fps=None, realtime=None, **kwargs):
        super().render(node, references, srgb=True, colorbits=8, **kwargs)
        if filename := node.get('filename', 1, str):
            path = SharedCache[filename]
            ext = path.suffix.lower()
            codec = node.get('codec', 1, str, 'h264')
            keep_alpha = node.get('keep_alpha', 1, bool, False)
            if ext in ('.mp4', '.mov', '.m4v', '.mkv', '.webm', '.ogg') or (ext == '.gif' and codec == 'gif'):
                pixfmt = node.get('pixfmt', 1, str, 'rgb8' if codec == 'gif' else ('yuva420p' if keep_alpha else 'yuv420p'))
                limit = node.get('limit', 1, float)
                options = {}
                for key, value in node.items():
                    if key not in {'filename', 'codec', 'pixfmt', 'limit', 'hidden'}:
                        options[key.replace('__', '-')] = str(value)
                path.write_video_frame(self._target.video_frame, time, fps=int(fps), realtime=realtime, codec=codec,
                                       pixfmt=pixfmt, options=options, limit=limit, alpha=keep_alpha)
            else:
                quality = node.get('quality', 1, int)
                path.write_image(self._target.image, quality=quality, alpha=keep_alpha)
