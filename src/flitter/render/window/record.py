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
                crf = node.get('crf', 1, int)
                profile = node.get('profile', 1, str)
                preset = node.get('preset', 1, str)
                limit = node.get('limit', 1, float)
                path.write_video_frame(self._target.video_frame, time, fps=int(fps), realtime=realtime, codec=codec,
                                       pixfmt=pixfmt, crf=crf, profile=profile, preset=preset, limit=limit, alpha=keep_alpha)
            else:
                quality = node.get('quality', 1, int)
                path.write_image(self._target.image, quality=quality, alpha=keep_alpha)
