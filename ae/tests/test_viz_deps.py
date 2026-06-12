"""The visualizer's new deps import and the imageio MP4 path round-trips."""
import os

import imageio
import numpy as np


def test_pillow_and_matplotlib_import():
    import PIL.Image  # noqa: F401
    import PIL.ImageDraw  # noqa: F401
    import matplotlib  # noqa: F401
    # use a non-interactive backend so plotting works headless
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401


def test_imageio_mp4_roundtrip(tmp_path):
    """imageio + imageio-ffmpeg can write and read back a 2-frame MP4.

    The env writes its replays the same way; Task 4 relies on this working.
    """
    path = os.path.join(str(tmp_path), "roundtrip.mp4")
    frames = [
        np.full((64, 64, 3), 30, dtype=np.uint8),
        np.full((64, 64, 3), 200, dtype=np.uint8),
    ]
    imageio.mimwrite(path, frames, fps=5, codec="libx264")
    assert os.path.exists(path) and os.path.getsize(path) > 0
    read_back = imageio.mimread(path)
    assert len(read_back) >= 1
    assert np.asarray(read_back[0]).shape[-1] == 3
