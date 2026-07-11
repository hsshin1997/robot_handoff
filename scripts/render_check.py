"""Headless render spot-check: saves camera images of the loaded scene."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pybullet as p

from scene import Scene


def snap(path: str, yaw: float, pitch: float = -20, dist: float = 2.6,
         target=(0.42, 0.0, 0.45), w: int = 960, h: int = 720) -> None:
    view = p.computeViewMatrixFromYawPitchRoll(list(target), dist, yaw, pitch, 0, 2)
    proj = p.computeProjectionMatrixFOV(55, w / h, 0.05, 10)
    img = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER)
    rgba = np.reshape(img[2], (h, w, 4)).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(rgba[:, :, :3]).save(path)
    except ImportError:
        np.save(path + ".npy", rgba)
    print("saved", path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    scene = Scene(load_visuals=True)
    snap(os.path.join(out, "cell_front.png"), yaw=90)   # looking along -y
    snap(os.path.join(out, "cell_iso.png"), yaw=45, pitch=-30)
    snap(os.path.join(out, "cell_side.png"), yaw=0)     # looking along -x
    scene.disconnect()
