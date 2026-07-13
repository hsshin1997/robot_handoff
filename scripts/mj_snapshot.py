"""Offscreen MuJoCo rendering — works with PLAIN python (no mjpython, no
GUI, no framework-build requirement). Use this when the interactive viewer
misbehaves, or for headless visual checks.

  python scripts/mj_snapshot.py                    # 3 PNGs of the scene
  python scripts/mj_snapshot.py --video out.mp4    # 4 s clip (needs: pip install imageio imageio-ffmpeg)
"""
import argparse
import os
import struct
import sys
import zlib

import mujoco
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
MODEL = os.path.join(ROOT, "mujoco_sim", "models", "scene.xml")


def make_cam(m, lookat, dist, azim, elev):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.azimuth = azim
    cam.elevation = elev
    return cam


def write_png(path, rgb):
    """Write uint8 RGB without adding an image-library dependency."""
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError(f"expected RGB image, got {rgb.shape}")
    raw = b"".join(b"\0" + rgb[row].tobytes() for row in range(height))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "mujoco_sim"))
    ap.add_argument("--video", metavar="MP4", help="render a 4 s settling clip")
    ap.add_argument("--wh", default="1280x800")
    args = ap.parse_args()
    w, h = (int(v) for v in args.wh.split("x"))

    from mujoco_sim.simulation.workcell import WorkcellSim
    sim = WorkcellSim(model_path=MODEL)
    m, d = sim.model, sim.data
    renderer = mujoco.Renderer(m, height=h, width=w)

    views = [("overview", [0.425, -0.175, 0.50], 3.15, 135, -18),
             ("front",    [0.425, -0.175, 0.50], 2.85, 90, -8),
             ("top",      [0.425, -0.175, 0.35], 3.00, 90, -88)]

    if args.video:
        import imageio
        cam = make_cam(m, *views[0][1:])
        frames = []
        fps = 30
        for i in range(4 * fps):
            for _ in range(int(1 / (fps * m.opt.timestep))):
                mujoco.mj_step(m, d)
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())
        imageio.mimsave(args.video, frames, fps=fps)
        print("wrote", args.video)
        return

    for name, lookat, dist, azim, elev in views:
        cam = make_cam(m, lookat, dist, azim, elev)
        renderer.update_scene(d, camera=cam)
        img = renderer.render()
        path = os.path.join(args.out, f"scene_{name}.png")
        os.makedirs(args.out, exist_ok=True)
        write_png(path, img)
        print("wrote", path)


if __name__ == "__main__":
    main()
