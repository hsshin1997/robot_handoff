"""After changing the part, gripper, or grasp set: run the full search once
and print the values to pin in tests/test_handoff.py (KNOWN_GOOD_XH /
KNOWN_GOOD_GRASP) and tests/test_rl.py (the action vector). Also rebuilds
stale caches. Run on a real machine (takes minutes, not sandbox-friendly).

  python scripts/repin_tests.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

import kin
from handoff import HandoffPlanner
from regrasp import RegraspPlanner
from scene import Scene


def main() -> None:
    scene = Scene()
    pl = HandoffPlanner(scene)
    NOM = np.eye(4); NOM[0, 3] = 0.200
    pl.T_fA_part = NOM
    pl.T_part_fA = kin.inv_T(NOM)
    print("G (width-filtered):", [n for n, _ in pl.G])

    rep = pl.search()
    if not rep.feasible:
        print("\nNo direct handoff for the nominal grasp with this config "
              f"(stats {dict(rep.stats)}).")
        print("Checking the regrasp branch instead...")
        rg = RegraspPlanner(scene, pl)
        rg.build_table()            # full rebuild, definitive verdicts
        plan = rg.find_regrasp(NOM)
        print("regrasp:", "FOUND" if plan else "none — inspect grasp set")
        return

    pn = rep.plan
    print("\nPin these in tests/test_handoff.py:")
    print(f"KNOWN_GOOD_GRASP = \"{pn.grasp_name}\"")
    print("KNOWN_GOOD_XH = np.array(", np.array2string(
        np.round(pn.X_h, 6), separator=", "), ")")
    sp = scene.cfg["handoff_search"]
    lo = np.array([sp["x"][0], sp["y"][0], sp["z"][0]])
    hi = np.array([sp["x"][1], sp["y"][1], sp["z"][1]])
    a = 2 * (pn.X_h[:3, 3] - lo) / (hi - lo) - 1
    yaw = float(np.arctan2(pn.X_h[1, 0], pn.X_h[0, 0]))
    print("\nPin this action in tests/test_rl.py "
          "(x, y, z, yaw/(pi/2), roll/pi):")
    print(np.round(np.array([a[0], a[1], a[2], yaw / (np.pi / 2), 0.0]), 3))
    print("\nRebuilding the regrasp table with the corrected grasp set...")
    RegraspPlanner(scene, pl).build_table()


if __name__ == "__main__":
    main()
