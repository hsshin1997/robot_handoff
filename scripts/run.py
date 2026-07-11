"""Load the cell, run the handoff search, print the result.

  python scripts/run.py            # headless: search + result summary
  python scripts/run.py --best     # scan every candidate, keep highest score
  python scripts/run.py --gui      # search, then replay the handoff visually
  python scripts/run.py --no-search  # just load the scene and print a summary
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pybullet as p

import kin
from handoff import HandoffPlanner
from scene import Scene


def print_scene_summary(scene: Scene) -> None:
    for r in (scene.robotA, scene.robotB):
        pos, quat = p.getBasePositionAndOrientation(r.body)
        print(f"robot {r.name}: base {np.round(pos, 3).tolist()} "
              f"yaw {np.degrees(p.getEulerFromQuaternion(quat)[2]):.0f} deg, "
              f"q(home) {np.round(r.get_q(), 2).tolist()}")
    print(f"bodies: workcell {scene.workcell_ids}, pcb {scene.pcb_id}, "
          f"bin {scene.bin_id}, nest {scene.nest_id}, part {scene.part_id}")


def print_report(rep, dt: float) -> None:
    print(f"\nsearch: {rep.n_candidates} candidates in {dt:.1f}s; "
          f"G* = {rep.G_star}")
    if rep.stats:
        print("stats:", dict(rep.stats))
    if not rep.feasible:
        print(f"\nNO FEASIBLE HANDOFF — dominant failing constraint: "
              f"{rep.dominant_failure()}")
        return
    pn = rep.plan
    np.set_printoptions(precision=4, suppress=True)
    print(f"\nFEASIBLE HANDOFF (score {pn.score:.3f}, "
          f"est. physical execution {pn.exec_time:.2f}s)")
    print(f"  grasp     : {pn.grasp_name}")
    print(f"  X_h (part pose at transfer):\n{pn.X_h}")
    print(f"  qA        : {np.round(pn.qA, 4).tolist()}")
    print(f"  qB_grasp  : {np.round(pn.qB_grasp, 4).tolist()}")
    print(f"  qB_insert : {np.round(pn.qB_insert, 4).tolist()}")


def _glide(scene, checker, robot, q_from, q_to, holder, T_fp, steps=40, dt=0.025):
    """Kinematic joint-space glide for the GUI replay."""
    q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
    hold = {"A": scene.robotA, "B": scene.robotB, None: None}[holder]
    for t in np.linspace(0.0, 1.0, steps):
        robot.set_q((1 - t) * q_from + t * q_to)
        if hold is not None:
            checker.place_part(hold, T_fp)
        time.sleep(dt)


def replay_regrasp(scene: Scene, checker, rplan, home_qA, home_qB) -> None:
    """A places the part on the nest -> retreats -> re-picks with the new
    grasp -> then the cached direct handoff."""
    T_old = np.asarray(scene.cfg["T_flangeA_part"], dtype=float)
    A = scene.robotA
    print("replay: A places the part on the nest")
    scene.robotB.set_q(home_qB)
    _glide(scene, checker, A, home_qA, rplan.qA_place, "A", T_old)
    checker.set_part_world(rplan.X_place)          # release: part rests
    print("replay: A retreats and approaches the re-pick grasp")
    _glide(scene, checker, A, rplan.qA_place, home_qA, None, None)
    _glide(scene, checker, A, home_qA, rplan.qA_pick, None, None)
    time.sleep(0.4)                                # close fingers
    print("replay: A re-picked; running the direct handoff")
    _glide(scene, checker, A, rplan.qA_pick, rplan.handoff.qA, "A", rplan.g_new)
    _replay_from_presented(scene, checker, rplan.handoff, rplan.g_new,
                           home_qA, home_qB)


def _glide_segment(scene, checker, robot, qs, holder, T_fp, steps=14):
    """Glide through a checked segment (list of configs)."""
    for a, b in zip(qs[:-1], qs[1:]):
        _glide(scene, checker, robot, a, b, holder, T_fp, steps=steps)


def _replay_from_presented(scene, checker, plan, T_fA, home_qA, home_qB):
    """Handoff replay starting with A already presenting at plan.qA."""
    A, B = scene.robotA, scene.robotB
    T_fB = kin.inv_T(plan.g)
    seg = plan.segments or {}
    print("replay: B approaches (pre-grasp) and grasps")
    if "B_approach" in seg:
        _glide(scene, checker, B, home_qB, seg["B_approach"][0], "A", T_fA)
        _glide_segment(scene, checker, B, seg["B_approach"], "A", T_fA)
    else:
        _glide(scene, checker, B, home_qB, plan.qB_grasp, "A", T_fA)
    time.sleep(0.4)                                   # co-grasp instant
    print("replay: A releases, retreats, returns home")
    if "A_retreat" in seg:
        _glide_segment(scene, checker, A, seg["A_retreat"], "B", T_fB)
        _glide(scene, checker, A, seg["A_retreat"][-1], home_qA, "B", T_fB)
    else:
        _glide(scene, checker, A, plan.qA, home_qA, "B", T_fB)
    print("replay: B transits to pre-insert and descends")
    if "B_to_preinsert" in seg:
        _glide_segment(scene, checker, B, seg["B_to_preinsert"], "B", T_fB, steps=10)
        _glide_segment(scene, checker, B, seg["B_insert_approach"], "B", T_fB, steps=10)
    else:
        qs = [plan.qB_grasp] + list(plan.waypoints) + [plan.qB_insert]
        _glide_segment(scene, checker, B, qs, "B", T_fB, steps=12)
    print("replay: done (insert pose reached)")


def replay(scene: Scene, checker, plan, home_qA, home_qB) -> None:
    """A presents -> B grasps -> A releases & retreats -> B moves to insert."""
    T_fA = np.asarray(scene.cfg["T_flangeA_part"], dtype=float)
    print("replay: A approaches (pre-present) and presents")
    scene.robotB.set_q(home_qB)
    seg = plan.segments or {}
    if "A_approach" in seg:
        _glide(scene, checker, scene.robotA, home_qA, seg["A_approach"][0], "A", T_fA)
        _glide_segment(scene, checker, scene.robotA, seg["A_approach"], "A", T_fA)
    else:
        _glide(scene, checker, scene.robotA, home_qA, plan.qA, "A", T_fA)
    _replay_from_presented(scene, checker, plan, T_fA, home_qA, home_qB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true", help="open the PyBullet GUI and replay")
    ap.add_argument("--best", action="store_true", help="scan all candidates, keep best score")
    ap.add_argument("--no-search", action="store_true", help="scene summary only")
    ap.add_argument("--no-regrasp", action="store_true",
                    help="skip the regrasp fallback when direct handoff fails")
    ap.add_argument("--thorough", action="store_true",
                    help="full exhaustive search instead of the fast "
                         "cache -> budgeted-search -> regrasp pipeline")
    ap.add_argument("--fastest", action="store_true",
                    help="exhaustive search minimizing estimated PHYSICAL "
                         "execution time of the handoff")
    ap.add_argument("--config", default="config/cell.yaml")
    args = ap.parse_args()

    scene = Scene(config_path=args.config, gui=args.gui)
    print_scene_summary(scene)
    if args.no_search:
        if args.gui:
            _spin()
        scene.disconnect()
        return

    planner = HandoffPlanner(scene)
    plan = rplan = None
    if args.thorough or args.best or args.fastest:
        t0 = time.time()
        rep = planner.search(return_best=args.best or args.fastest,
                             objective="time" if args.fastest else "margin")
        print_report(rep, time.time() - t0)
        plan = rep.plan
    else:
        from regrasp import RegraspPlanner
        rg = RegraspPlanner(scene, planner)
        t0 = time.time()
        kind, result, timings = planner.plan_fast(
            regrasp_planner=None if args.no_regrasp else rg)
        dt = time.time() - t0
        print(f"\nfast pipeline: {kind or 'REJECT'} in {dt:.2f}s  "
              f"({', '.join(f'{k} {v:.2f}s' for k, v in timings.items())})")
        if kind == "regrasp":
            rplan = result
            print_regrasp(rplan)
        elif kind is not None:
            plan = result
            np.set_printoptions(precision=4, suppress=True)
            print(f"FEASIBLE HANDOFF via {kind} (grasp {plan.grasp_name}, "
                  f"est. physical execution {plan.exec_time:.2f}s)")
            print(f"  X_h:\n{plan.X_h}")
            print(f"  qA        : {np.round(plan.qA, 4).tolist()}")
            print(f"  qB_grasp  : {np.round(plan.qB_grasp, 4).tolist()}")
            print(f"  qB_insert : {np.round(plan.qB_insert, 4).tolist()}")

    if plan is None and rplan is None and not args.no_regrasp and args.thorough:
        from regrasp import RegraspPlanner
        print("\ndirect handoff infeasible -> trying the regrasp branch "
              "(place on nest, re-pick)")
        t0 = time.time()
        rg = RegraspPlanner(scene, planner)
        rplan = rg.find_regrasp(planner.T_fA_part)
        if rplan is None:
            print(f"no regrasp plan either ({time.time()-t0:.1f}s) — part rejected")
        else:
            print_regrasp(rplan)

    if args.gui:
        if plan is not None or rplan is not None:
            p.resetDebugVisualizerCamera(cameraDistance=1.9, cameraYaw=55,
                                         cameraPitch=-25, cameraTargetPosition=[0.42, 0, 0.55])
            while p.isConnected():
                scene.robotA.set_q(planner.home_qA)
                scene.robotB.set_q(planner.home_qB)
                if plan is not None:
                    replay(scene, planner.c, plan, planner.home_qA, planner.home_qB)
                else:
                    replay_regrasp(scene, planner.c, rplan, planner.home_qA, planner.home_qB)
                time.sleep(1.5)
        else:
            _spin()
    scene.disconnect()


def print_regrasp(rplan) -> None:
    np.set_printoptions(precision=4, suppress=True)
    print("\nREGRASP PLAN")
    print(f"  placement : {rplan.placement} on the nest")
    print(f"  qA_place  : {np.round(rplan.qA_place, 4).tolist()}")
    print(f"  qA_pick   : {np.round(rplan.qA_pick, 4).tolist()}")
    print(f"  then handoff with grasp {rplan.handoff.grasp_name}:")
    print(f"  qA        : {np.round(rplan.handoff.qA, 4).tolist()}")
    print(f"  qB_grasp  : {np.round(rplan.handoff.qB_grasp, 4).tolist()}")
    print(f"  qB_insert : {np.round(rplan.handoff.qB_insert, 4).tolist()}")


def _spin() -> None:
    print("GUI running — Ctrl-C to exit")
    try:
        while p.isConnected():
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
