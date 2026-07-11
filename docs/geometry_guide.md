# Geometry Guide — Models, Constraints, and How to Change Them

What geometric models the system needs, exactly how each is used, every
geometric constraint baked into the planner, and step-by-step recipes for
swapping the part, the gripper, the robots, or the workcell.

Short answer: yes, the user supplies **three models** — robot arm (URDF +
meshes), gripper (URDF link geometry), part (STL) — plus the workcell
collision geometry. All are swappable without touching planner logic; the
oracle reads whatever geometry is loaded.

---

## 1. The geometry stack

| layer | file(s) | collision representation | used for |
|---|---|---|---|
| robot arms | `assets/gp7/gp7.urdf` + `meshes/collision/*.stl` | per-link meshes from ROS-Industrial | self/arm-arm/arm-env collision, FK/IK chain |
| gripper | `gripper` link inside `gp7.urdf` | 3 boxes: palm + 2 fingers | co-grasp clearance, env collision, defines the TCP |
| part | `parts/<name>/<file>.stl` (binary, meters) | **convex hull** of the mesh | part-vs-everything collision, grasp/placement geometry from its bbox |
| workcell | `assets/workcell/collision_boxes.yaml` | 30 hand-simplified boxes | environment collision (the 724k-tri STL is visual-only, never collision-checked) |
| fixtures | `config/cell.yaml` (pcb/bin/nest) | boxes built at load time | environment collision, placement surface |

Everything meets in `kin.CollisionChecker` — one clearance threshold
(default **2 mm**, `clearance` arg) applied to every pair, with three
explicit whitelists: robot base ↔ its pedestal, self-adjacent links, and the
part ↔ the holder's gripper (both grippers at the co-grasp instant).

## 2. The part model

**Requirements**: binary STL (pybullet's loader segfaults on ASCII), in
meters, watertight enough for a sensible convex hull. The part frame is the
**bbox center** — the loader recenters automatically, and all grasps,
placements, and the insert pose refer to that center.

**How its geometry is used**:
- *Collision*: convex hull. A strongly concave part (e.g. an L-bracket) is
  checked as its hull — conservative for the part itself but permissive for
  things entering its concavity. Decompose concave parts into convex pieces
  (V-HACD) if that matters.
- *Grasp modes* (`rl_env.GraspSampler`): derived from the bbox — every
  (approach axis ±x/±y/±z, closing axis) pair whose closing-width fits the
  finger gap. This is the "boxy part" assumption: grasps are axis-aligned
  antipodal pinches. Cylinders and boxes work naturally; freeform parts need
  hand-added entries in `grasp_set_G`.
- *Stable placements* (`regrasp.face_down_rotations`): the 6 axis-aligned
  face-down orientations, resting height = bbox half-extent of the down
  axis. Also a boxy-part assumption; parts that rest on edges/points (spheres,
  cones) would need `trimesh.poses.compute_stable_poses` instead.

**Graspability constraint** (hard): the part must have at least one bbox
extent `< finger gap − 2 mm` (gap is 24 mm with the current gripper → at
least one dimension under 22 mm). Quick check:

```python
from rl_env import GraspSampler; from scene import Scene
s = GraspSampler(Scene())          # raises if nothing fits
print(len(s.modes), "grasp modes") # >= 2 recommended
```

**To change the part**:
1. Put the binary-STL (meters) under `parts/<name>/`, set `part_mesh` in
   `cell.yaml`. (ASCII → binary converter example: `parts/conn_header/`.)
2. Update `T_world_insert` — part-frame pose at insertion (PCB top + part
   half-height + hover), and orient it so at least one grasp in G can
   approach.
3. Delete `models/regrasp_table.json` and `models/plan_cache.json`.
4. Run the graspability check above, then all four test suites, then
   `scripts/coverage.py --n 20`.

## 3. The gripper model

Defined entirely inside `gp7.urdf` (link `gripper` + fixed joints
`tool0-gripper` and `gripper-tcp`). Current geometry:

```
palm    : 90 x 95 x 100 mm box    (proportions from the original gripper CAD)
fingers : 5 x 2 x 100 mm boxes    (contact face 5 mm wide, 2 mm thin)
gap     : 24 mm between inner finger faces (centers at y = ±13 mm)
TCP     : z = 0.200 in gripper frame = fingertip-center plane
mount   : gripper z-axis = tool0 x-axis (rpy 0 90° 0)
```

**Geometric constraints it creates**:
- *Finger gap vs part width* — bounds which grasp modes exist (see §2).
- *Co-grasp interference* — at the transfer instant two grippers hold one
  small part at perpendicular approaches. Finger slimness is what makes this
  solvable: bulky fingers physically exclude each other (we hit exactly this —
  10 mm-thick fingers made every co-grasp collide; 2 mm fingers fixed it).
- *TCP length* — sets how far the flange sits from the part (0.200 m).
  Longer tools reach deeper but reduce effective arm reach; shorter tools
  force the flange closer to the arm's own body (a 55 mm shortening once
  pushed A into self-collision at the old search grid — retune
  `handoff_search` after changing it).
- *Palm bulk* — matters when re-picking from the nest with horizontal
  approaches (palm must clear the plate; edge placements exist for this).

**To change the gripper** (single source of truth is the URDF, but the TCP
offset leaks into three more places — update all of them):
1. Edit the palm/finger `<box>` sizes, finger `origin` y (= gap/2 +
   thickness/2) and the `gripper-tcp` joint z in `gp7.urdf`.
2. If the TCP offset changed from 0.200: update every ±0.200 translation in
   `cell.yaml` (`T_flangeA_part`, all `grasp_set_G` entries) and
   `TCP_OFFSET` in `src/rl_env.py`.
3. If the gap changed: `finger_gap` default in `rl_env.GraspSampler`.
4. Delete both caches; run tests; expect to retune `handoff_search`;
   re-measure coverage.

Note: the gripper is **fixed-open** — jaw motion is not simulated. Grasping
is modeled as a rigid attachment at the TCP; the fingers exist for collision
realism. (This is the project's scope decision; an articulated gripper would
need finger joints in the URDF plus open/close handling in `scene.py`.)

## 4. The robot model

`assets/gp7/gp7.urdf`, from ROS-Industrial `motoman_gp7_support`, with
visual meshes decimated for the GUI and the original collision meshes.
Joint limits are read from the URDF at load (`Robot.lower/upper`).

**Geometric/kinematic constraints it creates**:
- *Reach sphere* — prefilter bounds `reach_min/max = 0.18/0.90 m` in
  `src/handoff.py` (GP7 reach is 0.927 m; headroom left on purpose).
- *Joint limits + margin* — `joint_limit_margin` (0.09 rad) in config.
- *Rated joint speeds* — `GP7_SPEED` in `src/handoff.py`, used only for the
  physical execution-time estimate.
- *Self-collision* — checked from the URDF collision meshes with an
  auto-built allowed-pair matrix (adjacent links + pairs touching at home).

**To change the robot**:
1. Replace the URDF + meshes; keep link names `tool0` and `tcp` (or update
   `scene.Robot`'s expectations). Exactly 6 revolute joints are assumed.
2. Update `robotA_base` / `robotB_base` in config, and `home_qA/B`.
3. Update `reach_min/max` and `GP7_SPEED` in `src/handoff.py`.
4. Delete caches, run tests, retune `handoff_search` for the new workspace.

## 5. The workcell model

Visual: `assets/workcell/workcell.stl` (raw 724k tris, GUI only — headless
planning skips it). Collision: `assets/workcell/collision_boxes.yaml`, ~30
axis-aligned boxes (pedestals, extrusion frame, scanner heads) in mm
(scaled at load). **The raw mesh is never collision-checked** — that's a
deliberate performance/robustness rule.

**To change the workcell**: replace the STL and regenerate the box set
(each entry = `center` + `half_extents`, mm). Boxes were auto-extracted from
the STL's connected components and hand-checked; for a new cell, bounding
boxes of the major structures are enough. Keep them slightly generous —
collision-safe beats pixel-accurate. Fixture stands (pcb/bin/nest) are
auto-generated from config; don't add them here.

## 6. Constraint summary (one table)

| constraint | value | where defined |
|---|---|---|
| collision clearance | 2 mm | `kin.CollisionChecker(clearance=...)` |
| finger gap | 24 mm | URDF finger origins; `rl_env.GraspSampler(finger_gap=)` |
| graspable width | < gap − 2 mm | derived |
| TCP offset | 200 mm | URDF `gripper-tcp`; config translations; `rl_env.TCP_OFFSET` |
| reach prefilter | 0.18–0.90 m | `handoff.HandoffPlanner` |
| joint-limit margin | 0.09 rad | config `handoff_search` |
| IK acceptance | 1 mm / 0.6° | `kin.POS_TOL/ROT_TOL` |
| nest rest gap | 4 mm | `regrasp.REST_EPS` |
| re-pick approach | not from below (world z-component ≤ 0.3) | `regrasp.find_regrasp` |
| co-grasp dwell (time est.) | 0.3 s | `handoff.COGRASP_DWELL` |
| part↔gripper contact | whitelisted for holder(s) only | `kin.CollisionChecker` |
| singularity gate | w(q) ≥ 5th-percentile w over random configs | config `singularity_w_percentile`; `kin.manipulability` |
| pre-grasp / pre-present back-off | 40 mm along tool axis | config `approach.d_pre` |
| A retreat after release | 60 mm | config `approach.d_retreat` |
| pre-insert hover | 30 mm above insert pose | config `approach.d_app` |
| swept checks per segment | 3 interpolated configs | config `approach.n_sweep` |
| part symmetry orbit | per-part rotations (pin pattern, not body!) | config `part_symmetry_rpy_deg` |

## 7. Assumptions worth knowing (the honest list)

- **Boxy-part model**: grasp modes and stable placements come from the bbox.
  Fine for relays, headers, capacitors, most SMT/THT components; freeform
  parts need hand-authored grasps/placements.
- **Convex-hull part collision**; concave parts are approximated.
- **Fixed-open fingers**; grasp = rigid attachment at TCP, no contact
  mechanics, no grasp-force reasoning (per project scope).
- **Kinematic world**: no dynamics, no friction, no part settling on the
  nest beyond the geometric rest pose.
- **Endpoint + waypoint collision checking**, not swept volumes: paths are
  validated at interpolated configurations (`n_path_waypoints`), which can
  in principle miss a collision between waypoints — increase the count for
  tighter cells.
