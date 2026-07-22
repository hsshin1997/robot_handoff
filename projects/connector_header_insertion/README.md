# Connector-header insertion grasp project

This folder is the clean, scalable workspace for deriving insertion grasps,
then back-propagating them into handoff and reorientation decisions. The raw
connector and gripper STL files remain in the repository's shared `parts/` and
`assets/` libraries; this project references them by path and SHA-256 instead
of duplicating roughly 150 MB of CAD.

## Recommended representation

A dense lookup table over world poses is not the right primary representation.
The robot and board live in continuous space, so such a table either explodes
in size or leaves gaps with no valid guarantee.

Use a hybrid instead:

1. keep a connector-relative, low-dimensional grasp family `T_P_E`;
2. retain deterministic samples as reusable seeds, evidence, and cache keys;
3. compose them with the measured insertion target at runtime;
4. run exact robot IK, joint-limit, collision, and continuous insertion-path
   checks for the current world state; and
5. cache only validated trajectories together with their calibration
   fingerprint and certified neighborhood.

The current JSON/CSV library therefore remains useful, but it is not a database
of every possible world pose. A nearby cached row is a proposal unless the
whole neighborhood around it was explicitly certified. See
[contracts/runtime_pose_sets.md](contracts/runtime_pose_sets.md) for the
mathematical and claim-level contract.

## Three-layer implementation

The lookup-table idea is now implemented as three conservative mathematical
sets. The table of sampled grasps is only evidence attached to these sets.

| layer | represented set | generated artifact | current result |
|---|---|---|---|
| 1. task space | continuous cells in `(contact mode, u, v, roll)` constrained by the connector, gripper, insertion path, and PCB | `generated/sets/insertion_task_set.json` | 2,304 cells: 0 `SAFE`, 1,344 `REJECTED`, 960 `UNRESOLVED` |
| 2. robot conditioned | task cells intersected with a target-specific GP7 same-branch insertion-path search | `generated/sets/robot_insertion_set.json` | 24 stratified cell centers evaluated: 12 provisional path witnesses, 12 without pre-insert IK, 0 certified cells |
| 3. handoff preimage | current grasps with an evidenced direct handoff or transfer-surface route into a certified receiver set | `generated/sets/handoff_preimage_set.json` | 0 direct, 0 transfer, 0 uncovered, 1 unknown class |

The empty `SAFE` set is intentional. A sampled pose can prove that one pose
collides, or provide a useful IK witness, but it cannot prove that every pose
inside a continuous cell is safe. `UNRESOLVED` is therefore the outer search
domain, not a production-safe set. The definitions and proof obligations are
in [contracts/three_layer_feasible_sets.md](contracts/three_layer_feasible_sets.md).

Run the complete pipeline from the repository root:

```bash
.venv/bin/python scripts/generate_insertion_grasps.py
.venv/bin/python scripts/build_insertion_task_set.py
.venv/bin/python scripts/build_robot_insertion_set.py \
  --config projects/connector_header_insertion/config/robot_set.yaml
.venv/bin/python scripts/build_handoff_preimage_set.py \
  --config projects/connector_header_insertion/config/preimage_set.yaml
.venv/bin/python scripts/visualize_insertion_sets.py
```

Before layer 2, replace `board_world_pose`, `robot`, and the world-frame
calibration fingerprint in [config/robot_set.yaml](config/robot_set.yaml) with
the measured runtime values. Before layer 3, replace the intentionally
incomplete `current_grasp_domain` and provide provenance-bound trajectory
evidence in [config/preimage_set.yaml](config/preimage_set.yaml).

The final command writes the inline interactive visualization under
`.codex/visualizations/2026/07/21/connector-header-feasible-set/`. The current
standalone copy is
[generated/visualization/insertion-feasible-set.html](generated/visualization/insertion-feasible-set.html).
It uses the supplied PCB, connector, gripper-body, and both finger STL files.
All 2,304 constructive cell-center poses are available as pose glyphs; 1,728
also carry a sampled phase-1 witness. Rejected cells are hidden by default, and
the selected center pose shows the detailed gripper geometry. Drag to orbit,
scroll to zoom, and move the insertion slider from the 40 mm pre-insert state
to seating.

## What is implemented now

The supplied connector orientation is clear and encoded as follows:

- the nine-position row is `+X_P`;
- the long mating-post leg is `+Z_P`;
- the short bent PCB tail is `-Y_P`; and
- insertion/down is therefore `-Y_P`, or `+Z_I` in the insertion frame.

The PCB has now also been registered. The circled feature is the unique
nine-hole row at `X_B = -66.468001 mm`, with exact nominal pitch `3.9624 mm`
and hole diameter `1.778 mm`. The selected nominal seated transform maps the
long posts toward the nearest left board edge, maps the short tails through
`-Z_B`, and aligns all nine tail centerlines with the row. Its full SI contract,
hash, tolerances, and assumptions are in
[config/pcb_socket.yaml](config/pcb_socket.yaml).

See [contracts/frames.md](contracts/frames.md) and the generated
[side-view diagram](generated/renders/insertion_definition.svg).

The phase-1 evidence generator uses the connector STL for deterministic antipodal contact
sampling and all three movable gripper source files for PCB-plane collision
support. The complete gripper assembly is parsed and hash-checked as the
registration reference. At each candidate aperture, the two finger meshes are
moved symmetrically from their recovered reference planes.

Run it from the repository root:

```bash
.venv/bin/python scripts/generate_insertion_grasps.py
```

To compose the connector-relative seeds with a user-supplied board pose, edit
`board_world_pose` in
[config/query_example.yaml](config/query_example.yaml), then run:

```bash
.venv/bin/python scripts/query_insertion_poses.py \
  --config projects/connector_header_insertion/config/query_example.yaml
```

Set `robot: A` or `robot: B` in that YAML. If perception already reports the
desired seated connector pose rather than the PCB pose, remove
`board_world_pose` and `pcb_socket` and provide the 4x4
`world_part_insert_pose: T_W_P_insert` instead. Remove
`selection.max_candidates` to compose all 1,460 current pre-insert survivors;
remember that phase 1 has no seated-plane survivor yet.

This fast command does not load MuJoCo or claim robot feasibility. The checked-in
example explicitly uses `purpose: preinsert_diagnostic`: it reports 1,460
eligible finite-library seeds, retains the best 64, marks the selection as
truncated and continuous-incomplete, and writes `T_W_E_insert: null` for every
row because none passed the seated-plane gate. The arithmetic seated pose is
kept only as an explicitly non-executable nominal witness.

For a pre-insert endpoint-only GP7 diagnostic using the repository's current
compiled TCP site:

```bash
.venv/bin/python scripts/query_insertion_poses.py \
  --config projects/connector_header_insertion/config/query_example.yaml \
  --solve-ik --acknowledge-provisional-tcp
```

Keep `selection.max_candidates` small for an interactive IK run. This IK mode
is deliberately labelled provisional because the supplied long-finger CAD has
no calibrated flange-to-contact-frame transform. It also does not check a
common IK branch, collision, or the insertion path.

For normal use, set `selection.purpose: insertion` or omit `selection`, since
`insertion` is the safe default. That mode requires both pre-insert and seated
compatibility and currently returns zero rows. It will begin emitting
`T_W_E_insert` only after the full geometry gate finds survivors.

The current 3,200-surface-sample run produces:

| gate | count |
|---|---:|
| sampled antipodal poses inside the reserved provisional stroke | 4,798 |
| both contact points inside the authored plastic-housing proxy | 4,297 |
| gripper mesh clears an infinite PCB plane at the 40 mm pre-insert pose | 1,460 |
| gripper mesh clears that plane at the fully seated pose | **0** |

The best current pose reaches approximately 34.35 mm along the 40 mm straight
insertion before violating the 0.5 mm PCB clearance, leaving approximately
5.65 mm to the seated pose. Among the 1,460 pre-insert survivors, the limiting
component is the right/lower finger for 672 poses, the left/upper finger for
559, and the body for 229.

The top-ranked pose is `g_4eff122621ab2031`: it closes along `+Z_P`, uses a
3.175 mm aperture, and has approach direction approximately
`[-0.9663, -0.2573, 0]_P`. At seating, its body remains 4.98 mm above the PCB
plane, while both fingertips extend about 5.15 mm through the infinite-plane
model. Those violating finger vertices lie to the row-end side of the part,
approximately `X_P = -26.56..-4.35 mm` and `Z_P = 0.44..12.01 mm`. This
location-specific witness is in `generated/reports/grasp_summary.json` and is
retained as single-pose evidence by the layer-1 cell cover.

This does **not** yet prove that direct insertion is impossible. The phase-1
grasp generator itself uses an infinite plane. Layer 1 additionally performs
bounded actual finite-PCB vertex-penetration checks on 24 representatives: 14
have positive nominal interpenetration witnesses and 10 are inconclusive. A
missing vertex witness is not collision-free proof because triangle/edge
intersection, the complete gripper assembly, and the continuum between samples
remain unchecked. The present long-finger assembly therefore still cannot be
claimed to retain the connector safely through seating. A release-before-seat
concept would need passive guidance over roughly the final 5.65 mm for the best
phase-1 pose, so a lead-in fixture, board-edge clearance, or small insertion
tool remains worth evaluating.

## Folder contract

```text
projects/connector_header_insertion/
├── README.md                         human entry point and staged plan
├── config/
│   ├── grasp_generation.yaml        authored grasp assumptions and budgets
│   ├── pcb_socket.yaml              exact nominal PCB/socket registration
│   ├── query_example.yaml           user-editable runtime world target
│   ├── task_set.yaml                layer-1 continuous parameter cover
│   ├── robot_set.yaml               layer-2 target and GP7 search policy
│   └── preimage_set.yaml            layer-3 handoff evidence declarations
├── contracts/
│   ├── frames.md                    B/P/I/E/G frame and transform semantics
│   ├── runtime_pose_sets.md         continuous-family/cache/claim contract
│   └── three_layer_feasible_sets.md formal three-layer set contract
├── source/
│   └── asset_inventory.yaml         immutable CAD provenance and hashes
└── generated/                       reproducible; do not hand-edit
    ├── grasps/
    │   ├── phase1_pose_library.json all candidates, transforms, gates, clearances
    │   └── phase1_pose_table.csv    flat table for later current-grasp lookup
    ├── reports/
    │   └── grasp_summary.json       counts, ranking, assumptions, next gates
    ├── queries/
    │   └── query_example_result.json composed targets from example YAML
    ├── sets/
    │   ├── insertion_task_set.json  layer-1 inner/outer cell sets
    │   ├── robot_insertion_set.json layer-2 target-conditioned witnesses
    │   └── handoff_preimage_set.json layer-3 direct/transfer partition
    ├── visualization/
    │   └── insertion-feasible-set.html interactive CAD-backed viewer
    └── renders/
        ├── insertion_definition.svg orientation check
        └── grasp_summary.svg        gate/family overview
```

Generic code lives in
[`mujoco_sim/modeling/insertion_grasps.py`](../../mujoco_sim/modeling/insertion_grasps.py),
the command-line adapter lives in
[`scripts/generate_insertion_grasps.py`](../../scripts/generate_insertion_grasps.py),
and focused tests live in
[`tests/test_insertion_grasps.py`](../../tests/test_insertion_grasps.py). This
keeps part-specific numbers out of the reusable algorithm.

Runtime target composition lives in
[`mujoco_sim/modeling/insertion_query.py`](../../mujoco_sim/modeling/insertion_query.py),
with CLI
[`scripts/query_insertion_poses.py`](../../scripts/query_insertion_poses.py).
The PCB registration and runtime query are covered by
[`tests/test_connector_pcb_socket.py`](../../tests/test_connector_pcb_socket.py)
and [`tests/test_insertion_query.py`](../../tests/test_insertion_query.py).

The three set layers live in
[`mujoco_sim/modeling/insertion_task_set.py`](../../mujoco_sim/modeling/insertion_task_set.py),
[`mujoco_sim/planner/robot_insertion_set.py`](../../mujoco_sim/planner/robot_insertion_set.py),
and
[`mujoco_sim/planner/handoff_preimage_set.py`](../../mujoco_sim/planner/handoff_preimage_set.py).
Their focused tests are `tests/test_insertion_task_set.py`,
`tests/test_robot_insertion_set.py`, and `tests/test_handoff_preimage_set.py`.

## What “all poses” means

The physical solution is a continuous, often disconnected subset of `SE(3)`
and robot configuration space, so it cannot be listed literally. The primary
representation is now a cover of two authored contact modes by 2,304 bounded
`(u, v, roll)` cells:

- `u` and `v` locate the grasp centre on an opposing housing-patch family;
- `roll` is continuous and periodic on `[-pi, pi)`;
- `SAFE` cells form the certified inner approximation;
- `SAFE union UNRESOLVED` forms the conservative outer approximation; and
- sampled `T_P_E` rows are witnesses, numerical seeds, and visualization
  representatives attached to cells.

The underlying deterministic witness library still uses:

- 3,200 area-stratified connector surface samples;
- 24 uniform roll/approach samples (15-degree spacing) per opposing contact pair;
- antipodal/friction, reserved aperture, ideal pad support, and palm-depth
  filters; and
- deterministic SE(3) non-maximum suppression.

The library records these budgets and input hashes. Increasing its sample
budget broadens the evidence attached to cells, but failure to sample a pose is
not proof that no continuous pose exists. Refining the cell cover narrows the
outer approximation; it does not by itself turn an unresolved cell into a safe
one.

Every runtime result records the full pose-library SHA, connector SHA, socket
compatibility, world-frame ID/calibration fingerprint, eligible count before a
limit, and whether selection was truncated. A socket is rejected if its
connector hash, compatible project ID, or insertion-axis mapping does not match
the grasp library.

Each row stores `T_P_E`. A direct part target uses:

```text
T_W_E = T_W_P @ T_P_E
```

A board-pose target first uses the authored socket:

```text
T_W_P_insert = T_W_B @ T_B_P_insert
```

Each grasp ID is content-addressed from its quantized relative pose, contacts,
and required aperture, so unchanged poses retain their identity if report
ordering changes. The CSV is therefore a seed/index table for matching a
current grasp and proposing insertion or handoff searches. It is not a
certified handoff table and it never replaces runtime validation.

## Remaining work to certification

### 1. Close the physical gripper assumptions

Before treating rankings as hardware decisions:

1. measure total jaw aperture, per-finger motion, joint axes, and limit reserve;
2. calibrate the flange-to-contact-frame transform and its uncertainty;
3. confirm the recovered flat faces are the intended gripping pads;
4. identify or supply the eleven small components present only in the complete
   assembly STL; and
5. confirm the plastic-housing contact mask from the source CAD or a marked-up
   drawing.

### 2. Upgrade layer 1 from witnesses to whole-cell proofs

Build collision-ready gripper components at each aperture, then prove:

1. complete part-versus-body/finger separation away from intended contacts;
2. actual pad-footprint capture on plastic with no pin loading;
3. finite PCB, holes, nearby components, fixture, and board-edge clearance;
4. open-jaw approach and closing sweeps; and
5. the continuous insertion and correction-yaw envelope.

Use interval bounds, conservative distance fields, or adaptive subdivision to
prove every constraint over an entire cell and every continuous path state.
This will decide whether the current `0 seated` phase-1 result is a real tool
conflict or an artifact of the conservative plane.

### 3. Add insertion mechanics

For every geometry survivor, estimate or measure insertion force and moment,
then require sufficient frictional wrench margin under the available actuator
force. Include connector compliance, allowed housing pressure, pose tolerance,
manufacturing variation, and pin/hole lead-in.

### 4. Certify the robot-conditioned layer

The current GP7 layer composes the target and preserves a common numerical IK
branch across nine insertion samples. To certify it:

1. use the calibrated flange-to-contact-frame transform;
2. enumerate or conservatively cover the relevant IK branches;
3. certify robot, gripper, part, board, fixture, and other-arm swept collision;
4. prove continuity between path samples and over the complete task cell; and
5. bind cached paths to stable cell IDs and calibration fingerprints.

This is the backward-reachability idea in the existing
[detailed handoff pipeline](../../docs/handoff_pipeline_detailed.md), with the
insertion task supplying the receiving goal set. The
[MuJoCo handoff pipeline](../../docs/mujoco_handoff_pipeline.md) defines the
later planning and qualification boundary.

### 5. Supply layer-3 direct and transfer evidence

For each measured current-grasp class:

1. validate a synchronized donor/receiver handoff trajectory into a certified
   receiver cell; or
2. validate a directed place trajectory, transfer-surface placement, pick
   trajectory, and final handoff trajectory.

The transfer stage is a deliberate directed task-graph edge. Failed or absent
search remains `UNKNOWN`; only an exhaustive, domain-bound emptiness
certificate may produce `UNCOVERED`.

## Current certification boundary

No insertion cell is currently certified. Layer 1 has an empty `SAFE` inner
set; layer 2 has 12 useful but explicitly provisional GP7 path witnesses; and
layer 3 refuses to promote those witnesses to receiver goals. Full
part/gripper collision, complete pad capture, authoritative stroke and force,
continuous finite-board clearance, mechanics, TCP calibration, and swept
workcell collision remain open. No downstream executor should interpret the
outer set or visualization as production-safe.
