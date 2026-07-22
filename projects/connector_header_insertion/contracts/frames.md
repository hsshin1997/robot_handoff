# Connector-header insertion frame contract

All generated translations are metres. The source STL files remain untouched
and are converted from millimetres at ingestion.

## Frames

- `P`: native connector STL frame. `+X_P` follows the nine-pin row, `+Z_P`
  follows the long mating-post leg, and `-Y_P` follows the short bent PCB tail.
- `B`: native PCB STL frame, normalized so the top surface is `Z_B = 0`.
  `-Z_B` points through the board and `-X_B` points toward the nearest left
  edge at the selected socket.
- `I`: insertion frame. Its origin lies on the nominal PCB seating plane,
  `+X_I` follows the row, and `+Z_I` is down/into the PCB. Therefore the user’s
  side view is a long horizontal leg followed by a short downward leg.
- `E`: ideal contact frame. `+Y_E` is the jaw-closing line, `+Z_E` points from
  palm toward the contact plane, and `+X_E` is pad width.
- `G`: reference complete-gripper STL frame.

`T_X_Y` maps coordinates in `Y` into `X`. A generated grasp stores `T_P_E`.
For a measured part pose, use:

```text
T_W_E = T_W_P @ T_P_E
```

Do not invert `T_P_E` in that composition.

For the user-circled nine-hole socket, the authored nominal seated pose is:

```text
T_B_P_insert =
  [ 0  0 -1 -0.065819158]
  [-1  0  0  0.039797663]
  [ 0  1  0 -0.003252597]
  [ 0  0  0  1           ]
```

It maps `-Y_P` to `-Z_B` for insertion, `+Z_P` to `-X_B` so the long posts
point toward the nearest board edge, and all nine tail centerlines to the
circled holes. Given a measured board pose:

```text
T_W_P_insert = T_W_B @ T_B_P_insert
T_W_E_insert = T_W_P_insert @ T_P_E
```

The exact socket centers, nominal clearances, transform provenance, and the
remaining manufacturing/calibration unknowns are recorded in
[`../config/pcb_socket.yaml`](../config/pcb_socket.yaml).

The insertion-frame transform is:

```text
T_I_P =
  [ 1  0  0  0          ]
  [ 0  0  1  0          ]
  [ 0 -1  0  0.003252597]
  [ 0  0  0  1          ]
```

Thus `+Z_I = -Y_P`, and the nominal PCB plane is `Z_I = 0`.

## PCB half-space

The free side is authored independently of the insertion-frame origin:

```text
[0, 1, 0]_P dot p_P >= 0.00325259662 m
```

Every supplied body/finger mesh vertex must remain at least 0.5 mm on that
free side to pass the seated screen. For the configured fixed-orientation
straight insertion, endpoint support is sufficient for the infinite plane;
rotating correction moves require a swept check later.

## Scope of a generated pose

A surviving entry is only a `phase1_geometric_candidate`. It does not certify
robot reachability, complete part/gripper collision, pad capture, actuator
force, insertion wrench, neighboring-board clearance, or uncertainty margin.
