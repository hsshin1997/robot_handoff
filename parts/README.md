# parts/

Candidate "part" geometries for the A→B handoff test (one of these plugs into
`config/cell.yaml`'s `T_flangeA_part` / part mesh once mesh loading is wired
into `Scene.spawn_part`). All STL files are in **meters**, matching
`parts/conn_header` — no additional 0.001 scale factor needed (unlike the
workcell STL, which is mm).

| folder | source | notes |
|---|---|---|
| `conn_dsub` | real manufacturer STEP (DS1037-9F) | pre-existing, D-sub connector |
| `conn_header` | real manufacturer STEP/STL | pre-existing, 1x11 THT header, 2.54mm pitch |
| `resistor` | generated | 1/4W axial THT, 2.3mm dia x 6.3mm body, 4mm leads each side |
| `capacitor` | generated | radial electrolytic THT, 10mm dia x 12mm body, 2.5mm lead pitch |
| `terminal_block` | generated | 2-position PCB screw terminal, 5.08mm pitch |
| `relay` | generated | PCB-mount power relay, ~19x15.5x15.2mm body, 5 THT pins (Songle SRD / Omron G5LE footprint size) |

## Why generated instead of downloaded

Real manufacturer STEP models for these four exist on GrabCAD/SnapEDA/TraceParts,
but all three require a logged-in account to download and there was no active
Chrome session to authenticate with. Since PyBullet consumes STL/OBJ (not STEP)
anyway, and this project's collision philosophy is already "simplified
primitives, not raw CAD meshes" (see workcell collision_boxes.yaml), these were
built directly as box/cylinder assemblies at realistic THT package dimensions
via `gen_parts.py` (kept outside this repo — regenerate on request rather than
carrying a script file here per the project's fixed structure).

If you'd rather have the real manufacturer STEP files (e.g. to match a specific
BOM part number), log into GrabCAD/SnapEDA in Chrome and I can drive the
download from there.
