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
| `relay` | manufacturer STEP | CIT J1021C family model (`small_relay.stp`) |

## BOM-matched STEP models

The following STEP files were added for the photographed/BOM components. STEP
geometry is in millimetres, as is conventional for MCAD exchange files.

| requested component | file | match quality | source / mechanical notes |
|---|---|---|---|
| CIT Relay & Switch `J1021CS312VDC.20` | `relay/small_relay.stp` | exact J102 family | Manufacturer-authored `J1021C` assembly already present in the workspace; 15.5 x 10.5 x 11.25 mm package family. |
| Rubycon `80ZLH560MEFC18X20` | `capacitor/rubycon_80ZLH560/CP_Radial_D18mm_P7.5mm.step` | dimensional proxy | Official KiCad generic radial-can model. Correct 18 mm diameter and 7.5 mm lead pitch; verify/adjust height to the Rubycon drawing's 20 mm nominal (22 mm seated maximum) before production-clearance work. |
| `KF301-3P` / `KF128-3P` | `terminal_block/KF301-3P_equivalent/KF301-3P_5.08mm_equivalent.step` | family equivalent | Official KiCad Phoenix MKDS 3-position, 5.08 mm-pitch horizontal screw-terminal model. Use for envelope/collision work; screw and housing details differ from commodity KF301/KF128 variants. |
| right-angle PCB DE-9 female | `conn_dsub/de9_female_right_angle/DE9_Female_RightAngle.step` | family match | Official KiCad DSUB-9 female horizontal model, standard 2.77 x 2.84 mm contact grid and 9.40 mm PCB-edge offset. Compare mounting-post style with the physical connector. |
| JST `S9B-XH-A` | `conn_header/JST_S9B-XH-A/JST_S9B-XH-A.step` | exact part family | Official KiCad model for JST XH `S9B-XH-A`, 1x09, 2.50 mm pitch, horizontal/right-angle. |
| DIN0411 / DIN0414 axial resistor | `resistor/DIN0411_DIN0414/DIN0411_1W_axial.step`, `resistor/DIN0411_DIN0414/DIN0414_2W_axial.step` | exact standard envelopes | Official KiCad models: DIN0411 body 9.9 x 3.6 mm, 15.24 mm formed pitch; DIN0414 body 11.9 x 4.5 mm, 20.32 mm formed pitch. |

KiCad model source: <https://github.com/KiCad/kicad-packages3D> (CC BY-SA
4.0; see that repository for attribution and license details). Product/package
dimensions were checked against the Rubycon and CIT manufacturer data linked
from their DigiKey product pages on 2026-07-11.

## Why generated instead of downloaded

Real manufacturer STEP models for these four exist on GrabCAD/SnapEDA/TraceParts,
but all three require a logged-in account to download and there was no active
Chrome session to authenticate with. Since PyBullet consumes STL/OBJ (not STEP)
anyway, and this project's collision philosophy is already "simplified
primitives, not raw CAD meshes" (see workcell collision_boxes.yaml), these were
built directly as box/cylinder assemblies at realistic THT package dimensions
via `gen_parts.py` (kept outside this repo — regenerate on request rather than
carrying a script file here per the project's fixed structure).

The generated files above remain useful as lightweight simulation meshes. For
the newly added BOM-matched models, prefer the STEP files listed in the table
and convert a copy to a metre-scaled STL before loading it in PyBullet.
