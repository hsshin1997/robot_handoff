# Building MuJoCo offline tables and policies

This guide covers the production-oriented MuJoCo preprocessing path. It builds
the reusable geometry, reachability, downstream, direct-handoff,
reorientation, and qualification artifacts used by the online planner. For
scene configuration and day-to-day operation, see
[mujoco_user_guide.md](mujoco_user_guide.md).

Run every command below from the repository root with the project virtual
environment active:

```bash
source .venv/bin/activate
```

## What is computed offline

The dependency order is:

```text
project.yaml + CAD/model bytes + declared units
        |
        +--> prepared SI triangle meshes --> antipodal surface grasps
        |                              \--> COM-stable stage placements
        |
        +--> compiled scene + solver policy --> receiver downstream witnesses
        |                                      (scanner, pre-insert, insertion,
        |                                       correction envelope and paths)
        |
known initial grasp + downstream witnesses --> direct handoff policy
stable placements + downstream witnesses ----> reorientation policy, if needed
declared initial-grasp domain + policies ------> coverage certificate
```

The outputs have different roles:

| Output | Default location | Meaning |
|---|---|---|
| Prepared CAD | `mujoco_sim/models/generated_cad/` | Deterministic STL/OBJ chunks and metadata in declared SI scale; visual triangles are not decimated |
| Compiled scene | `mujoco_sim/models/scene.xml` | MJCF paired with the selected project |
| Reachability tables | `mujoco_sim/cache/reachability_A.npz`, `reachability_B.npz` | TCP position/direction voxel lookup used as a cheap G1 accelerator; exact IK remains authoritative |
| Grasp artifacts | `mujoco_sim/cache/grasp/*.json` | Actual-triangle antipodal contact pairs and required jaw openings |
| Stable-pose artifacts | `mujoco_sim/cache/stable-pose/*.json` | COM/support-polygon-valid poses instantiated on the declared stage |
| Task-policy artifacts | `mujoco_sim/cache/task-policy/*.json` | Downstream witnesses and direct/reorientation plans with checked paths |
| Project snapshot | `mujoco_sim/cache/project-metadata.json` | Canonical manifest and referenced-asset fingerprints plus precompute summary |
| Qualification report | `mujoco_sim/cache/coverage-certificate.json` | Coverage of the explicitly declared initial-grasp domain and separate physical prerequisites |

The current implementation does not publish a general reusable motion roadmap.
The task-policy witnesses contain the paths required by the policy that was
actually certified.

Optional XYZ/ASCII-PCD `proposal_templates` do not produce a mesh or a raw
grasp artifact. They only reorder the contact-valid grasp library before
downstream/direct/reorientation computation; see
[Optional XYZ and ASCII-PCD grasp proposal templates](mujoco_user_guide.md#optional-xyz-and-ascii-pcd-grasp-proposal-templates).

## Default project: complete build

### 1. Prepare CAD and compile the scene

The scene compiler invokes the same CAD preparation code itself, so the first
command is optional for STL/OBJ. It is useful as a unit/fidelity audit and is
the easiest way to give an explicit FreeCAD executable for STEP/STP input.

```bash
# Optional standalone audit/prewarm.
python scripts/prepare_project_cad.py --project mujoco_sim/config/project.yaml

# Required after changing project assets, frames, fixtures, or initial joints.
python scripts/build_mujoco_scene.py \
  --project mujoco_sim/config/project.yaml \
  --output mujoco_sim/models/scene.xml
```

For STEP/STP:

```bash
python scripts/prepare_project_cad.py \
  --project mujoco_sim/config/project.yaml \
  --freecad /absolute/path/to/FreeCADCmd \
  --linear-deflection-mm 0.05 \
  --angular-deflection-deg 5.0
python scripts/build_mujoco_scene.py
```

`FreeCADCmd` may instead be on `PATH` or named by `FREECADCMD`. MuJoCo does not
load STEP directly: the explicit deflections define the B-rep tessellation,
and the resulting triangles are then preserved without decimation. Every STL
entry must declare units or an explicit scale because STL contains no unit
metadata.

### 2. Build reachability lookup tables

```bash
python scripts/build_reachability.py \
  --project mujoco_sim/config/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --out mujoco_sim/cache
```

By default this samples the count and voxel size from
`solver_defaults.yaml` (`offline.reachability_samples` and
`offline.voxel_fraction_of_reach`; the latter is currently used as a voxel
size in metres). Development-sized overrides are available:

```bash
python scripts/build_reachability.py \
  --project mujoco_sim/config/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --out mujoco_sim/cache \
  --samples 10000 \
  --voxel 0.04
```

An undersampled map cannot make a pose safe: a map miss falls back
conservatively and every survivor still passes numerical IK, joint-limit,
singularity, collision, and motion gates. Production timing benefits from the
full configured density.

Unlike the JSON artifact store, the reachability files currently have fixed
names and do not carry a project fingerprint. Use a separate cache directory
per project, and rebuild them whenever robot kinematics, joint limits, base
calibration, TCP, sample count, or voxel size changes.

### 3. Build geometry and task policies

```bash
python scripts/precompute_pipeline.py \
  --project mujoco_sim/config/project.yaml \
  --project-root . \
  --model mujoco_sim/models/scene.xml \
  --cache-dir mujoco_sim/cache \
  --production
```

The production hook materializes, in dependency order:

1. surface-contact grasp candidates;
2. stable stage placements;
3. B's scanner/insertion/correction-envelope witnesses;
4. the known-start direct handoff policy; and
5. the backward reorientation policy only when direct handoff is unavailable.

Use `--json` to emit the complete deterministic project snapshot and hook
summary. Without it, the command prints the project and manifest fingerprints,
asset count, and metadata path.

Repeat the same command once. The second invocation should be a verified warm
cache hit. A warm online check is:

```bash
python -m mujoco_sim.pipeline \
  --project mujoco_sim/config/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --cache mujoco_sim/cache \
  --json
```

Look for `downstream_cache_hit`, `direct_cache_hit`, or
`regrasp_cache_hit` in the planning statistics. `--best` intentionally uses a
different policy key from first-feasible mode. `--production` warms the normal
first-feasible policy; if deployment uses exhaustive scoring, also run
`python -m mujoco_sim.pipeline --best` during the offline build to populate its
separate policy key.

### 4. Qualify the declared input domain

First run a one-class smoke check:

```bash
python scripts/qualify_pipeline.py \
  --project mujoco_sim/config/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --cache mujoco_sim/cache \
  --max-classes 1 \
  --output mujoco_sim/cache/coverage-smoke.json
```

Then omit `--max-classes` for the complete declared domain:

```bash
python scripts/qualify_pipeline.py \
  --project mujoco_sim/config/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --cache mujoco_sim/cache \
  --required 1.0 \
  --output mujoco_sim/cache/coverage-certificate.json
```

The smoke prefix is not a certificate for omitted classes. Its report sets
`truncated_prefix_smoke: true`, `evaluated_domain_complete: false`, and keeps
both mathematical and physical certification false; `prefix_smoke_passed`
reports only whether the evaluated prefix succeeded. The complete domain
comes from `qualification.initial_grasp_domain.source`:

- `known_start` certifies only the configured known startup grasp;
- `known_start_plus_geometry_library` also enumerates the deterministic
  geometry-grasp library and can be much more expensive. The current library is
  reusable for the sender domain only when A and B have identical gripper
  geometry/capability; asymmetric grippers fail closed until an A-specific
  library is generated.

`mathematical_coverage_certified: true` means the configured fraction of that
finite domain has a verified direct or reorientation policy. It does not mean
the current model is physically certified. Physical certification additionally
requires articulated grippers and their scene adapter, calibrated gripper and
part contacts, pin and hole collision CAD, calibrated pin/hole materials, and
a physical contact execution backend. The current static-gripper,
virtual-aperture-PCB,
ideal-weld reference project correctly reports `physical_certified: false`.
Certificates use schema version 2 and record UTC generation time plus project,
solver, compiled-model, part, and both gripper fingerprints for staleness
auditing. Physical prerequisites use one exact strict-boolean schema; missing,
partial, unknown, or string-valued fields cannot certify physical truth.
The validated descriptor boundary and remaining model-specific scene-import
work are documented in
[mujoco_gripper_integration.md](mujoco_gripper_integration.md).

## Alternate projects

Keep one manifest, compiled MJCF, generated-CAD directory, cache, and coverage
report together. This prevents accidentally using a scene or fixed-name
reachability map from another project.

```bash
PROJECT=/absolute/path/to/cell/project.yaml
PROJECT_ROOT=/absolute/path/to/cell
MODEL=/absolute/path/to/cell/build/scene.xml
CACHE=/absolute/path/to/cell/cache

python scripts/prepare_project_cad.py \
  --project "$PROJECT" \
  --project-root "$PROJECT_ROOT" \
  --generated-dir /absolute/path/to/cell/build/generated_cad

python scripts/build_mujoco_scene.py \
  --project "$PROJECT" \
  --output "$MODEL"

python scripts/build_reachability.py \
  --project "$PROJECT" \
  --model "$MODEL" \
  --out "$CACHE"

python scripts/precompute_pipeline.py \
  --project "$PROJECT" \
  --project-root "$PROJECT_ROOT" \
  --model "$MODEL" \
  --cache-dir "$CACHE" \
  --production

python scripts/qualify_pipeline.py \
  --project "$PROJECT" \
  --model "$MODEL" \
  --cache "$CACHE" \
  --required 1.0 \
  --output "$CACHE/coverage-certificate.json"
```

The scene compiler writes its generated CAD beside `MODEL`; the planner reuses
that exact directory. If an external manifest uses relative asset paths, put
the manifest at `PROJECT_ROOT` or use absolute paths so both the scene compiler
and the metadata pass resolve the same bytes. Always pass all three of
`--project`, `--model`, and `--cache` to planning and visualization.

## Fingerprints and automatic invalidation

The JSON cache is append-only and content-addressed. Each artifact filename is
the SHA-256 digest of a key containing:

- cache-key schema and producer/artifact version;
- relevant CAD, gripper, robot, and scene-semantic fingerprints;
- physical task frames and the known initial state;
- numerical, collision, uncertainty, and motion policy parameters; and
- serialized upstream candidates or dependencies where applicable.

Changing a relevant input generates a new filename; stale data is not returned
under the new key. YAML key order and formatting do not change the canonical
manifest fingerprint, while physical values and referenced file bytes do.
`project-metadata.json` records both the canonical and source-file hashes.

Pose proposals have an additional, deliberately separate identity. The planner
hashes each template's file bytes together with its format, frame, role, length
and angle units, RPY convention, and proposal-only usage. That semantic
fingerprint is carried into direct and reorientation task-policy keys; a changed
ordering also flows into downstream computation. The antipodal grasp cache and
stable-pose cache remain reusable because templates cannot change CAD contacts,
part geometry, or collision geometry.

After editing a template, rerun `precompute_pipeline.py --production`; rebuilding
`scene.xml` is unnecessary unless a scene input also changed. The canonical
manifest records the template declaration, while the generic
`project-metadata.json` asset list currently fingerprints model/CAD path fields,
not same-path proposal-file bytes. Therefore do not use an unchanged top-level
`project_fingerprint` alone to infer that template-backed policies are warm—the
planner's semantic template fingerprint is the authoritative policy
invalidation input.

Do not clear the entire cache merely because a project changed. Rebuild the
scene and rerun preprocessing; automatic key invalidation preserves still-valid
artifacts. Old unreferenced digests can be archived later when no planner or
precompute process is running.

Cache entries are atomically replaced and protected by per-key locks. On read,
the key digest and value fingerprint are verified. A corrupt entry raises an
error rather than being trusted. Stop concurrent builders, move only the named
corrupt JSON aside, and rerun the producer. Lock files older than the configured
stale-lock interval are reclaimed automatically; do not delete a fresh lock
while another build is active.

## Cold versus warm timing

A **cold build** means the relevant digest does not exist. It may perform mesh
surface sampling, many numerical IK restarts, correction-envelope checks,
swept collision checks, and bounded RRT searches. Downstream feasibility is
normally the most expensive stage.

A **warm decision** means the same project bytes, frames, start grasp, solver
policy, scene semantics, search mode, and cache directory produce an existing
digest. The cache still parses and verifies the entry before returning it.
Changing even one of those inputs can legitimately produce a cold miss.

Measure both explicitly:

```bash
# Build or refresh every production artifact.
time python scripts/precompute_pipeline.py --production

# Repeat without changing inputs: integrity-checked warm pass.
time python scripts/precompute_pipeline.py --production

# Measure the deployed online query separately.
time python -m mujoco_sim.pipeline --cache mujoco_sim/cache --json
```

Never describe a cache-hit timing as the cold solve time, and never qualify a
continuous state outside the enumerated offline domain by analogy to a nearby
cached state. A new state still has to pass all exact gates or be explicitly
included in a new offline coverage domain.

## Collision/contact audit before qualification

Audit the direct plan's reorientation and insertion interfaces:

```bash
python -m mujoco_sim.audit_contacts \
  --project mujoco_sim/config/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --cache mujoco_sim/cache
```

Audit the forced reorientation example:

```bash
python -m mujoco_sim.audit_contacts \
  --project mujoco_sim/config/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --cache mujoco_sim/cache \
  --reorientation-demo
```

Add `--json` for machine-readable results. Signed MuJoCo contact distance is
reported: negative values are penetration, zero is touching, and positive
values are near contacts exposed by a configured margin. Part-to-stage support
contact during placement is expected but bounded. Gripper-to-stage
penetration, robot/fixture contact, and all contacts outside their named phase
are forbidden. The audit also states whether insertion uses exact fixture/hole
collision CAD or the current segmented virtual-aperture placeholder.

Do not qualify a placeholder insertion as physical contact behavior. The
bounded fallback aperture is a known geometry-only approximation, not evidence
that collision checking was skipped. Supply separate
pin/hole collision CAD and calibrated contact data before physical
certification.

The contact audit reports part/support, gripper/stage, surrounding PCB-ring,
and insertion contacts for the selected route. Treat those run-specific values
as regression evidence for the compiled model, not a universal tolerance or a
physical insertion result.

## Tests for the offline subsystem

```bash
python tests/test_mujoco_cad_preprocess.py
python tests/test_mujoco_part_mesh.py
python tests/test_geometry_grasps.py
python tests/test_mujoco_placements.py
python tests/test_mujoco_offline.py
python tests/test_mujoco_reachability.py
python tests/test_mujoco_qualification.py
python tests/test_mujoco_cli_paths.py
```

For a release candidate, also run the complete MuJoCo test loop documented in
[mujoco_user_guide.md](mujoco_user_guide.md#verification-and-release-checklist).
