# Downstream-constrained dual-robot handoff: mathematics and methodology

This document defines the implemented planning method independently of the
current connector example. The executable workflow, commands, cache timings,
and present simulation limitations are summarized in
[mujoco_handoff_pipeline.md](mujoco_handoff_pipeline.md).

The current task assumes the startup part pose is known. In-gripper pose
estimation is therefore outside this implementation: the caller supplies
$^{W}T_P^{start}$, or the equivalent startup TCP-to-part transform together
with robot FK. All other stages—geometry grasps, downstream filtering, direct
handoff, stable placement, reorientation, motion validation, execution
monitoring, and coverage accounting—remain active.

## 0. Scope and input contract

The method separates physical/task facts from solver policy.

The user-owned `project.yaml` supplies:

- articulated robot URDF/MJCF assets and calibrated base transforms;
- reusable gripper asset and flange/TCP semantics;
- workstation visual and collision CAD;
- part CAD, mass, and the part-to-pin feature transform;
- a known initial part/grasp state;
- the finite source for offline admissible initial-grasp classes;
- bounded handoff, scanner, reorientation, and insertion regions; and
- PCB pose plus PCB-to-hole feature transforms.

The system-owned `solver_defaults.yaml` supplies sampling budgets, numerical
tolerances, uncertainty/clearance policy, motion-planning bounds, and execution
interlocks. These are versioned system policies, not part-specific rules.
`pipeline_config.yaml` and `grasp_config.yaml` are migration tombstones. The
planner does not read them, and they must not become new tuning surfaces.

A robot or gripper cannot be reconstructed from unstructured CAD alone.
Articulation requires bodies, joints, axes, limits, and collision/contact
semantics in URDF/MJCF or an equivalent descriptor. STEP/STL/OBJ supplies shape,
not kinematics.

The reference executable currently has a dual-GP7 scene/kinematics adapter and
a static-gripper scene adapter. The geometry/task-graph interfaces are reusable,
but a different robot or articulated gripper needs its corresponding adapter;
changing only a manifest asset path is not yet an end-to-end robot conversion.

## 1. Frames, transforms, and perturbations

Every pose is an element of $SE(3)$:

$$
T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix},
\qquad R\in SO(3),\ t\in\mathbb R^3.
$$

$^{X}T_Y$ maps coordinates expressed in frame $Y$ into frame $X$. Relevant
frames are:

| Symbol | Meaning |
|---|---|
| $W$ | calibrated world/cell frame |
| $A_0,B_0$ | robot base frames |
| $E_A,E_B$ | robot TCP frames |
| $P$ | native part CAD frame |
| $F$ | functional part pin frame |
| $C$ | PCB frame |
| $H_k$ | hole frame for insertion target $k$ |
| $N$ | reorientation support frame |

Robot FK is $^{R_0}T_E=f_R(q_R)$. World FK is

$$
{}^W T_E = {}^W T_{R_0}\,f_R(q_R).
$$

Twists and pose errors are ordered $(v,\omega)$: translation first, rotation
second. The implementation uses spatial/left perturbations when transporting
covariance. For $T=\exp(\xi^\wedge)\bar T$ and a left composition $Y=AT$,

$$
\Sigma_Y=\operatorname{Ad}_A\Sigma_T\operatorname{Ad}_A^\top,
\qquad
\operatorname{Ad}_A=
\begin{bmatrix}R &[t]_\times R\\0&R\end{bmatrix}.
$$

Metres and radians are not combined in an unscaled norm. The insertion
correction test uses a 0.10 m characteristic length for angular error.

### 1.1 Grasp convention

A grasp is the pose of TCP $E$ in the part frame:

$$
\boxed{g\equiv{}^P T_E}.
$$

For part world pose $X={}^W T_P$,

$$
{}^W T_E=Xg.
$$

With a known initial part pose and measured/current robot joints,

$$
\boxed{g_A^{start}=(X^{start})^{-1}
{}^W T_{E_A}(q_A^{start})}.
$$

The project manifest may store the inverse $^{E}T_P$ because that is often the
measurement/calibration convention; it is inverted once at the boundary.

### 1.2 Symmetry action

For `g = ^P T_E`, a part-frame symmetry acts on the left. If $S$ maps a grasp
contact set to its symmetric copy in the chosen part-frame convention,

$$
g' = Sg.
$$

If the stored $S$ instead maps the relabeled frame in the inverse direction,
use $S^{-1}g$. The essential rule is left action. The former expression
$gS$ is wrong: it rotates about the TCP and changes a different frame.
Only symmetries of the functional pin pattern are admissible; a visually
symmetric body with an asymmetric pin pattern is not task-symmetric.

## 2. CAD preprocessing and collision representation

Visual CAD is preprocessed deterministically.

- STL is normalized to binary STL with every triangle preserved.
- A large STL is split below MuJoCo's per-asset face limit without decimation,
  welding, simplification, or reordering.
- OBJ is preserved byte-for-byte.
- STEP/STP is tessellated by FreeCAD with explicit deflection settings, after
  which every tessellated triangle is preserved.

This is exact polygon preservation, not analytic STEP/B-rep collision.

MuJoCo does not load STEP directly. The preprocessing index records the
generated STL chunks that a scene compiler must reference. The reference scene
builder now invokes this preparation automatically and emits MJCF references to
those chunks. `FreeCADCmd`/`freecadcmd` must be discoverable for a cold STEP
build; an explicit standalone preparation can prewarm the same content cache.

Visual and collision models are intentionally distinct. MuJoCo collision for a
mesh geom uses its convex hull. Concave CAD must therefore be represented by
primitives or separate convex pieces. A connected-component split is not a
general convex decomposition, although it is still safer than replacing a
multi-body tool with one coarse palm box.

The current static gripper STL contains eight connected surface components.
All eight are loaded for each gripper's collision checks. They are fixed to one
body and each has convex-hull contact. Consequently the model can reject
gripper-to-gripper and gripper-to-robot overlap, but cannot simulate finger
motion or certify pad capture.

## 3. Geometry-derived parallel-jaw grasp library

No part axes, center point, approach rolls, or per-part aperture are specified.
Let the triangulated part surface be $\mathcal M$ and let a reusable gripper
capability contain

$$
d_{min},\ d_{max},\quad
(w_{pad},h_{pad}),\quad l_{finger},\quad \mu.
$$

The implemented deterministic generator:

1. samples points on $\mathcal M$ with strata proportional to triangle area;
2. at sample $p_0$ with outward normal $n_0$, casts a ray along $-n_0$;
3. uses the first opposing hit $(p_1,n_1)$;
4. validates antipodal/friction-cone geometry and aperture;
5. derives roll about the closing line from surface covariance;
6. estimates pad support and palm/finger-depth clearance; and
7. ranks, SE(3)-deduplicates, and applies a coverage term so elongated parts
   retain grasps at spatially distinct locations.

Define

$$
c=\frac{p_1-p_0}{\lVert p_1-p_0\rVert},
\qquad d=\lVert p_1-p_0\rVert,
\qquad \alpha=\tan^{-1}\mu.
$$

The hard contact tests are

$$
d_{min}\le d\le d_{max},
\qquad n_0^\top(-c)\ge\cos\alpha,
\qquad n_1^\top c\ge\cos\alpha.
$$

For a chosen approach direction $a\perp c$, construct

$$
y_E=c,\qquad z_E=a,\qquad x_E=c\times a,
$$

and

$$
g={}^P T_E=
\begin{bmatrix}
x_E&y_E&z_E&\frac12(p_0+p_1)\\
0&0&0&1
\end{bmatrix}.
$$

Thus `+E_Y` is the jaw-closing line, `+E_Z` is palm-to-contact approach, and
the required opening is $d$. A downstream planner receives both contacts,
normals, required aperture, directions, and quality terms; it never needs to
infer them from a part name.

Antipodality is necessary but not sufficient. Robot IK, the complete gripper
geometry, part collision, environment collision, co-grasp contact-patch
separation, and approach/retreat paths remain hard gates.

### 3.1 Static and articulated aperture truth

For an articulated gripper, prismatic/slide joint limits define the aperture
range and the requested $d$ maps to coordinated joint positions. For a static
STL, declared manufacturer capability can filter candidates but does not create
motion. The current static model therefore uses virtual close/capture and weld
ownership; it is not physically certifiable.

## 4. Functional insertion targets

The user supplies a part pin feature $^{P}T_F$, a PCB pose $^{W}T_C$, and each
hole feature $^{C}T_{H_k}$. Insertion aligns the feature frames:

$$
{}^W T_P^{ins,k}\,{}^P T_F
= {}^W T_C\,{}^C T_{H_k}.
$$

Therefore

$$
\boxed{{}^W T_P^{ins,k}
= {}^W T_C\,{}^C T_{H_k}\,({}^P T_F)^{-1}}.
$$

This equation replaces per-part hard-coded insertion world poses.

Let

$$
{}^W T_{H_k}={}^W T_C{}^C T_{H_k},
\qquad a_k={}^W R_{H_k}e_z,
$$

where hole `+Z` is defined to point into the hole. The pre-insertion pose keeps
the insertion orientation and moves opposite the physical hole axis:

$$
R_P^{pre,k}=R_P^{ins,k},
\qquad
t_P^{pre,k}=t_P^{ins,k}-d_{app}a_k.
$$

The former shortcut $X_{ins}\operatorname{Trans}(0,0,-d)$ is not generally
correct because it offsets along a part-frame axis, which need not equal the
hole axis.

Both $t_P^{pre,k}$ and $t_P^{ins,k}$ must lie inside the manifest-declared
insertion region. This is a task-space admissibility bound, not a replacement
for the pin/hole frame equality or collision checking.

### 4.1 Axis-relative correction envelope

For a correction vertex
$\delta p_H=(\delta x,\delta y,\delta z)$ and hole-axis yaw $\delta\psi$,

$$
t'_P=t_P^{ins}+{}^W R_H\delta p_H,
$$

$$
R'_P=\exp([a_H]_\times\delta\psi)R_P^{ins}.
$$

The current four-dimensional box has positive/negative lateral, axial, and yaw
limits, producing $2^4=16$ vertices. It is relative to each configured hole,
not world `Z`.

At nominal insertion joint state $q^{ins}$, a first-order gain screen uses

$$
\delta q=J^+(q^{ins})\delta x,
$$

but acceptance also requires seeded IK at every exact perturbed pose, the same
joint/singularity gates, and branch continuity. Wrist dither is reserved in the
joint-6 margin.

## 5. Downstream factorization

For receiver grasp $g_B$, all downstream TCP targets are fixed:

$$
{}^W T_{E_B}^{task}={}^W T_P^{task}g_B.
$$

Therefore scanner and insertion feasibility depends on $g_B$ and the fixed
task/cell model, not on the handoff pose. Define

$$
\mathcal G_B^*=
\{g_B:\operatorname{DownstreamFeasible}(g_B)\}.
$$

`DownstreamFeasible` requires:

- scanner IK and margins;
- pre-insertion and insertion IK at every hole;
- wrist-dither reserve;
- all 16 correction vertices with branch-continuous IK;
- scanner-to-preinsert and insertion motion paths;
- collision policy compliance; and
- a minimum singular value/manipulability witness.

This entire witness is content-addressed and cached. On the reference connector
project, 23 of the 128 generated receiver grasps currently survive. The online
handoff loop considers only those 23.

The scanner currently uses a system-derived presentation orientation based on
the first insertion target and a position inside the declared scanner region.
A future sensor-specific presentation constraint can become another semantic
feature/region input without changing the grasp convention.

## 6. Direct handoff problem

At transfer, A cannot change its measured grasp. A direct candidate is

$$
h=(X_h,g_A^{start},g_B),
\qquad X_h={}^W T_P^{handoff},
\qquad g_B\in\mathcal G_B^*.
$$

It induces

$$
{}^W T_{E_A}=X_hg_A^{start},
\qquad
{}^W T_{E_B}=X_hg_B.
$$

$X_h$ is sampled only inside the user-declared handoff region. Position
resolution, orientation family, and candidate count are system solver policy.
The current deterministic orientation family is task-informed from insertion
orientation rather than a part-name rule.

### 6.1 Coarse-to-fine hard gates

The implemented order is:

**G1 — reachability.** Query each induced TCP pose in its robot base frame:

$$
\boxed{{}^{R_0}T_E
=({}^W T_{R_0})^{-1}X_hg_R}.
$$

Querying only $X_h$'s part origin is mathematically wrong. Optional reachability
maps provide an O(1) rejector; exact IK remains authoritative. A conservative
reach-sphere fallback prevents sparse-map false negatives in the present
prototype.

**G2 — exact verified IK.** Solve all requested TCP poses. The current GP7
solver uses deterministic multi-seed numerical IK and verifies every solution
with FK position/orientation tolerances. It is not analytically branch-complete.

**G3 — joint and singularity margins.** Require

$$
q_i^{min}+m_i\le q_i\le q_i^{max}-m_i,
$$

and calibrated manipulability/singular-value thresholds. Yoshikawa
manipulability is

$$
w(q)=\sqrt{\det(JJ^\top)}=\prod_i\sigma_i(J),
$$

but its absolute magnitude is unit/model dependent; thresholds are calibrated
per robot rather than copied across assets.

**G4 — co-grasp and collision.** Reject overlapping occupied contact patches,
then check the complete MuJoCo state: both robots, all gripper collision
components, part, workstation, and fixtures. Exact component collision is
authoritative; a contact-patch heuristic can reject early but never accept.
Clearance includes the configured calibration 3-sigma allowance.

**G5 — downstream correction.** This is already folded into the cached
$\mathcal G_B^*$ witness and includes exact axis-relative vertices and branch
continuity.

**G6 — path existence.** Validate A current-to-pre-handoff, A approach, B
current-to-pre-handoff, B approach, A retreat, and B-to-scanner, plus cached
downstream paths. Every segment uses the correct held/fixed part state.

### 6.2 Direct-first latency path

The low-latency search first tries branch-continuous warm-start IK over the
bounded grid and returns the first candidate passing every gate. If warm starts
fail, complete configured multi-seed enumeration remains as a correctness
fallback. Cached direct policies bypass both.

### 6.3 Optional best-plan score

Hard validity is never traded against a score. For `--best`, already-valid
candidates receive normalized terms:

$$
\Phi=
\lambda_m\phi_m+
\lambda_j\phi_j+
\lambda_c\phi_c+
\lambda_r\phi_r+
\lambda_t\phi_t,
$$

with

$$
\phi_r=1-\frac{d_{SO(3)}(R_h,R_{ins})}{\pi},
\qquad
\phi_t=\exp(-L_q/L_0).
$$

The reorientation term compares the two **part** orientations $R_h$ and
$R_{ins}$. The former expression $d(R_hR_{g_B},R_{ins})$ compared a TCP
orientation with a part orientation and was invalid. Manipulability, joint
margin, clearance, and cycle travel are similarly normalized before weighting.

## 7. Collision policy and motion planning

### 7.1 Semantic, bounded contact

Expected contact is permitted only for named phase semantics:

- a holder may contact the part only with its gripper collision components;
- it may never hide part contact with link 6, wrist, or another arm link;
- current static-holder penetration is bounded to 0.75 mm;
- placement permits only the part/support-surface pair; and
- geometric insertion permits only the part/PCB pair.

Every other contact remains a hard failure. An allowed pair is not a global
collision disable and does not exempt deep penetration.

With an articulated gripper, named pad geoms should replace the broad static
`gripper_collision_*` holder allowance.

### 7.2 Adaptive edge validation and RRT-Connect

For joint edge $(q_0,q_1)$, the validator recursively/subdivisively samples so
that no joint changes by more than the system maximum $\Delta q_{max}$ between
checks. This fixes the old fixed-eight-sample method, whose collision resolution
degraded as an edge became longer.

If the direct edge collides, deterministic bidirectional RRT-Connect searches
within joint limits and explicit time/node budgets. Its collision callback
poses the part rigidly with the holder or holds it fixed on the support, so it
checks the same coupled state that execution will replay. Successful sparse
paths are shortcut-smoothed and densified/revalidated before use.

The checker exposes a non-force-producing MuJoCo contact margin of 0.5 mm for
all non-authorized swept-path pairs. The co-grasp state separately requires
$5.0+1.5=6.5$ mm (nominal clearance plus calibration 3-sigma). Placement phases
authorize bounded part/support contact and positive finger/support proximity;
the latter has zero permitted penetration. Thus necessary table approach is
not confused with an unrestricted collision exemption.

## 8. Stable placement and backward reorientation

### 8.1 Stable pose generation

For each approximately coplanar extreme support facet, construct its 2-D convex
support polygon $\mathcal S$. A placement is quasistatically stable if the COM
projection lies inside with margin:

$$
\operatorname{sdist}
\left(\pi_g(c_P),\partial\mathcal S\right)
\ge\epsilon_{stab}.
$$

The implementation evaluates connected CAD components independently. Closed
components contribute uniform-density signed-volume COM estimates. If the mesh
is open/nonmanifold or closed geometry is not representative, it explicitly
falls back to the CAD bounding-box center; this lowers physical confidence and
must not be presented as measured mass properties.

The selected support normal maps to support-frame $-Z$, the plane is $N.z=0$,
and `+N.z` points away from the table. A stage instance is valid only if the
complete projected part footprint fits within the declared rectangular region
with edge margin. Yaw samples are a system search policy.

For characteristic part scale $s_P=\lVert p_{max}-p_{min}\rVert_2$ and stage
scale $s_N=\min(w_N,h_N)$, the reference policy uses
$\epsilon_{stab}=0.005s_P$. Its dimensionless placement robustness is

$$
\rho_{place}=\min\!\left(
\operatorname{clip}\!\left(\frac{2m_{support}}{s_P},0,1\right),
\operatorname{clip}\!\left(\frac{2m_{edge}}{s_N},0,1\right)
\right).
$$

This cached value is propagated to both the place and re-pick task-graph edges;
it is not a hard-coded score.

### 8.2 Feasibility graph

Let

- $\mathcal A$ be candidate A grasps;
- $\mathcal B^*=\mathcal G_B^*$ be insertion-feasible B grasps;
- $\mathcal P$ be stable stage placements;
- $D\subseteq\mathcal A\times\mathcal B^*$ be fully validated direct co-grasp
  edges; and
- $F\subseteq\mathcal P\times\mathcal A$ be validated placement/grasp edges.

Each edge stores cycle cost, bottleneck robustness, and its continuous
trajectory witness. A reorientation transition $a_i\to p\to a_j$ exists only
when both $(p,a_i)$ and $(p,a_j)$ are in $F$.

The search proceeds backward from $\mathcal B^*$ through $D$ and then through
shared placements in $F$. A successful sequence is

$$
a_0,p_1,a_1,\ldots,p_k,a_k,b,
\qquad (a_k,b)\in D,\ b\in\mathcal B^*.
$$

This terminal condition is important: a grasp that merely looks different
after re-pick is not useful unless it connects to an insertion-feasible B
grasp.

Selection is lexicographic:

1. if $a_0$ has any direct edge into $\mathcal B^*$, use direct handoff and do
   not reorient;
2. otherwise minimize summed cycle cost within the maximum hop bound;
3. break ties with fewer hops;
4. then maximize minimum edge robustness; and
5. use stable ID ordering for reproducibility.

The generic `TaskGraph` supports bounded multi-hop paths. The current reference
integration finds a one-placement solution for the forced adverse grasp and
connects it to a separately verified direct goal.

### 8.3 Pose uncertainty through placement

A flat surface does not reset XY/yaw uncertainty and may add settling error. A
kinematic simulation cannot certify that error distribution. A production
system should use a locating nest or remeasure the part on the support. A
fixture can reduce uncertainty to its tolerance; an unobserved flat placement
cannot be assumed to do so.

## 9. Handoff and insertion execution

For selected direct plan $h^*$:

1. A follows its checked current-to-pre-handoff and approach paths.
2. B follows its checked pre-handoff and guarded approach paths.
3. B closes to the geometry-required aperture; aperture/force capture is
   verified before A release.
4. No robot moves during the short co-grasp dwell.
5. Ownership transfers transactionally from A to B.
6. A follows its checked retreat.
7. B moves to the scanner.
8. The measured post-scan $g_B$ recomputes every downstream TCP target.
9. Correction and path gates are rerun if the measured grasp changes.
10. B follows pre-insertion and insertion paths.

For reorientation, prepend checked place, release, retreat/re-approach,
re-pick, capture, and lift/transit paths.

The executor checks collision at every replay waypoint. An unexpected contact
raises an abort and prevents the next state transition. This continuous monitor
is independent of the planner's prior result and protects against corrupted or
modified trajectories.

The current close, force guard, scanner, and weld transfer are idealized because
the gripper is static and no sensor model exists. The execution state machine is
verified; the corresponding physical signals are not.

### 9.1 Capture uncertainty

For independent grasp and robot-to-robot calibration covariances in a common
tangent convention,

$$
\Sigma_{ho}=\Sigma_{grasp}+\Sigma_{r2r}.
$$

If a covariance originates in another frame it is first transported with the
appropriate adjoint. Componentwise 3-sigma translation/rotation must fit inside
the receiver's declared capture region. Planning cannot compensate for a
capture region smaller than calibrated uncertainty; that requires better
metrology or compliant geometry.

## 10. Offline artifacts and cycle time

The cache key is content-addressed from artifact/schema version, CAD/scene
fingerprints, project/task transforms, relevant solver policy, and declared
upstream dependencies. Entries are atomically written and integrity checked.

The dependency order is:

```text
CAD/scene identity
  -> geometry grasp library
  -> downstream receiver feasibility
  -> direct co-grasp/motion policy
  -> stable placement instances
  -> backward reorientation policy
  -> coverage report
```

Reachability maps and reusable motion roadmaps can be added at the same offline
tier. They are rejectors/proposals; exact online gates remain authoritative.

The 2026-07-12 Mac Studio snapshot for project fingerprint
`aaff9b2dfcdbb71721f6fe8776d8bf0fbdceb892ab55ac403f04cb47acfef9f0`
and solver fingerprint
`d652ff9f31a7181d1dbdb6ba37bd2c201d8a76a3afddbb1dc9d656accd451139`
measured:

- 23 downstream-valid receiver grasps from 128 geometry candidates;
- downstream filter cold computation 149.9 s;
- direct policy cold computation 2.93 s;
- direct policy cache hit 26.9 ms;
- adverse-grasp reorientation cold computation 4.39 s;
- reorientation policy cache hit 4.58 ms; and
- stable-placement cache hit 4.59 ms.

These measurements are not a real-time guarantee and do not include robot
motion. Production CT is obtained by running the production precompute after
every content change and covering the expected initial-grasp domain. A truly
new continuous initial state may still require cold numeric IK/collision/RRT.

## 11. Coverage certificate and physical certification

Let $\mathcal D$ be the finite, explicitly declared set of admissible initial
grasp classes. The task graph partitions it into

$$
\mathcal D=
\mathcal D_{direct}\ \dot\cup\
\mathcal D_{reorient}\ \dot\cup\
\mathcal D_{uncovered}.
$$

Coverage is

$$
\gamma=
\frac{|\mathcal D_{direct}|+|\mathcal D_{reorient}|}{|\mathcal D|}.
$$

The implementation marks a report certified only when $\gamma$ meets or exceeds
the requested target, normally 1.0. “100%” is therefore scoped to $\mathcal D$
and the fingerprinted model/policy. It is not a claim over all continuous
poses, unmodeled obstacles, arbitrary parts, or calibration drift.

The ordinary demo/run covers only its known-start singleton. The project
manifest separately declares whether the offline domain is only that state or
the known state plus the deterministic geometry-grasp library; qualification
enumerates exactly the declared source.

Logical coverage is also not physical certification. The current project must
report `physical_certified: false` even when its singleton is feasible,
because:

- its gripper is a static STL without finger joints or physical aperture/
  contact validation;
- its PCB is a solid collision board without actual hole/chamfer collision CAD;
- the part has no separate pin collision model or calibrated contact materials;
  and
- the current executor still uses ideal weld ownership and virtual capture/
  insertion predicates.

Physical certification additionally needs calibrated transforms and uncertainty,
measured COM/friction, real gripper/force feedback, sensor error models,
pin/hole geometry, hardware stopping-distance validation, and coverage over the
declared production domain.

## 12. Learning: proposal acceleration only

The best first learning application is supervised proposal ordering. Offline
deterministic planning already produces labels such as:

- hard-gate success/failure and first failing gate;
- IK/collision/RRT solve time;
- clearance and bottleneck robustness;
- path/cycle cost; and
- downstream insertion outcome.

A gradient-boosted model or small neural ranker can prioritize grasps, handoff
poses, and placement/re-pick edges. Evaluate top-k feasible recall, latency, and
distribution shift on held-out part/cell variations. Preserve deterministic
fallback ordering.

Learning has no authority to relax or replace geometry, IK, joint, collision,
uncertainty, motion, or execution-monitor gates. The current
`SafetyGatedRanker` enforces this boundary by excluding proposals marked
hard-invalid.

RL is overkill for the current mostly static combinatorial problem. It may be
useful later for high-level scheduling or a separately safety-wrapped contact
controller, but RL reward/value is not evidence of collision freedom,
reachability, capture, or physical certification.

## 13. Complete direct-first algorithm

```text
OFFLINE_OR_AFTER_CONTENT_CHANGE(project):
    fingerprint CAD, calibration, feature frames, and solver policy
    normalize/tessellate exact visual CAD; prepare collision models
    G_A, G_B <- geometry antipodal contact grasps
    B_star <- downstream_filter(G_B, scanner, pin/hole targets,
                                correction envelope, IK, collision, motion)
    P <- COM/support/footprint-valid stable stage placements
    cache all artifacts by content key

PLAN_ONE_KNOWN_START(X_W_P_start, q_A_start):
    g_A <- inverse(X_W_P_start) * FK_W_A(q_A_start)

    direct <- search_handoff_region(g_A, B_star,
                                    induced-TCP reachability,
                                    IK/margins,
                                    distinct contact patches,
                                    full component collision,
                                    adaptive-edge/RRT motion)
    if direct exists:
        return DIRECT(direct)               # hard branch preference

    direct_goals <- A grasps having verified edges to B_star
    F <- collision/IK/motion-valid placement-grasp edges
    graph <- backward_graph(B_star, direct_goals, P, F)
    reorientation <- minimum_cycle_path(graph, g_A, max_hops)
    if reorientation exists:
        return REORIENT_THEN_DIRECT(reorientation)

    return UNCOVERED(reason and gate statistics)

EXECUTE(plan):
    replay only checked trajectories
    monitor every waypoint for unexpected contact
    enforce capture-before-release transaction
    scan/recompute held grasp and downstream TCP targets
    revalidate correction/path when the measured grasp changes
    approach/insert along configured hole axis
```

## 14. Corrections relative to the original prototype

| Defect | Correct method |
|---|---|
| G1 queried the part origin | Query each induced TCP pose $({}^WT_{R_0})^{-1}X_hg_R$ |
| Symmetries right-multiplied `g` | For `g = ^P T_E`, apply symmetry on the left |
| Both grippers targeted the part center | Generate distinct antipodal contact pairs across the native surface |
| Co-grasp used a palm proxy/ignored gripper collision | Check all current eight static components plus contact-patch occupancy |
| Holder allowance hid wrist/part contact | Permit only named/bounded gripper-part contact |
| Pre-insertion assumed world/part Z | Offset opposite each semantic hole `+Z` axis |
| Correction yaw assumed world Z | Rotate about each physical hole axis |
| Reorientation score mixed TCP and part frames | Compare $R_h$ with $R_{ins}$ |
| Reorientation interpolated unchecked joints | Adaptive edge validation, then bounded RRT-Connect |
| Re-pick was not tied to downstream success | Search backward from insertion-feasible B grasps |
| A feasible demo implied general coverage | Certify only an explicitly declared admissible domain |

## 15. Validation and commands

The current direct visualization and the forced adverse-grasp reorientation
visualization both use planner-produced, collision-checked paths. The forced
demo refuses to run if the initial grasp is not rejected as intended, if no
stable placement/re-pick path exists, or if the re-picked A grasp lacks a
verified edge to an insertion-feasible B grasp. Regression tests also inject
unexpected part/fixture collision and verify execution abort.

Run the canonical workflow from the repository root:

```bash
source .venv/bin/activate
python scripts/prepare_project_cad.py --project mujoco_sim/project.yaml
python scripts/build_mujoco_scene.py
python scripts/precompute_pipeline.py --project mujoco_sim/project.yaml \
  --model mujoco_sim/models/scene.xml --production
python -m mujoco_sim.pipeline --execute --json
```

On macOS, visualize with:

```bash
mjpython -m mujoco_sim.visualize_pipeline --hold -1
mjpython -m mujoco_sim.visualize_reorientation_demo --hold -1
```

Use ordinary `python` for those passive-viewer modules on Linux. The complete
focused direct-run test command list is maintained in
[mujoco_handoff_pipeline.md](mujoco_handoff_pipeline.md#8-exact-commands).
