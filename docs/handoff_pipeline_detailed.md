# Downstream-Constrained Dual-Robot Handoff: Detailed Pipeline with Math

## 0. Notation and coordinate frames

All poses are elements of SE(3), written as homogeneous transforms

$$
T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}, \quad R \in SO(3),\ t \in \mathbb{R}^3 .
$$

We write $^{X}T_{Y}$ for the pose of frame $Y$ expressed in frame $X$. The relevant frames:

| Symbol | Frame |
|---|---|
| $W$ | World / cell frame |
| $A_0, B_0$ | Robot A / B base frames |
| $E_A, E_B$ | Robot A / B TCP (flange→tool already folded in) |
| $P$ | Part frame (CAD origin) |
| $S$ | Scanner frame |
| $C$ | PCB frame |
| $N$ | Reorientation nest / flat surface frame |

Calibration gives the static transforms $^{W}T_{A_0}$, $^{W}T_{B_0}$, $^{B_0}T_{S}$, $^{B_0}T_{C}$, $^{A_0}T_{N}$. The robot-to-robot calibration is the composition $^{A_0}T_{B_0} = ({^{W}T_{A_0}})^{-1}\, {^{W}T_{B_0}}$ and its error budget propagates directly into handoff clearance — keep it under ~0.5 mm / 0.1° for gripper-to-gripper transfer of small parts.

Forward kinematics: $^{A_0}T_{E_A} = f_A(q_A)$, $q_A \in \mathbb{R}^{6}$, similarly for B. Manipulator Jacobian $J(q) \in \mathbb{R}^{6\times 6}$ maps $\dot q \mapsto (v, \omega)$.

**Grasp definition.** A grasp $g$ on the part is a fixed transform from part frame to TCP frame:

$$
g \equiv {^{P}T_{E}} \in SE(3),
$$

so that if the part is at world pose $X = {^{W}T_{P}}$ and held with grasp $g$, the TCP must be at

$$
{^{W}T_{E}} = X \, g .
$$

Each robot has a finite grasp set defined offline in the CAD frame: $\mathcal{G}_A = \{g_A^{(1)}, \dots\}$, $\mathcal{G}_B = \{g_B^{(1)}, \dots\}$. For parts with symmetry group $\mathcal{S} \subset SO(3)$ (e.g., a relay with 180° body symmetry), every grasp expands to the orbit $\{g \cdot \sigma : \sigma \in \mathcal{S}\}$ — exploit this, it can double or quadruple the feasible set for free. (If the pin pattern breaks the symmetry, restrict $\mathcal{S}$ to symmetries of the *pin pattern*, not the body.)

---

## 1. Problem statement, formally

At handoff time, Robot A holds the part with (measured) grasp $\hat g_A$, and the part must end up in Robot B's gripper with some grasp $g_B \in \mathcal{G}_B$ such that B can complete the downstream sequence. A **handoff candidate** is a tuple

$$
h = (X_h,\ g_A,\ g_B), \qquad X_h = {^{W}T_{P}^{\,\text{handoff}}} \in SE(3),
$$

i.e., a world pose of the part at the transfer instant plus the two grasps. It induces required TCP poses

$$
{^{W}T_{E_A}} = X_h\, g_A, \qquad {^{W}T_{E_B}} = X_h\, g_B .
$$

The downstream task for B is a sequence of required **part** poses: the scanner presentation pose $X_{scan} = {^{W}T_{B_0}}\, {^{B_0}T_S}\, {^{S}T_P^{\,\text{present}}}$ and the insertion poses $\{X_{ins}^{(k)}\}_{k=1..K}$, one per PCB hole/placement, plus pre-insertion approach poses $X_{app}^{(k)} = X_{ins}^{(k)} \cdot \text{Trans}(0,0,-d_{app})$ along the insertion axis. Because the grasp is rigid during the whole downstream phase, every task pose maps to a TCP pose through the **same** $g_B$:

$$
{^{W}T_{E_B}^{\,task}} = X_{task}\, g_B .
$$

This is the crux: **choosing $(X_h, g_B)$ fixes B's entire downstream TCP trajectory family.** The handoff problem is therefore:

$$
\begin{aligned}
\max_{h = (X_h, g_A, g_B)} \quad & \Phi(h) \\
\text{s.t.} \quad
& \exists\, q_A:\ f_A(q_A) = {^{A_0}T_W}\, X_h\, g_A,\ q_A \in \mathcal{Q}_A^{free} \\
& \exists\, q_B:\ f_B(q_B) = {^{B_0}T_W}\, X_h\, g_B,\ q_B \in \mathcal{Q}_B^{free} \\
& \text{Feas}_B(g_B, X_{scan}) \wedge \bigwedge_k \text{Feas}_B(g_B, X_{ins}^{(k)}, X_{app}^{(k)}) \\
& \text{CoGraspClearance}(X_h, g_A, g_B) \geq \delta_{min} \\
& \text{path-connected: collision-free trajectories exist between all waypoints}
\end{aligned}
$$

where $\mathcal{Q}^{free}$ denotes joint-limit-satisfying, collision-free configurations and $\Phi$ is the quality score defined in §4.

---

## 2. Stage 1 — In-gripper part pose estimation

The nominal grasp $g_A^{nom}$ is corrupted by pick error (part shifted in the pile, jaw closing dynamics). Model the true grasp as

$$
\hat g_A = g_A^{nom} \cdot \exp(\hat\xi^{\wedge}), \qquad \hat\xi \in \mathbb{R}^6,\ \hat\xi \sim \mathcal{N}(0, \Sigma_g),
$$

with $\exp: \mathfrak{se}(3) \to SE(3)$ the exponential map and $\xi = (\rho, \phi)$ the translation/rotation twist. Estimate $\hat\xi$ by one of:

1. **Mechanical determinism** (baseline): form-fitting fingers + hard stop ⇒ $\Sigma_g$ small enough (< 0.2 mm / 0.5°) to skip measurement. Validate once with a scan.
2. **Quick-look camera** (upgrade): Robot A presents the gripped part to a 2D/3D camera; run PPF+ICP restricted to $SE(3)$ perturbations near $g_A^{nom}$ (a local, seeded registration — fast and unambiguous).
3. **Full scanner re-measure** after handoff (this always happens anyway at B's scanner station and corrects residual error before insertion; the handoff planner only needs $\hat g_A$ accurate to ~1 mm / 2° so the grippers don't fight each other).

Propagate uncertainty through frame composition with the adjoint: if $Y = T X$ and $X$ has covariance $\Sigma_X$ in its local tangent space, then $\Sigma_Y = \mathrm{Ad}_T\, \Sigma_X\, \mathrm{Ad}_T^{\top}$, where

$$
\mathrm{Ad}_T = \begin{bmatrix} R & [t]_\times R \\ 0 & R \end{bmatrix} \in \mathbb{R}^{6\times 6}.
$$

Use this to convert grasp covariance into gripper-to-gripper misalignment covariance at the handoff pose and check it against the receiver jaw's capture region (chamfer / compliance funnel) — exactly analogous to a peg-in-hole clearance test.

---

## 3. Stage 2 — Candidate generation

### 3.1 Handoff pose parameterization

Decompose $X_h = (t_h, R_h)$. Restrict the search:

- **Position** $t_h \in \mathcal{V}$ — the *dual-reachability volume*, precomputed as the intersection of both robots' reachability maps (§3.2). Typically a lens-shaped region between the two bases. Discretize at 2–5 cm.
- **Orientation** $R_h$ — sample from a deterministic $SO(3)$ grid (e.g., 60–576 rotations via HEALPix/icosahedral sampling), or, much better, sample **task-informed orientations**: since $X_{ins}$ requires pins-down, prefer $R_h$ within a bounded geodesic distance of the orientation that minimizes B's reorientation:

$$
d_{SO(3)}(R_h, R_{ins}) = \left\| \log\!\left(R_h^{\top} R_{ins}\right)^{\vee} \right\| \leq \theta_{max}.
$$

In practice a horizontal "part presented sideways at chest height between the robots" band covers most industrial cases; the grid just formalizes it.

### 3.2 Reachability map precomputation (offline, per robot)

For robot $R \in \{A, B\}$, sample joint space (or sample TCP poses and run IK), and build a voxel map over $(x, y, z)$ with, per voxel, a set of discretized approach directions and a scalar quality:

$$
M_R(v, \hat n) = \max_{q\,:\, f_R(q) \in v,\ \hat z_{TCP}(q) \approx \hat n} \; w(q)\, P_{jl}(q),
$$

where $w(q)$ is Yoshikawa manipulability and $P_{jl}$ the joint-limit penalty (both defined in §4). Store as a dense array; queries are O(1).

Also precompute **B's downstream feasibility as a function of $g_B$ alone**. Since $X_{scan}$ and $X_{ins}^{(k)}$ are fixed cell poses, for each $g_B \in \mathcal{G}_B$:

$$
Q_{down}(g_B) = \min\Big( \tilde w\big(X_{scan}\, g_B\big),\ \min_k \tilde w\big(X_{ins}^{(k)}\, g_B\big),\ \min_k \tilde w\big(X_{app}^{(k)}\, g_B\big) \Big),
$$

where $\tilde w(\cdot)$ is the best penalized manipulability over the full IK solution set at that TCP pose (with full collision checking).

**This is the single biggest computational win in the whole pipeline**: the downstream feasibility of a receiver grasp $g_B$ does not depend on the handoff pose at all — it depends only on $g_B$. So you filter $\mathcal{G}_B$ down to a *downstream-feasible grasp subset*

$$
\mathcal{G}_B^{\star} = \{\, g_B : Q_{down}(g_B) > \epsilon \,\}
$$

**once, offline**, and the online problem shrinks to: find $(X_h, g_A, g_B \in \mathcal{G}_B^{\star})$ jointly reachable.

(Caveat: if PCB placements vary board-to-board, precompute $Q_{down}$ over the hole-pose envelope, or refresh it when a new board program loads.)

### 3.3 Grasp-pair compatibility

Not every $(g_A, g_B)$ pair works: the two grippers must not collide while co-grasping. Precompute a compatibility matrix

$$
\mathcal{C}[i,j] = \mathbb{1}\!\left[\ \mathrm{dist}\big(\mathcal{B}_{grip_A}(g_A^{(i)}),\ \mathcal{B}_{grip_B}(g_B^{(j)})\big) \geq \delta_{min}\ \right],
$$

where $\mathcal{B}_{grip}(g)$ is the gripper body volume posed on the part via $g$, and $\delta_{min}$ ≈ 5–10 mm **plus** the 3σ gripper-to-gripper misalignment from §2. Also require **approach/retreat cone separation**: A's retreat direction $\hat r_A$ and B's approach direction $\hat a_B$ should satisfy $\hat r_A \cdot \hat a_B \leq \cos\theta_{sep}$ (grippers depart/arrive without crossing) — typically opposing or orthogonal faces of the part.

The candidate set is then

$$
\mathcal{H} = \Big\{ (X_h, g_A^{(i)}, g_B^{(j)}) : X_h \in \text{grid},\ g_A^{(i)} \in \mathcal{G}_A^{reach}(\hat g_A),\ g_B^{(j)} \in \mathcal{G}_B^{\star},\ \mathcal{C}[i,j]=1 \Big\}.
$$

Note $\mathcal{G}_A^{reach}(\hat g_A)$: in the **direct** handoff branch, Robot A cannot change its grasp — $g_A = \hat g_A$ is whatever came out of the bin. Only the regrasp branch (§6) re-opens the choice of $g_A$.

---

## 4. Stage 3 — Feasibility filtering and scoring

Process candidates coarse-to-fine; each test is strictly cheaper than the next.

### 4.1 Hard gates (in order)

**(G1) Map lookup.** $M_A(t_h, \hat n_A) > 0$ and $M_B(t_h, \hat n_B) > 0$ with the approach directions implied by $g_A, g_B$. O(1); kills ~90% of the grid.

**(G2) Exact IK.** Solve $f_A(q_A) = {^{A_0}T_W}\, X_h\, g_A$ and $f_B(q_B) = {^{B_0}T_W}\, X_h\, g_B$. For 6-DOF industrial arms with spherical-ish wrists (GP7 class), use the analytic IK (up to 8 branches each); enumerate **all** branches, don't take the first.

**(G3) Joint limits & singularity.** For every surviving branch pair, require

$$
q_i \in [q_i^{min} + m_{jl},\ q_i^{max} - m_{jl}] \ \ \forall i, \qquad w(q) = \sqrt{\det\!\big(J(q)J(q)^{\top}\big)} = \prod_i \sigma_i(J) \geq w_{min},
$$

with margin $m_{jl}$ ≈ 5–10° and $w_{min}$ calibrated per robot (as a percentile of the reachability map's $w$ distribution — absolute values of $w$ are not comparable across robots or units).

**(G4) Collision.** Full FCL/Bullet check of both arms + grippers + part + cell at the co-grasp configuration $(q_A, q_B)$, plus swept-volume checks along A's approach-to-handoff and B's approach-to-handoff segments.

**(G5) Correction margin at insertion** (the subtle one). It is not enough that $X_{ins}^{(k)}$ has an IK solution — the force-guided search will command perturbations $\delta x \in \mathcal{E}$ (e.g., ±2 mm lateral, ±3° about the pin axis) around it. First-order test: for the chosen IK branch $q^{ins}$,

$$
\delta q = J^{+}(q^{ins})\, \delta x, \qquad \text{require } q^{ins} + \delta q \in \mathcal{Q}^{lim} \ \text{ and } \ \|\delta q\| \leq \kappa \|\delta x\|
$$

for all vertices $\delta x$ of the correction polytope $\mathcal{E}$ (8–16 vertices suffice). The gain bound $\kappa$ rejects near-singular branches where tiny Cartesian corrections demand large joint motion (equivalently, require $\sigma_{min}(J) \geq \sigma_0$). A more exact version solves IK at each perturbed pose and requires **branch continuity** (no IK-branch flip inside $\mathcal{E}$). Fold this test into $Q_{down}(g_B)$ offline — it is a property of $g_B$, not of $X_h$.

**(G6) Path existence.** Plan (or table-lookup) collision-free trajectories: A: current → handoff; B: home → handoff → scanner → PCB. At the library/production tier, pre-plan and cache these per ($X_h$-cell, $g_B$); online planning (RRT-Connect) is the fallback.

### 4.2 Soft score

For survivors, compute

$$
\Phi(h) = \lambda_1\, \Phi_{manip} + \lambda_2\, \Phi_{jl} + \lambda_3\, \Phi_{clear} - \lambda_4\, \Phi_{reorient} - \lambda_5\, \Phi_{cycle}
$$

with

- $\Phi_{manip} = \min\big(\bar w_A(q_A),\ \bar w_B(q_B),\ Q_{down}(g_B)\big)$ — worst-case penalized manipulability across the handoff and downstream chain;
- $\Phi_{jl} = \min_i \dfrac{\min(q_i - q_i^{min},\ q_i^{max} - q_i)}{\tfrac{1}{2}(q_i^{max} - q_i^{min})}$ — normalized joint-limit margin over both robots' branches;
- $\Phi_{clear} = d_{clear}(h)$ — minimum collision clearance at co-grasp;
- $\Phi_{reorient} = d_{SO(3)}\big(R_h R_{g_B},\ R_{ins}\big)$ — B's residual reorientation after receiving;
- $\Phi_{cycle} = \|q_A - q_A^{now}\|_{W} + \|q_B^{ho} - q_B^{home}\|_{W}$ — weighted joint travel as a cycle-time proxy.

The penalized manipulability combines Yoshikawa's index with a joint-limit penalty (Tsai's measure):

$$
\bar w(q) = w(q) \cdot \Big(1 - \exp\Big(-k \prod_i \tfrac{(q_i - q_i^{min})(q_i^{max} - q_i)}{(q_i^{max} - q_i^{min})^2}\Big)\Big).
$$

Two refinements worth using on a real cell:

- **Directional manipulability at insertion.** Yoshikawa's $w$ is isotropic, but insertion cares about specific directions: translation along the pin axis $\hat z_{ins}$ and rotation about it. Use the task-direction ellipsoid measure
$$
\alpha(\hat u) = \big(\hat u^{\top} (J J^{\top})^{-1} \hat u\big)^{-1/2}
$$
evaluated at $\hat u = \hat z_{ins}$ (velocity form) — and its force-ellipsoid dual for the preload direction, since force and velocity ellipsoids are reciprocal (the force ellipsoid uses $(JJ^\top)$ in place of $(JJ^\top)^{-1}$).
- **Wrist-rotation margin, explicitly.** Insertion yaw-dither consumes joint 6. Require $|q_6^{ins}| \leq q_6^{max} - \theta_{dither} - m_{jl}$ across all $k$ holes; parts with rotational pin symmetry (e.g., 2-pin at 180°) let you fold $q_6$ by the symmetry angle — check both.

Weights: keep G1–G6 hard, then a weighted-sum or lexicographic $\Phi$; start with $\lambda = (1,\ 0.5,\ 0.3,\ 0.5,\ 0.2)$ after normalizing each term to $[0,1]$, and tune on logged cycles.

---

## 5. Stage 4 — Handoff execution (the mechanics)

Let $h^{\star} = (X_h, g_A, g_B)$ be the winner. Execution sequence with force interlocks:

1. A moves to $X_h g_A$ (already there, or a final linear approach ≤ 50 mm).
2. B moves to a pre-handoff pose $X_h\, g_B\, \text{Trans}(0,0,-d_{pre})$, approaching along its gripper $\hat z$; $d_{pre}$ ≈ 30–50 mm.
3. B advances linearly to $X_h g_B$ at reduced speed with (if available) a force guard $|F| < F_{guard}$; a contact spike ⇒ misalignment beyond the capture region ⇒ abort, retreat, re-plan.
4. **Co-grasp**: B closes; verify B's aperture $d_B \approx d_{expected} \pm \tau$ (part actually captured) **before** A releases. During the co-grasp instant the part is dually constrained — keep it short (< 300 ms) and command no motion; if either robot has active compliance, soften it here to avoid internal-force buildup from the residual misalignment $\exp(\hat\xi^{\wedge})$.
5. A opens fully and retreats along $\hat r_A$ (linear, ≥ jaw depth + margin); then B is free to move.
6. B aperture + (optional) wrist-force sanity check ⇒ handoff confirmed ⇒ proceed to scanner.

The **capture region** analysis from §2 governs the tolerances here: gripper-to-gripper misalignment covariance

$$
\Sigma_{ho} = \mathrm{Ad}\,\Sigma_g\,\mathrm{Ad}^{\top} + \Sigma_{r2r}
$$

(grasp uncertainty + robot-to-robot calibration) must satisfy $3\sigma$ lateral < jaw chamfer / compliance funnel. If it doesn't, fix the calibration or add compliant jaw geometry — do not try to plan your way out of a metrology problem.

---

## 6. Stage 5 — Regrasp branch (when direct handoff fails)

Triggered when $\mathcal{H} = \emptyset$ after §4 (typically: the bin pick forced an unfavorable $\hat g_A$ — part gripped by the face B needs, or pins pointing at A's wrist).

### 6.1 Stable placements

Precompute the part's stable placement set $\mathcal{P} = \{p_1, \dots, p_m\}$ on a horizontal surface: each $p_l$ is an equivalence class of part orientations resting on a facet of the convex hull, valid iff the gravity projection of the center of mass falls inside the support polygon with margin:

$$
\pi_{\hat g}(\mathbf{c}) \in \mathrm{ConvHull}(\text{contact facet}), \qquad \text{margin} \geq \epsilon_{stab}.
$$

(`trimesh.poses.compute_stable_poses` computes these, with quasi-static probabilities, directly from the CAD mesh.) Each placement is a part pose on the nest, free in yaw:

$$
X_{place}(l, \theta) = {^{W}T_N}\cdot \text{Rot}_z(\theta) \cdot T_{p_l}.
$$

### 6.2 Regrasp graph

Build the bipartite **placement–grasp feasibility table** offline:

$$
F[l, i] = \mathbb{1}\big[\ \text{grasp } g_A^{(i)} \text{ on placement } p_l \text{ is surface-collision-free and IK-feasible for A over some } \theta\ \big].
$$

A regrasp is a path: current grasp $\hat g_A$ → place at some $p_l$ with $F[l, \hat i] = 1$ (A can *put it down* from the current grasp) → re-pick with $g_A'$ such that $F[l, i'] = 1$ **and** $g_A'$ admits a non-empty $\mathcal{H}$ (re-run §4 with $g_A'$; cache which grasps are handoff-viable — this too is precomputable). One placement hop suffices for almost all industrial parts; if not, search the graph (nodes = grasps ∪ placements, edges = $F$) with Dijkstra, cost = cycle time per hop. Choose $\theta$ (yaw on the nest) to maximize the re-pick's $\bar w$ — a free 1-DOF optimization, solved by a 10–20-point line search.

**Pose uncertainty across the regrasp**: placing on a flat plate adds settle error (~0.5–1 mm, part-dependent). If the nest is a machined V-groove / pocket, placement instead *reduces* uncertainty to fixture tolerance (~0.05–0.1 mm) — which is why a fixture nest beats a flat plate whenever you can afford the tooling: it turns the regrasp from an error-accumulating step into an error-*resetting* step. With a flat plate, add a quick 2D camera shot over the nest to re-measure yaw + XY before re-picking.

---

## 7. Full decision algorithm

```
procedure PLAN_AND_EXECUTE_HANDOFF(ĝ_A):
    # -- offline, already available --
    #   M_A, M_B          reachability maps                          (§3.2)
    #   G_B*, Q_down      downstream-feasible receiver grasps + quality
    #   C[i,j]            gripper co-grasp compatibility              (§3.3)
    #   F[l,i], P         placement–grasp table, stable placements    (§6)

    for attempt in 1..MAX_REGRASP+1:
        H ← ∅
        for X_h in POSE_GRID ∩ dual_reach(M_A, M_B):                 # G1
          for g_B in G_B* with C[current_grasp, g_B] = 1:
            if not exact_IK_all_branches(X_h, ĝ_A, g_B): continue    # G2
            if not joint_sing_margins(q_A, q_B):        continue     # G3
            if not collision_free_cograsp_and_sweeps(): continue     # G4
            # G5 folded into Q_down(g_B) offline
            if not paths_exist():                       continue     # G6
            H ← H ∪ {(X_h, ĝ_A, g_B, Φ)}                              # §4.2
        if H ≠ ∅:
            h* ← argmax_Φ H
            execute_direct_handoff(h*)                                # §5
            return SUCCESS

        # -- regrasp branch --
        if attempt > MAX_REGRASP: return FAIL_REJECT_PART
        (p_l, θ, g_A') ← best_regrasp(ĝ_A, F, P)                      # §6
        if none: return FAIL_REJECT_PART
        place(p_l, θ); re-measure pose (fixture nest or camera)
        pick with g_A'; verify aperture/force
        ĝ_A ← g_A' ⊕ measured correction

    # after handoff:
    scan → refine part-in-B-gripper pose → recompute insertion TCP targets → insert
```

Online cost with the precomputation in place: G1 lookups are microseconds; the dominant cost is G2/G4 on the ~10–100 survivors — well under 100 ms with analytic IK + FCL, negligible against a multi-second pick cycle. Without the offline $\mathcal{G}_B^{\star}$ factorization, the same search would need scanner + insertion IK/collision checks *per candidate* and would be 50–100× slower — that factorization is the difference between "online handoff optimization" being a research topic and a production feature.

---

## 8. What to log and how to validate

Per cycle, log: $\hat g_A$ (and its source), $|\mathcal{H}|$, chosen $h^{\star}$ with all score components, regrasp invocations, co-grasp force/aperture traces, and downstream outcome (scan residual, insertion retries). Then validate:

- **Direct-handoff rate** $= P(\mathcal{H} \neq \emptyset)$ vs. pick pose — if low, the fix is usually adding grasps to $\mathcal{G}_B$ or widening the orientation band, not more search.
- **Score → outcome correlation**: regress insertion retries against $Q_{down}(g_B)$ and the correction-margin term; if uncorrelated, your $w_{min}$ and margins are mis-set.
- **Handoff misalignment**: periodically scan the part in B's gripper immediately post-handoff and compare with the planned $g_B$ — this measures $\Sigma_{ho}$ empirically and closes the loop on the §2/§5 tolerances.
- **Ablation**: run a period with fixed handoff poses vs. the planner; the planner should win specifically on the tail (unfavorable picks) — that tail is where the yield lives.

---

## 9. Practical implementation stack

Analytic IK: IKFast or the closed-form GP7 solution (Yaskawa publishes the DH parameters); numeric fallback TRAC-IK. Collision: FCL (via MoveIt or standalone); represent grippers + part as convex decompositions (VHACD). Reachability maps: the `reuleaux` / MoveIt reachability tooling, or roll your own — it's about a day of GPU-batched FK. Stable poses: `trimesh.poses.compute_stable_poses`. $SO(3)$ sampling: HEALPix-based or the 72-rotation icosahedral set. All of §3.2's precomputation is embarrassingly parallel — batch FK/IK on GPU if the grids grow.

In ROS 2, structure it as a `handoff_planner` node (service: plan from $\hat g_A$ → returns $h^{\star}$ or a regrasp plan) feeding a BehaviorTree.CPP tree that sequences pick → validate grasp → plan → [direct handoff | place–regrasp–replan] → co-grasp interlock → scan → insert, with the force/aperture interlocks as condition nodes — this drops straight into the BT structure of an existing ROS 2 manipulation pipeline.
