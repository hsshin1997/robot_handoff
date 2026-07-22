"""Robot-independent continuous task-set approximation for insertion grasps.

The pose library used by this project is a collection of *witnesses*.  It is
not interpreted as an enumeration of a continuous pose space.  This module
instead covers explicitly authored contact modes with axis-aligned cells in a
low-dimensional parameter space::

    theta = (u, v, roll)

``u`` and ``v`` locate the grasp centre on an opposing pair of housing
patches, and ``roll`` rotates the gripper about the nominal closing axis.  A
cell is one range of these parameters.  The representation deliberately
separates three-valued cell claims from sampled evidence:

``SAFE``
    Every pose in the cell has been proved to satisfy every declared
    robot-independent task constraint.
``REJECTED``
    Every pose in the cell violates at least one declared constraint.
``UNRESOLVED``
    Neither statement has been proved.

The current connector project can analytically reject some edge cells because
the rectangular pad cannot fit inside the authored housing proxy for *any*
roll.  It cannot yet certify a safe cell: manufacturing tolerances, complete
pad contact, exact continuous mesh collision, insertion wrench, and compliant
pin/hole mechanics are not available.  Sampled library poses and finite-PCB
vertex-penetration checks are therefore stored only as witnesses attached to
``UNRESOLVED`` cells.

Frame convention follows the repository: ``T_X_Y`` maps coordinates in frame
``Y`` into frame ``X``.  A representative grasp is ``T_P_E``.  The seated
socket is ``T_B_P_insert`` and the straight insertion direction is expressed
in ``P``.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.se3 import inverse, validate_transform


CELL_SAFE = "SAFE"
CELL_REJECTED = "REJECTED"
CELL_UNRESOLVED = "UNRESOLVED"
CELL_CLASSIFICATIONS = (CELL_SAFE, CELL_REJECTED, CELL_UNRESOLVED)

_EPS = np.finfo(float).eps


def _unit(value: Any, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a finite three-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 64.0 * _EPS:
        raise ValueError(f"{label} must be nonzero")
    return vector / norm


def _pair(value: Any, *, label: str) -> tuple[float, float]:
    bounds = np.asarray(value, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
        raise ValueError(f"{label} must be a finite [minimum, maximum] pair")
    if not bounds[0] < bounds[1]:
        raise ValueError(f"{label} minimum must be less than maximum")
    return float(bounds[0]), float(bounds[1])


def _positive_integer(value: Any, *, label: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _rounded(value: float) -> float:
    result = round(float(value), 12)
    return 0.0 if result == -0.0 else result


def _array(value: np.ndarray) -> list[Any]:
    array = np.asarray(value, dtype=float)
    return np.vectorize(_rounded, otypes=[float])(array).tolist()


def sha256_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 digest without loading a large CAD file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_cell_id(
    mode_id: str,
    u_index: int,
    v_index: int,
    roll_index: int,
) -> str:
    payload = f"{mode_id}:{u_index}:{v_index}:{roll_index}".encode("utf-8")
    return "task_cell_" + hashlib.sha256(payload).hexdigest()[:16]


def wrap_periodic_angle(angle_rad: float) -> float:
    """Map an angle into the half-open interval ``[-pi, pi)``."""
    value = float(angle_rad)
    if not np.isfinite(value):
        raise ValueError("angle must be finite")
    wrapped = (value + np.pi) % (2.0 * np.pi) - np.pi
    return -np.pi if np.isclose(wrapped, np.pi) else float(wrapped)


@dataclass(frozen=True)
class ContactMode:
    """One opposing-patch family and its parameter-cell resolution."""

    mode_id: str
    description: str
    closing_axis_P: np.ndarray
    position_u_axis_P: np.ndarray
    position_v_axis_P: np.ndarray
    contact_midplane_coordinate_P_m: float
    roll_zero_approach_axis_P: np.ndarray
    aperture_model: Mapping[str, Any]
    u_bounds_P_m: tuple[float, float]
    v_bounds_P_m: tuple[float, float]
    possible_aperture_range_m: tuple[float, float]
    u_cells: int
    v_cells: int
    roll_cells: int

    def __post_init__(self) -> None:
        if not isinstance(self.mode_id, str) or not self.mode_id:
            raise ValueError("contact mode id must be non-empty")
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("contact mode description must be non-empty")
        closing = _unit(self.closing_axis_P, label="closing_axis_P")
        u_axis = _unit(self.position_u_axis_P, label="position_u_axis_P")
        v_axis = _unit(self.position_v_axis_P, label="position_v_axis_P")
        axes = np.column_stack((u_axis, closing, v_axis))
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-10, rtol=0.0):
            raise ValueError("contact-mode closing/u/v axes must be orthogonal")
        if abs(float(np.linalg.det(axes))) < 1.0 - 1e-10:
            raise ValueError("contact-mode axes must form a complete basis")
        object.__setattr__(self, "closing_axis_P", closing)
        object.__setattr__(self, "position_u_axis_P", u_axis)
        object.__setattr__(self, "position_v_axis_P", v_axis)
        midplane = float(self.contact_midplane_coordinate_P_m)
        if not np.isfinite(midplane):
            raise ValueError("contact_midplane_coordinate_P_m must be finite")
        object.__setattr__(self, "contact_midplane_coordinate_P_m", midplane)
        roll_zero = _unit(
            self.roll_zero_approach_axis_P,
            label="roll_zero_approach_axis_P",
        )
        if abs(float(roll_zero @ closing)) > 1e-10:
            raise ValueError(
                "roll_zero_approach_axis_P must be orthogonal to closing_axis_P"
            )
        object.__setattr__(self, "roll_zero_approach_axis_P", roll_zero)
        aperture_model = dict(self.aperture_model)
        if aperture_model.get("type") != "constant":
            raise ValueError(
                "this task-set schema currently requires aperture_model.type=constant"
            )
        aperture_value = float(aperture_model.get("value_m", np.nan))
        if not np.isfinite(aperture_value) or aperture_value <= 0.0:
            raise ValueError("constant aperture_model.value_m must be positive")
        aperture_model["value_m"] = aperture_value
        object.__setattr__(self, "aperture_model", aperture_model)
        object.__setattr__(self, "u_bounds_P_m", _pair(
            self.u_bounds_P_m, label=f"{self.mode_id}.u_bounds_P_m"))
        object.__setattr__(self, "v_bounds_P_m", _pair(
            self.v_bounds_P_m, label=f"{self.mode_id}.v_bounds_P_m"))
        aperture = np.asarray(self.possible_aperture_range_m, dtype=float)
        if (aperture.shape != (2,) or not np.all(np.isfinite(aperture))
                or aperture[0] <= 0.0 or aperture[0] > aperture[1]):
            raise ValueError(
                f"{self.mode_id}.possible_aperture_range_m is invalid")
        object.__setattr__(self, "possible_aperture_range_m", (
            float(aperture[0]), float(aperture[1])))
        if not aperture[0] - 1e-12 <= aperture_value <= aperture[1] + 1e-12:
            raise ValueError(
                "aperture_model.value_m must lie in possible_aperture_range_m")
        for name in ("u_cells", "v_cells", "roll_cells"):
            object.__setattr__(self, name, _positive_integer(
                getattr(self, name), label=f"{self.mode_id}.{name}"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ContactMode":
        try:
            cells = value["cells"]
            return cls(
                mode_id=str(value["id"]),
                description=str(value["description"]),
                closing_axis_P=np.asarray(value["closing_axis_P"], dtype=float),
                position_u_axis_P=np.asarray(
                    value["position_u_axis_P"], dtype=float),
                position_v_axis_P=np.asarray(
                    value["position_v_axis_P"], dtype=float),
                contact_midplane_coordinate_P_m=float(
                    value["contact_midplane_coordinate_P_m"]),
                roll_zero_approach_axis_P=np.asarray(
                    value["roll_zero_approach_axis_P"], dtype=float),
                aperture_model=value["aperture_model"],
                u_bounds_P_m=_pair(
                    value["u_bounds_P_m"], label="u_bounds_P_m"),
                v_bounds_P_m=_pair(
                    value["v_bounds_P_m"], label="v_bounds_P_m"),
                possible_aperture_range_m=_pair(
                    value["possible_aperture_range_m"],
                    label="possible_aperture_range_m",
                ),
                u_cells=cells["u"],
                v_cells=cells["v"],
                roll_cells=cells["roll"],
            )
        except KeyError as error:
            raise ValueError(
                f"contact mode is missing {error.args[0]}") from error

    @classmethod
    def from_artifact_mapping(cls, value: Mapping[str, Any]) -> "ContactMode":
        """Reconstruct the executable map stored in a generated artifact."""
        try:
            mapping = value["constructive_map"]
            counts = value["cell_counts"]
            mode = cls(
                mode_id=str(value["id"]),
                description=str(value["description"]),
                closing_axis_P=np.asarray(
                    mapping["closing_axis_P"], dtype=float),
                position_u_axis_P=np.asarray(
                    mapping["position_u_axis_P"], dtype=float),
                position_v_axis_P=np.asarray(
                    mapping["position_v_axis_P"], dtype=float),
                contact_midplane_coordinate_P_m=float(
                    mapping["contact_midplane_coordinate_P_m"]),
                roll_zero_approach_axis_P=np.asarray(
                    mapping["roll_zero_approach_axis_P"], dtype=float),
                aperture_model=mapping["aperture_model"],
                u_bounds_P_m=_pair(
                    value["u_bounds_P_m"], label="u_bounds_P_m"),
                v_bounds_P_m=_pair(
                    value["v_bounds_P_m"], label="v_bounds_P_m"),
                possible_aperture_range_m=_pair(
                    value["possible_aperture_range_m"],
                    label="possible_aperture_range_m",
                ),
                u_cells=counts["u"],
                v_cells=counts["v"],
                roll_cells=counts["roll"],
            )
            if mapping.get("positive_roll_rule") != (
                    "right_hand_about_closing_axis_P"):
                raise ValueError(
                    "artifact contact mode has unsupported positive_roll_rule")
            declared_quadrature = _unit(
                mapping["positive_roll_quadrature_axis_P"],
                label="positive_roll_quadrature_axis_P",
            )
            if not np.allclose(
                    declared_quadrature,
                    mode.positive_roll_quadrature_axis_P,
                    atol=1e-12,
                    rtol=0.0):
                raise ValueError(
                    "artifact positive-roll quadrature is inconsistent with "
                    "cross(closing, roll_zero_approach)")
            return mode
        except KeyError as error:
            raise ValueError(
                f"artifact contact mode is missing {error.args[0]}") from error

    @property
    def positive_roll_quadrature_axis_P(self) -> np.ndarray:
        """Right-hand positive-roll direction at zero roll."""
        return _unit(
            np.cross(self.closing_axis_P, self.roll_zero_approach_axis_P),
            label="positive_roll_quadrature_axis_P",
        )

    def aperture_m(self, u_P_m: float, v_P_m: float) -> float:
        """Evaluate the authored aperture model.

        ``u`` and ``v`` are accepted explicitly so a future schema can add a
        verified piecewise model without changing the constructive-map API.
        """
        if not np.all(np.isfinite((u_P_m, v_P_m))):
            raise ValueError("aperture query coordinates must be finite")
        return float(self.aperture_model["value_m"])

    def construct_pose(
        self,
        u_P_m: float,
        v_P_m: float,
        roll_rad: float,
    ) -> tuple[np.ndarray, float]:
        """Construct ``T_P_E(theta)`` using right-hand roll about closing.

        Positive roll follows Rodrigues' right-hand rule around
        ``closing_axis_P``.  The parameter-position ``v`` axis is intentionally
        independent of this roll quadrature, preventing a hidden handedness
        flip when a mode chooses ``+v`` opposite to positive angular motion.
        """
        u = float(u_P_m)
        v = float(v_P_m)
        roll = float(roll_rad)
        if not np.all(np.isfinite((u, v, roll))):
            raise ValueError("constructive-map parameters must be finite")
        zero = self.roll_zero_approach_axis_P
        quadrature = self.positive_roll_quadrature_axis_P
        approach = _unit(
            np.cos(roll) * zero + np.sin(roll) * quadrature,
            label="constructed approach axis",
        )
        x_axis = _unit(
            np.cross(self.closing_axis_P, approach),
            label="constructed E x axis",
        )
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack((
            x_axis,
            self.closing_axis_P,
            approach,
        ))
        transform[:3, 3] = (
            u * self.position_u_axis_P
            + v * self.position_v_axis_P
            + self.contact_midplane_coordinate_P_m * self.closing_axis_P
        )
        return validate_transform(transform), self.aperture_m(u, v)

    def parameterize_seed(
        self,
        record: Mapping[str, Any],
        *,
        minimum_closing_alignment: float,
    ) -> tuple[float, float, float] | None:
        """Project one sampled grasp witness into this mode's coordinates."""
        transform = validate_transform(np.asarray(record["T_P_E"], dtype=float))
        contacts = record.get("contact_points_P_m")
        if contacts is None:
            centre = transform[:3, 3]
        else:
            points = np.asarray(contacts, dtype=float)
            if points.shape != (2, 3) or not np.all(np.isfinite(points)):
                raise ValueError("contact_points_P_m must have shape (2, 3)")
            centre = np.mean(points, axis=0)
        closing = _unit(
            record.get("closing_direction_P", transform[:3, 1]),
            label="seed closing_direction_P",
        )
        if abs(float(closing @ self.closing_axis_P)) < minimum_closing_alignment:
            return None
        approach = _unit(
            record.get("approach_direction_P", transform[:3, 2]),
            label="seed approach_direction_P",
        )
        zero_component = float(approach @ self.roll_zero_approach_axis_P)
        quadrature_component = float(
            approach @ self.positive_roll_quadrature_axis_P)
        if np.hypot(zero_component, quadrature_component) <= 1e-8:
            return None
        u = float(centre @ self.position_u_axis_P)
        v = float(centre @ self.position_v_axis_P)
        roll = wrap_periodic_angle(np.arctan2(
            quadrature_component, zero_component))
        return u, v, roll

    def cell_indices(
        self,
        parameters: tuple[float, float, float],
    ) -> tuple[int, int, int] | None:
        u, v, roll = parameters

        def index(value: float, bounds: tuple[float, float], count: int) -> int | None:
            low, high = bounds
            tolerance = 1e-10 * max(1.0, abs(low), abs(high))
            if value < low - tolerance or value > high + tolerance:
                return None
            fraction = (min(max(value, low), high) - low) / (high - low)
            return min(int(fraction * count), count - 1)

        u_index = index(u, self.u_bounds_P_m, self.u_cells)
        v_index = index(v, self.v_bounds_P_m, self.v_cells)
        roll_index = index(
            wrap_periodic_angle(roll), (-np.pi, np.pi), self.roll_cells)
        if u_index is None or v_index is None or roll_index is None:
            return None
        return u_index, v_index, roll_index

    def cell_bounds(
        self,
        u_index: int,
        v_index: int,
        roll_index: int,
    ) -> dict[str, list[float]]:
        def bounds(pair: tuple[float, float], count: int, index: int) -> list[float]:
            low, high = pair
            step = (high - low) / count
            return [_rounded(low + index * step),
                    _rounded(low + (index + 1) * step)]

        return {
            "u_P_m": bounds(self.u_bounds_P_m, self.u_cells, u_index),
            "v_P_m": bounds(self.v_bounds_P_m, self.v_cells, v_index),
            "roll_rad": bounds((-np.pi, np.pi), self.roll_cells, roll_index),
        }


def _best_symmetric_edge_clearance(
    centre_bounds: Sequence[float],
    patch_bounds: Sequence[float],
) -> float:
    """Maximum achievable clearance to both patch edges in a centre cell."""
    centre_low, centre_high = map(float, centre_bounds)
    patch_low, patch_high = map(float, patch_bounds)
    midpoint = 0.5 * (patch_low + patch_high)
    centre = min(max(midpoint, centre_low), centre_high)
    return max(0.0, min(centre - patch_low, patch_high - centre))


def analytic_cell_rejection(
    mode: ContactMode,
    bounds: Mapping[str, Sequence[float]],
    *,
    pad_size_m: Sequence[float],
    usable_opening_range_m: Sequence[float],
) -> str | None:
    """Prove a whole cell invalid using conservative necessary conditions.

    For every in-plane rotation, the projection of a rectangular pad onto
    either tangent axis is at least half of its shorter side.  If even the
    best centre in a cell cannot provide this much edge clearance, no roll in
    that cell can keep the full pad inside the authored rectangular patch.
    This test is deliberately weak but is a valid whole-cell rejection.
    """
    pad = np.asarray(pad_size_m, dtype=float)
    if pad.shape != (2,) or not np.all(np.isfinite(pad)) or np.any(pad <= 0.0):
        raise ValueError("pad_size_m must contain two positive dimensions")
    opening = np.asarray(usable_opening_range_m, dtype=float)
    if (opening.shape != (2,) or not np.all(np.isfinite(opening))
            or opening[0] < 0.0 or opening[0] >= opening[1]):
        raise ValueError("usable_opening_range_m is invalid")
    possible = mode.possible_aperture_range_m
    if possible[1] < opening[0] or possible[0] > opening[1]:
        return "MODE_APERTURE_RANGE_DISJOINT_FROM_USABLE_GRIPPER_RANGE"
    minimum_projected_half_extent = 0.5 * float(np.min(pad))
    available_u = _best_symmetric_edge_clearance(
        bounds["u_P_m"], mode.u_bounds_P_m)
    available_v = _best_symmetric_edge_clearance(
        bounds["v_P_m"], mode.v_bounds_P_m)
    if available_u + 1e-12 < minimum_projected_half_extent:
        return "PAD_FOOTPRINT_CANNOT_FIT_U_PATCH_FOR_ANY_ROLL"
    if available_v + 1e-12 < minimum_projected_half_extent:
        return "PAD_FOOTPRINT_CANNOT_FIT_V_PATCH_FOR_ANY_ROLL"
    return None


def query_constructive_task_pose(
    task_set_document: Mapping[str, Any],
    *,
    contact_mode: str,
    u_P_m: float,
    v_P_m: float,
    roll_rad: float,
) -> dict[str, Any]:
    """Evaluate an arbitrary pose inside an artifact's authored mode domain.

    This is the continuous query interface.  It does not snap to a cell or a
    sampled library witness.  Cell membership is returned only so callers can
    find the associated SAFE/REJECTED/UNRESOLVED range claim.
    """
    if task_set_document.get("artifact_type") != (
            "robot_independent_insertion_task_set"):
        raise ValueError("document is not a robot-independent insertion task set")
    parameterization = task_set_document.get("parameterization")
    if not isinstance(parameterization, Mapping):
        raise ValueError("task set requires parameterization")
    values = parameterization.get("contact_modes")
    if not isinstance(values, list):
        raise ValueError("task set requires parameterization.contact_modes")
    record = next(
        (value for value in values
         if isinstance(value, Mapping) and value.get("id") == contact_mode),
        None,
    )
    if record is None:
        raise ValueError(f"unknown contact mode {contact_mode!r}")
    mode = ContactMode.from_artifact_mapping(record)
    u = float(u_P_m)
    v = float(v_P_m)
    roll = wrap_periodic_angle(roll_rad)
    tolerance = 1e-12
    if not (mode.u_bounds_P_m[0] - tolerance <= u
            <= mode.u_bounds_P_m[1] + tolerance):
        raise ValueError("u_P_m is outside the authored contact-mode domain")
    if not (mode.v_bounds_P_m[0] - tolerance <= v
            <= mode.v_bounds_P_m[1] + tolerance):
        raise ValueError("v_P_m is outside the authored contact-mode domain")
    indices = mode.cell_indices((u, v, roll))
    if indices is None:
        raise ValueError("query could not be assigned to an authored cell")
    cell_id = stable_cell_id(mode.mode_id, *indices)
    cells = task_set_document.get("cells")
    if not isinstance(cells, list):
        raise ValueError("task set requires cells")
    cell = next((value for value in cells
                 if isinstance(value, Mapping) and value.get("id") == cell_id), None)
    if cell is None:
        raise ValueError("task set is missing the query's parameter cell")
    transform, aperture = mode.construct_pose(u, v, roll)
    return {
        "contact_mode": mode.mode_id,
        "theta": {
            "u_P_m": _rounded(u),
            "v_P_m": _rounded(v),
            "roll_rad": _rounded(roll),
        },
        "T_P_E": _array(transform),
        "required_aperture_m": _rounded(aperture),
        "cell_id": cell_id,
        "cell_classification": str(cell["classification"]),
        "claim_scope": (
            "classification applies to the containing parameter cell; "
            "the outer approximation is limited to this authored mode domain"
        ),
    }


def _representative_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed_grasp_id": str(record["id"]),
        "seed_library_index": int(record["library_index"]),
        "seed_status": str(record["status"]),
        "T_P_E": _array(validate_transform(np.asarray(
            record["T_P_E"], dtype=float))),
        "required_aperture_m": _rounded(record["required_aperture_m"]),
        "quality": _rounded(record["quality"]),
    }


def _library_witness(record: Mapping[str, Any]) -> dict[str, Any]:
    status = str(record["status"])
    if bool(record.get("seated_compatible", False)):
        claim = "SAMPLED_PHASE1_SEATED_GEOMETRIC_WITNESS"
    elif bool(record.get("preinsert_compatible", False)):
        claim = "SAMPLED_PREINSERT_ONLY_WITNESS"
    else:
        claim = "SAMPLED_LIBRARY_REJECTION_WITNESS"
    return {
        "claim": claim,
        "seed_grasp_id": str(record["id"]),
        "seed_status": status,
        "scope": "single sampled T_P_E only; never a whole-cell claim",
        "infinite_plane_seated_clearance_m": (
            None if record.get("seated_pcb_clearance_m") is None
            else _rounded(record["seated_pcb_clearance_m"])
        ),
        "infinite_plane_preinsert_clearance_m": (
            None if record.get("preinsert_pcb_clearance_m") is None
            else _rounded(record["preinsert_pcb_clearance_m"])
        ),
    }


def _seed_rank(record: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        -float(bool(record.get("seated_compatible", False))),
        -float(bool(record.get("preinsert_compatible", False))),
        -float(record.get("quality", 0.0)),
        float(record.get("library_index", 0)),
    )


def build_parameter_cells(
    modes: Sequence[ContactMode],
    seed_records: Iterable[Mapping[str, Any]],
    *,
    pad_size_m: Sequence[float],
    usable_opening_range_m: Sequence[float],
    minimum_closing_alignment: float,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    """Cover all authored mode domains and attach sampled witnesses.

    The return value contains the JSON-compatible cells and a mapping from
    cell ID to the original representative record.  The latter is only for
    optional diagnostic geometry; callers must not infer continuum validity
    from it.
    """
    alignment = float(minimum_closing_alignment)
    if not np.isfinite(alignment) or not 0.0 < alignment <= 1.0:
        raise ValueError("minimum_closing_alignment must be in (0, 1]")
    if not modes:
        raise ValueError("at least one contact mode is required")
    if len({mode.mode_id for mode in modes}) != len(modes):
        raise ValueError("contact mode IDs must be unique")

    assigned: dict[tuple[str, int, int, int], list[Mapping[str, Any]]] = {}
    for record in seed_records:
        best: tuple[float, ContactMode, tuple[float, float, float]] | None = None
        transform = validate_transform(np.asarray(record["T_P_E"], dtype=float))
        closing = _unit(
            record.get("closing_direction_P", transform[:3, 1]),
            label="seed closing axis",
        )
        for mode in modes:
            parameters = mode.parameterize_seed(
                record, minimum_closing_alignment=alignment)
            if parameters is None:
                continue
            score = abs(float(closing @ mode.closing_axis_P))
            if best is None or score > best[0]:
                best = score, mode, parameters
        if best is None:
            continue
        _, mode, parameters = best
        indices = mode.cell_indices(parameters)
        if indices is not None:
            assigned.setdefault((mode.mode_id, *indices), []).append(record)

    cells: list[dict[str, Any]] = []
    representatives: dict[str, Mapping[str, Any]] = {}
    for mode in modes:
        for u_index in range(mode.u_cells):
            for v_index in range(mode.v_cells):
                for roll_index in range(mode.roll_cells):
                    bounds = mode.cell_bounds(u_index, v_index, roll_index)
                    theta = {
                        name: 0.5 * (float(pair[0]) + float(pair[1]))
                        for name, pair in bounds.items()
                    }
                    center_transform, center_aperture = mode.construct_pose(
                        theta["u_P_m"],
                        theta["v_P_m"],
                        theta["roll_rad"],
                    )
                    reason = analytic_cell_rejection(
                        mode,
                        bounds,
                        pad_size_m=pad_size_m,
                        usable_opening_range_m=usable_opening_range_m,
                    )
                    classification = (
                        CELL_REJECTED if reason is not None
                        else CELL_UNRESOLVED
                    )
                    key = (mode.mode_id, u_index, v_index, roll_index)
                    records = sorted(assigned.get(key, []), key=_seed_rank)
                    representative = records[0] if records else None
                    cell_id = stable_cell_id(
                        mode.mode_id, u_index, v_index, roll_index)
                    if representative is not None:
                        representatives[cell_id] = representative
                    witnesses = (
                        [] if representative is None
                        else [_library_witness(representative)]
                    )
                    cells.append({
                        "id": cell_id,
                        "contact_mode": mode.mode_id,
                        "grid_index": {
                            "u": u_index,
                            "v": v_index,
                            "roll": roll_index,
                        },
                        "bounds": bounds,
                        "classification": classification,
                        "classification_reason": (
                            reason if reason is not None
                            else "NO_WHOLE_CELL_PROOF_AVAILABLE"
                        ),
                        "center_pose": {
                            "theta": {
                                key: _rounded(value)
                                for key, value in theta.items()
                            },
                            "T_P_E": _array(center_transform),
                            "required_aperture_m": _rounded(center_aperture),
                            "source": "contact_mode_constructive_map",
                            "constructive_map_version": 1,
                        },
                        "representative": (
                            None if representative is None
                            else _representative_record(representative)
                        ),
                        "seed_witness_count": len(records),
                        "witnesses": witnesses,
                    })
    return cells, representatives


class FinitePCBFootprint:
    """Nominal finite-board solid query based on the PCB's top triangulation.

    A watertight, constant-thickness PCB STL has planar top triangles whose
    union is the exact nominal XY footprint, including through holes.  The
    query only returns true for points strictly inside both that footprint and
    the board Z slab.  It is useful for *positive* vertex-penetration
    witnesses.  A false result is not a mesh-clearance certificate because
    gripper triangle faces or edges can intersect the PCB without enclosing a
    sampled vertex.
    """

    def __init__(
        self,
        triangles_B_m: np.ndarray,
        *,
        top_surface_tolerance_m: float = 1e-8,
        spatial_bin_size_m: float = 0.004,
    ) -> None:
        triangles = np.asarray(triangles_B_m, dtype=float)
        if (triangles.ndim != 3 or triangles.shape[1:] != (3, 3)
                or len(triangles) == 0 or not np.all(np.isfinite(triangles))):
            raise ValueError("PCB triangles must have finite shape (N, 3, 3)")
        tolerance = float(top_surface_tolerance_m)
        bin_size = float(spatial_bin_size_m)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("top_surface_tolerance_m must be positive")
        if not np.isfinite(bin_size) or bin_size <= 0.0:
            raise ValueError("spatial_bin_size_m must be positive")
        self.z_min_m = float(np.min(triangles[:, :, 2]))
        self.z_max_m = float(np.max(triangles[:, :, 2]))
        if self.z_max_m - self.z_min_m <= 2.0 * tolerance:
            raise ValueError("PCB mesh has no resolvable thickness")
        top = np.all(
            np.abs(triangles[:, :, 2] - self.z_max_m) <= tolerance,
            axis=1,
        )
        if not np.any(top):
            raise ValueError("PCB mesh has no planar top triangles")
        self.top_triangles_xy_m = triangles[top, :, :2].copy()
        self.xy_min_m = np.min(self.top_triangles_xy_m.reshape(-1, 2), axis=0)
        self.xy_max_m = np.max(self.top_triangles_xy_m.reshape(-1, 2), axis=0)
        self.spatial_bin_size_m = bin_size
        self._bins: dict[tuple[int, int], list[int]] = {}
        for triangle_index, triangle in enumerate(self.top_triangles_xy_m):
            low = self._bin_index(np.min(triangle, axis=0))
            high = self._bin_index(np.max(triangle, axis=0))
            for i in range(low[0], high[0] + 1):
                for j in range(low[1], high[1] + 1):
                    self._bins.setdefault((i, j), []).append(triangle_index)

    @property
    def top_triangle_count(self) -> int:
        return int(len(self.top_triangles_xy_m))

    def _bin_index(self, point_xy: np.ndarray) -> tuple[int, int]:
        index = np.floor(
            (np.asarray(point_xy, dtype=float) - self.xy_min_m)
            / self.spatial_bin_size_m
        ).astype(int)
        return int(index[0]), int(index[1])

    @staticmethod
    def _strictly_inside_triangle(
        point: np.ndarray,
        triangle: np.ndarray,
        *,
        tolerance_m: float,
    ) -> bool:
        a, b, c = triangle
        v0 = b - a
        v1 = c - a
        v2 = point - a
        determinant = float(v0[0] * v1[1] - v0[1] * v1[0])
        if abs(determinant) <= 64.0 * _EPS:
            return False
        first = float((v2[0] * v1[1] - v2[1] * v1[0]) / determinant)
        second = float((v0[0] * v2[1] - v0[1] * v2[0]) / determinant)
        third = 1.0 - first - second
        # Convert the metric interior tolerance into a conservative
        # barycentric tolerance using the triangle's longest edge.
        scale = max(
            float(np.linalg.norm(b - a)),
            float(np.linalg.norm(c - b)),
            float(np.linalg.norm(a - c)),
            1e-12,
        )
        barycentric_tolerance = tolerance_m / scale
        return min(first, second, third) > barycentric_tolerance

    def contains_nominal_solid(
        self,
        points_B_m: np.ndarray,
        *,
        interior_tolerance_m: float,
    ) -> np.ndarray:
        points = np.asarray(points_B_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_B_m must have shape (N, 3)")
        tolerance = float(interior_tolerance_m)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("interior_tolerance_m must be positive")
        if 2.0 * tolerance >= self.z_max_m - self.z_min_m:
            raise ValueError("interior tolerance consumes the PCB thickness")
        result = np.zeros(len(points), dtype=bool)
        slab_indices = np.flatnonzero(
            (points[:, 2] > self.z_min_m + tolerance)
            & (points[:, 2] < self.z_max_m - tolerance)
            & np.all(points[:, :2] > self.xy_min_m + tolerance, axis=1)
            & np.all(points[:, :2] < self.xy_max_m - tolerance, axis=1)
        )
        for point_index in slab_indices:
            point = points[point_index, :2]
            candidates = self._bins.get(self._bin_index(point), ())
            if any(self._strictly_inside_triangle(
                    point,
                    self.top_triangles_xy_m[triangle_index],
                    tolerance_m=tolerance,
            ) for triangle_index in candidates):
                result[point_index] = True
        return result


@dataclass(frozen=True)
class GripperComponentVertices:
    """A deterministic subset of component vertices in component frame C."""

    name: str
    vertices_C_m: np.ndarray
    T_G_C_reference: np.ndarray
    aperture_multiplier: float
    source_unique_vertex_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("component name must be non-empty")
        vertices = np.asarray(self.vertices_C_m, dtype=float)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0
                or not np.all(np.isfinite(vertices))):
            raise ValueError("vertices_C_m must have non-empty shape (N, 3)")
        transform = validate_transform(self.T_G_C_reference)
        multiplier = float(self.aperture_multiplier)
        if not np.isfinite(multiplier):
            raise ValueError("aperture_multiplier must be finite")
        source_count = _positive_integer(
            self.source_unique_vertex_count,
            label="source_unique_vertex_count",
        )
        if source_count < len(vertices):
            raise ValueError("source vertex count cannot be smaller than sample")
        object.__setattr__(self, "vertices_C_m", vertices.copy())
        object.__setattr__(self, "T_G_C_reference", transform)
        object.__setattr__(self, "aperture_multiplier", multiplier)
        object.__setattr__(self, "source_unique_vertex_count", source_count)


@dataclass(frozen=True)
class SampledGripperGeometry:
    """Registered component-vertex samples for positive collision witnesses."""

    T_G_E: np.ndarray
    reference_aperture_m: float
    opening_axis_G: np.ndarray
    components: tuple[GripperComponentVertices, ...]

    def __post_init__(self) -> None:
        transform = validate_transform(self.T_G_E)
        aperture = float(self.reference_aperture_m)
        if not np.isfinite(aperture) or aperture <= 0.0:
            raise ValueError("reference_aperture_m must be positive")
        axis = _unit(self.opening_axis_G, label="opening_axis_G")
        if not self.components:
            raise ValueError("sampled gripper requires components")
        object.__setattr__(self, "T_G_E", transform)
        object.__setattr__(self, "reference_aperture_m", aperture)
        object.__setattr__(self, "opening_axis_G", axis)

    def component_points_P(
        self,
        T_P_E: np.ndarray,
        required_aperture_m: float,
    ) -> Iterable[tuple[GripperComponentVertices, np.ndarray]]:
        T_P_G = validate_transform(T_P_E) @ inverse(self.T_G_E)
        aperture_delta = float(required_aperture_m) - self.reference_aperture_m
        for component in self.components:
            transform = component.T_G_C_reference
            translation_G = (
                transform[:3, 3]
                + self.opening_axis_G
                * component.aperture_multiplier
                * aperture_delta
            )
            points_G = (
                component.vertices_C_m @ transform[:3, :3].T
                + translation_G
            )
            points_P = points_G @ T_P_G[:3, :3].T + T_P_G[:3, 3]
            yield component, points_P


def finite_pcb_vertex_witness(
    *,
    footprint: FinitePCBFootprint,
    gripper: SampledGripperGeometry,
    T_P_E: np.ndarray,
    required_aperture_m: float,
    T_B_P_insert: np.ndarray,
    insertion_axis_P: np.ndarray,
    preinsert_distance_m: float,
    path_samples: int,
    interior_tolerance_m: float,
) -> dict[str, Any]:
    """Search a straight insertion sweep for a positive penetration witness."""
    seated = validate_transform(T_B_P_insert)
    axis_P = _unit(insertion_axis_P, label="insertion_axis_P")
    axis_B = seated[:3, :3] @ axis_P
    distance = float(preinsert_distance_m)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("preinsert_distance_m must be positive")
    samples = _positive_integer(path_samples, label="path_samples")
    if samples < 2:
        raise ValueError("path_samples must be at least two")

    component_points = tuple(gripper.component_points_P(
        T_P_E, required_aperture_m))
    tested_points = 0
    for path_index, progress in enumerate(np.linspace(0.0, 1.0, samples)):
        T_B_P = seated.copy()
        T_B_P[:3, 3] -= axis_B * distance * (1.0 - progress)
        for component, points_P in component_points:
            points_B = points_P @ T_B_P[:3, :3].T + T_B_P[:3, 3]
            tested_points += len(points_B)
            inside = footprint.contains_nominal_solid(
                points_B, interior_tolerance_m=interior_tolerance_m)
            if np.any(inside):
                point_index = int(np.flatnonzero(inside)[0])
                return {
                    "claim": "FINITE_PCB_INTERPENETRATION_WITNESS",
                    "scope": (
                        "one sampled gripper vertex at one sampled path state; "
                        "positive nominal-CAD collision evidence only"
                    ),
                    "component": component.name,
                    "path_sample_index": path_index,
                    "path_progress": _rounded(progress),
                    "point_B_m": _array(points_B[point_index]),
                    "tested_vertex_state_count_until_witness": tested_points,
                    "component_vertex_sample_count": len(component.vertices_C_m),
                    "component_source_unique_vertex_count": (
                        component.source_unique_vertex_count),
                    "pcb_top_triangle_count": footprint.top_triangle_count,
                    "interior_tolerance_m": _rounded(interior_tolerance_m),
                }
    return {
        "claim": "NO_FINITE_PCB_COLLISION_OBSERVED_IN_VERTEX_PATH_SAMPLE",
        "scope": (
            "absence of a sampled vertex inside the PCB is inconclusive; "
            "triangle/edge intersection and between-sample collision remain unchecked"
        ),
        "path_sample_count": samples,
        "tested_vertex_state_count": tested_points,
        "component_vertex_samples": {
            component.name: {
                "sampled": len(component.vertices_C_m),
                "source_unique": component.source_unique_vertex_count,
            }
            for component, _ in component_points
        },
        "pcb_top_triangle_count": footprint.top_triangle_count,
        "interior_tolerance_m": _rounded(interior_tolerance_m),
    }


def attach_finite_pcb_witnesses(
    cells: list[dict[str, Any]],
    representatives: Mapping[str, Mapping[str, Any]],
    *,
    footprint: FinitePCBFootprint,
    gripper: SampledGripperGeometry,
    T_B_P_insert: np.ndarray,
    insertion_axis_P: np.ndarray,
    preinsert_distance_m: float,
    maximum_representatives: int,
    path_samples: int,
    interior_tolerance_m: float,
) -> dict[str, int]:
    """Attach bounded finite-board evidence without changing cell claims."""
    budget = _positive_integer(
        maximum_representatives, label="maximum_representatives")
    cell_by_id = {cell["id"]: cell for cell in cells}
    ranked = sorted(
        representatives.items(),
        key=lambda item: _seed_rank(item[1]),
    )[:budget]
    collision_count = 0
    no_observation_count = 0
    for cell_id, record in ranked:
        witness = finite_pcb_vertex_witness(
            footprint=footprint,
            gripper=gripper,
            T_P_E=np.asarray(record["T_P_E"], dtype=float),
            required_aperture_m=float(record["required_aperture_m"]),
            T_B_P_insert=T_B_P_insert,
            insertion_axis_P=insertion_axis_P,
            preinsert_distance_m=preinsert_distance_m,
            path_samples=path_samples,
            interior_tolerance_m=interior_tolerance_m,
        )
        cell_by_id[cell_id]["witnesses"].append(witness)
        if witness["claim"] == "FINITE_PCB_INTERPENETRATION_WITNESS":
            collision_count += 1
        else:
            no_observation_count += 1
    return {
        "representatives_evaluated": len(ranked),
        "finite_pcb_interpenetration_witnesses": collision_count,
        "no_collision_observed_samples": no_observation_count,
        "representatives_not_evaluated": max(0, len(representatives) - len(ranked)),
    }


def build_task_set_document(
    *,
    project_id: str,
    modes: Sequence[ContactMode],
    pose_library: Mapping[str, Any],
    socket_contract: Mapping[str, Any],
    pad_size_m: Sequence[float],
    usable_opening_range_m: Sequence[float],
    minimum_closing_alignment: float,
    input_provenance: Mapping[str, Any],
    finite_pcb: tuple[
        FinitePCBFootprint,
        SampledGripperGeometry,
        Mapping[str, Any],
    ] | None = None,
) -> dict[str, Any]:
    """Build the self-contained layer-1 JSON artifact."""
    if not isinstance(project_id, str) or not project_id:
        raise ValueError("project_id must be non-empty")
    if int(pose_library.get("schema_version", 0)) != 1:
        raise ValueError("pose library schema_version must be 1")
    records = pose_library.get("candidates")
    if not isinstance(records, list):
        raise ValueError("pose library candidates must be a list")
    task_geometry = pose_library.get("task_geometry")
    if not isinstance(task_geometry, Mapping):
        raise ValueError("pose library task_geometry must be a mapping")
    if int(socket_contract.get("schema_version", 0)) != 1:
        raise ValueError("PCB socket schema_version must be 1")
    try:
        insertion_axis_P = _unit(
            task_geometry["insertion_axis_P"], label="insertion_axis_P")
        preinsert_distance_m = float(task_geometry["preinsert_distance_m"])
        T_B_P_insert = validate_transform(np.asarray(
            socket_contract["T_B_P_insert"], dtype=float))
    except KeyError as error:
        raise ValueError(
            f"input contract is missing {error.args[0]}") from error
    if not np.isfinite(preinsert_distance_m) or preinsert_distance_m <= 0.0:
        raise ValueError("preinsert_distance_m must be positive")
    connector_library_sha = str(
        pose_library.get("asset_stats", {}).get("part", {}).get("sha256", ""))
    connector_socket_sha = str(
        socket_contract.get("assets", {}).get("connector", {}).get("sha256", ""))
    if (not connector_library_sha or
            connector_library_sha != connector_socket_sha):
        raise ValueError("pose library and PCB socket connector hashes differ")

    cells, representatives = build_parameter_cells(
        modes,
        records,
        pad_size_m=pad_size_m,
        usable_opening_range_m=usable_opening_range_m,
        minimum_closing_alignment=minimum_closing_alignment,
    )
    finite_summary = {
        "enabled": False,
        "representatives_evaluated": 0,
        "finite_pcb_interpenetration_witnesses": 0,
        "no_collision_observed_samples": 0,
        "representatives_not_evaluated": len(representatives),
    }
    if finite_pcb is not None:
        footprint, gripper, options = finite_pcb
        finite_summary.update({
            "enabled": True,
            **attach_finite_pcb_witnesses(
                cells,
                representatives,
                footprint=footprint,
                gripper=gripper,
                T_B_P_insert=T_B_P_insert,
                insertion_axis_P=insertion_axis_P,
                preinsert_distance_m=preinsert_distance_m,
                maximum_representatives=int(options["maximum_representatives"]),
                path_samples=int(options["path_samples"]),
                interior_tolerance_m=float(options["interior_tolerance_m"]),
            ),
        })

    safe_ids = [cell["id"] for cell in cells
                if cell["classification"] == CELL_SAFE]
    rejected_ids = [cell["id"] for cell in cells
                    if cell["classification"] == CELL_REJECTED]
    unresolved_ids = [cell["id"] for cell in cells
                      if cell["classification"] == CELL_UNRESOLVED]
    occupied = sum(cell["representative"] is not None for cell in cells)
    return {
        "schema_version": 1,
        "artifact_type": "robot_independent_insertion_task_set",
        "project_id": project_id,
        "task_identity": {
            "connector_sha256": connector_library_sha,
            "pcb_sha256": str(
                socket_contract.get("assets", {}).get("pcb", {}).get(
                    "sha256", "")),
        },
        "claim_model": {
            "cell_classifications": list(CELL_CLASSIFICATIONS),
            "safe_inner_approximation": "union of SAFE cells only",
            "possible_outer_approximation": (
                "union of SAFE and UNRESOLVED cells within the explicitly "
                "authored contact-mode domains only; poses outside those "
                "domains are not classified"),
            "sampled_witness_rule": (
                "a representative or collision witness applies to one sampled "
                "pose/path state and never promotes an entire cell to SAFE or REJECTED"
            ),
        },
        "frame_contract": {
            "pose": "T_P_E maps ideal gripper contact frame E into connector frame P",
            "socket": "T_B_P_insert maps the seated connector P into PCB frame B",
            "composition": "T_B_E = T_B_P_insert @ T_P_E",
            "parameter_coordinates": (
                "u_P_m and v_P_m are signed projections in connector frame P; "
                "roll_rad is periodic about the contact-mode closing axis"
            ),
        },
        "inputs": dict(input_provenance),
        "parameterization": {
            "variables": [
                {"name": "u_P_m", "type": "continuous", "unit": "metre"},
                {"name": "v_P_m", "type": "continuous", "unit": "metre"},
                {
                    "name": "roll_rad",
                    "type": "continuous_periodic",
                    "unit": "radian",
                    "canonical_interval": [-_rounded(np.pi), _rounded(np.pi)],
                },
            ],
            "periodic_variables": ["roll_rad"],
            "minimum_seed_closing_alignment": _rounded(
                minimum_closing_alignment),
            "contact_modes": [
                {
                    "id": mode.mode_id,
                    "description": mode.description,
                    "u_bounds_P_m": list(mode.u_bounds_P_m),
                    "v_bounds_P_m": list(mode.v_bounds_P_m),
                    "possible_aperture_range_m": list(
                        mode.possible_aperture_range_m),
                    "cell_counts": {
                        "u": mode.u_cells,
                        "v": mode.v_cells,
                        "roll": mode.roll_cells,
                    },
                    "constructive_map": {
                        "position_u_axis_P": _array(
                            mode.position_u_axis_P),
                        "position_v_axis_P": _array(
                            mode.position_v_axis_P),
                        "closing_axis_P": _array(mode.closing_axis_P),
                        "contact_midplane_coordinate_P_m": _rounded(
                            mode.contact_midplane_coordinate_P_m),
                        "roll_zero_approach_axis_P": _array(
                            mode.roll_zero_approach_axis_P),
                        "positive_roll_quadrature_axis_P": _array(
                            mode.positive_roll_quadrature_axis_P),
                        "positive_roll_rule": (
                            "right_hand_about_closing_axis_P"),
                        "aperture_model": dict(mode.aperture_model),
                        "formula": (
                            "p=u*position_u_axis_P+v*position_v_axis_P+"
                            "contact_midplane_coordinate_P_m*closing_axis_P; "
                            "q=cross(closing_axis_P,roll_zero_approach_axis_P); "
                            "E_z=cos(roll)*roll_zero_approach_axis_P+sin(roll)*q; "
                            "E_x=cross(closing_axis_P,E_z); "
                            "R_P_E=[E_x,closing_axis_P,E_z]"
                        ),
                    },
                }
                for mode in modes
            ],
        },
        "gripper_task_capability": {
            "pad_size_m": list(map(float, pad_size_m)),
            "usable_opening_range_m": list(map(float, usable_opening_range_m)),
            "status": "provisional_geometry_only",
        },
        "insertion_trajectory": {
            "type": "straight_fixed_orientation",
            "path_parameter_range": [0.0, 1.0],
            "insertion_axis_P": _array(insertion_axis_P),
            "preinsert_distance_m": _rounded(preinsert_distance_m),
            "T_B_P_insert": _array(T_B_P_insert),
            "formula": (
                "R_B_P(s)=R_B_P_insert; "
                "t_B_P(s)=t_B_P_insert-R_B_P_insert*axis_P*d*(1-s)"
            ),
        },
        "finite_pcb_witness_evaluation": finite_summary,
        "counts": {
            "cells": len(cells),
            "safe": len(safe_ids),
            "rejected": len(rejected_ids),
            "unresolved": len(unresolved_ids),
            "cells_with_seed_representative": occupied,
            "pose_library_seed_records": len(records),
        },
        "safe_inner_cell_ids": safe_ids,
        "rejected_cell_ids": rejected_ids,
        "unresolved_cell_ids": unresolved_ids,
        "cells": cells,
        "certification_boundary": {
            "certified_safe_set_available": bool(safe_ids),
            "checked": [
                "Complete authored (u, v, roll) domains are covered by cells.",
                "A whole cell is rejected when no pad roll can fit inside the authored rectangular housing proxy.",
                "The pose-library connector hash equals the PCB socket connector hash.",
                "Selected representative gripper vertices are tested against the actual finite PCB top-triangle footprint and thickness slab.",
            ],
            "not_checked": [
                "Interval-bounded exact triangle collision for every pose in each cell and every continuous path state.",
                "Complete pad footprint contact on true plastic CAD rather than an authored rectangular proxy.",
                "Part-versus-gripper collision away from intended contacts.",
                "Manufacturing, calibration, board-pose, hole-position, and gripper-registration uncertainty.",
                "Insertion wrench, friction/slip margin, pin compliance, board flex, solder, and contact dynamics.",
                "Robot inverse kinematics, joint limits, singularity, robot/fixture collision, and trajectory continuity.",
            ],
            "conclusion": (
                "SAFE is intentionally empty unless every declared constraint "
                "is proved over a complete parameter cell. UNRESOLVED cells form "
                "a conservative outer search domain, not an executable set."
            ),
        },
    }


def _refresh_classification_summary(document: dict[str, Any]) -> None:
    cells = document.get("cells")
    if not isinstance(cells, list):
        raise ValueError("task-set document requires cells for classification")
    identifiers: dict[str, list[str]] = {
        CELL_SAFE: [], CELL_REJECTED: [], CELL_UNRESOLVED: [],
    }
    for cell in cells:
        classification = cell.get("classification")
        if classification not in identifiers:
            raise ValueError(f"invalid cell classification {classification!r}")
        identifiers[classification].append(str(cell["id"]))
    document["safe_inner_cell_ids"] = identifiers[CELL_SAFE]
    document["rejected_cell_ids"] = identifiers[CELL_REJECTED]
    document["unresolved_cell_ids"] = identifiers[CELL_UNRESOLVED]
    counts = document.setdefault("counts", {})
    counts["cells"] = len(cells)
    counts["safe"] = len(identifiers[CELL_SAFE])
    counts["rejected"] = len(identifiers[CELL_REJECTED])
    counts["unresolved"] = len(identifiers[CELL_UNRESOLVED])
    boundary = document.get("certification_boundary")
    if isinstance(boundary, dict):
        boundary["certified_safe_set_available"] = bool(identifiers[CELL_SAFE])


def apply_whole_cell_task_certificates(
    document: dict[str, Any],
    certificate_imports: Sequence[Mapping[str, Any]],
    *,
    required_proved_constraints: Sequence[str],
) -> dict[str, Any]:
    """Fail-closed import of independently produced whole-cell certificates.

    The certificate binds the exact canonical certificate-binding hash of the
    current *uncertified* task definition.  It may promote only an
    ``UNRESOLVED`` cell and
    only after its file hash, project/connector bindings, and required proof
    obligations all match.  This verifier does not create proof; it merely
    prevents an unbound or incomplete external result from becoming ``SAFE``.

    Each import mapping must contain ``path``, ``expected_sha256``,
    ``actual_sha256``, and parsed JSON ``document``.  File IO stays with the
    caller so this interface is reusable by CLI, tests, or another build
    system.
    """
    required = tuple(required_proved_constraints)
    if (not required or any(not isinstance(item, str) or not item for item in required)
            or len(set(required)) != len(required)):
        raise ValueError(
            "required_proved_constraints must be unique non-empty strings")
    if any(cell.get("classification") == CELL_SAFE
           for cell in document.get("cells", [])):
        raise ValueError(
            "uncertified task-set input already contains SAFE cells")

    base_sha = certificate_binding_sha256(document)
    identity = document.get("task_identity", {})
    expected_bindings = {
        "base_artifact_certificate_binding_sha256": base_sha,
        "project_id": str(document.get("project_id", "")),
        "connector_sha256": str(identity.get("connector_sha256", "")),
    }
    cell_by_id = {str(cell["id"]): cell for cell in document.get("cells", [])}
    promoted: set[str] = set()
    imported: list[dict[str, Any]] = []
    for import_index, supplied in enumerate(certificate_imports):
        if not isinstance(supplied, Mapping):
            raise ValueError(f"certificate import {import_index} must be a mapping")
        path = str(supplied.get("path", ""))
        expected_file_sha = str(supplied.get("expected_sha256", ""))
        actual_file_sha = str(supplied.get("actual_sha256", ""))
        certificate = supplied.get("document")
        valid_expected = (
            len(expected_file_sha) == 64
            and all(character in "0123456789abcdef"
                    for character in expected_file_sha)
        )
        valid_actual = (
            len(actual_file_sha) == 64
            and all(character in "0123456789abcdef"
                    for character in actual_file_sha)
        )
        if not path or not valid_expected or not valid_actual:
            raise ValueError(
                f"certificate import {import_index} requires path and exact SHA-256")
        if expected_file_sha != actual_file_sha:
            raise ValueError(
                f"certificate file SHA mismatch for {path}: expected "
                f"{expected_file_sha}, got {actual_file_sha}")
        if not isinstance(certificate, Mapping):
            raise ValueError(f"certificate {path} must be a JSON object")
        if int(certificate.get("schema_version", 0)) != 1:
            raise ValueError(f"certificate {path} schema_version must be 1")
        if certificate.get("artifact_type") != (
                "robot_independent_insertion_task_whole_cell_certificate"):
            raise ValueError(f"certificate {path} has wrong artifact_type")
        certificate_id = certificate.get("certificate_id")
        if not isinstance(certificate_id, str) or not certificate_id:
            raise ValueError(f"certificate {path} requires certificate_id")
        bindings = certificate.get("bindings")
        if bindings != expected_bindings:
            raise ValueError(
                f"certificate {path} bindings do not exactly match the "
                "uncertified task-set artifact")
        proved = certificate.get("proved_constraints")
        if (not isinstance(proved, list)
                or any(not isinstance(item, str) or not item for item in proved)):
            raise ValueError(
                f"certificate {path} proved_constraints must be strings")
        missing = sorted(set(required) - set(proved))
        if missing:
            raise ValueError(
                f"certificate {path} is missing required proofs: {missing}")
        claims = certificate.get("cell_claims")
        if not isinstance(claims, list) or not claims:
            raise ValueError(f"certificate {path} requires non-empty cell_claims")
        local_ids: set[str] = set()
        for claim in claims:
            if not isinstance(claim, Mapping):
                raise ValueError(f"certificate {path} cell claim must be an object")
            cell_id = str(claim.get("cell_id", ""))
            if claim.get("classification") != CELL_SAFE:
                raise ValueError(
                    f"certificate {path} may only import SAFE cell claims")
            if cell_id not in cell_by_id:
                raise ValueError(
                    f"certificate {path} references unknown cell {cell_id!r}")
            if cell_id in local_ids or cell_id in promoted:
                raise ValueError(
                    f"cell {cell_id!r} has duplicate certificate claims")
            cell = cell_by_id[cell_id]
            if cell.get("classification") != CELL_UNRESOLVED:
                raise ValueError(
                    f"certificate cannot promote non-UNRESOLVED cell {cell_id!r}")
            if any(
                witness.get("claim") == "FINITE_PCB_INTERPENETRATION_WITNESS"
                for witness in cell.get("witnesses", [])
                if isinstance(witness, Mapping)
            ):
                raise ValueError(
                    f"certificate SAFE claim contradicts a finite-PCB "
                    f"penetration witness in cell {cell_id!r}")
            local_ids.add(cell_id)
        for cell_id in local_ids:
            cell = cell_by_id[cell_id]
            cell["classification"] = CELL_SAFE
            cell["classification_reason"] = (
                "IMPORTED_BOUND_WHOLE_CELL_TASK_CERTIFICATE")
            cell["whole_cell_task_certificate"] = {
                "certificate_id": certificate_id,
                "path": path,
                "sha256": actual_file_sha,
                "proved_constraints": sorted(set(proved)),
                "base_artifact_certificate_binding_sha256": base_sha,
            }
        promoted.update(local_ids)
        imported.append({
            "certificate_id": certificate_id,
            "path": path,
            "sha256": actual_file_sha,
            "promoted_cell_ids": sorted(local_ids),
        })

    document["whole_cell_task_certificates"] = {
        "verification_policy": "fail_closed_exact_file_and_artifact_binding",
        "base_artifact_certificate_binding_sha256": base_sha,
        "expected_bindings": expected_bindings,
        "required_proved_constraints": list(required),
        "imports": imported,
        "promoted_safe_cell_count": len(promoted),
    }
    _refresh_classification_summary(document)
    return document


def certificate_binding_sha256(document: Mapping[str, Any]) -> str:
    """Hash the exact uncertified task definition used by certificates.

    Certificate-import metadata is analogous to a signature block and is
    excluded from its signed payload.  The task-set config's raw file digest
    is also excluded because that file may contain the expected certificate
    digest; all effective task parameters and every other input digest remain
    in the payload.  This avoids a certificate/config hash cycle while binding
    every generated cell bound and constructive centre pose exactly.
    """
    normalized = deepcopy(dict(document))
    normalized.pop("semantic_sha256", None)
    normalized.pop("whole_cell_task_certificates", None)
    inputs = normalized.get("inputs")
    if isinstance(inputs, dict):
        task_config = inputs.get("task_set_config")
        if isinstance(task_config, dict):
            task_config.pop("sha256", None)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_sha256(document: Mapping[str, Any]) -> str:
    """Fingerprint the semantic JSON document independent of indentation."""
    normalized = dict(document)
    # The embedded digest describes the rest of the document and is therefore
    # excluded to avoid a self-referential hash.
    normalized.pop("semantic_sha256", None)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CELL_CLASSIFICATIONS",
    "CELL_REJECTED",
    "CELL_SAFE",
    "CELL_UNRESOLVED",
    "ContactMode",
    "FinitePCBFootprint",
    "GripperComponentVertices",
    "SampledGripperGeometry",
    "apply_whole_cell_task_certificates",
    "certificate_binding_sha256",
    "analytic_cell_rejection",
    "artifact_sha256",
    "attach_finite_pcb_witnesses",
    "build_parameter_cells",
    "build_task_set_document",
    "finite_pcb_vertex_witness",
    "query_constructive_task_pose",
    "sha256_file",
    "stable_cell_id",
    "wrap_periodic_angle",
]
