"""Contract tests for the connector-header PCB socket registration."""
from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "projects" / "connector_header_insertion" / "config" / "pcb_socket.yaml"
)


def _contract() -> dict:
    with CONTRACT_PATH.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points @ transform[:3, :3].T + transform[:3, 3]


def test_pcb_asset_provenance_and_socket_row() -> None:
    config = _contract()
    asset = config["assets"]["pcb"]
    path = ROOT / asset["path"]
    assert _sha256(path) == asset["sha256"]
    assert path.stat().st_size == asset["byte_count"]
    with path.open("rb") as stream:
        stream.seek(80)
        triangle_count = struct.unpack("<I", stream.read(4))[0]
    assert triangle_count == asset["triangle_count"] == 12124

    holes = np.asarray(config["socket"]["hole_centers_B_m"], dtype=float)
    assert holes.shape == (9, 3)
    assert np.allclose(holes[:, 0], config["socket"]["centerline_x_B_m"])
    assert np.allclose(holes[:, 2], config["board"]["nominal_top_surface_z_B_m"])
    index = np.arange(9, dtype=float)
    fit = np.linalg.lstsq(
        np.column_stack((np.ones(9), index)), holes[:, 1], rcond=None,
    )[0]
    assert np.isclose(fit[1], config["socket"]["nominal_pitch_m"], atol=1e-10)


def test_insert_transform_is_proper_se3_and_maps_authored_axes() -> None:
    config = _contract()
    transform = np.asarray(config["T_B_P_insert"], dtype=float)
    assert transform.shape == (4, 4)
    assert np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0])
    rotation = transform[:3, :3]
    atol = config["tolerances_and_margins"]["rotation_orthonormal_atol"]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0.0)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=atol, rtol=0.0)

    frame_B = config["frames"]["B"]
    frame_P = config["frames"]["P"]
    assert np.allclose(
        rotation @ np.asarray(frame_P["short_tail_axis"]),
        frame_B["insertion_direction"],
    )
    assert np.allclose(
        rotation @ np.asarray(frame_P["long_mating_post_axis"]),
        frame_B["nearest_outward_edge_axis"],
    )
    assert np.allclose(
        rotation @ np.asarray(frame_P["pin_row_axis"]),
        -np.asarray(frame_B["row_axis"]),
    )


def test_transform_maps_all_pin_centers_and_seating_plane_to_socket() -> None:
    config = _contract()
    transform = np.asarray(config["T_B_P_insert"], dtype=float)
    geometry = config["connector_socket_geometry"]
    pin_x = np.asarray(geometry["pin_centers_x_P_m"], dtype=float)
    seating_y = float(geometry["seating_plane_y_P_m"])
    tail_z = float(geometry["short_tail_center_z_P_m"])
    pin_centers_at_board_P = np.column_stack((
        pin_x,
        np.full(9, seating_y),
        np.full(9, tail_z),
    ))
    mapped_B = _transform_points(transform, pin_centers_at_board_P)
    holes_B = np.asarray(config["socket"]["hole_centers_B_m"], dtype=float)

    # +X_P maps to -Y_B, so increasing pin order maps to reversed hole order.
    errors = np.linalg.norm(mapped_B - holes_B[::-1], axis=1)
    tolerance = config["tolerances_and_margins"][
        "maximum_pin_to_hole_center_fit_error_m"
    ]
    assert np.max(errors) <= tolerance
    assert np.allclose(
        mapped_B[:, 2], config["board"]["nominal_top_surface_z_B_m"],
        atol=config["tolerances_and_margins"]["seating_plane_fit_tolerance_m"],
        rtol=0.0,
    )

    tail_tip_P = np.array([[pin_x[0], geometry["tail_tip_plane_y_P_m"], tail_z]])
    tail_tip_B = _transform_points(transform, tail_tip_P)[0]
    board_bottom = float(config["board"]["bottom_surface_z_B_m"])
    expected_protrusion = float(
        config["derived_seated_checks"]["tail_protrusion_below_board_m"]
    )
    assert np.isclose(
        board_bottom - tail_tip_B[2], expected_protrusion, atol=1e-9, rtol=0.0,
    )


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
