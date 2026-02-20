"""Shared transient control-mesh and bounded-transform helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apexsim.utils.constants import SMALL_EPS


@dataclass(frozen=True)
class ControlInterpolationMap:
    """Linear interpolation map from control nodes to full sample support.

    Args:
        left_index: Left node index per sample.
        right_index: Right node index per sample.
        right_weight: Right-node interpolation weight per sample.
    """

    left_index: np.ndarray
    right_index: np.ndarray
    right_weight: np.ndarray


def build_control_node_count(
    *,
    sample_count: int,
    control_interval: int,
) -> int:
    """Return number of control nodes used for mesh-based control variables.

    Args:
        sample_count: Number of full track samples.
        control_interval: Desired spacing between control nodes in samples.

    Returns:
        Control node count bounded to ``[1, sample_count]``.
    """
    if sample_count <= 1:
        return 1
    control_count = int(np.ceil((sample_count - 1) / max(control_interval, 1))) + 1
    return min(max(control_count, 2), sample_count)


def build_control_mesh_positions(
    *,
    sample_count: int,
    control_interval: int,
) -> np.ndarray:
    """Build evenly spaced control-node positions over sample indices.

    Args:
        sample_count: Number of full track samples.
        control_interval: Desired spacing between control nodes in samples.

    Returns:
        Monotonic control-node positions.
    """
    if sample_count <= 1:
        return np.zeros(1, dtype=float)
    control_count = build_control_node_count(
        sample_count=sample_count,
        control_interval=control_interval,
    )
    return np.linspace(0.0, float(sample_count - 1), control_count, dtype=float)


def build_control_interpolation_map(
    *,
    sample_count: int,
    mesh_positions: np.ndarray,
) -> ControlInterpolationMap:
    """Build per-sample linear interpolation weights for control expansion.

    Args:
        sample_count: Number of full track samples.
        mesh_positions: Control-node positions in sample-index coordinates.

    Returns:
        Interpolation map from node values to full sample support.
    """
    positions = np.asarray(mesh_positions, dtype=float)
    if sample_count <= 1:
        zero = np.zeros(1, dtype=np.int64)
        return ControlInterpolationMap(
            left_index=zero,
            right_index=zero,
            right_weight=np.zeros(1, dtype=float),
        )

    sample_positions = np.arange(sample_count, dtype=float)
    upper = np.searchsorted(positions, sample_positions, side="right")
    upper = np.clip(upper, 1, positions.size - 1)
    lower = upper - 1

    x0 = positions[lower]
    x1 = positions[upper]
    denom = np.maximum(x1 - x0, SMALL_EPS)
    right_weight = (sample_positions - x0) / denom
    right_weight = np.clip(right_weight, 0.0, 1.0)

    return ControlInterpolationMap(
        left_index=np.asarray(lower, dtype=np.int64),
        right_index=np.asarray(upper, dtype=np.int64),
        right_weight=np.asarray(right_weight, dtype=float),
    )


def sample_seed_on_mesh(
    seed: np.ndarray,
    mesh_positions: np.ndarray,
) -> np.ndarray:
    """Sample a full-resolution seed signal onto control-node positions.

    Args:
        seed: Full-resolution signal.
        mesh_positions: Control-node positions in sample-index coordinates.

    Returns:
        Seed values on control nodes.
    """
    sample_positions = np.arange(seed.size, dtype=float)
    return np.asarray(np.interp(mesh_positions, sample_positions, seed), dtype=float)


def expand_mesh_controls(
    *,
    node_values: np.ndarray,
    sample_count: int,
    mesh_positions: np.ndarray,
    interpolation_map: ControlInterpolationMap | None = None,
) -> np.ndarray:
    """Expand node values to full sample support by linear interpolation.

    Args:
        node_values: Control values at node support.
        sample_count: Number of full track samples.
        mesh_positions: Control-node positions in sample-index coordinates.
        interpolation_map: Optional precomputed interpolation map.

    Returns:
        Full-resolution control signal.
    """
    if sample_count <= 1:
        return np.asarray(node_values[:1], dtype=float)
    if interpolation_map is None:
        sample_positions = np.arange(sample_count, dtype=float)
        return np.asarray(np.interp(sample_positions, mesh_positions, node_values), dtype=float)

    left = np.asarray(node_values[interpolation_map.left_index], dtype=float)
    right = np.asarray(node_values[interpolation_map.right_index], dtype=float)
    weight = np.asarray(interpolation_map.right_weight, dtype=float)
    return np.asarray(left * (1.0 - weight) + right * weight, dtype=float)


def bounded_artanh(
    value: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Stable inverse-tanh transform for bounded signals in ``[-1, 1]``.

    Args:
        value: Input vector in ``[-1, 1]``.
        eps: Boundary clipping epsilon.

    Returns:
        Inverse-tanh transformed values.
    """
    clipped = np.clip(value, -1.0 + eps, 1.0 - eps)
    return np.asarray(np.arctanh(clipped), dtype=float)
