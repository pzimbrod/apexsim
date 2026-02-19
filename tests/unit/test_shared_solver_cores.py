"""Unit tests for shared solver-core helper modules."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from apexsim.simulation._profile_core import (
    ProfileOps,
    SpeedProfileCoreCallbacks,
    SpeedProfileCoreInputs,
    lateral_speed_limit_core,
    resolve_profile_start_speed,
    solve_speed_profile_core,
)
from apexsim.simulation._transient_controls_core import (
    bounded_artanh,
    build_control_interpolation_map,
    build_control_mesh_positions,
    build_control_node_count,
    expand_mesh_controls,
    sample_seed_on_mesh,
)
from apexsim.simulation._transient_pid_core import (
    bounded_pid_command,
    clamp_integral_scalar,
    clamp_integral_value,
    pid_error_derivative,
    resolve_initial_speed,
    segment_time_partition,
    segment_time_partition_torch,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


NUMPY_OPS = ProfileOps(
    full=lambda size, value: np.full((size,), value, dtype=float),
    copy=lambda value: np.copy(value),
    scalar=lambda value: float(value),
    abs=lambda value: np.asarray(np.abs(value), dtype=float),
    maximum=lambda left, right: np.maximum(left, right),
    minimum=lambda left, right: np.minimum(left, right),
    clip=lambda value, low, high: np.clip(value, low, high),
    sqrt=lambda value: np.sqrt(value),
    where=lambda condition, left, right: np.where(condition, left, right),
    stack=lambda values: np.asarray(values, dtype=float),
    cat_tail=lambda core: np.concatenate((core, core[-1:])),
    zeros_like=lambda ref: np.zeros_like(ref),
    max=lambda value: float(np.max(value)),
    sum=lambda value: float(np.sum(value)),
    to_float=lambda value: float(value),
)


class ProfileCoreTests(unittest.TestCase):
    """Validate shared quasi-static core primitives."""

    def test_resolve_profile_start_speed_uses_override(self) -> None:
        """Resolve explicit start speed when runtime override is set."""
        self.assertEqual(
            resolve_profile_start_speed(max_speed=30.0, initial_speed=None),
            30.0,
        )
        self.assertEqual(
            resolve_profile_start_speed(max_speed=30.0, initial_speed=4.0),
            4.0,
        )

    def test_lateral_speed_limit_handles_zero_curvature(self) -> None:
        """Clamp straight segments to max speed in lateral-speed limit."""
        curvature_abs = np.asarray([0.0, 0.25], dtype=float)
        ay_limit = np.asarray([8.0, 8.0], dtype=float)
        limited = lateral_speed_limit_core(
            curvature_abs=curvature_abs,
            lateral_accel_limit=ay_limit,
            max_speed=40.0,
            ops=NUMPY_OPS,
        )
        self.assertAlmostEqual(float(limited[0]), 40.0, places=9)
        self.assertAlmostEqual(float(limited[1]), np.sqrt(8.0 / 0.25), places=9)

    def test_solve_speed_profile_core_supports_single_sample_track(self) -> None:
        """Run single-sample edge case and return finite degenerate outputs."""
        inputs = SpeedProfileCoreInputs(
            ds=np.asarray([], dtype=float),
            curvature=np.asarray([0.0], dtype=float),
            grade=np.asarray([0.0], dtype=float),
            banking=np.asarray([0.0], dtype=float),
            max_speed=30.0,
            min_speed=0.5,
            start_speed=5.0,
            lateral_envelope_max_iterations=6,
            lateral_envelope_convergence_tolerance=1e-10,
        )
        callbacks = SpeedProfileCoreCallbacks(
            lateral_accel_limit=lambda *, speed, banking: np.full_like(speed, 12.0),
            max_longitudinal_accel=lambda **_: 3.0,
            max_longitudinal_decel=lambda **_: 4.0,
        )
        result = solve_speed_profile_core(inputs=inputs, callbacks=callbacks, ops=NUMPY_OPS)
        self.assertEqual(result.speed.shape, (1,))
        self.assertEqual(result.longitudinal_accel.shape, (1,))
        self.assertEqual(float(result.lap_time), 0.0)


class TransientControlsCoreTests(unittest.TestCase):
    """Validate shared control-mesh helpers."""

    def test_build_control_node_count_and_mesh_positions(self) -> None:
        """Compute bounded node count and monotonic mesh positions."""
        self.assertEqual(build_control_node_count(sample_count=1, control_interval=5), 1)
        self.assertEqual(build_control_node_count(sample_count=10, control_interval=3), 4)
        mesh = build_control_mesh_positions(sample_count=10, control_interval=3)
        self.assertEqual(mesh.shape[0], 4)
        self.assertAlmostEqual(float(mesh[0]), 0.0, places=9)
        self.assertAlmostEqual(float(mesh[-1]), 9.0, places=9)

    def test_control_interpolation_map_handles_single_sample(self) -> None:
        """Build zeroed interpolation map for degenerate sample grids."""
        interpolation_map = build_control_interpolation_map(
            sample_count=1,
            mesh_positions=np.asarray([0.0], dtype=float),
        )
        self.assertTrue(
            np.array_equal(interpolation_map.left_index, np.asarray([0], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(interpolation_map.right_index, np.asarray([0], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(interpolation_map.right_weight, np.asarray([0.0], dtype=float))
        )

    def test_expand_mesh_controls_matches_mapless_path(self) -> None:
        """Expand mesh controls identically with and without cached map."""
        sample_count = 17
        mesh_positions = build_control_mesh_positions(sample_count=sample_count, control_interval=4)
        node_values = np.linspace(-1.0, 1.0, mesh_positions.size, dtype=float)
        interpolation_map = build_control_interpolation_map(
            sample_count=sample_count,
            mesh_positions=mesh_positions,
        )
        expanded_direct = expand_mesh_controls(
            node_values=node_values,
            sample_count=sample_count,
            mesh_positions=mesh_positions,
        )
        expanded_map = expand_mesh_controls(
            node_values=node_values,
            sample_count=sample_count,
            mesh_positions=mesh_positions,
            interpolation_map=interpolation_map,
        )
        self.assertTrue(np.allclose(expanded_direct, expanded_map, atol=1e-12, rtol=0.0))

    def test_sample_seed_and_bounded_artanh_are_finite(self) -> None:
        """Sample seed on mesh and keep bounded inverse tanh finite."""
        seed = np.linspace(-0.95, 0.95, 11, dtype=float)
        mesh_positions = np.asarray([0.0, 3.2, 6.6, 10.0], dtype=float)
        sampled = sample_seed_on_mesh(seed=seed, mesh_positions=mesh_positions)
        self.assertEqual(sampled.shape[0], mesh_positions.shape[0])
        transformed = bounded_artanh(np.asarray([-1.0, -0.3, 0.0, 0.7, 1.0], dtype=float))
        self.assertTrue(np.all(np.isfinite(transformed)))


class TransientPidCoreTests(unittest.TestCase):
    """Validate shared transient PID helper primitives."""

    def test_pid_scalar_and_generic_helpers(self) -> None:
        """Clamp integrals and form bounded PID command deterministically."""
        self.assertEqual(resolve_initial_speed(max_speed=20.0, initial_speed=None), 20.0)
        self.assertEqual(resolve_initial_speed(max_speed=20.0, initial_speed=0.0), 0.0)
        self.assertAlmostEqual(clamp_integral_scalar(3.0, 1.5), 1.5, places=9)
        self.assertAlmostEqual(clamp_integral_scalar(-3.0, 1.5), -1.5, places=9)

        clamped = clamp_integral_value(
            value=4.0,
            limit=1.25,
            clamp_fn=np.clip,
            as_value_fn=float,
        )
        self.assertAlmostEqual(float(clamped), 1.25, places=9)

        derivative = pid_error_derivative(
            error=1.0,
            previous_error=0.5,
            dt=0.0,
            denominator_floor=1e-3,
            clamp_min_fn=np.maximum,
            as_value_fn=float,
        )
        self.assertAlmostEqual(float(derivative), 500.0, places=6)

        command = bounded_pid_command(
            reference_feedforward=0.2,
            kp=2.0,
            ki=0.5,
            kd=0.1,
            error=0.4,
            error_integral=0.2,
            error_derivative=0.3,
            lower_limit=-0.5,
            upper_limit=0.6,
            clamp_fn=np.clip,
        )
        self.assertAlmostEqual(float(command), 0.6, places=9)

    def test_segment_time_partition_respects_substep_limit(self) -> None:
        """Split segments into bounded substeps with optional hard cap."""
        total, dt, substeps = segment_time_partition(
            segment_length=12.0,
            speed=6.0,
            min_time_step=0.02,
            max_time_step=0.6,
            max_integration_step=0.15,
        )
        self.assertAlmostEqual(total, 0.6, places=9)
        self.assertEqual(substeps, 4)
        self.assertAlmostEqual(dt, 0.15, places=9)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch backend not available")
    def test_segment_time_partition_torch_matches_scalar_partition(self) -> None:
        """Keep torch partitioning consistent with scalar helper behavior."""
        import torch

        total, dt, substeps = segment_time_partition_torch(
            torch=torch,
            segment_length=torch.tensor(12.0, dtype=torch.float64),
            speed=torch.tensor(6.0, dtype=torch.float64),
            min_time_step=0.02,
            max_time_step=0.6,
            max_integration_step=0.15,
        )
        self.assertEqual(substeps, 4)
        self.assertAlmostEqual(float(total.detach().cpu().item()), 0.6, places=9)
        self.assertAlmostEqual(float(dt.detach().cpu().item()), 0.15, places=9)
