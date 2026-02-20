"""Shared transient-lap-solver configuration and helper utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from apexsim.utils.constants import SMALL_EPS
from apexsim.utils.exceptions import ConfigurationError

DEFAULT_SOLVER_MODE = "quasi_static"
VALID_SOLVER_MODES = ("quasi_static", "transient_oc")

DEFAULT_TRANSIENT_INTEGRATION_METHOD = "rk4"
VALID_TRANSIENT_INTEGRATION_METHODS = ("euler", "rk4")

DEFAULT_TRANSIENT_ODE_BACKEND_POLICY = "auto"
VALID_TRANSIENT_ODE_BACKEND_POLICIES = ("auto", "external", "internal")

DEFAULT_TRANSIENT_OPTIMIZER_BACKEND_POLICY = "auto"
VALID_TRANSIENT_OPTIMIZER_BACKEND_POLICIES = ("auto", "external", "internal")

DEFAULT_TRANSIENT_DRIVER_MODEL = "pid"
VALID_TRANSIENT_DRIVER_MODELS = ("pid", "optimal_control")

DEFAULT_TRANSIENT_MAX_ITERATIONS = 80
DEFAULT_TRANSIENT_TOLERANCE = 1e-4
DEFAULT_TRANSIENT_MIN_TIME_STEP = 1e-4
DEFAULT_TRANSIENT_MAX_TIME_STEP = 0.2
DEFAULT_TRANSIENT_LATERAL_CONSTRAINT_WEIGHT = 100.0
DEFAULT_TRANSIENT_TRACKING_WEIGHT = 5.0
DEFAULT_TRANSIENT_CONTROL_SMOOTHNESS_WEIGHT = 1e-3
DEFAULT_TRANSIENT_OPTIMIZER_LEARNING_RATE = 0.2
DEFAULT_TRANSIENT_OPTIMIZER_LBFGS_MAX_ITER = 20
DEFAULT_TRANSIENT_OPTIMIZER_ADAM_STEPS = 120
DEFAULT_TRANSIENT_OPTIMIZER_VERBOSE = 0
DEFAULT_TRANSIENT_CONTROL_INTERVAL = 8
DEFAULT_TRANSIENT_PID_LONGITUDINAL_KP = 0.8
DEFAULT_TRANSIENT_PID_LONGITUDINAL_KI = 0.0
DEFAULT_TRANSIENT_PID_LONGITUDINAL_KD = 0.05
DEFAULT_TRANSIENT_PID_STEER_KP = 1.6
DEFAULT_TRANSIENT_PID_STEER_KI = 0.0
DEFAULT_TRANSIENT_PID_STEER_KD = 0.12
DEFAULT_TRANSIENT_PID_STEER_VY_DAMPING = 0.25
DEFAULT_TRANSIENT_PID_LONGITUDINAL_INTEGRAL_LIMIT = 30.0
DEFAULT_TRANSIENT_PID_STEER_INTEGRAL_LIMIT = 2.0
DEFAULT_TRANSIENT_PID_GAIN_SCHEDULING_MODE = "off"
VALID_TRANSIENT_PID_GAIN_SCHEDULING_MODES = ("off", "physics_informed", "custom")


@dataclass(frozen=True)
class PidSpeedSchedule:
    """Piecewise-linear speed-dependent gain schedule.

    Args:
        speed_nodes_mps: Monotonic speed nodes [m/s].
        values: Gain values at ``speed_nodes_mps``.
    """

    speed_nodes_mps: tuple[float, ...]
    values: tuple[float, ...]

    def validate(self) -> None:
        """Validate PWL node/value vectors.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If nodes or values are
                inconsistent.
        """
        if len(self.speed_nodes_mps) < 2:
            msg = "PidSpeedSchedule.speed_nodes_mps must contain at least two nodes"
            raise ConfigurationError(msg)
        if len(self.speed_nodes_mps) != len(self.values):
            msg = "PidSpeedSchedule.speed_nodes_mps and values must have the same length"
            raise ConfigurationError(msg)
        speeds = np.asarray(self.speed_nodes_mps, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if np.any(~np.isfinite(speeds)):
            msg = "PidSpeedSchedule.speed_nodes_mps must be finite"
            raise ConfigurationError(msg)
        if np.any(~np.isfinite(values)):
            msg = "PidSpeedSchedule.values must be finite"
            raise ConfigurationError(msg)
        if np.any(speeds < 0.0):
            msg = "PidSpeedSchedule.speed_nodes_mps must be greater than or equal to 0"
            raise ConfigurationError(msg)
        if np.any(np.diff(speeds) <= 0.0):
            msg = "PidSpeedSchedule.speed_nodes_mps must be strictly increasing"
            raise ConfigurationError(msg)

    def evaluate(self, speed_mps: float) -> float:
        """Evaluate schedule at one speed by linear interpolation with clamping.

        Args:
            speed_mps: Evaluation speed [m/s].

        Returns:
            Interpolated schedule value.
        """
        if not np.isfinite(speed_mps):
            msg = "speed_mps must be finite"
            raise ConfigurationError(msg)
        speeds = np.asarray(self.speed_nodes_mps, dtype=float)
        values = np.asarray(self.values, dtype=float)
        return float(np.interp(float(speed_mps), speeds, values))

    def evaluate_many(self, speeds_mps: np.ndarray) -> np.ndarray:
        """Evaluate schedule at multiple speeds.

        Args:
            speeds_mps: Evaluation speeds [m/s].

        Returns:
            Interpolated values for each input speed.
        """
        speeds_query = np.asarray(speeds_mps, dtype=float)
        if np.any(~np.isfinite(speeds_query)):
            msg = "speeds_mps must be finite"
            raise ConfigurationError(msg)
        speeds = np.asarray(self.speed_nodes_mps, dtype=float)
        values = np.asarray(self.values, dtype=float)
        return np.asarray(np.interp(speeds_query, speeds, values), dtype=float)


@dataclass(frozen=True)
class TransientPidGainSchedulingConfig:
    """Speed-dependent PID gain schedules.

    Longitudinal schedules are used by point-mass and single-track PID drivers.
    Steering schedules are used by single-track PID drivers.
    """

    longitudinal_kp: PidSpeedSchedule | None = None
    longitudinal_ki: PidSpeedSchedule | None = None
    longitudinal_kd: PidSpeedSchedule | None = None
    steer_kp: PidSpeedSchedule | None = None
    steer_ki: PidSpeedSchedule | None = None
    steer_kd: PidSpeedSchedule | None = None
    steer_vy_damping: PidSpeedSchedule | None = None

    def validate(self) -> None:
        """Validate scheduling config and node-grid consistency.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If schedules are
                inconsistent.
        """
        for name, schedule in (
            ("longitudinal_kp", self.longitudinal_kp),
            ("longitudinal_ki", self.longitudinal_ki),
            ("longitudinal_kd", self.longitudinal_kd),
            ("steer_kp", self.steer_kp),
            ("steer_ki", self.steer_ki),
            ("steer_kd", self.steer_kd),
            ("steer_vy_damping", self.steer_vy_damping),
        ):
            if schedule is None:
                continue
            try:
                schedule.validate()
            except ConfigurationError as exc:
                msg = f"Invalid schedule {name!r}: {exc}"
                raise ConfigurationError(msg) from exc

        self._validate_shared_grid(
            group_name="longitudinal",
            schedules=(
                self.longitudinal_kp,
                self.longitudinal_ki,
                self.longitudinal_kd,
            ),
        )
        self._validate_shared_grid(
            group_name="steering",
            schedules=(
                self.steer_kp,
                self.steer_ki,
                self.steer_kd,
                self.steer_vy_damping,
            ),
        )

    def has_longitudinal_schedules(self) -> bool:
        """Return whether all longitudinal schedules are present.

        Returns:
            ``True`` when longitudinal ``kp``, ``ki``, and ``kd`` schedules are
            all configured.
        """
        return (
            self.longitudinal_kp is not None
            and self.longitudinal_ki is not None
            and self.longitudinal_kd is not None
        )

    def has_steering_schedules(self) -> bool:
        """Return whether all steering schedules are present.

        Returns:
            ``True`` when steering ``kp``, ``ki``, ``kd``, and lateral-velocity
            damping schedules are all configured.
        """
        return (
            self.steer_kp is not None
            and self.steer_ki is not None
            and self.steer_kd is not None
            and self.steer_vy_damping is not None
        )

    def _validate_shared_grid(
        self,
        *,
        group_name: str,
        schedules: tuple[PidSpeedSchedule | None, ...],
    ) -> None:
        """Require equal speed grids within one schedule group.

        Args:
            group_name: Human-readable group identifier used in error messages.
            schedules: Schedule tuple belonging to one gain group.
        """
        present = [item for item in schedules if item is not None]
        if len(present) <= 1:
            return
        reference_nodes = present[0].speed_nodes_mps
        for candidate in present[1:]:
            if candidate.speed_nodes_mps != reference_nodes:
                msg = (
                    f"All {group_name} schedules must share identical speed_nodes_mps "
                    "for consistent interpolation."
                )
                raise ConfigurationError(msg)


@dataclass(frozen=True)
class TransientNumericsConfig:
    """Numerical controls for the transient OC lap-time solver.

    Args:
        integration_method: Integration method used in state propagation
            (``"euler"`` or ``"rk4"``).
        max_iterations: Maximum optimization iterations.
        tolerance: Optimization convergence tolerance.
        min_time_step: Minimum integration step used during arc-length to time
            conversion [s].
        max_time_step: Maximum integration step used during arc-length to time
            conversion [s].
        lateral_constraint_weight: Penalty weight for lateral feasibility
            violations.
        tracking_weight: Penalty weight for transient path-tracking terms.
        control_smoothness_weight: Penalty weight for control variation between
            neighboring samples.
        control_interval: Arc-length sample interval used for optimization
            controls before interpolation to full track resolution.
        optimizer_learning_rate: Initial optimizer learning rate.
        optimizer_lbfgs_max_iter: Maximum iterations for one LBFGS step.
        optimizer_adam_steps: Number of Adam fallback steps.
        pid_longitudinal_kp: Proportional gain for longitudinal speed tracking.
        pid_longitudinal_ki: Integral gain for longitudinal speed tracking.
        pid_longitudinal_kd: Derivative gain for longitudinal speed tracking.
        pid_steer_kp: Proportional gain for yaw-rate steering feedback.
        pid_steer_ki: Integral gain for yaw-rate steering feedback.
        pid_steer_kd: Derivative gain for yaw-rate steering feedback.
        pid_steer_vy_damping: Damping term on lateral velocity for steering
            stabilization.
        pid_longitudinal_integral_limit: Absolute clamp for longitudinal PID
            integrator state.
        pid_steer_integral_limit: Absolute clamp for steering PID integrator
            state.
        pid_gain_scheduling_mode: PID gain scheduling mode:
            ``"off"``, ``"physics_informed"``, or ``"custom"``.
        pid_gain_scheduling: Optional custom schedule table set used when
            ``pid_gain_scheduling_mode="custom"``.
    """

    integration_method: str = DEFAULT_TRANSIENT_INTEGRATION_METHOD
    max_iterations: int = DEFAULT_TRANSIENT_MAX_ITERATIONS
    tolerance: float = DEFAULT_TRANSIENT_TOLERANCE
    min_time_step: float = DEFAULT_TRANSIENT_MIN_TIME_STEP
    max_time_step: float = DEFAULT_TRANSIENT_MAX_TIME_STEP
    lateral_constraint_weight: float = DEFAULT_TRANSIENT_LATERAL_CONSTRAINT_WEIGHT
    tracking_weight: float = DEFAULT_TRANSIENT_TRACKING_WEIGHT
    control_smoothness_weight: float = DEFAULT_TRANSIENT_CONTROL_SMOOTHNESS_WEIGHT
    control_interval: int = DEFAULT_TRANSIENT_CONTROL_INTERVAL
    optimizer_learning_rate: float = DEFAULT_TRANSIENT_OPTIMIZER_LEARNING_RATE
    optimizer_lbfgs_max_iter: int = DEFAULT_TRANSIENT_OPTIMIZER_LBFGS_MAX_ITER
    optimizer_adam_steps: int = DEFAULT_TRANSIENT_OPTIMIZER_ADAM_STEPS
    pid_longitudinal_kp: float = DEFAULT_TRANSIENT_PID_LONGITUDINAL_KP
    pid_longitudinal_ki: float = DEFAULT_TRANSIENT_PID_LONGITUDINAL_KI
    pid_longitudinal_kd: float = DEFAULT_TRANSIENT_PID_LONGITUDINAL_KD
    pid_steer_kp: float = DEFAULT_TRANSIENT_PID_STEER_KP
    pid_steer_ki: float = DEFAULT_TRANSIENT_PID_STEER_KI
    pid_steer_kd: float = DEFAULT_TRANSIENT_PID_STEER_KD
    pid_steer_vy_damping: float = DEFAULT_TRANSIENT_PID_STEER_VY_DAMPING
    pid_longitudinal_integral_limit: float = (
        DEFAULT_TRANSIENT_PID_LONGITUDINAL_INTEGRAL_LIMIT
    )
    pid_steer_integral_limit: float = DEFAULT_TRANSIENT_PID_STEER_INTEGRAL_LIMIT
    pid_gain_scheduling_mode: str = DEFAULT_TRANSIENT_PID_GAIN_SCHEDULING_MODE
    pid_gain_scheduling: TransientPidGainSchedulingConfig | None = None

    def validate(self) -> None:
        """Validate transient numerical controls.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If numerical controls
                violate required bounds.
        """
        if self.integration_method not in VALID_TRANSIENT_INTEGRATION_METHODS:
            msg = (
                "integration_method must be one of "
                f"{VALID_TRANSIENT_INTEGRATION_METHODS}, got: {self.integration_method!r}"
            )
            raise ConfigurationError(msg)
        if self.max_iterations < 1:
            msg = "max_iterations must be at least 1"
            raise ConfigurationError(msg)
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            msg = "tolerance must be a positive finite value"
            raise ConfigurationError(msg)
        if not np.isfinite(self.min_time_step) or self.min_time_step <= 0.0:
            msg = "min_time_step must be a positive finite value"
            raise ConfigurationError(msg)
        if not np.isfinite(self.max_time_step) or self.max_time_step <= self.min_time_step:
            msg = "max_time_step must be finite and greater than min_time_step"
            raise ConfigurationError(msg)
        if (
            not np.isfinite(self.lateral_constraint_weight)
            or self.lateral_constraint_weight < 0.0
        ):
            msg = "lateral_constraint_weight must be a non-negative finite value"
            raise ConfigurationError(msg)
        if not np.isfinite(self.tracking_weight) or self.tracking_weight < 0.0:
            msg = "tracking_weight must be a non-negative finite value"
            raise ConfigurationError(msg)
        if (
            not np.isfinite(self.control_smoothness_weight)
            or self.control_smoothness_weight < 0.0
        ):
            msg = "control_smoothness_weight must be a non-negative finite value"
            raise ConfigurationError(msg)
        if self.control_interval < 1:
            msg = "control_interval must be at least 1"
            raise ConfigurationError(msg)
        if (
            not np.isfinite(self.optimizer_learning_rate)
            or self.optimizer_learning_rate <= 0.0
        ):
            msg = "optimizer_learning_rate must be a positive finite value"
            raise ConfigurationError(msg)
        if self.optimizer_lbfgs_max_iter < 1:
            msg = "optimizer_lbfgs_max_iter must be at least 1"
            raise ConfigurationError(msg)
        if self.optimizer_adam_steps < 1:
            msg = "optimizer_adam_steps must be at least 1"
            raise ConfigurationError(msg)
        for name, value in (
            ("pid_longitudinal_kp", self.pid_longitudinal_kp),
            ("pid_longitudinal_ki", self.pid_longitudinal_ki),
            ("pid_longitudinal_kd", self.pid_longitudinal_kd),
            ("pid_steer_kp", self.pid_steer_kp),
            ("pid_steer_ki", self.pid_steer_ki),
            ("pid_steer_kd", self.pid_steer_kd),
            ("pid_steer_vy_damping", self.pid_steer_vy_damping),
        ):
            if not np.isfinite(value):
                msg = f"{name} must be finite"
                raise ConfigurationError(msg)
        if (
            not np.isfinite(self.pid_longitudinal_integral_limit)
            or self.pid_longitudinal_integral_limit < 0.0
        ):
            msg = "pid_longitudinal_integral_limit must be a non-negative finite value"
            raise ConfigurationError(msg)
        if (
            not np.isfinite(self.pid_steer_integral_limit)
            or self.pid_steer_integral_limit < 0.0
        ):
            msg = "pid_steer_integral_limit must be a non-negative finite value"
            raise ConfigurationError(msg)
        if self.pid_gain_scheduling_mode not in VALID_TRANSIENT_PID_GAIN_SCHEDULING_MODES:
            msg = (
                "pid_gain_scheduling_mode must be one of "
                f"{VALID_TRANSIENT_PID_GAIN_SCHEDULING_MODES}, got: "
                f"{self.pid_gain_scheduling_mode!r}"
            )
            raise ConfigurationError(msg)
        if self.pid_gain_scheduling is not None:
            self.pid_gain_scheduling.validate()
        if self.pid_gain_scheduling_mode == "custom":
            if self.pid_gain_scheduling is None:
                msg = (
                    "pid_gain_scheduling must be provided when "
                    "pid_gain_scheduling_mode='custom'"
                )
                raise ConfigurationError(msg)
            if not self.pid_gain_scheduling.has_longitudinal_schedules():
                msg = (
                    "Custom PID gain scheduling requires longitudinal schedules "
                    "(kp/ki/kd)."
                )
                raise ConfigurationError(msg)


@dataclass(frozen=True)
class TransientRuntimeConfig:
    """Runtime controls for transient OC solve behavior.

    Args:
        ode_backend_policy: ODE-backend selection policy.
        optimizer_backend_policy: Optimizer-backend selection policy.
        driver_model: Driver/control strategy used inside the transient solver.
            ``"pid"`` applies deterministic closed-loop PID control.
            ``"optimal_control"`` runs full control optimization.
        deterministic_seed: Optional deterministic seed used by stochastic
            optimizer fallbacks.
        verbosity: Verbosity level for transient solver logging.
    """

    ode_backend_policy: str = DEFAULT_TRANSIENT_ODE_BACKEND_POLICY
    optimizer_backend_policy: str = DEFAULT_TRANSIENT_OPTIMIZER_BACKEND_POLICY
    driver_model: str = DEFAULT_TRANSIENT_DRIVER_MODEL
    deterministic_seed: int | None = None
    verbosity: int = DEFAULT_TRANSIENT_OPTIMIZER_VERBOSE

    def validate(self) -> None:
        """Validate transient runtime controls.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If runtime controls
                violate required bounds.
        """
        if self.ode_backend_policy not in VALID_TRANSIENT_ODE_BACKEND_POLICIES:
            msg = (
                "ode_backend_policy must be one of "
                f"{VALID_TRANSIENT_ODE_BACKEND_POLICIES}, got: {self.ode_backend_policy!r}"
            )
            raise ConfigurationError(msg)
        if self.optimizer_backend_policy not in VALID_TRANSIENT_OPTIMIZER_BACKEND_POLICIES:
            msg = (
                "optimizer_backend_policy must be one of "
                f"{VALID_TRANSIENT_OPTIMIZER_BACKEND_POLICIES}, got: "
                f"{self.optimizer_backend_policy!r}"
            )
            raise ConfigurationError(msg)
        if self.driver_model not in VALID_TRANSIENT_DRIVER_MODELS:
            msg = (
                "driver_model must be one of "
                f"{VALID_TRANSIENT_DRIVER_MODELS}, got: {self.driver_model!r}"
            )
            raise ConfigurationError(msg)
        if self.deterministic_seed is not None and self.deterministic_seed < 0:
            msg = "deterministic_seed must be greater than or equal to 0 when provided"
            raise ConfigurationError(msg)
        if self.verbosity < 0:
            msg = "verbosity must be greater than or equal to 0"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class TransientConfig:
    """Top-level transient OC configuration container.

    Args:
        numerics: Transient numerical controls.
        runtime: Transient runtime controls.
    """

    numerics: TransientNumericsConfig = field(default_factory=TransientNumericsConfig)
    runtime: TransientRuntimeConfig = field(default_factory=TransientRuntimeConfig)

    def validate(self) -> None:
        """Validate transient configuration.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If transient settings
                are inconsistent.
        """
        self.numerics.validate()
        self.runtime.validate()


@dataclass(frozen=True)
class TransientProfileResult:
    """Transient speed-profile and state traces in arc-length sample order.

    Args:
        speed: Total speed trace [m/s].
        longitudinal_accel: Net longitudinal acceleration trace [m/s^2].
        lateral_accel: Lateral acceleration trace [m/s^2].
        lap_time: Integrated lap time [s].
        time: Cumulative time trace [s].
        vx: Body-frame longitudinal speed trace [m/s].
        vy: Body-frame lateral speed trace [m/s].
        yaw_rate: Yaw-rate trace [rad/s].
        steer_cmd: Steering command trace [rad].
        ax_cmd: Longitudinal acceleration command trace [m/s^2].
        objective_value: Final OC objective value.
    """

    speed: np.ndarray
    longitudinal_accel: np.ndarray
    lateral_accel: np.ndarray
    lap_time: float
    time: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    yaw_rate: np.ndarray
    steer_cmd: np.ndarray
    ax_cmd: np.ndarray
    objective_value: float


def segment_time_step(
    *,
    segment_length: float,
    speed: float,
    min_time_step: float,
    max_time_step: float,
) -> float:
    """Convert one arc-length step to total segment travel time.

    Args:
        segment_length: Arc-length step [m].
        speed: Positive speed [m/s].
        min_time_step: Lower segment travel-time bound [s].
        max_time_step: Deprecated upper integration-step bound [s].

    Returns:
        Segment travel time [s].
    """
    del max_time_step
    raw_step = segment_length / max(abs(speed), SMALL_EPS)
    return float(max(raw_step, min_time_step))
