"""Simulation configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

from lap_time_sim.utils.exceptions import ConfigurationError

DEFAULT_MIN_SPEED_MPS = 8.0
DEFAULT_LATERAL_ENVELOPE_MAX_ITERATIONS = 20
DEFAULT_LATERAL_ENVELOPE_CONVERGENCE_TOL_MPS = 0.1
DEFAULT_TRANSIENT_DT_S = 0.01
DEFAULT_MAX_SPEED_MPS = 115.0
DEFAULT_ENABLE_TRANSIENT_REFINEMENT = False


@dataclass(frozen=True)
class NumericsConfig:
    """Numerical controls for the lap-time solver.

    Attributes:
        min_speed_mps: Numerical floor for speed to avoid singular divisions.
        lateral_envelope_max_iterations: Maximum fixed-point iterations for the
            lateral-speed envelope solver.
        lateral_envelope_convergence_tol_mps: Early-stop threshold for the
            lateral envelope fixed-point update (`max |v_k - v_{k-1}|`).
        transient_dt_s: Integration step for optional transient refinement.
    """

    min_speed_mps: float = DEFAULT_MIN_SPEED_MPS
    lateral_envelope_max_iterations: int = DEFAULT_LATERAL_ENVELOPE_MAX_ITERATIONS
    lateral_envelope_convergence_tol_mps: float = DEFAULT_LATERAL_ENVELOPE_CONVERGENCE_TOL_MPS
    transient_dt_s: float = DEFAULT_TRANSIENT_DT_S

    def validate(self) -> None:
        """Validate numerical solver settings.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If any solver
            configuration value violates its bound.
        """
        if self.min_speed_mps <= 0.0:
            msg = "min_speed_mps must be positive"
            raise ConfigurationError(msg)
        if self.lateral_envelope_max_iterations < 1:
            msg = "lateral_envelope_max_iterations must be at least 1"
            raise ConfigurationError(msg)
        if self.lateral_envelope_convergence_tol_mps <= 0.0:
            msg = "lateral_envelope_convergence_tol_mps must be positive"
            raise ConfigurationError(msg)
        if self.transient_dt_s <= 0.0:
            msg = "transient_dt_s must be positive"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime controls that define simulation scenario boundaries.

    Attributes:
        max_speed_mps: Hard speed cap used by the quasi-steady profile solver.
        enable_transient_refinement: Flag for optional second-pass transient solve.
    """

    max_speed_mps: float
    enable_transient_refinement: bool = DEFAULT_ENABLE_TRANSIENT_REFINEMENT

    def validate(self, numerics: NumericsConfig) -> None:
        """Validate runtime controls against solver numerics.

        Args:
            numerics: Numerical parameter set used by the solver.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If runtime controls
                are inconsistent with solver numerics.
        """
        if self.max_speed_mps <= numerics.min_speed_mps:
            msg = "max_speed_mps must be greater than numerics.min_speed_mps"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level solver config composed of runtime and numerics.

    Attributes:
        runtime: Scenario and runtime controls, independent of physical car data.
        numerics: Discretization and convergence controls for numerical solving.
    """

    runtime: RuntimeConfig
    numerics: NumericsConfig

    def validate(self) -> None:
        """Validate combined simulation settings.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If runtime or
                numerical configuration values violate their bounds.
        """
        self.numerics.validate()
        self.runtime.validate(self.numerics)


def build_simulation_config(
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
    numerics: NumericsConfig | None = None,
    enable_transient_refinement: bool = DEFAULT_ENABLE_TRANSIENT_REFINEMENT,
) -> SimulationConfig:
    """Build a validated simulation config with sensible numerical defaults.

    Args:
        max_speed_mps: Runtime speed cap for the quasi-steady profile solver.
        numerics: Optional numerical settings. Defaults to :class:`NumericsConfig`.
        enable_transient_refinement: Flag for optional transient post-processing.

    Returns:
        Fully validated simulation configuration.

    Raises:
        lap_time_sim.utils.exceptions.ConfigurationError: If assembled runtime or
            numerical settings are inconsistent.
    """
    config = SimulationConfig(
        runtime=RuntimeConfig(
            max_speed_mps=max_speed_mps,
            enable_transient_refinement=enable_transient_refinement,
        ),
        numerics=numerics or NumericsConfig(),
    )
    config.validate()
    return config
