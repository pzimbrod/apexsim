"""Simulation configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

from lap_time_sim.utils.exceptions import ConfigurationError

DEFAULT_MIN_SPEED = 8.0
DEFAULT_LATERAL_ENVELOPE_MAX_ITERATIONS = 20
DEFAULT_LATERAL_ENVELOPE_CONVERGENCE_TOLERANCE = 0.1
DEFAULT_TRANSIENT_STEP = 0.01
DEFAULT_MAX_SPEED = 115.0
DEFAULT_ENABLE_TRANSIENT_REFINEMENT = False


@dataclass(frozen=True)
class NumericsConfig:
    """Numerical controls for the lap-time solver.

    Args:
        min_speed: Numerical floor for speed to avoid singular divisions [m/s].
        lateral_envelope_max_iterations: Maximum fixed-point iterations for the
            lateral-speed envelope solver.
        lateral_envelope_convergence_tolerance: Early-stop threshold for the
            lateral envelope fixed-point update (`max |v_k - v_{k-1}|`) [m/s].
        transient_step: Integration step for optional transient refinement [s].
    """

    min_speed: float = DEFAULT_MIN_SPEED
    lateral_envelope_max_iterations: int = DEFAULT_LATERAL_ENVELOPE_MAX_ITERATIONS
    lateral_envelope_convergence_tolerance: float = DEFAULT_LATERAL_ENVELOPE_CONVERGENCE_TOLERANCE
    transient_step: float = DEFAULT_TRANSIENT_STEP

    def validate(self) -> None:
        """Validate numerical solver settings.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If any solver
            configuration value violates its bound.
        """
        if self.min_speed <= 0.0:
            msg = "min_speed must be positive"
            raise ConfigurationError(msg)
        if self.lateral_envelope_max_iterations < 1:
            msg = "lateral_envelope_max_iterations must be at least 1"
            raise ConfigurationError(msg)
        if self.lateral_envelope_convergence_tolerance <= 0.0:
            msg = "lateral_envelope_convergence_tolerance must be positive"
            raise ConfigurationError(msg)
        if self.transient_step <= 0.0:
            msg = "transient_step must be positive"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime controls that define simulation scenario boundaries.

    Args:
        max_speed: Hard speed cap used by the quasi-steady profile solver [m/s].
        enable_transient_refinement: Flag for optional second-pass transient solve.
    """

    max_speed: float
    enable_transient_refinement: bool = DEFAULT_ENABLE_TRANSIENT_REFINEMENT

    def validate(self, numerics: NumericsConfig) -> None:
        """Validate runtime controls against solver numerics.

        Args:
            numerics: Numerical parameter set used by the solver.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If runtime controls
                are inconsistent with solver numerics.
        """
        if self.max_speed <= numerics.min_speed:
            msg = "max_speed must be greater than numerics.min_speed"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level solver config composed of runtime and numerics.

    Args:
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
    max_speed: float = DEFAULT_MAX_SPEED,
    numerics: NumericsConfig | None = None,
    enable_transient_refinement: bool = DEFAULT_ENABLE_TRANSIENT_REFINEMENT,
) -> SimulationConfig:
    """Build a validated simulation config with sensible numerical defaults.

    Args:
        max_speed: Runtime speed cap for the quasi-steady profile solver [m/s].
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
            max_speed=max_speed,
            enable_transient_refinement=enable_transient_refinement,
        ),
        numerics=numerics or NumericsConfig(),
    )
    config.validate()
    return config
