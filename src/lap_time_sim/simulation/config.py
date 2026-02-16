"""Simulation configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

from lap_time_sim.utils.exceptions import ConfigurationError


@dataclass(frozen=True)
class SimulationNumerics:
    """Numerical controls for the lap-time solver.

    Attributes:
        min_speed_mps: Numerical floor for speed to avoid singular divisions.
        lateral_envelope_max_iterations: Maximum fixed-point iterations for the
            lateral-speed envelope solver.
        lateral_envelope_convergence_tol_mps: Early-stop threshold for the
            lateral envelope fixed-point update (`max |v_k - v_{k-1}|`).
        transient_dt_s: Integration step for optional transient refinement.
    """

    min_speed_mps: float = 8.0
    lateral_envelope_max_iterations: int = 20
    lateral_envelope_convergence_tol_mps: float = 0.1
    transient_dt_s: float = 0.01

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
class SimulationRuntime:
    """Runtime controls that define simulation scenario boundaries.

    Attributes:
        max_speed_mps: Hard speed cap used by the quasi-steady profile solver.
        enable_transient_refinement: Flag for optional second-pass transient solve.
    """

    max_speed_mps: float = 115.0
    enable_transient_refinement: bool = False

    def validate(self, numerics: SimulationNumerics) -> None:
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

    runtime: SimulationRuntime = field(default_factory=SimulationRuntime)
    numerics: SimulationNumerics = field(default_factory=SimulationNumerics)

    def validate(self) -> None:
        """Validate combined simulation settings.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If runtime or
                numerical configuration values violate their bounds.
        """
        self.numerics.validate()
        self.runtime.validate(self.numerics)
