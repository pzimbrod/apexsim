"""Dataclasses for tire model parameters."""

from __future__ import annotations

from dataclasses import dataclass

from lap_time_sim.utils.exceptions import ConfigurationError


@dataclass(frozen=True)
class PacejkaParameters:
    """Pacejka lateral model coefficients and load-sensitivity behavior.

    The lateral force model uses coefficients `B, C, D, E` in a Magic Formula style
    equation. `D` is treated as an effective friction coefficient at the current
    reference condition.

    Args:
        B: Pacejka stiffness factor (-).
        C: Pacejka shape factor (-).
        D: Peak friction-like factor at reference load (-).
        E: Pacejka curvature factor (-).
        reference_load: Reference normal load used by load sensitivity [N].
        load_sensitivity: Linear scaling of effective friction with load deviation
            from reference (-/N).
        min_mu_scale: Lower bound for load-scaled friction multiplier (-).
    """

    B: float
    C: float
    D: float
    E: float
    reference_load: float
    load_sensitivity: float
    min_mu_scale: float

    def validate(self) -> None:
        """Validate physical and numerical constraints for coefficients.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If any coefficient
                violates required bounds.
        """
        if self.B <= 0.0:
            msg = "Pacejka B must be positive"
            raise ConfigurationError(msg)
        if self.C <= 0.0:
            msg = "Pacejka C must be positive"
            raise ConfigurationError(msg)
        if self.D <= 0.0:
            msg = "Pacejka D must be positive"
            raise ConfigurationError(msg)
        if self.reference_load <= 0.0:
            msg = "Reference normal load must be positive"
            raise ConfigurationError(msg)
        if self.min_mu_scale <= 0.0:
            msg = "min_mu_scale must be positive"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class AxleTireParameters:
    """Separate tire coefficients for front and rear axle.

    Args:
        front: Front-axle Pacejka coefficients.
        rear: Rear-axle Pacejka coefficients.
    """

    front: PacejkaParameters
    rear: PacejkaParameters

    def validate(self) -> None:
        """Validate both axle parameter sets.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If front or rear
                coefficient sets are invalid.
        """
        self.front.validate()
        self.rear.validate()


def default_axle_tire_parameters() -> AxleTireParameters:
    """Create a default high-downforce race-car tire parameterization.

    Returns:
        Front and rear Pacejka parameter sets tuned for a high-downforce car.
    """
    return AxleTireParameters(
        front=PacejkaParameters(
            B=9.5,
            C=1.35,
            D=1.85,
            E=0.97,
            reference_load=3500.0,
            load_sensitivity=-0.08,
            min_mu_scale=0.4,
        ),
        rear=PacejkaParameters(
            B=10.2,
            C=1.33,
            D=1.95,
            E=0.95,
            reference_load=4200.0,
            load_sensitivity=-0.08,
            min_mu_scale=0.4,
        ),
    )
