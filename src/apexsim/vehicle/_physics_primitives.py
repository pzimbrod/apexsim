"""Internal physics primitives shared across vehicle model backends."""

from __future__ import annotations

from dataclasses import dataclass

from apexsim.utils.exceptions import ConfigurationError


@dataclass(frozen=True)
class EnvelopePhysics:
    """Shared longitudinal envelope limits used by lap-time vehicle models.

    Args:
        max_drive_accel: Maximum forward tire acceleration on flat road and zero
            lateral demand, excluding drag and grade [m/s^2].
        max_brake_accel: Maximum braking deceleration magnitude on flat road and
            zero lateral demand, excluding drag and grade [m/s^2].
        reference_mass: Optional reference mass [kg] used to interpret
            acceleration limits as force-equivalent capabilities. If set,
            effective limits are scaled by ``reference_mass / vehicle.mass``.
    """

    max_drive_accel: float
    max_brake_accel: float
    reference_mass: float | None = None

    def validate(self) -> None:
        """Validate shared longitudinal envelope limits.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If limits are not
                strictly positive.
        """
        if self.max_drive_accel <= 0.0:
            msg = "max_drive_accel must be positive"
            raise ConfigurationError(msg)
        if self.max_brake_accel <= 0.0:
            msg = "max_brake_accel must be positive"
            raise ConfigurationError(msg)
        if self.reference_mass is not None and self.reference_mass <= 0.0:
            msg = "reference_mass must be positive when provided"
            raise ConfigurationError(msg)
