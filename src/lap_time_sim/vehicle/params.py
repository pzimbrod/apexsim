"""Vehicle parameter definitions for the 3-DOF bicycle model."""

from __future__ import annotations

from dataclasses import dataclass

from lap_time_sim.utils.exceptions import ConfigurationError

MIN_STATIC_FRONT_WEIGHT_FRACTION = 0.05
MAX_STATIC_FRONT_WEIGHT_FRACTION = 0.95


@dataclass(frozen=True)
class VehicleParameters:
    """Vehicle and chassis parameters for lap-time simulation."""

    mass_kg: float
    yaw_inertia_kgm2: float
    h_cg_m: float
    wheelbase_m: float
    track_front_m: float
    track_rear_m: float
    static_front_weight_fraction: float
    cop_position_m: float
    c_l: float
    c_d: float
    frontal_area_m2: float
    roll_rate_nm_per_deg: float
    spring_rate_front_npm: float
    spring_rate_rear_npm: float
    arb_distribution_front: float
    ride_height_front_m: float
    ride_height_rear_m: float
    air_density_kgpm3: float

    @property
    def cg_to_rear_axle_m(self) -> float:
        """Distance from center of gravity to rear axle.

        Returns:
            Rear axle distance from center of gravity in meters.
        """
        return self.static_front_weight_fraction * self.wheelbase_m

    @property
    def cg_to_front_axle_m(self) -> float:
        """Distance from center of gravity to front axle.

        Returns:
            Front axle distance from center of gravity in meters.
        """
        return self.wheelbase_m - self.cg_to_rear_axle_m

    def validate(self) -> None:
        """Validate configuration values before simulation.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If any parameter
                violates its defined bound.
        """
        if self.mass_kg <= 0.0:
            msg = "mass_kg must be positive"
            raise ConfigurationError(msg)
        if self.yaw_inertia_kgm2 <= 0.0:
            msg = "yaw_inertia_kgm2 must be positive"
            raise ConfigurationError(msg)
        if self.wheelbase_m <= 0.0:
            msg = "wheelbase_m must be positive"
            raise ConfigurationError(msg)
        if self.track_front_m <= 0.0 or self.track_rear_m <= 0.0:
            msg = "track widths must be positive"
            raise ConfigurationError(msg)
        if not (
            MIN_STATIC_FRONT_WEIGHT_FRACTION
            < self.static_front_weight_fraction
            < MAX_STATIC_FRONT_WEIGHT_FRACTION
        ):
            msg = (
                "static_front_weight_fraction must be between "
                f"{MIN_STATIC_FRONT_WEIGHT_FRACTION} and {MAX_STATIC_FRONT_WEIGHT_FRACTION}"
            )
            raise ConfigurationError(msg)
        if self.frontal_area_m2 <= 0.0:
            msg = "frontal_area_m2 must be positive"
            raise ConfigurationError(msg)
        if not 0.0 <= self.arb_distribution_front <= 1.0:
            msg = "arb_distribution_front must be between 0 and 1"
            raise ConfigurationError(msg)
        if self.air_density_kgpm3 <= 0.0:
            msg = "air_density_kgpm3 must be positive"
            raise ConfigurationError(msg)
