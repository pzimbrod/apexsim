"""Vehicle parameter definitions for the 3-DOF single_track model."""

from __future__ import annotations

from dataclasses import dataclass

from apexsim.utils.exceptions import ConfigurationError

MIN_STATIC_FRONT_WEIGHT_FRACTION = 0.05
MAX_STATIC_FRONT_WEIGHT_FRACTION = 0.95


@dataclass(frozen=True)
class VehicleParameters:
    """Vehicle and chassis parameters for lap-time simulation.

    Args:
        mass: Vehicle mass [kg].
        yaw_inertia: Yaw moment of inertia [kg*m^2].
        cg_height: Center-of-gravity height above ground [m].
        wheelbase: Wheelbase [m].
        front_track: Front track width [m].
        rear_track: Rear track width [m].
        front_weight_fraction: Static front axle weight fraction in [0, 1].
        cop_position: Center-of-pressure position relative to CoG [m].
        lift_coefficient: Aerodynamic lift/downforce coefficient.
        drag_coefficient: Aerodynamic drag coefficient.
        frontal_area: Frontal reference area [m^2].
        roll_rate: Roll rate [N*m/deg].
        front_spring_rate: Front spring rate [N/m].
        rear_spring_rate: Rear spring rate [N/m].
        front_arb_distribution: Front anti-roll-bar distribution in [0, 1].
        front_ride_height: Front ride height [m].
        rear_ride_height: Rear ride height [m].
        air_density: Air density [kg/m^3].
    """

    mass: float
    yaw_inertia: float
    cg_height: float
    wheelbase: float
    front_track: float
    rear_track: float
    front_weight_fraction: float
    cop_position: float
    lift_coefficient: float
    drag_coefficient: float
    frontal_area: float
    roll_rate: float
    front_spring_rate: float
    rear_spring_rate: float
    front_arb_distribution: float
    front_ride_height: float
    rear_ride_height: float
    air_density: float

    @property
    def cg_to_rear_axle(self) -> float:
        """Distance from center of gravity to rear axle.

        Returns:
            Rear axle distance from center of gravity [m].
        """
        return self.front_weight_fraction * self.wheelbase

    @property
    def cg_to_front_axle(self) -> float:
        """Distance from center of gravity to front axle.

        Returns:
            Front axle distance from center of gravity [m].
        """
        return self.wheelbase - self.cg_to_rear_axle

    def validate(self) -> None:
        """Validate configuration values before simulation.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If any parameter
                violates its defined bound.
        """
        if self.mass <= 0.0:
            msg = "mass must be positive"
            raise ConfigurationError(msg)
        if self.yaw_inertia <= 0.0:
            msg = "yaw_inertia must be positive"
            raise ConfigurationError(msg)
        if self.wheelbase <= 0.0:
            msg = "wheelbase must be positive"
            raise ConfigurationError(msg)
        if self.front_track <= 0.0 or self.rear_track <= 0.0:
            msg = "track widths must be positive"
            raise ConfigurationError(msg)
        if not (
            MIN_STATIC_FRONT_WEIGHT_FRACTION
            < self.front_weight_fraction
            < MAX_STATIC_FRONT_WEIGHT_FRACTION
        ):
            msg = (
                "front_weight_fraction must be between "
                f"{MIN_STATIC_FRONT_WEIGHT_FRACTION} and {MAX_STATIC_FRONT_WEIGHT_FRACTION}"
            )
            raise ConfigurationError(msg)
        if self.frontal_area <= 0.0:
            msg = "frontal_area must be positive"
            raise ConfigurationError(msg)
        if not 0.0 <= self.front_arb_distribution <= 1.0:
            msg = "front_arb_distribution must be between 0 and 1"
            raise ConfigurationError(msg)
        if self.air_density <= 0.0:
            msg = "air_density must be positive"
            raise ConfigurationError(msg)
