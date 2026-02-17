"""Track data models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lap_time_sim.utils.exceptions import TrackDataError

MIN_TRACK_POINT_COUNT = 4


@dataclass(frozen=True)
class TrackData:
    """Processed track representation in arc-length domain.

    Args:
        x_m: Global x-coordinate samples along the centerline (m).
        y_m: Global y-coordinate samples along the centerline (m).
        elevation_m: Elevation samples along the centerline (m).
        banking_rad: Banking angle samples (rad).
        s_m: Monotonic arc-length coordinate (m).
        heading_rad: Centerline heading angle (rad).
        curvature_1pm: Signed curvature along arc length (1/m).
        grade: Longitudinal grade ``dz/ds`` (-).
    """

    x_m: np.ndarray
    y_m: np.ndarray
    elevation_m: np.ndarray
    banking_rad: np.ndarray
    s_m: np.ndarray
    heading_rad: np.ndarray
    curvature_1pm: np.ndarray
    grade: np.ndarray

    @property
    def length_m(self) -> float:
        """Track length in meters.

        Returns:
            Final arc-length value of the discretized track in meters.
        """
        return float(self.s_m[-1])

    def validate(self) -> None:
        """Validate consistency of all track arrays.

        Raises:
            lap_time_sim.utils.exceptions.TrackDataError: If array lengths, arc
                length monotonicity, or numeric validity checks fail.
        """
        arrays = [
            self.x_m,
            self.y_m,
            self.elevation_m,
            self.banking_rad,
            self.s_m,
            self.heading_rad,
            self.curvature_1pm,
            self.grade,
        ]
        size = arrays[0].size
        if size < MIN_TRACK_POINT_COUNT:
            msg = f"Track must contain at least {MIN_TRACK_POINT_COUNT} points"
            raise TrackDataError(msg)
        if any(arr.size != size for arr in arrays):
            msg = "All track arrays must have equal length"
            raise TrackDataError(msg)
        if np.any(~np.isfinite(self.s_m)):
            msg = "Arc-length array contains non-finite values"
            raise TrackDataError(msg)
        if not np.all(np.diff(self.s_m) > 0.0):
            msg = "Arc length must be strictly increasing"
            raise TrackDataError(msg)
