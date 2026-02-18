"""Track data models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apexsim.utils.exceptions import TrackDataError

MIN_TRACK_POINT_COUNT = 4


@dataclass(frozen=True)
class TrackData:
    """Processed track representation in arc-length domain.

    Args:
        x: Global x-coordinate samples along the centerline [m].
        y: Global y-coordinate samples along the centerline [m].
        elevation: Elevation samples along the centerline [m].
        banking: Banking angle samples [rad].
        arc_length: Monotonic arc-length coordinate [m].
        heading: Centerline heading angle [rad].
        curvature: Signed curvature along arc length [1/m].
        grade: Longitudinal grade ``dz/ds`` (-).
    """

    x: np.ndarray
    y: np.ndarray
    elevation: np.ndarray
    banking: np.ndarray
    arc_length: np.ndarray
    heading: np.ndarray
    curvature: np.ndarray
    grade: np.ndarray

    @property
    def length(self) -> float:
        """Track length [m].

        Returns:
            Final arc-length value of the discretized track [m].
        """
        return float(self.arc_length[-1])

    def validate(self) -> None:
        """Validate consistency of all track arrays.

        Raises:
            apexsim.utils.exceptions.TrackDataError: If array lengths, arc
                length monotonicity, or numeric validity checks fail.
        """
        arrays = [
            self.x,
            self.y,
            self.elevation,
            self.banking,
            self.arc_length,
            self.heading,
            self.curvature,
            self.grade,
        ]
        size = arrays[0].size
        if size < MIN_TRACK_POINT_COUNT:
            msg = f"Track must contain at least {MIN_TRACK_POINT_COUNT} points"
            raise TrackDataError(msg)
        if any(arr.size != size for arr in arrays):
            msg = "All track arrays must have equal length"
            raise TrackDataError(msg)
        if np.any(~np.isfinite(self.arc_length)):
            msg = "Arc-length array contains non-finite values"
            raise TrackDataError(msg)
        if not np.all(np.diff(self.arc_length) > 0.0):
            msg = "Arc length must be strictly increasing"
            raise TrackDataError(msg)
