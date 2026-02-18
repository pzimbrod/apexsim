"""Track geometry processing utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from apexsim.track.models import TrackData
from apexsim.utils.constants import SMALL_EPS

FloatArray = npt.NDArray[np.float64]


def cumulative_arc_length(x: FloatArray, y: FloatArray) -> FloatArray:
    """Compute cumulative arc length from Cartesian points.

    Args:
        x: Track x-coordinate samples [m].
        y: Track y-coordinate samples [m].

    Returns:
        Cumulative arc-length samples [m].
    """
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    s = np.zeros(x.shape[0], dtype=np.float64)
    s[1:] = np.cumsum(ds)
    return np.asarray(s, dtype=np.float64)


def heading_from_xy(x: FloatArray, y: FloatArray) -> FloatArray:
    """Compute heading angle along the track.

    Args:
        x: Track x-coordinate samples [m].
        y: Track y-coordinate samples [m].

    Returns:
        Heading angle samples [rad].
    """
    dx = np.gradient(x)
    dy = np.gradient(y)
    return np.asarray(np.arctan2(dy, dx), dtype=np.float64)


def curvature_from_xy(x: FloatArray, y: FloatArray, arc_length: FloatArray) -> FloatArray:
    """Compute signed curvature from first and second derivatives.

    Args:
        x: Track x-coordinate samples [m].
        y: Track y-coordinate samples [m].
        arc_length: Monotonic arc-length samples [m].

    Returns:
        Signed curvature samples [1/m].
    """
    dx_ds = np.gradient(x, arc_length)
    dy_ds = np.gradient(y, arc_length)
    d2x_ds2 = np.gradient(dx_ds, arc_length)
    d2y_ds2 = np.gradient(dy_ds, arc_length)

    denominator = np.power(dx_ds * dx_ds + dy_ds * dy_ds, 1.5)
    denominator = np.maximum(denominator, SMALL_EPS)
    numerator = dx_ds * d2y_ds2 - dy_ds * d2x_ds2
    return np.asarray(numerator / denominator, dtype=np.float64)


def grade_from_elevation(elevation: FloatArray, arc_length: FloatArray) -> FloatArray:
    """Compute slope ``dz/ds`` along the track.

    Args:
        elevation: Elevation profile [m].
        arc_length: Monotonic arc-length samples [m].

    Returns:
        Dimensionless grade values.
    """
    return np.asarray(np.gradient(elevation, arc_length), dtype=np.float64)


def build_track_data(
    x: FloatArray,
    y: FloatArray,
    elevation: FloatArray,
    banking: FloatArray,
) -> TrackData:
    """Build a complete ``TrackData`` object from raw coordinate arrays.

    Args:
        x: Track x-coordinate samples [m].
        y: Track y-coordinate samples [m].
        elevation: Elevation profile [m].
        banking: Banking profile [rad].

    Returns:
        Validated track data in arc-length domain.

    Raises:
        apexsim.utils.exceptions.TrackDataError: If generated track arrays
            are inconsistent or numerically invalid.
    """
    arc_length = cumulative_arc_length(x, y)
    heading = heading_from_xy(x, y)
    curvature = curvature_from_xy(x, y, arc_length)
    grade = grade_from_elevation(elevation, arc_length)

    data = TrackData(
        x=x,
        y=y,
        elevation=elevation,
        banking=banking,
        arc_length=arc_length,
        heading=heading,
        curvature=curvature,
        grade=grade,
    )
    data.validate()
    return data
