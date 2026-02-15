"""Track geometry processing utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from lap_time_sim.track.models import TrackData
from lap_time_sim.utils.constants import SMALL_EPS

FloatArray = npt.NDArray[np.float64]


def cumulative_arc_length(x_m: FloatArray, y_m: FloatArray) -> FloatArray:
    """Compute cumulative arc length from Cartesian points."""
    dx = np.diff(x_m)
    dy = np.diff(y_m)
    ds = np.hypot(dx, dy)
    s = np.zeros(x_m.shape[0], dtype=np.float64)
    s[1:] = np.cumsum(ds)
    return np.asarray(s, dtype=np.float64)


def heading_from_xy(x_m: FloatArray, y_m: FloatArray) -> FloatArray:
    """Compute heading angle along the track."""
    dx = np.gradient(x_m)
    dy = np.gradient(y_m)
    return np.asarray(np.arctan2(dy, dx), dtype=np.float64)


def curvature_from_xy(x_m: FloatArray, y_m: FloatArray, s_m: FloatArray) -> FloatArray:
    """Compute signed curvature from first and second derivatives."""
    dx_ds = np.gradient(x_m, s_m)
    dy_ds = np.gradient(y_m, s_m)
    d2x_ds2 = np.gradient(dx_ds, s_m)
    d2y_ds2 = np.gradient(dy_ds, s_m)

    denominator = np.power(dx_ds * dx_ds + dy_ds * dy_ds, 1.5)
    denominator = np.maximum(denominator, SMALL_EPS)
    numerator = dx_ds * d2y_ds2 - dy_ds * d2x_ds2
    return np.asarray(numerator / denominator, dtype=np.float64)


def grade_from_elevation(elevation_m: FloatArray, s_m: FloatArray) -> FloatArray:
    """Compute slope dz/ds along the track."""
    return np.asarray(np.gradient(elevation_m, s_m), dtype=np.float64)


def build_track_data(
    x_m: FloatArray,
    y_m: FloatArray,
    elevation_m: FloatArray,
    banking_rad: FloatArray,
) -> TrackData:
    """Build a complete `TrackData` object from raw coordinate arrays."""
    s_m = cumulative_arc_length(x_m, y_m)
    heading_rad = heading_from_xy(x_m, y_m)
    curvature_1pm = curvature_from_xy(x_m, y_m, s_m)
    grade = grade_from_elevation(elevation_m, s_m)

    data = TrackData(
        x_m=x_m,
        y_m=y_m,
        elevation_m=elevation_m,
        banking_rad=banking_rad,
        s_m=s_m,
        heading_rad=heading_rad,
        curvature_1pm=curvature_1pm,
        grade=grade,
    )
    data.validate()
    return data
