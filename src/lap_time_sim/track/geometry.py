"""Track geometry processing utilities."""

from __future__ import annotations

import numpy as np

from lap_time_sim.track.models import TrackData
from lap_time_sim.utils.constants import SMALL_EPS


def cumulative_arc_length(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """Compute cumulative arc length from Cartesian points."""
    dx = np.diff(x_m)
    dy = np.diff(y_m)
    ds = np.hypot(dx, dy)
    s = np.zeros(x_m.shape[0], dtype=float)
    s[1:] = np.cumsum(ds)
    return s


def heading_from_xy(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """Compute heading angle along the track."""
    dx = np.gradient(x_m)
    dy = np.gradient(y_m)
    return np.arctan2(dy, dx)


def curvature_from_xy(x_m: np.ndarray, y_m: np.ndarray, s_m: np.ndarray) -> np.ndarray:
    """Compute signed curvature from first and second derivatives."""
    dx_ds = np.gradient(x_m, s_m)
    dy_ds = np.gradient(y_m, s_m)
    d2x_ds2 = np.gradient(dx_ds, s_m)
    d2y_ds2 = np.gradient(dy_ds, s_m)

    denominator = np.power(dx_ds * dx_ds + dy_ds * dy_ds, 1.5)
    denominator = np.maximum(denominator, SMALL_EPS)
    numerator = dx_ds * d2y_ds2 - dy_ds * d2x_ds2
    return numerator / denominator


def grade_from_elevation(elevation_m: np.ndarray, s_m: np.ndarray) -> np.ndarray:
    """Compute slope dz/ds along the track."""
    return np.gradient(elevation_m, s_m)


def build_track_data(
    x_m: np.ndarray,
    y_m: np.ndarray,
    elevation_m: np.ndarray,
    banking_rad: np.ndarray,
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
