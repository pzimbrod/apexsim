"""Synthetic track layout builders for physics-focused test scenarios."""

from __future__ import annotations

import numpy as np

from apexsim.track.geometry import build_track_data
from apexsim.track.models import MIN_TRACK_POINT_COUNT, TrackData
from apexsim.utils.exceptions import TrackDataError

DEFAULT_STRAIGHT_LENGTH = 1_000.0
DEFAULT_STRAIGHT_SAMPLE_COUNT = 501
DEFAULT_CIRCLE_RADIUS = 50.0
DEFAULT_CIRCLE_SAMPLE_COUNT = 721
DEFAULT_FIGURE_EIGHT_RADIUS = 80.0
DEFAULT_FIGURE_EIGHT_SAMPLE_COUNT = 1_201


def _validate_positive(name: str, value: float) -> None:
    """Validate that a scalar parameter is strictly positive.

    Args:
        name: Parameter name used in error messages.
        value: Parameter value to validate.

    Raises:
        apexsim.utils.exceptions.TrackDataError: If ``value`` is not
            strictly positive.
    """
    if value <= 0.0:
        msg = f"{name} must be positive"
        raise TrackDataError(msg)


def _validate_sample_count(sample_count: int) -> None:
    """Validate that sample count is sufficient for geometry derivatives.

    Args:
        sample_count: Number of unique centerline samples.

    Raises:
        apexsim.utils.exceptions.TrackDataError: If ``sample_count`` is
            below the minimum required count.
    """
    if sample_count < MIN_TRACK_POINT_COUNT:
        msg = (
            "sample_count must be at least "
            f"{MIN_TRACK_POINT_COUNT}"
        )
        raise TrackDataError(msg)


def _closed_loop(points: np.ndarray) -> np.ndarray:
    """Append the first point to the end to create a closed loop.

    Args:
        points: One-dimensional point coordinate array.

    Returns:
        Closed-loop coordinate array with repeated start point at the end.
    """
    return np.concatenate([points, points[:1]])


def build_straight_track(
    length: float = DEFAULT_STRAIGHT_LENGTH,
    sample_count: int = DEFAULT_STRAIGHT_SAMPLE_COUNT,
    elevation: float = 0.0,
    banking: float = 0.0,
) -> TrackData:
    """Build a straight track centerline with constant elevation and banking.

    Args:
        length: Total straight-line track length [m].
        sample_count: Number of centerline samples.
        elevation: Constant elevation value along the track [m].
        banking: Constant banking angle along the track [rad].

    Returns:
        Validated straight ``TrackData`` representation.

    Raises:
        apexsim.utils.exceptions.TrackDataError: If geometric input values are
            outside valid bounds.
    """
    _validate_positive("length", length)
    _validate_sample_count(sample_count)

    x = np.linspace(0.0, float(length), int(sample_count), dtype=float)
    y = np.zeros_like(x)
    elevation_profile = np.full_like(x, float(elevation), dtype=float)
    banking_profile = np.full_like(x, float(banking), dtype=float)

    return build_track_data(x=x, y=y, elevation=elevation_profile, banking=banking_profile)


def build_circular_track(
    radius: float = DEFAULT_CIRCLE_RADIUS,
    sample_count: int = DEFAULT_CIRCLE_SAMPLE_COUNT,
    clockwise: bool = False,
    elevation: float = 0.0,
    banking: float = 0.0,
) -> TrackData:
    """Build a circular closed-loop track with approximately constant curvature.

    Args:
        radius: Circle radius [m].
        sample_count: Number of unique samples around the circle.
        clockwise: Whether to traverse the circle clockwise.
        elevation: Constant elevation value along the track [m].
        banking: Constant banking angle along the track [rad].

    Returns:
        Validated circular ``TrackData`` representation.

    Raises:
        apexsim.utils.exceptions.TrackDataError: If geometric input values are
            outside valid bounds.
    """
    _validate_positive("radius", radius)
    _validate_sample_count(sample_count)

    angle = np.linspace(0.0, 2.0 * np.pi, int(sample_count), endpoint=False, dtype=float)
    if clockwise:
        angle = -angle

    x = _closed_loop(float(radius) * np.cos(angle))
    y = _closed_loop(float(radius) * np.sin(angle))
    elevation_profile = np.full_like(x, float(elevation), dtype=float)
    banking_profile = np.full_like(x, float(banking), dtype=float)

    return build_track_data(x=x, y=y, elevation=elevation_profile, banking=banking_profile)


def build_figure_eight_track(
    lobe_radius: float = DEFAULT_FIGURE_EIGHT_RADIUS,
    sample_count: int = DEFAULT_FIGURE_EIGHT_SAMPLE_COUNT,
    elevation: float = 0.0,
    banking: float = 0.0,
) -> TrackData:
    """Build a closed figure-eight track using a Gerono lemniscate centerline.

    Args:
        lobe_radius: Characteristic lobe radius scaling the layout [m].
        sample_count: Number of unique samples along the centerline.
        elevation: Constant elevation value along the track [m].
        banking: Constant banking angle along the track [rad].

    Returns:
        Validated figure-eight ``TrackData`` representation.

    Raises:
        apexsim.utils.exceptions.TrackDataError: If geometric input values are
            outside valid bounds.
    """
    _validate_positive("lobe_radius", lobe_radius)
    _validate_sample_count(sample_count)

    angle = np.linspace(0.0, 2.0 * np.pi, int(sample_count), endpoint=False, dtype=float)

    x = _closed_loop(float(lobe_radius) * np.sin(angle))
    y = _closed_loop(0.5 * float(lobe_radius) * np.sin(2.0 * angle))
    elevation_profile = np.full_like(x, float(elevation), dtype=float)
    banking_profile = np.full_like(x, float(banking), dtype=float)

    return build_track_data(x=x, y=y, elevation=elevation_profile, banking=banking_profile)
