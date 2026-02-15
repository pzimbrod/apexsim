"""Track data loading from CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from lap_time_sim.track.geometry import build_track_data
from lap_time_sim.track.models import TrackData
from lap_time_sim.utils.exceptions import TrackDataError

REQUIRED_COLUMNS = ("x", "y", "elevation", "banking")


def load_track_csv(path: str | Path) -> TrackData:
    """Load a track CSV into ``TrackData``.

    Args:
        path: Path to a CSV containing ``x``, ``y``, ``elevation``, and ``banking``.

    Returns:
        Parsed and validated track representation.

    Raises:
        lap_time_sim.utils.exceptions.TrackDataError: If the file does not exist,
            has an invalid schema, or contains too few rows.
    """
    file_path = Path(path)
    if not file_path.exists():
        msg = f"Track file not found: {file_path}"
        raise TrackDataError(msg)

    x_vals: list[float] = []
    y_vals: list[float] = []
    elevation_vals: list[float] = []
    banking_vals: list[float] = []

    with file_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            msg = f"CSV has no header: {file_path}"
            raise TrackDataError(msg)

        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            msg = f"Track CSV missing required columns: {missing}"
            raise TrackDataError(msg)

        for row in reader:
            x_vals.append(float(row["x"]))
            y_vals.append(float(row["y"]))
            elevation_vals.append(float(row["elevation"]))
            banking_vals.append(float(row["banking"]))

    if len(x_vals) < 4:
        msg = "Track CSV must contain at least 4 data rows"
        raise TrackDataError(msg)

    return build_track_data(
        x_m=np.asarray(x_vals, dtype=float),
        y_m=np.asarray(y_vals, dtype=float),
        elevation_m=np.asarray(elevation_vals, dtype=float),
        banking_rad=np.asarray(banking_vals, dtype=float),
    )
