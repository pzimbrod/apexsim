"""Import and convert Spa centerline data from TUMFTM racetrack-database.

Usage:
    python scripts/import_spa_from_tumftm.py

The script downloads `tracks/Spa.csv` and converts it into the internal format:
`x,y,elevation,banking,w_tr_right_m,w_tr_left_m`.
Elevation and banking are currently not supplied by the source and are set to 0.0.
"""

from __future__ import annotations

import csv
import logging
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/tracks/Spa.csv"
OUT_PATH = Path("data/spa_francorchamps.csv")
RAW_COPY_PATH = Path("data/sources/tumftm_spa_raw.csv")


def _download_text(url: str) -> str:
    """Download a UTF-8 text resource via HTTP.

    Args:
        url: Source URL.

    Returns:
        Decoded response body.
    """
    with urllib.request.urlopen(url, timeout=30) as response:  # nosec B310
        data = response.read().decode("utf-8")
    return data


def main() -> None:
    """Download and convert Spa centerline data into internal CSV schema."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger("import_spa_from_tumftm")
    text = _download_text(SOURCE_URL)
    RAW_COPY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_COPY_PATH.write_text(text, encoding="utf-8")

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        msg = "Source track file is empty"
        raise RuntimeError(msg)

    header = lines[0].lstrip("# ").strip()
    fieldnames = [name.strip() for name in header.split(",")]
    required = {"x_m", "y_m", "w_tr_right_m", "w_tr_left_m"}
    if not required.issubset(set(fieldnames)):
        msg = "Unexpected source schema for Spa track file"
        raise RuntimeError(msg)
    reader = csv.DictReader(lines[1:], fieldnames=fieldnames)

    rows: list[dict[str, float]] = []
    for row in reader:
        rows.append(
            {
                "x": float(row["x_m"]),
                "y": float(row["y_m"]),
                "elevation": 0.0,
                "banking": 0.0,
                "w_tr_right_m": float(row["w_tr_right_m"]),
                "w_tr_left_m": float(row["w_tr_left_m"]),
            }
        )

    if len(rows) < 4:
        msg = "Source track has too few points"
        raise RuntimeError(msg)

    # Close the lap explicitly by appending the first point.
    rows.append(rows[0].copy())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["x", "y", "elevation", "banking", "w_tr_right_m", "w_tr_left_m"],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %s with %d points", OUT_PATH, len(rows))


if __name__ == "__main__":
    main()
