"""Run a single-track parameter sensitivity study on a synthetic circular track."""

from __future__ import annotations

import logging

from common import run_single_track_sensitivity_study, sensitivity_output_root

from apexsim.track import build_circular_track
from apexsim.utils import configure_logging

TRACK_LABEL = "Synthetic circle (R=50 m)"


def main() -> None:
    """Run and export synthetic-track sensitivity artifacts."""
    configure_logging(logging.INFO)
    logger = logging.getLogger("synthetic_single_track_sensitivity")

    track = build_circular_track(radius=50.0, sample_count=721)
    output_dir = sensitivity_output_root() / "synthetic_circle_single_track"

    long_table, pivot_table = run_single_track_sensitivity_study(
        track=track,
        track_label=TRACK_LABEL,
        output_dir=output_dir,
    )

    logger.info(
        "Wrote %d long-form rows to %s",
        len(long_table),
        output_dir / "sensitivities_long.csv",
    )
    logger.info(
        "Wrote %d pivot rows to %s",
        len(pivot_table),
        output_dir / "sensitivities_pivot.csv",
    )
    logger.info("Sensitivity bar plot: %s", output_dir / "sensitivity_bars.png")


if __name__ == "__main__":
    main()
