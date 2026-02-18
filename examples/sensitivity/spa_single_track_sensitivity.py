"""Run a single-track parameter sensitivity study on Spa-Francorchamps."""

from __future__ import annotations

import logging

from common import run_single_track_sensitivity_study, sensitivity_output_root, spa_track_path

from apexsim.track import load_track_csv
from apexsim.utils import configure_logging

TRACK_LABEL = "Spa-Francorchamps"


def main() -> None:
    """Run and export Spa sensitivity artifacts."""
    configure_logging(logging.INFO)
    logger = logging.getLogger("spa_single_track_sensitivity")

    track = load_track_csv(spa_track_path())
    output_dir = sensitivity_output_root() / "spa_single_track"

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
