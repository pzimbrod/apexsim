"""Shared progress-bar helpers for long-running solver loops."""

from __future__ import annotations

import sys

import numpy as np

DEFAULT_PROGRESS_BAR_WIDTH = 30
DEFAULT_TRACK_PROGRESS_FRACTION_STEP = 0.10


def render_progress_line(
    *,
    prefix: str,
    fraction: float,
    suffix: str,
    final: bool = False,
    bar_width: int = DEFAULT_PROGRESS_BAR_WIDTH,
) -> None:
    """Render one in-place text progress line to stderr.

    Args:
        prefix: Prefix shown before the progress bar.
        fraction: Progress fraction in ``[0, 1]``.
        suffix: Additional text shown after the percentage.
        final: If ``True``, end the line with a newline.
        bar_width: Number of characters used by the progress bar.
    """
    clamped = float(np.clip(fraction, 0.0, 1.0))
    filled = int(clamped * bar_width)
    bar = "#" * filled + "-" * (bar_width - filled)
    end = "\n" if final else ""
    print(
        f"\r{prefix} [{bar}] {100.0 * clamped:5.1f}% {suffix}",
        end=end,
        file=sys.stderr,
        flush=True,
    )


def maybe_emit_track_progress(
    *,
    progress_prefix: str | None,
    segment_idx: int,
    segment_count: int,
    next_fraction_threshold: float,
    fraction_step: float = DEFAULT_TRACK_PROGRESS_FRACTION_STEP,
    bar_width: int = DEFAULT_PROGRESS_BAR_WIDTH,
) -> float:
    """Emit throttled progress updates for track-integration loops.

    Args:
        progress_prefix: Prefix for progress output; ``None`` disables output.
        segment_idx: Current segment index (0-based).
        segment_count: Total segment count.
        next_fraction_threshold: Next completion fraction that should trigger
            progress output.
        fraction_step: Step used to advance ``next_fraction_threshold``.
        bar_width: Number of characters used by the progress bar.

    Returns:
        Updated threshold for the next progress emission.
    """
    if progress_prefix is None or segment_count <= 0:
        return next_fraction_threshold

    completed = segment_idx + 1
    fraction = completed / segment_count
    is_final = completed == segment_count
    if fraction < next_fraction_threshold and not is_final:
        return next_fraction_threshold

    render_progress_line(
        prefix=progress_prefix,
        fraction=fraction,
        suffix=f"segment {completed}/{segment_count}",
        final=is_final,
        bar_width=bar_width,
    )

    threshold = next_fraction_threshold
    while threshold <= fraction:
        threshold += fraction_step
    return threshold
