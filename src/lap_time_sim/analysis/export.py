"""Export helpers for simulation outputs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from lap_time_sim.analysis.kpi import KpiSummary


def export_kpi_json(kpis: KpiSummary, path: str | Path) -> None:
    """Persist KPI summary as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(kpis), indent=2), encoding="utf-8")
