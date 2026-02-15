"""Simulation solvers and runners."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lap_time_sim.simulation.config import SimulationConfig

if TYPE_CHECKING:
    from lap_time_sim.simulation.model_api import LapTimeVehicleModel, VehicleModelDiagnostics
    from lap_time_sim.simulation.runner import LapSimulationResult

__all__ = [
    "LapSimulationResult",
    "LapTimeVehicleModel",
    "SimulationConfig",
    "VehicleModelDiagnostics",
    "simulate_lap",
]


def __getattr__(name: str) -> Any:
    if name == "LapTimeVehicleModel":
        from lap_time_sim.simulation.model_api import LapTimeVehicleModel

        return LapTimeVehicleModel
    if name == "VehicleModelDiagnostics":
        from lap_time_sim.simulation.model_api import VehicleModelDiagnostics

        return VehicleModelDiagnostics
    if name == "LapSimulationResult":
        from lap_time_sim.simulation.runner import LapSimulationResult

        return LapSimulationResult
    if name == "simulate_lap":
        from lap_time_sim.simulation.runner import simulate_lap

        return simulate_lap
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
