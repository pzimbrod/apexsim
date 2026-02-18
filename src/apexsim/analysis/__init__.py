"""Simulation analysis tools."""

from apexsim.analysis.kpi import KpiSummary, compute_kpis
from apexsim.analysis.performance_envelope import (
    PerformanceEnvelopeConfig,
    PerformanceEnvelopeNumerics,
    PerformanceEnvelopePhysics,
    PerformanceEnvelopeResult,
    PerformanceEnvelopeRuntime,
    build_performance_envelope_config,
    compute_performance_envelope,
)
from apexsim.analysis.plots import export_standard_plots
from apexsim.analysis.sensitivity import (
    SensitivityConfig,
    SensitivityNumerics,
    SensitivityParameter,
    SensitivityResult,
    SensitivityRuntime,
    SensitivityStudyParameter,
    SensitivityStudyResult,
    build_sensitivity_config,
    compute_sensitivities,
    register_sensitivity_model_adapter,
    run_lap_sensitivity_study,
)

__all__ = [
    "KpiSummary",
    "PerformanceEnvelopeConfig",
    "PerformanceEnvelopeNumerics",
    "PerformanceEnvelopePhysics",
    "PerformanceEnvelopeResult",
    "PerformanceEnvelopeRuntime",
    "SensitivityConfig",
    "SensitivityNumerics",
    "SensitivityParameter",
    "SensitivityResult",
    "SensitivityRuntime",
    "SensitivityStudyParameter",
    "SensitivityStudyResult",
    "build_performance_envelope_config",
    "build_sensitivity_config",
    "compute_kpis",
    "compute_performance_envelope",
    "compute_sensitivities",
    "export_standard_plots",
    "register_sensitivity_model_adapter",
    "run_lap_sensitivity_study",
]
