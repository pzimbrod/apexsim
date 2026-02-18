"""Simulation configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

from apexsim.utils.exceptions import ConfigurationError

DEFAULT_MIN_SPEED = 8.0
DEFAULT_LATERAL_ENVELOPE_MAX_ITERATIONS = 20
DEFAULT_LATERAL_ENVELOPE_CONVERGENCE_TOLERANCE = 0.1
DEFAULT_TRANSIENT_STEP = 0.01
DEFAULT_MAX_SPEED = 115.0
DEFAULT_ENABLE_TRANSIENT_REFINEMENT = False
DEFAULT_COMPUTE_BACKEND = "numpy"
DEFAULT_TORCH_DEVICE = "cpu"
DEFAULT_ENABLE_TORCH_COMPILE = False
VALID_COMPUTE_BACKENDS = ("numpy", "numba", "torch")


@dataclass(frozen=True)
class NumericsConfig:
    """Numerical controls for the lap-time solver.

    Args:
        min_speed: Numerical floor for speed to avoid singular divisions [m/s].
        lateral_envelope_max_iterations: Maximum fixed-point iterations for the
            lateral-speed envelope solver.
        lateral_envelope_convergence_tolerance: Early-stop threshold for the
            lateral envelope fixed-point update (``max |v_k - v_{k-1}|``) [m/s].
        transient_step: Integration step for optional transient refinement [s].
    """

    min_speed: float = DEFAULT_MIN_SPEED
    lateral_envelope_max_iterations: int = DEFAULT_LATERAL_ENVELOPE_MAX_ITERATIONS
    lateral_envelope_convergence_tolerance: float = DEFAULT_LATERAL_ENVELOPE_CONVERGENCE_TOLERANCE
    transient_step: float = DEFAULT_TRANSIENT_STEP

    def validate(self) -> None:
        """Validate numerical solver settings.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If any solver
                configuration value violates its bound.
        """
        if self.min_speed <= 0.0:
            msg = "min_speed must be positive"
            raise ConfigurationError(msg)
        if self.lateral_envelope_max_iterations < 1:
            msg = "lateral_envelope_max_iterations must be at least 1"
            raise ConfigurationError(msg)
        if self.lateral_envelope_convergence_tolerance <= 0.0:
            msg = "lateral_envelope_convergence_tolerance must be positive"
            raise ConfigurationError(msg)
        if self.transient_step <= 0.0:
            msg = "transient_step must be positive"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime controls that define simulation scenario boundaries.

    Args:
        max_speed: Hard speed cap used by the quasi-steady profile solver [m/s].
        enable_transient_refinement: Flag for optional second-pass transient solve.
        compute_backend: Numerical backend identifier (``numpy``, ``numba``,
            or ``torch``).
        torch_device: Torch device identifier. ``numpy`` and ``numba`` are
            CPU-only backends and therefore require ``torch_device='cpu'``.
        torch_compile: Enable ``torch.compile`` for the ``torch`` backend.
    """

    max_speed: float
    enable_transient_refinement: bool = DEFAULT_ENABLE_TRANSIENT_REFINEMENT
    compute_backend: str = DEFAULT_COMPUTE_BACKEND
    torch_device: str = DEFAULT_TORCH_DEVICE
    torch_compile: bool = DEFAULT_ENABLE_TORCH_COMPILE

    def validate(self, numerics: NumericsConfig) -> None:
        """Validate runtime controls against solver numerics.

        Args:
            numerics: Numerical parameter set used by the solver.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If runtime controls
                are inconsistent with solver numerics or backend requirements.
        """
        if self.max_speed <= numerics.min_speed:
            msg = "max_speed must be greater than numerics.min_speed"
            raise ConfigurationError(msg)
        if self.compute_backend not in VALID_COMPUTE_BACKENDS:
            msg = (
                "compute_backend must be one of "
                f"{VALID_COMPUTE_BACKENDS}, got: {self.compute_backend!r}"
            )
            raise ConfigurationError(msg)
        if not self.torch_device:
            msg = "torch_device must be a non-empty string"
            raise ConfigurationError(msg)
        if not isinstance(self.torch_compile, bool):
            msg = "torch_compile must be a boolean"
            raise ConfigurationError(msg)

        if self.compute_backend != "torch":
            if self.torch_device != DEFAULT_TORCH_DEVICE:
                msg = (
                    "torch_device is only meaningful for compute_backend='torch'. "
                    "Use torch_device='cpu' for numpy/numba backends."
                )
                raise ConfigurationError(msg)
            if self.torch_compile:
                msg = "torch_compile can only be enabled for compute_backend='torch'"
                raise ConfigurationError(msg)

        if self.compute_backend == "numba":
            self._validate_numba_runtime()
        if self.compute_backend == "torch":
            self._validate_torch_runtime()

    def _validate_numba_runtime(self) -> None:
        """Validate availability of numba runtime.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If numba is not
                installed in the active environment.
        """
        try:
            import numba  # type: ignore[import-untyped]  # noqa: F401
        except ModuleNotFoundError as exc:
            msg = (
                "compute_backend='numba' requires Numba. "
                "Install with `pip install -e '.[numba]'` or add `numba` to your environment."
            )
            raise ConfigurationError(msg) from exc

    def _validate_torch_runtime(self) -> None:
        """Validate availability of the requested torch runtime.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If torch is not
                installed or selected CUDA device is not available.
        """
        try:
            import torch
        except ModuleNotFoundError as exc:
            msg = (
                "compute_backend='torch' requires PyTorch. "
                "Install with `pip install -e '.[torch]'` or add `torch` to your environment."
            )
            raise ConfigurationError(msg) from exc

        if self.torch_device.startswith("cuda") and not torch.cuda.is_available():
            msg = (
                "torch_device requests CUDA but no CUDA device is available: "
                f"{self.torch_device!r}"
            )
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level solver config composed of runtime and numerics.

    Args:
        runtime: Scenario and runtime controls, independent of physical car data.
        numerics: Discretization and convergence controls for numerical solving.
    """

    runtime: RuntimeConfig
    numerics: NumericsConfig

    def validate(self) -> None:
        """Validate combined simulation settings.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If runtime or
                numerical configuration values violate their bounds.
        """
        self.numerics.validate()
        self.runtime.validate(self.numerics)


def build_simulation_config(
    max_speed: float = DEFAULT_MAX_SPEED,
    numerics: NumericsConfig | None = None,
    enable_transient_refinement: bool = DEFAULT_ENABLE_TRANSIENT_REFINEMENT,
    compute_backend: str = DEFAULT_COMPUTE_BACKEND,
    torch_device: str = DEFAULT_TORCH_DEVICE,
    torch_compile: bool = DEFAULT_ENABLE_TORCH_COMPILE,
) -> SimulationConfig:
    """Build a validated simulation config with sensible numerical defaults.

    Args:
        max_speed: Runtime speed cap for the quasi-steady profile solver [m/s].
        numerics: Optional numerical settings. Defaults to :class:`NumericsConfig`.
        enable_transient_refinement: Flag for optional transient post-processing.
        compute_backend: Numerical backend identifier (``numpy``, ``numba``,
            or ``torch``).
        torch_device: Torch device identifier. ``numpy`` and ``numba`` are
            CPU-only backends and therefore require ``torch_device='cpu'``.
        torch_compile: Enable ``torch.compile`` for the ``torch`` backend.

    Returns:
        Fully validated simulation configuration.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If assembled runtime or
            numerical settings are inconsistent.
    """
    config = SimulationConfig(
        runtime=RuntimeConfig(
            max_speed=max_speed,
            enable_transient_refinement=enable_transient_refinement,
            compute_backend=compute_backend,
            torch_device=torch_device,
            torch_compile=torch_compile,
        ),
        numerics=numerics or NumericsConfig(),
    )
    config.validate()
    return config
