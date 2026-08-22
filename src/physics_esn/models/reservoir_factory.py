from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from physics_esn.models.nonlinear_wc_reservoir import (
    NonlinearWilsonCowanReservoir,
    generate_sparse_coupling_graph,
)
from physics_esn.models.physics_reservoir import (
    COUPLED_NONLINEAR_WC_MODE,
    DETERMINISTIC_POLES_MODE,
    DISTRIBUTED_POLES_MODE,
    INDEPENDENT_NONLINEAR_WC_MODE,
    PhysicsInformedReservoir,
    build_physics_informed_reservoir,
    generate_continuous_reservoir_modes,
    normalize_reservoir_mode,
)
from physics_esn.models.wilson_cowan import (
    WilsonCowanParameters,
    discrete_reservoir_eigenvalues,
)


def _configured_seed(values: dict[str, Any], fallback: int = 0) -> int:
    return int(values.get("seed", values.get("random_seed", fallback)))


@dataclass
class ReservoirBuildResult:
    """A common readout interface plus mode-specific immutable metadata."""

    model: PhysicsInformedReservoir
    mode: str
    requested_mode: str
    random_seed: int
    num_wc_blocks: int
    reservoir_units: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
    continuous_eigenvalues: np.ndarray | None = None
    discrete_eigenvalues: np.ndarray | None = None

    @property
    def reservoir_state_dimension(self) -> int:
        return self.model.reservoir_state_dimension

    def fit_one_step(
        self,
        signal_values: np.ndarray,
        ridge: float,
        washout_samples: int = 0,
    ) -> Any:
        return self.model.fit_one_step(signal_values, ridge=ridge, washout_samples=washout_samples)

    def predict_one_step(
        self,
        signal_values: np.ndarray,
        warmup_values: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.model.predict_one_step(signal_values, warmup_values=warmup_values)


def build_reservoir(
    config: dict[str, Any],
    fitted_parameters: WilsonCowanParameters,
    wc_eigenvalues: np.ndarray,
    sample_interval_s: float,
    *,
    reservoir_mode: str | None = None,
) -> ReservoirBuildResult:
    """Build one of the four fixed-dynamics reservoir architectures.

    ``sample_interval_s`` is the processed EEG sample interval in seconds. It is
    deliberately independent of the integration interval used during WC fitting.
    """
    if not np.isfinite(sample_interval_s) or sample_interval_s <= 0.0:
        raise ValueError("sample_interval_s must be finite and positive.")
    reservoir_config = dict(config.get("reservoir", {}))
    requested_mode = str(
        reservoir_mode
        if reservoir_mode is not None
        else reservoir_config.get("reservoir_mode", DETERMINISTIC_POLES_MODE)
    )
    mode = normalize_reservoir_mode(requested_mode)

    if mode in (DETERMINISTIC_POLES_MODE, DISTRIBUTED_POLES_MODE):
        reservoir_seed = _configured_seed(reservoir_config)
        if mode == DETERMINISTIC_POLES_MODE:
            mode_seed = input_seed = reservoir_seed
        else:
            mode_seed_sequence, input_seed_sequence = np.random.SeedSequence(reservoir_seed).spawn(2)
            mode_seed = int(mode_seed_sequence.generate_state(1, dtype=np.uint64)[0])
            input_seed = int(input_seed_sequence.generate_state(1, dtype=np.uint64)[0])
        continuous_modes = generate_continuous_reservoir_modes(
            wc_eigenvalues,
            reservoir_mode=mode,
            reservoir_size=int(reservoir_config.get("reservoir_size", 2)),
            eigenvalue_sigma_real=float(reservoir_config.get("eigenvalue_sigma_real", 0.0)),
            eigenvalue_sigma_imag=float(reservoir_config.get("eigenvalue_sigma_imag", 0.0)),
            seed=mode_seed,
        )
        discrete_modes = discrete_reservoir_eigenvalues(continuous_modes, dt=sample_interval_s)
        if not np.all(np.isfinite(discrete_modes)) or np.any(np.abs(discrete_modes) >= 1.0):
            raise RuntimeError("Stable continuous reservoir modes must map inside the unit circle.")
        model = build_physics_informed_reservoir(
            discrete_modes,
            input_scale=float(reservoir_config.get("input_scale", 0.2)),
            seed=input_seed,
        )
        diagnostics = {
            "criterion": "linear_stability",
            "continuous_max_real_part_per_s": float(np.max(continuous_modes.real)),
            "discrete_max_modulus": float(np.max(np.abs(discrete_modes))),
            "stable": True,
        }
        return ReservoirBuildResult(
            model=model,
            mode=mode,
            requested_mode=requested_mode,
            random_seed=reservoir_seed,
            num_wc_blocks=0,
            reservoir_units=int(discrete_modes.size),
            diagnostics=diagnostics,
            continuous_eigenvalues=continuous_modes,
            discrete_eigenvalues=discrete_modes,
        )

    nonlinear_config = dict(config.get("nonlinear_reservoir", {}))
    nonlinear_seed = _configured_seed(
        nonlinear_config,
        fallback=_configured_seed(reservoir_config),
    )
    num_blocks = int(
        nonlinear_config.get("num_blocks", reservoir_config.get("reservoir_size", 50))
    )
    coupling_graph = None
    coupling_strength = 0.0
    coupling_enabled = False
    graph_seed: int | None = None
    if mode == COUPLED_NONLINEAR_WC_MODE:
        coupling_config = dict(config.get("coupling", {}))
        configured_enabled = coupling_config.get("enabled", True)
        if not isinstance(configured_enabled, bool):
            raise ValueError("coupling.enabled must be a boolean when provided.")
        coupling_enabled = configured_enabled
        graph_seed = _configured_seed(coupling_config, fallback=nonlinear_seed)
        coupling_strength = (
            float(coupling_config.get("strength", 0.05)) if coupling_enabled else 0.0
        )
        coupling_graph = generate_sparse_coupling_graph(
            num_blocks=num_blocks,
            degree=int(coupling_config.get("degree", 4)),
            seed=graph_seed,
        )
    elif mode != INDEPENDENT_NONLINEAR_WC_MODE:
        raise AssertionError(f"Unhandled normalized reservoir mode: {mode}")

    configured_state_bounds = nonlinear_config.get("state_bounds")
    if configured_state_bounds is None:
        configured_state_bounds = (
            float(nonlinear_config.get("state_lower_bound", -1.0e-6)),
            float(nonlinear_config.get("state_upper_bound", 1.0 + 1.0e-6)),
        )
    bounds = np.asarray(configured_state_bounds, dtype=np.float64)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        raise ValueError("nonlinear_reservoir.state_bounds must be [lower, upper].")
    dynamics = NonlinearWilsonCowanReservoir.from_fitted_parameters(
        fitted_parameters,
        num_blocks=num_blocks,
        sample_dt=sample_interval_s,
        parameter_jitter=nonlinear_config.get("parameter_jitter", 0.05),
        input_scale=float(nonlinear_config.get("input_scale", 0.1)),
        seed=nonlinear_seed,
        rk4_substeps=int(nonlinear_config.get("rk4_substeps", 1)),
        initial_state=tuple(nonlinear_config.get("initial_state", (0.1, 0.1))),
        preflight_duration_s=float(nonlinear_config.get("preflight_duration_s", 0.25)),
        max_resample_attempts=int(nonlinear_config.get("max_resample_attempts", 100)),
        state_bounds=(float(bounds[0]), float(bounds[1])),
        coupling_graph=coupling_graph,
        coupling_strength=coupling_strength,
    )
    diagnostics = dict(dynamics.diagnostics)
    diagnostics.update(
        {
            "criterion": "nonlinear_trajectory_boundedness",
            "linear_equilibrium_stability_required": False,
            "coupling_enabled": coupling_enabled,
            "coupling_strength": coupling_strength,
            "coupling_graph_seed": graph_seed,
        }
    )
    return ReservoirBuildResult(
        model=PhysicsInformedReservoir(dynamics),
        mode=mode,
        requested_mode=requested_mode,
        random_seed=nonlinear_seed,
        num_wc_blocks=num_blocks,
        reservoir_units=num_blocks,
        diagnostics=diagnostics,
    )
