from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from physics_esn.models.wilson_cowan import (
    WILSON_COWAN_PARAMETER_BOUNDS,
    WilsonCowanParameters,
    wilson_cowan_vector_field,
)


PARAMETER_NAMES = (
    "tau_e",
    "tau_i",
    "w_ee",
    "w_ei",
    "w_ie",
    "w_ii",
    "p",
    "q",
)
_LOG_SPACE_PARAMETERS = frozenset(("tau_e", "tau_i", "w_ee", "w_ei", "w_ie", "w_ii"))
_SIGNED_PARAMETERS = frozenset(("p", "q"))


def _readonly_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class WilsonCowanParameterPopulation:
    """Broadcastable, immutable parameters for a population of WC blocks."""

    tau_e: np.ndarray
    tau_i: np.ndarray
    w_ee: np.ndarray
    w_ei: np.ndarray
    w_ie: np.ndarray
    w_ii: np.ndarray
    p: np.ndarray
    q: np.ndarray
    sigmoid_gain: np.ndarray
    sigmoid_theta: np.ndarray

    def __post_init__(self) -> None:
        lengths: set[int] = set()
        for name in (*PARAMETER_NAMES, "sigmoid_gain", "sigmoid_theta"):
            values = _readonly_vector(getattr(self, name), name=name)
            object.__setattr__(self, name, values)
            lengths.add(values.size)
        if len(lengths) != 1:
            raise ValueError("All Wilson-Cowan parameter arrays must have the same length.")

        for name in PARAMETER_NAMES:
            lower, upper = WILSON_COWAN_PARAMETER_BOUNDS[name]
            values = getattr(self, name)
            if np.any(values < lower) or np.any(values > upper):
                raise ValueError(f"Sampled {name} values must remain within [{lower}, {upper}].")
        if np.any(self.sigmoid_gain <= 0.0):
            raise ValueError("sigmoid_gain values must be positive.")

    @property
    def num_blocks(self) -> int:
        return int(self.tau_e.size)

    def parameter_matrix(self) -> np.ndarray:
        return np.column_stack([getattr(self, name) for name in PARAMETER_NAMES])


def _population_from_matrix(
    matrix: np.ndarray,
    sigmoid_gain: float,
    sigmoid_theta: float,
) -> WilsonCowanParameterPopulation:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(PARAMETER_NAMES):
        raise ValueError("Parameter matrix must have one row per block and eight columns.")
    repeated_gain = np.full(values.shape[0], sigmoid_gain, dtype=np.float64)
    repeated_theta = np.full(values.shape[0], sigmoid_theta, dtype=np.float64)
    return WilsonCowanParameterPopulation(
        **{name: values[:, index] for index, name in enumerate(PARAMETER_NAMES)},
        sigmoid_gain=repeated_gain,
        sigmoid_theta=repeated_theta,
    )


def _validate_center(parameters: WilsonCowanParameters) -> None:
    for name in PARAMETER_NAMES:
        value = float(getattr(parameters, name))
        lower, upper = WILSON_COWAN_PARAMETER_BOUNDS[name]
        if not lower <= value <= upper:
            raise ValueError(
                f"Fitted parameter {name}={value} lies outside the sampling bounds "
                f"[{lower}, {upper}]."
            )


def _parameter_jitter_scales(
    parameter_jitter: float | Mapping[str, float],
) -> dict[str, float]:
    if isinstance(parameter_jitter, Mapping):
        unknown = set(parameter_jitter) - set(PARAMETER_NAMES) - {"default"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown Wilson-Cowan jitter parameters: {names}.")
        default = float(parameter_jitter.get("default", 0.0))
        scales = {
            name: float(parameter_jitter.get(name, default)) for name in PARAMETER_NAMES
        }
    else:
        scale = float(parameter_jitter)
        scales = {name: scale for name in PARAMETER_NAMES}
    if any(not np.isfinite(value) or value < 0.0 for value in scales.values()):
        raise ValueError("Parameter jitter scales must be finite and non-negative.")
    return scales


def _draw_bounded_parameter(
    rng: np.random.Generator,
    name: str,
    center: float,
    jitter: float,
    *,
    max_attempts: int = 10_000,
) -> tuple[float, int]:
    """Draw one parameter and return it with its boundary-rejection count.

    Time constants and nonzero nonnegative couplings use multiplicative
    log-space jitter. A coupling centered exactly at zero uses a half-normal
    scale relative to its allowed range. Signed drives use bounded additive
    jitter relative to their allowed range.
    """

    lower, upper = WILSON_COWAN_PARAMETER_BOUNDS[name]
    if jitter == 0.0:
        return center, 0
    for rejection_count in range(max_attempts):
        if name in _LOG_SPACE_PARAMETERS and center > 0.0:
            candidate = center * float(np.exp(rng.normal(0.0, jitter)))
        elif name in _LOG_SPACE_PARAMETERS:
            candidate = abs(float(rng.normal(0.0, jitter * (upper - lower))))
        elif name in _SIGNED_PARAMETERS:
            candidate = float(rng.normal(center, jitter * (upper - lower)))
        else:  # pragma: no cover - PARAMETER_NAMES exhausts the supported cases.
            raise AssertionError(f"Unhandled parameter sampling rule for {name}.")
        if np.isfinite(candidate) and lower <= candidate <= upper:
            return candidate, rejection_count
    raise RuntimeError(f"Unable to sample {name} within its configured bounds.")


def _sample_parameter_rows(
    center: WilsonCowanParameters,
    count: int,
    jitter_scales: Mapping[str, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    rows = np.empty((count, len(PARAMETER_NAMES)), dtype=np.float64)
    boundary_rejections = 0
    for row in range(count):
        for column, name in enumerate(PARAMETER_NAMES):
            sampled, rejected = _draw_bounded_parameter(
                rng,
                name,
                float(getattr(center, name)),
                float(jitter_scales[name]),
            )
            rows[row, column] = sampled
            boundary_rejections += rejected
    return rows, boundary_rejections


def generate_sparse_coupling_graph(
    num_blocks: int,
    degree: int,
    seed: int = 0,
) -> sparse.csr_matrix:
    """Generate a directed fixed-in-degree graph with unit row sums.

    ``A[k, j]`` is the influence of source block ``j`` on target block ``k``.
    Every nonempty row has exactly ``degree`` entries of weight ``1 / degree``;
    therefore reservoir size does not change the scale of the graph current.
    """

    if isinstance(num_blocks, bool) or not isinstance(num_blocks, (int, np.integer)):
        raise ValueError("num_blocks must be a positive integer.")
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
        raise ValueError("degree must be a non-negative integer.")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be a positive integer.")
    if degree < 0 or degree >= num_blocks:
        raise ValueError("degree must satisfy 0 <= degree < num_blocks.")
    if degree == 0:
        return sparse.csr_matrix((num_blocks, num_blocks), dtype=np.float64)

    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(num_blocks, dtype=np.int64), degree)
    columns = np.empty(num_blocks * degree, dtype=np.int64)
    candidates = np.arange(num_blocks, dtype=np.int64)
    for target in range(num_blocks):
        available = np.concatenate((candidates[:target], candidates[target + 1 :]))
        columns[target * degree : (target + 1) * degree] = rng.choice(
            available,
            size=degree,
            replace=False,
        )
    data = np.full(rows.size, 1.0 / degree, dtype=np.float64)
    return sparse.csr_matrix((data, (rows, columns)), shape=(num_blocks, num_blocks))


def _normalized_coupling_graph(
    graph: sparse.spmatrix | np.ndarray | None,
    num_blocks: int,
) -> sparse.csr_matrix | None:
    if graph is None:
        return None
    matrix = sparse.csr_matrix(graph, dtype=np.float64, copy=True)
    if matrix.shape != (num_blocks, num_blocks):
        raise ValueError("coupling_graph must have shape (num_blocks, num_blocks).")
    matrix.eliminate_zeros()
    if matrix.nnz and (
        not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0.0)
    ):
        raise ValueError("coupling_graph weights must be finite and non-negative.")
    if np.any(np.asarray(matrix.diagonal()) != 0.0):
        raise ValueError("coupling_graph must not contain self-edges.")
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    nonempty = row_sums > 0.0
    if np.any(nonempty):
        matrix = sparse.diags(
            np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=nonempty)
        ) @ matrix
        matrix = matrix.tocsr()
    return matrix


def rk4_wilson_cowan_step(
    e: np.ndarray,
    i: np.ndarray,
    parameters: WilsonCowanParameterPopulation,
    dt: float,
    *,
    external_e: np.ndarray | float = 0.0,
    external_i: np.ndarray | float = 0.0,
    coupling_graph: sparse.csr_matrix | None = None,
    coupling_strength: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance a vectorized WC population by one fixed RK4 step in seconds."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive.")
    e0 = np.asarray(e, dtype=np.float64)
    i0 = np.asarray(i, dtype=np.float64)
    if e0.shape != (parameters.num_blocks,) or i0.shape != (parameters.num_blocks,):
        raise ValueError("E and I state arrays must have one value per WC block.")

    def derivatives(e_state: np.ndarray, i_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coupling_current: np.ndarray | float = 0.0
        if coupling_graph is not None and coupling_strength != 0.0:
            coupling_current = coupling_strength * (coupling_graph @ e_state)
        return wilson_cowan_vector_field(
            e_state,
            i_state,
            parameters,
            external_e=np.asarray(external_e) + coupling_current,
            external_i=external_i,
        )

    k1_e, k1_i = derivatives(e0, i0)
    k2_e, k2_i = derivatives(e0 + 0.5 * dt * k1_e, i0 + 0.5 * dt * k1_i)
    k3_e, k3_i = derivatives(e0 + 0.5 * dt * k2_e, i0 + 0.5 * dt * k2_i)
    k4_e, k4_i = derivatives(e0 + dt * k3_e, i0 + dt * k3_i)
    next_e = e0 + (dt / 6.0) * (k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e)
    next_i = i0 + (dt / 6.0) * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i)
    return np.asarray(next_e, dtype=np.float64), np.asarray(next_i, dtype=np.float64)


def _preflight_population(
    parameters: WilsonCowanParameterPopulation,
    *,
    sample_dt: float,
    rk4_substeps: int,
    initial_state: tuple[float, float],
    duration_s: float,
    state_bounds: tuple[float, float],
    coupling_graph: sparse.csr_matrix | None = None,
    coupling_strength: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    e = np.full(parameters.num_blocks, initial_state[0], dtype=np.float64)
    i = np.full(parameters.num_blocks, initial_state[1], dtype=np.float64)
    valid = np.ones(parameters.num_blocks, dtype=bool)
    minimum = float(min(initial_state))
    maximum = float(max(initial_state))
    step_count = max(1, int(np.ceil(duration_s / sample_dt)))
    substep_dt = sample_dt / rk4_substeps
    lower, upper = state_bounds

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(step_count):
            for _ in range(rk4_substeps):
                e, i = rk4_wilson_cowan_step(
                    e,
                    i,
                    parameters,
                    substep_dt,
                    coupling_graph=coupling_graph,
                    coupling_strength=coupling_strength,
                )
                finite = np.isfinite(e) & np.isfinite(i)
                bounded = (e >= lower) & (e <= upper) & (i >= lower) & (i <= upper)
                valid &= finite & bounded
                finite_values = np.concatenate((e[finite], i[finite]))
                if finite_values.size:
                    minimum = min(minimum, float(np.min(finite_values)))
                    maximum = max(maximum, float(np.max(finite_values)))
                # Invalid independent blocks are quarantined so they cannot
                # overflow repeatedly. For a coupled preflight we terminate,
                # since one invalid trajectory can affect its neighbors.
                if not np.all(valid):
                    if coupling_graph is not None and coupling_strength != 0.0:
                        return valid, minimum, maximum
                    e[~valid] = initial_state[0]
                    i[~valid] = initial_state[1]
    return valid, minimum, maximum


def _graph_diagnostics(graph: sparse.csr_matrix | None, num_blocks: int) -> dict[str, Any]:
    if graph is None:
        return {
            "coupling_graph_nnz": 0,
            "coupling_graph_density": 0.0,
            "coupling_in_degree_min": 0,
            "coupling_in_degree_max": 0,
            "coupling_in_degree_mean": 0.0,
            "coupling_nonempty_row_sum_min": 0.0,
            "coupling_nonempty_row_sum_max": 0.0,
        }
    row_counts = np.diff(graph.indptr)
    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    nonempty_sums = row_sums[row_sums > 0.0]
    return {
        "coupling_graph_nnz": int(graph.nnz),
        "coupling_graph_density": float(graph.nnz / (num_blocks * num_blocks)),
        "coupling_in_degree_min": int(row_counts.min()),
        "coupling_in_degree_max": int(row_counts.max()),
        "coupling_in_degree_mean": float(row_counts.mean()),
        "coupling_nonempty_row_sum_min": (
            float(nonempty_sums.min()) if nonempty_sums.size else 0.0
        ),
        "coupling_nonempty_row_sum_max": (
            float(nonempty_sums.max()) if nonempty_sums.size else 0.0
        ),
    }


@dataclass
class NonlinearWilsonCowanReservoir:
    """Fixed ensemble of driven nonlinear WC blocks with a ridge-ready state."""

    parameters: WilsonCowanParameterPopulation
    input_weights_e: np.ndarray
    input_weights_i: np.ndarray
    sample_dt: float
    rk4_substeps: int = 1
    initial_state: tuple[float, float] = (0.1, 0.1)
    state_bounds: tuple[float, float] = (-1.0e-6, 1.0 + 1.0e-6)
    coupling_graph: sparse.csr_matrix | np.ndarray | None = None
    coupling_strength: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    state: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.sample_dt) or self.sample_dt <= 0.0:
            raise ValueError("sample_dt must be finite and positive seconds.")
        if (
            isinstance(self.rk4_substeps, bool)
            or not isinstance(self.rk4_substeps, (int, np.integer))
            or self.rk4_substeps < 1
        ):
            raise ValueError("rk4_substeps must be a positive integer.")
        if len(self.initial_state) != 2 or not np.all(np.isfinite(self.initial_state)):
            raise ValueError("initial_state must contain two finite activity values.")
        if len(self.state_bounds) != 2 or not np.all(np.isfinite(self.state_bounds)):
            raise ValueError("state_bounds must contain two finite values.")
        lower, upper = (float(self.state_bounds[0]), float(self.state_bounds[1]))
        if lower >= upper:
            raise ValueError("state_bounds must be ordered from lower to upper.")
        if not lower <= self.initial_state[0] <= upper or not lower <= self.initial_state[1] <= upper:
            raise ValueError("initial_state must lie inside state_bounds.")
        if not np.isfinite(self.coupling_strength) or self.coupling_strength < 0.0:
            raise ValueError("coupling_strength must be finite and non-negative.")

        self.input_weights_e = _readonly_vector(self.input_weights_e, name="input_weights_e")
        self.input_weights_i = _readonly_vector(self.input_weights_i, name="input_weights_i")
        expected_shape = (self.parameters.num_blocks,)
        if self.input_weights_e.shape != expected_shape or self.input_weights_i.shape != expected_shape:
            raise ValueError("Input-weight arrays must have one value per WC block.")
        self.coupling_graph = _normalized_coupling_graph(
            self.coupling_graph,
            self.parameters.num_blocks,
        )
        if self.coupling_strength > 0.0 and self.coupling_graph is None:
            raise ValueError("A positive coupling_strength requires a coupling_graph.")
        self.state_bounds = (lower, upper)
        self.initial_state = (float(self.initial_state[0]), float(self.initial_state[1]))
        self.reset()

    @property
    def num_blocks(self) -> int:
        return self.parameters.num_blocks

    @classmethod
    def from_fitted_parameters(
        cls,
        fitted_parameters: WilsonCowanParameters,
        *,
        num_blocks: int,
        sample_dt: float,
        parameter_jitter: float | Mapping[str, float] = 0.05,
        input_scale: float = 0.1,
        seed: int = 0,
        rk4_substeps: int = 1,
        initial_state: tuple[float, float] = (0.1, 0.1),
        preflight_duration_s: float = 0.25,
        max_resample_attempts: int = 100,
        state_bounds: tuple[float, float] = (-1.0e-6, 1.0 + 1.0e-6),
        coupling_graph: sparse.csr_matrix | np.ndarray | None = None,
        coupling_strength: float = 0.0,
    ) -> NonlinearWilsonCowanReservoir:
        if isinstance(num_blocks, bool) or not isinstance(num_blocks, (int, np.integer)):
            raise ValueError("num_blocks must be a positive integer.")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be a positive integer.")
        if not np.isfinite(input_scale) or input_scale < 0.0:
            raise ValueError("input_scale must be finite and non-negative.")
        if not np.isfinite(preflight_duration_s) or preflight_duration_s <= 0.0:
            raise ValueError("preflight_duration_s must be finite and positive.")
        if (
            isinstance(max_resample_attempts, bool)
            or not isinstance(max_resample_attempts, (int, np.integer))
            or max_resample_attempts < 1
        ):
            raise ValueError("max_resample_attempts must be a positive integer.")
        if (
            isinstance(rk4_substeps, bool)
            or not isinstance(rk4_substeps, (int, np.integer))
            or rk4_substeps < 1
        ):
            raise ValueError("rk4_substeps must be a positive integer.")
        if not np.isfinite(sample_dt) or sample_dt <= 0.0:
            raise ValueError("sample_dt must be finite and positive seconds.")
        if len(initial_state) != 2 or not np.all(np.isfinite(initial_state)):
            raise ValueError("initial_state must contain two finite values.")
        if len(state_bounds) != 2 or not np.all(np.isfinite(state_bounds)):
            raise ValueError("state_bounds must contain two finite values.")
        if float(state_bounds[0]) >= float(state_bounds[1]):
            raise ValueError("state_bounds must be ordered from lower to upper.")
        if not np.isfinite(coupling_strength) or coupling_strength < 0.0:
            raise ValueError("coupling_strength must be finite and non-negative.")

        _validate_center(fitted_parameters)
        jitter_scales = _parameter_jitter_scales(parameter_jitter)
        parameter_seed, input_seed = np.random.SeedSequence(seed).spawn(2)
        parameter_rng = np.random.default_rng(parameter_seed)
        input_rng = np.random.default_rng(input_seed)
        matrix, boundary_rejections = _sample_parameter_rows(
            fitted_parameters,
            num_blocks,
            jitter_scales,
            parameter_rng,
        )

        invalid_rejections = 0
        independent_minimum = float("nan")
        independent_maximum = float("nan")
        for attempt in range(max_resample_attempts):
            population = _population_from_matrix(
                matrix,
                fitted_parameters.sigmoid_gain,
                fitted_parameters.sigmoid_theta,
            )
            valid, independent_minimum, independent_maximum = _preflight_population(
                population,
                sample_dt=sample_dt,
                rk4_substeps=rk4_substeps,
                initial_state=initial_state,
                duration_s=preflight_duration_s,
                state_bounds=state_bounds,
            )
            if np.all(valid):
                break
            invalid_count = int(np.count_nonzero(~valid))
            invalid_rejections += invalid_count
            replacements, rejected = _sample_parameter_rows(
                fitted_parameters,
                invalid_count,
                jitter_scales,
                parameter_rng,
            )
            boundary_rejections += rejected
            matrix[~valid] = replacements
        else:
            raise RuntimeError(
                "Unable to construct a numerically bounded nonlinear WC parameter population "
                f"after {max_resample_attempts} preflight attempts."
            )

        normalized_graph = _normalized_coupling_graph(coupling_graph, num_blocks)
        coupled_valid, coupled_minimum, coupled_maximum = _preflight_population(
            population,
            sample_dt=sample_dt,
            rk4_substeps=rk4_substeps,
            initial_state=initial_state,
            duration_s=preflight_duration_s,
            state_bounds=state_bounds,
            coupling_graph=normalized_graph,
            coupling_strength=coupling_strength,
        )
        if not np.all(coupled_valid):
            raise RuntimeError(
                "The coupled nonlinear WC preflight left the configured state bounds; "
                "reduce coupling strength or increase rk4_substeps."
            )

        input_weights_e = input_scale * input_rng.standard_normal(num_blocks)
        input_weights_i = input_scale * input_rng.standard_normal(num_blocks)
        parameter_summary = {
            name: {
                "min": float(np.min(getattr(population, name))),
                "max": float(np.max(getattr(population, name))),
                "mean": float(np.mean(getattr(population, name))),
            }
            for name in PARAMETER_NAMES
        }
        diagnostics: dict[str, Any] = {
            "bounded": True,
            "preflight_duration_s": float(preflight_duration_s),
            "preflight_samples": max(1, int(np.ceil(preflight_duration_s / sample_dt))),
            "preflight_independent_state_min": independent_minimum,
            "preflight_independent_state_max": independent_maximum,
            "preflight_coupled_state_min": coupled_minimum,
            "preflight_coupled_state_max": coupled_maximum,
            "preflight_state_bounds": [float(state_bounds[0]), float(state_bounds[1])],
            "parameter_boundary_rejections": int(boundary_rejections),
            "parameter_preflight_rejections": int(invalid_rejections),
            "parameter_jitter": dict(jitter_scales),
            "parameter_summary": parameter_summary,
            "sample_interval_s": float(sample_dt),
            "rk4_substeps": int(rk4_substeps),
            "rk4_substep_s": float(sample_dt / rk4_substeps),
        }
        diagnostics.update(_graph_diagnostics(normalized_graph, num_blocks))
        return cls(
            parameters=population,
            input_weights_e=input_weights_e,
            input_weights_i=input_weights_i,
            sample_dt=float(sample_dt),
            rk4_substeps=int(rk4_substeps),
            initial_state=initial_state,
            state_bounds=state_bounds,
            coupling_graph=normalized_graph,
            coupling_strength=float(coupling_strength),
            diagnostics=diagnostics,
        )

    def reset(self) -> None:
        self.state = np.empty(2 * self.num_blocks, dtype=np.float64)
        self.state[0::2] = self.initial_state[0]
        self.state[1::2] = self.initial_state[1]

    def step(self, value: float) -> np.ndarray:
        scalar_value = float(value)
        if not np.isfinite(scalar_value):
            raise ValueError("Reservoir input must be finite.")
        e = self.state[0::2].copy()
        i = self.state[1::2].copy()
        external_e = self.input_weights_e * scalar_value
        external_i = self.input_weights_i * scalar_value
        substep_dt = self.sample_dt / self.rk4_substeps
        for _ in range(self.rk4_substeps):
            e, i = rk4_wilson_cowan_step(
                e,
                i,
                self.parameters,
                substep_dt,
                external_e=external_e,
                external_i=external_i,
                coupling_graph=self.coupling_graph,
                coupling_strength=self.coupling_strength,
            )
            lower, upper = self.state_bounds
            if (
                not np.all(np.isfinite(e))
                or not np.all(np.isfinite(i))
                or np.any(e < lower)
                or np.any(e > upper)
                or np.any(i < lower)
                or np.any(i > upper)
            ):
                raise FloatingPointError(
                    "Nonlinear Wilson-Cowan reservoir left its finite bounded state range."
                )
        self.state[0::2] = e
        self.state[1::2] = i
        return self.state.copy()

    def run(self, signal_values: np.ndarray) -> np.ndarray:
        values = np.asarray(signal_values, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("signal_values must be a finite one-dimensional array.")
        outputs = np.empty((values.size, self.state.size), dtype=np.float64)
        for index, value in enumerate(values):
            outputs[index] = self.step(float(value))
        return outputs
