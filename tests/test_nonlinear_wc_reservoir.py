from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

from physics_esn.models.nonlinear_wc_reservoir import (
    PARAMETER_NAMES,
    NonlinearWilsonCowanReservoir,
    WilsonCowanParameterPopulation,
    generate_sparse_coupling_graph,
    rk4_wilson_cowan_step,
)
from physics_esn.models.wilson_cowan import (
    WILSON_COWAN_PARAMETER_BOUNDS,
    WilsonCowanParameters,
    continuous_eigenvalues,
    find_equilibrium,
    jacobian_at_equilibrium,
    wilson_cowan_vector_field,
)


def _params() -> WilsonCowanParameters:
    return WilsonCowanParameters(
        tau_e=0.01,
        tau_i=0.02,
        w_ee=10.0,
        w_ei=12.0,
        w_ie=10.0,
        w_ii=0.0,
        p=0.5,
        q=0.0,
        sigmoid_gain=1.5,
        sigmoid_theta=2.5,
    )


def _reservoir(
    *,
    seed: int = 7,
    num_blocks: int = 6,
    coupling_graph: sparse.csr_matrix | None = None,
    coupling_strength: float = 0.0,
    parameter_jitter: float = 0.03,
    input_scale: float = 0.1,
) -> NonlinearWilsonCowanReservoir:
    return NonlinearWilsonCowanReservoir.from_fitted_parameters(
        _params(),
        num_blocks=num_blocks,
        sample_dt=0.004,
        parameter_jitter=parameter_jitter,
        input_scale=input_scale,
        seed=seed,
        rk4_substeps=2,
        preflight_duration_s=0.04,
        state_bounds=(-0.05, 1.05),
        coupling_graph=coupling_graph,
        coupling_strength=coupling_strength,
    )


def _copied_population(
    population: WilsonCowanParameterPopulation,
    **replacements: np.ndarray,
) -> WilsonCowanParameterPopulation:
    values = {
        name: np.array(replacements.get(name, getattr(population, name)), copy=True)
        for name in (*PARAMETER_NAMES, "sigmoid_gain", "sigmoid_theta")
    }
    return WilsonCowanParameterPopulation(**values)


def test_nonlinear_ensemble_dimension_seed_and_parameter_bounds() -> None:
    first = _reservoir(seed=12, num_blocks=8)
    second = _reservoir(seed=12, num_blocks=8)
    different = _reservoir(seed=13, num_blocks=8)

    assert first.state.shape == (16,)
    assert first.run(np.zeros(5)).shape == (5, 16)
    assert np.array_equal(first.parameters.parameter_matrix(), second.parameters.parameter_matrix())
    assert np.array_equal(first.input_weights_e, second.input_weights_e)
    assert np.array_equal(first.input_weights_i, second.input_weights_i)
    assert not np.array_equal(first.parameters.parameter_matrix(), different.parameters.parameter_matrix())
    for name in PARAMETER_NAMES:
        lower, upper = WILSON_COWAN_PARAMETER_BOUNDS[name]
        values = getattr(first.parameters, name)
        assert np.all((values >= lower) & (values <= upper))


def test_zero_input_is_finite_and_eeg_input_changes_state() -> None:
    reservoir = _reservoir(seed=3)
    zero_states = reservoir.run(np.zeros(40))
    reservoir.reset()
    driven_states = reservoir.run(np.ones(40))

    assert np.all(np.isfinite(zero_states))
    assert np.all((-0.05 <= zero_states) & (zero_states <= 1.05))
    assert not np.allclose(zero_states, driven_states)


def test_changing_one_uncoupled_block_parameter_is_block_local() -> None:
    baseline = _reservoir(
        seed=2,
        num_blocks=3,
        parameter_jitter=0.0,
        input_scale=0.0,
    )
    modified_w_ee = np.array(baseline.parameters.w_ee, copy=True)
    modified_w_ee[1] *= 1.05
    modified = NonlinearWilsonCowanReservoir(
        parameters=_copied_population(baseline.parameters, w_ee=modified_w_ee),
        input_weights_e=np.zeros(3),
        input_weights_i=np.zeros(3),
        sample_dt=baseline.sample_dt,
        rk4_substeps=baseline.rk4_substeps,
        initial_state=baseline.initial_state,
        state_bounds=baseline.state_bounds,
    )

    baseline_state = baseline.step(0.0)
    modified_state = modified.step(0.0)

    assert np.array_equal(baseline_state[[0, 1, 4, 5]], modified_state[[0, 1, 4, 5]])
    assert not np.array_equal(baseline_state[2:4], modified_state[2:4])


def test_sparse_graph_has_reproducible_fixed_connectivity_and_unit_rows() -> None:
    graph = generate_sparse_coupling_graph(num_blocks=20, degree=4, seed=21)
    repeated = generate_sparse_coupling_graph(num_blocks=20, degree=4, seed=21)
    changed = generate_sparse_coupling_graph(num_blocks=20, degree=4, seed=22)

    assert sparse.isspmatrix_csr(graph)
    assert graph.nnz == 80
    assert np.all(np.diff(graph.indptr) == 4)
    assert np.allclose(np.asarray(graph.sum(axis=1)).ravel(), 1.0)
    assert np.all(graph.diagonal() == 0.0)
    assert (graph != repeated).nnz == 0
    assert (graph != changed).nnz > 0


def test_graph_normalization_is_size_independent() -> None:
    small = generate_sparse_coupling_graph(num_blocks=10, degree=3, seed=1)
    large = generate_sparse_coupling_graph(num_blocks=100, degree=3, seed=1)

    assert np.allclose(np.asarray(small.sum(axis=1)).ravel(), 1.0)
    assert np.allclose(np.asarray(large.sum(axis=1)).ravel(), 1.0)
    assert np.allclose(small.data, 1.0 / 3.0)
    assert np.allclose(large.data, 1.0 / 3.0)


def test_zero_coupling_exactly_reproduces_independent_ensemble() -> None:
    graph = generate_sparse_coupling_graph(num_blocks=6, degree=2, seed=5)
    independent = _reservoir(seed=9, coupling_graph=None, coupling_strength=0.0)
    zero_coupled = _reservoir(seed=9, coupling_graph=graph, coupling_strength=0.0)
    signal_values = np.sin(np.linspace(0.0, 2.0 * np.pi, 60))

    assert np.array_equal(
        independent.parameters.parameter_matrix(),
        zero_coupled.parameters.parameter_matrix(),
    )
    assert np.array_equal(independent.input_weights_e, zero_coupled.input_weights_e)
    assert np.array_equal(independent.input_weights_i, zero_coupled.input_weights_i)
    assert np.array_equal(independent.run(signal_values), zero_coupled.run(signal_values))


def test_nonzero_coupling_changes_trajectory() -> None:
    graph = generate_sparse_coupling_graph(num_blocks=6, degree=2, seed=5)
    independent = _reservoir(seed=9, coupling_graph=None, coupling_strength=0.0)
    coupled = _reservoir(seed=9, coupling_graph=graph, coupling_strength=0.08)
    signal_values = np.zeros(60)

    assert np.array_equal(
        independent.parameters.parameter_matrix(),
        coupled.parameters.parameter_matrix(),
    )
    assert not np.allclose(independent.run(signal_values), coupled.run(signal_values))


def test_vectorized_rk4_matches_high_accuracy_reference() -> None:
    params = _params()
    population = WilsonCowanParameterPopulation(
        **{
            name: np.array([getattr(params, name)], dtype=np.float64)
            for name in PARAMETER_NAMES
        },
        sigmoid_gain=np.array([params.sigmoid_gain]),
        sigmoid_theta=np.array([params.sigmoid_theta]),
    )
    initial = np.array([0.17, 0.09])
    external_e = np.array([0.04])
    external_i = np.array([-0.02])
    dt = 0.0005
    next_e, next_i = rk4_wilson_cowan_step(
        initial[:1],
        initial[1:],
        population,
        dt,
        external_e=external_e,
        external_i=external_i,
    )

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        de_dt, di_dt = wilson_cowan_vector_field(
            state[0],
            state[1],
            params,
            external_e=float(external_e[0]),
            external_i=float(external_i[0]),
        )
        return np.array([de_dt, di_dt])

    reference = solve_ivp(
        rhs,
        (0.0, dt),
        initial,
        rtol=1.0e-12,
        atol=1.0e-14,
    ).y[:, -1]
    assert np.allclose(np.array([next_e[0], next_i[0]]), reference, atol=1.0e-10)


def test_bounded_nonlinear_preflight_does_not_require_linear_stability() -> None:
    hopf_side_parameters = WilsonCowanParameters(
        tau_e=0.03898646481723582,
        tau_i=0.04427439322441954,
        w_ee=8.505622052599906,
        w_ei=16.65161840034504,
        w_ie=17.588126998864123,
        w_ii=0.12468307361318542,
        p=2.796664841380178,
        q=-2.5364883326614382,
        sigmoid_gain=1.5,
        sigmoid_theta=2.5,
    )
    equilibrium = find_equilibrium(hopf_side_parameters)
    eigenvalues = continuous_eigenvalues(
        jacobian_at_equilibrium(equilibrium, hopf_side_parameters)
    )
    assert np.max(eigenvalues.real) > 0.0

    reservoir = NonlinearWilsonCowanReservoir.from_fitted_parameters(
        hopf_side_parameters,
        num_blocks=2,
        sample_dt=0.004,
        parameter_jitter=0.0,
        input_scale=0.1,
        seed=1,
        rk4_substeps=2,
        preflight_duration_s=0.5,
        state_bounds=(-0.05, 1.05),
    )
    assert reservoir.diagnostics["bounded"] is True
