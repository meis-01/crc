# Wilson-Cowan EEG Proof of Concept

The first model is

```text
tau_E dE/dt = -E + S(w_EE E - w_EI I + P)
tau_I dI/dt = -I + S(w_IE E - w_II I + Q)
```

## Scope and invariants

- Use the two-population Wilson-Cowan excitatory/inhibitory model only.
- Use channel `Oz`; do not build a multichannel or multi-region model yet.
- Read an existing local EDF dataset from the YAML `data_root`; never download data.
- Discover `sub-001` through `sub-010` and keep every subject's signals, splits,
  fitted parameters, eigenvalues, and outputs separate.
- Inspect each raw EDF before optional resampling or preprocessing.
- Fit and validate the complete pipeline on `sub-001` before running any fitting
  optimization across all ten subjects.

## Milestones

1. `sub-001` raw inspection -> `Oz` -> preprocessing -> PSD -> Wilson-Cowan simulation.
2. Fit Wilson-Cowan parameters to the training portion of `sub-001` spectral dynamics.
3. Find an equilibrium, calculate its 2x2 Jacobian and continuous eigenvalues
   `lambda`, map them with `mu = exp(lambda * dt)`, and construct the complex reservoir.
4. Perform a chronological, subject-local split and evaluate one-step-ahead `Oz`
   prediction for `sub-001`.
5. Reuse the same subject-isolated pipeline for `sub-001` through `sub-010`, saving
   one output directory per subject. This milestone remains deferred until milestones
   1-4 have been verified and must not launch a ten-subject optimization implicitly.
