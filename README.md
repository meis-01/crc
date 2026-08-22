## Reservoir modes

The four reservoir modes form a controlled progression from a highly constrained linear representation of the fitted Wilson-Cowan dynamics to a heterogeneous nonlinear interacting network. The purpose of this hierarchy is to isolate where predictive improvement comes from: **physiological time scales, diversity of time scales, nonlinear dynamics, or network interactions**.

### `deterministic_poles`

This is the most constrained reservoir and serves as the linear physics-informed baseline.

After fitting a Wilson-Cowan model, an equilibrium $x^*$ is identified and the Jacobian of the WC vector field is evaluated at that point. Its continuous-time eigenvalues $\lambda_k$ describe the local decay and oscillation modes of small perturbations around the equilibrium.

These continuous-time modes are converted to discrete-time poles at the EEG sampling interval $\Delta t$:

$$
\mu_k = \exp(\lambda_k\Delta t).
$$

The reservoir then evolves according to

$$
z_{t+1} = \mathrm{diag}(\mu_k)z_t + W_{\mathrm{in}}u_t.
$$

Thus, each reservoir state behaves approximately like a fixed linear filter whose memory and oscillatory behavior are determined by the fitted WC dynamics.

If

$$
\lambda_k = a_k + i b_k,
$$

then $a_k$ determines the decay time scale and $b_k$ determines the oscillation frequency. After discretization, the corresponding pole $\mu_k$ carries the same information into the sampled-time reservoir.

This mode therefore asks:

> **Are the local time scales implied by a fitted Wilson-Cowan model already useful for predicting EEG?**

It is important that this model is not interpreted as a simulation of the full Wilson-Cowan equations. It only retains the dynamics obtained by linearizing the system around one equilibrium.

---

### `distributed_poles`

The deterministic model provides only the small number of modes associated with one fitted equilibrium. The distributed-pole reservoir introduces **heterogeneity in those time scales** while remaining completely linear.

Instead of using the fitted eigenvalues directly, a population of nearby continuous-time eigenvalues is sampled:

$$
\widetilde{\lambda}_j =
(\mathrm{Re},\lambda_k+\epsilon_r)
+
i(\mathrm{Im},\lambda_k+\epsilon_i),
$$

where

$$
\epsilon_r \sim \mathcal{N}(0,\sigma_r^2),
\qquad
\epsilon_i \sim \mathcal{N}(0,\sigma_i^2).
$$

Each sampled eigenvalue is then discretized:

$$
\widetilde{\mu}_j =
\exp(\widetilde{\lambda}_j\Delta t).
$$

Instead of representing only one fitted decay rate and oscillation frequency, the reservoir therefore contains a **bank of nearby dynamical modes**.

Some modes decay slightly faster or slower, while others oscillate at slightly different frequencies. The ridge readout can combine these fixed responses to construct a richer temporal representation of the EEG.

Sampling is performed in continuous-time eigenvalue space because the quantities being perturbed then have a direct dynamical interpretation. Conjugate pairs are preserved so that oscillatory modes remain physically consistent, and only stable samples satisfying

$$
\mathrm{Re},\widetilde{\lambda}_j < 0
$$

or equivalently

$$
|\widetilde{\mu}_j| < 1
$$

are retained.

This mode asks:

> **Is a heterogeneous population of physiologically motivated time scales more useful than the small deterministic set obtained from one fitted equilibrium?**

The important point is that this reservoir is still linear. Any gain relative to `deterministic_poles` comes from **dynamical diversity**, not from nonlinear computation.

---

### `independent_nonlinear_wc`

This mode makes the major transition from a linear filter bank to a true nonlinear dynamical reservoir.

Instead of keeping only the Jacobian eigenvalues, it retains the complete Wilson-Cowan equations. The reservoir contains $K$ independent excitatory-inhibitory WC systems:

$$
\tau_E^{(k)}\dot E_k =
-E_k +
S!\left(
w_{EE}^{(k)}E_k
---------------

w_{EI}^{(k)}I_k
+
P_k
+
B_{E,k}u_t
\right),
$$

$$
\tau_I^{(k)}\dot I_k =
-I_k +
S!\left(
w_{IE}^{(k)}E_k
---------------

w_{II}^{(k)}I_k
+
Q_k
+
B_{I,k}u_t
\right).
$$

Each block contains an excitatory state $E_k$ and an inhibitory state $I_k$. The observed EEG sample $u_t$ acts as an external current driving the system; the EEG itself is therefore **not identified with either $E$ or $I$**.

The fitted WC parameter vector acts as the center of the population. Individual blocks receive small fixed parameter perturbations, so the ensemble contains slightly different nonlinear systems rather than $K$ identical copies.

The input projections $B_{E,k}$ and $B_{I,k}$ are also fixed random coefficients. Consequently, different WC blocks respond differently to the same EEG input.

The reservoir state presented to the readout is

$$
h_t =
[E_1,I_1,E_2,I_2,\ldots,E_K,I_K],
$$

with dimension $2K$.

Unlike the pole-based reservoirs, these states are not fixed linear convolutions of the input. Their response depends on their current state through the WC sigmoid. This introduces effects such as saturation, state-dependent gain, nonlinear mixing, and potentially oscillatory behavior that cannot be represented by the linearized model.

The blocks remain mutually independent:

$$
(E_k,I_k)
\not\rightarrow
(E_j,I_j),
\qquad k\neq j.
$$

All interaction between blocks occurs only indirectly when the final ridge regression combines their states.

This mode therefore asks:

> **Does retaining the full nonlinear Wilson-Cowan vector field provide predictive information that cannot be obtained from a sufficiently rich linear population of WC-derived time scales?**

The comparison

`distributed_poles` $\rightarrow$ `independent_nonlinear_wc`

is consequently the key experiment for testing the value of **nonlinearity itself**.

---

### `coupled_nonlinear_wc`

The final mode adds interactions between the nonlinear Wilson-Cowan blocks.

The local WC dynamics remain unchanged, but the excitatory populations are connected through a fixed sparse graph:

$$
\tau_E^{(k)}\dot E_k =
-E_k +
S!\left(
w_{EE}^{(k)}E_k
---------------

w_{EI}^{(k)}I_k
+
P_k
+
B_{E,k}u_t
+
\gamma\sum_j A_{kj}E_j
\right).
$$

Here $A_{kj}$ describes a directed connection from excitatory population $j$ to population $k$, while $\gamma$ controls the overall coupling strength.

The additional term

$$
\gamma\sum_j A_{kj}E_j
$$

means that the state of one WC block can now alter the future trajectory of another block.

This changes the reservoir qualitatively. The independent ensemble provides a collection of parallel nonlinear responses to the same signal, whereas the coupled reservoir can generate **collective dynamics** through recurrent interactions between those responses.

The graph is sparse, has no self-connections, and is row-normalized. Row normalization prevents the effective coupling magnitude from automatically increasing when the reservoir size or graph degree changes.

Only excitatory populations are coupled in the current implementation; inhibitory dynamics remain local.

The coupling strength provides a direct bridge to the independent model:

$$
\gamma = 0
\quad\Longrightarrow\quad
\text{independent nonlinear WC ensemble}.
$$

Therefore, using the same reservoir seed, the comparison

`independent_nonlinear_wc` $\rightarrow$ `coupled_nonlinear_wc`

isolates the effect of **network topology and cross-block recurrence**.

This mode asks:

> **Does interaction between heterogeneous nonlinear WC populations create useful temporal representations beyond those obtained from independent nonlinear populations?**

---

### Interpretation of the four-mode ablation

The hierarchy can be summarized as:

| Mode                       | WC time scales | Heterogeneity | Full WC nonlinearity | Cross-block recurrence |
| -------------------------- | -------------: | ------------: | -------------------: | ---------------------: |
| `deterministic_poles`      |              ✓ |             — |                    — |                      — |
| `distributed_poles`        |              ✓ |             ✓ |                    — |                      — |
| `independent_nonlinear_wc` |              ✓ |             ✓ |                    ✓ |                      — |
| `coupled_nonlinear_wc`     |              ✓ |             ✓ |                    ✓ |                      ✓ |

This makes the experiment an incremental ablation:

$$
\text{deterministic poles}
\rightarrow
\text{distributed poles}
\rightarrow
\text{independent nonlinear WC}
\rightarrow
\text{coupled nonlinear WC}.
$$

Each transition introduces one major source of representational capacity:

1. **Deterministic → distributed:** diversity of decay rates and oscillatory frequencies.
2. **Distributed → independent nonlinear:** nonlinear state-dependent dynamics.
3. **Independent → coupled nonlinear:** interaction and collective recurrent dynamics.

Because the reservoir dynamics remain fixed and only the ridge readout is trained, improvements across this hierarchy can be attributed more directly to the reservoir representation rather than to additional learned recurrent parameters.
