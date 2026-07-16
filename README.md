# Compressed Traffic Assignment with Augmented Lagrangian Method

This repository contains the complete implementation for the submitted manuscript: ***["Compressed Traffic Assignment with Augmented Lagrangian Method"](https://arxiv.org/abs/2604.23101)***.

This work proposes a compressed representation of the path-flow space using low-rank Singular Value Decomposition (SVD), effectively projecting the high-dimensional path-flow variables onto a compact subspace. It separates paths into major
paths (retained explicitly) and minor paths (captured implicitly via the low-rank basis),
drastically reducing the number of decision variables.

The resulting compressed problem is solved using an Augmented Lagrangian Method (ALM), which
handles OD flow conservation constraints and non-negativity of compressed path flows as soft
penalties. The dual multipliers are updated iteratively in an outer loop, while each inner
subproblem is solved efficiently with L-BFGS-B — a quasi-Newton method well-suited for
large-scale bound-constrained optimization.

## Gradient Computation: Paper vs. Implementation

### Augmented Lagrangian (eq. 3)

$$
L_c(y,z,\lambda,\mu) = \hat{f}(y,z) + \lambda'(A_1 y + Mz - d) + \frac{c_1}{2}\lVert A_1 y + Mz - d \rVert^2 + \frac{1}{2c_2}\sum_{i=1}^{n-s}\left[( \max\{0,\mu_i - c_2[U_r z]_i\})^2 - \mu_i^2 \right].
$$

### Gradients from the Paper (eq. 4)

$$
\nabla_y L_c = \nabla_y \hat{f}(y,z) + A_1'\left(\lambda + c_1(A_1 y + Mz - d)\right)
$$

$$
\nabla_z L_c = \nabla_z \hat{f}(y,z) + M'\left(\lambda + c_1(A_1 y + Mz - d)\right) - U_r'\left(\mu + c_2\, h^+(z,\mu,c_2)\right)
$$

where $h_i^+(z,\mu,c_2) = \max\left(-[U_r z]_i, -\mu_i/c_2\right)$.

### Gradients from the Implementation

Given that $g_v = \nabla_v f(v)$ and $v_0$ is constant, expanding $\nabla \hat{f}$ via the chain rule on $v = v_0 + B_1' y + Dz$ leads to:

$$
\nabla_y \hat{f} = B_1 g_v, \qquad \nabla_z \hat{f} = D' g_v.
$$

This gives the fully expanded gradients:

$$
\nabla_y L_c = B_1\, g_v + A_1'(\lambda + c_1\,\delta), \qquad \delta = A_1 y + Mz - d,
$$

$$
\nabla_z L_c = D' g_v + M'(\lambda + c_1\,\delta) - U_r'\,\phi, \qquad \phi = \max(\mathbf{0}, \mu - c_2 u), u = U_r z.
$$

Expanding $-U_r'(\mu + c_2 h^+)$ element-wise:

$$
\mu_i + c_2 h_i^+
= \mu_i + c_2 \max\left(-u_i,-\tfrac{\mu_i}{c_2}\right)
= \max(\mu_i - c_2 u_i, 0)
= \phi_i.
$$

Therefore $-U_r'(\mu + c_2 h^+) \equiv -U_r' \phi$.

### Code mapping (`objective_and_gradient_direct`)

| Math expression | Python code |
| --- | --- |
| $v = v_0 + B_1' y + Dz$ | `v = v0 + B1.T @ y + D @ z` |
| $u = U_r z$ | `u = U_r @ z` |
| $\delta = A_1 y + Mz - d$ | `od_error = A1 @ y + M @ z - d_multi` |
| $\phi = \max(0, \mu - c_2 u)$ | `max_term_minor = np.maximum(0, mu - c2 * u)` |
| $g_v = \nabla_v f(v)$ | `grad_v = bpr_gradient(v, capacity, t_0, alpha, beta)` |
| $\nabla_y L_c = B_1 g_v + A_1'(\lambda + c_1\delta)$ | `grad_y = B1 @ grad_v + A1.T @ (lambda_od + c1 * od_error)` |
| $\nabla_z L_c = D' g_v + M'(\lambda + c_1\delta) - U_r'\phi$ | `grad_z = D.T @ grad_v + M.T @ (lambda_od + c1 * od_error) - U_r.T @ max_term_minor` |

The implementation is fully consistent with the paper. Besides, the code also includes three other gradient implementations for testing and benchmarking purposes. Please refer to ***[E. Gradients Computation and Overhead](https://arxiv.org/abs/2604.23101)*** for complexity analysis and performance comparison.

1. `objective_and_gradient_chain_rule`
2. `objective_and_gradient_factored`
3. `objective_and_gradient_mixed`

---

## How to Run

### Data

Place GMNS-format input files under `data/<network>/`:

| File | Contents |
| --- | --- |
| `link.csv` | Link attributes: capacity, length, free speed, reference link flow |
| `node.csv` | Node coordinates and associated zones |
| `demand.csv` | OD demand matrix (optional; inferred from path flows (columns.csv) if absent) |
| `columns.csv` | Route assignment generated from the reference user-equilibrium solver [(OpenDTA)](https://github.com/jdlph/OpenDTA) |

Two sample networks are included at `data/`, namely `two_corridor` and `chicago_sketch`. More networks will be added in the future.

### Parameters

Key runtime parameters are configured in `config.json`.

| Parameter | Default | Description |
| --- | --- | --- |
| `data_dir` | `"two_corridor"` | Network folder under `data/` |
| `rank` | `50` | SVD rank $r$ for minor-path compression |
| `tolerance` | `1e-4` | ALM convergence tolerance |
| `threshold_index` | `0` | Index into thresholds returned by `setup_thresholds` |
| `demand_file` | `null` | Optional demand file path; auto-detected if omitted |
| `link_perf_file` | `null` | Optional link performance file path; auto-detected if omitted |
| `alm.c1_init` | `1e3` | Initial penalty for OD conservation |
| `alm.c2_init` | `1e3` | Initial penalty for minor non-negativity |
| `alm.beta_penalty` | `4.0` | Penalty growth factor |
| `alm.max_outer_iter` | `20` | Maximum ALM outer iterations |
| `alm.max_inner_iter` | `200` | Maximum L-BFGS-B inner iterations per outer step |
| `alm.stagnation_tol` | `1e-6` | Relative stagnation tolerance |
| `alm.stagnation_window` | `3` | Number of outer iterations for stagnation detection |

### Dependencies

The code is implemented in Python, and requires the following libraries.

```text
numpy
scipy
pandas
scikit-learn
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

### Running

Default mode loads `config.json` from the repository root:

```bash
python compressed_tap.py
```

Use a custom config file:

```bash
python compressed_tap.py --config path/to/your_config.json
```

To run a different setup, edit `config.json` (or pass a different config file path).

Results are printed to the console including link volume accuracy ($R^2$), OD
conservation violation, minor path flow non-negativity violation, and Reference Objective Difference (%) against the reference solution.
