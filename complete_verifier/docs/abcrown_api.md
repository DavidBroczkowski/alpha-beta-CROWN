# alpha-beta-CROWN — high-level API overview

The high-level API turns formal verification into a tiny Python surface. Provide a model (PyTorch `nn.Module` or ONNX path), describe the input/output region with `IOConstraints` written in plain Python expressions, optionally tweak the config, then call one of four methods:

| Method | Use it for |
|---|---|
| `solver.verify(...)`              | prove an output constraint holds on the whole input region |
| `solver.compute_bounds(...)`      | get **sound + tight** numeric bounds for output objectives |
| `solver.minimize(...)`            | find a feasible primal optimum that **minimizes** a scalar objective |
| `solver.maximize(...)`            | find a feasible primal optimum that **maximizes** a scalar objective |

`verify` returns `safe` / `unsafe-pgd` / `unsafe-bab` / `safe-incomplete` / `unknown`; the other three return their numerical result with a status field.

The notebook demos under `complete_verifier/examples_abcrown` are runnable Colab examples of every method, with end-to-end visualizations.

---

## 1. Quick start
To install `alpha-beta-CROWN`, follow these steps:

1. Clone the repository:
```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
```

2. (Optional) Enable CPLEX Cuts: If you require CPLEX cuts for verification (e.g. GCP-CROWN), you must manually install CPLEX and compile the `get_cuts` executable now. See the instructions in [`complete_verifier/cuts/CPLEX_cuts/README.md`](../cuts/CPLEX_cuts/README.md).

3. Install the package:
```bash
pip install .
```

Then the API can be imported and called via

```python
from abcrown import (
    ABCrownSolver, IOConstraints, ConfigBuilder, input_vars, output_vars
)

# Construct constraints with L-infinity box and logit ordering
x = input_vars((1, 28, 28))
y = output_vars(3)
input_constraint = (x >= base - eps) & (x <= base + eps)
output_constraint = (y[0] > y[1]) & (y[0] > y[2])

constraints = IOConstraints(
    input_vars=x,
    output_vars=y,
    input_constraint=input_constraint,
    output_constraint=output_constraint,
)
# Config (defaults)
config = ConfigBuilder.from_defaults()

# Verify
solver = ABCrownSolver(model, x, y, config=config)
result = solver.verify(constraints=constraints)
print(result.status, result.success)
```

---

## 2. Function reference

### `input_vars(shape)` / `output_vars(dim)`
Creates symbolic `VariableVector` objects used by the expression Domain-Specific Language (DSL).
- `shape`: int / tuple / `torch.Size` describing the input tensor.
- `dim`: integer length of the output tensor.

### `IOConstraints(...)`
Recommended high-level constraint entry point.

Single entry point that dispatches automatically:
- Bounds + clauses: pass `lower`, `upper`, `clauses`.
- Expression DSL: pass `input_vars`, optional `output_vars`, and boolean combinations of
  comparisons (see cheat sheet below).
- `force_simplify`: optional bool to override DNF simplification (`True` forces simplify, `False` skips; default auto for <=10 outputs).
- `vnnlib_path="path/to/spec.vnnlib"`: load an existing VNNLIB property.
- For output constraints, only **strict output specification** is accepted, i.e., **<, >** are allowed while **>=, <=** are not.

Internally constraints are converted and stored in an OR-of-AND list of `(C, rhs)` inequalities with
`C @ output < rhs` (Disjunction Normal Form), but you don't need to manually build the DNF.

### `VerificationSpec.build_spec(...)`
Still supported for backward compatibility. New code should prefer `IOConstraints(...)`.

### DSL cheat sheet

| Goal               | DSL example                                             | Notes                                         |
|-------------------|----------------------------------------------------------|-----------------------------------------------|
| 1-D bound         | `(x[0] > -0.1) & (x[0] < 0.1)`                           | combine predicates with `&`                   |
| Tensor L∞ box     | `(x > lower_tensor) & (x < upper_tensor)`                | tensors broadcast to the symbolic shape       |
| Logit ordering    | `(y[0] > y[1]) & (y[0] > y[2])`                          | indexing syntax mirrors PyTorch               |
| Linear inequality | `y[0] - y[1] > 0`                                        | auto-converted to `(C, rhs)`                  |
| OR between specs  | `(y[0] > 0) \| (y[1] > 0)`                               | produces two OR clauses internally            |
| Pin to a point    | `point = torch.tensor([1.0]); (x > point) & (x < point)` | handy for single-sample demos                 |

Comparisons must involve `input_vars` or `output_vars`. The opposite operand can be a
scalar, list, NumPy array, or torch tensor.

### `default_config()` / `ConfigBuilder`
- `default_config()` returns a deep copy of the built-in dict. See `complete_verifier/arguments.py` for the default values of all configuration options. **NOTE:** `"complete_verifier"` defaults to `"auto"` when the verifier is invoked through this API. 
- `ConfigBuilder.from_defaults()` provides a chainable helper:
  ```python
  cfg = (
      ConfigBuilder.from_defaults()
      .set("general/device", "cpu")
      .set("attack/pgd_order", "skip")
  )
  ```
- `.update()` merges nested dicts, `.from_yaml(path)` loads overrides from YAML,
  `.from_config(cfg)` clones an existing configuration.
- `.set(path, value)` is the preferred API for one config path at a time; use
  `.update()` when you need to inject nested dicts or callables:
  ```python
  builder = ConfigBuilder.from_defaults()
  builder.set("attack/pgd_order", "skip")            # preferred path-style override
  builder.update({"attack": {"pgd_order": "skip"}})  # deep-merge mapping
  ```

### `ABCrownSolver`
Recommended constructor:
- `ABCrownSolver(computing_graph, input_vars, output_vars, config=None, name=None)`

Compatibility forms are still accepted:
- `ABCrownSolver(constraint, computing_graph, config=None, name=None)`
- `ABCrownSolver(..., constraint=...)`
- `spec=` alias for `constraint=` (legacy)

`verify(constraints=None, interm_bounds=None, return_reference=True)` returns `SolveResult` with:
- `status`: `safe`, `unsafe-pgd`, `unsafe-bab`, `safe-incomplete`, `unknown`, …
- `success`: boolean (`True` if the property is satisfied, or a counterexample is
  confirmed when unsafety is expected).
- `reference`: optional dict of intermediate data (bounds, attack traces, etc.).
- `stats`: metadata such as elapsed time, PGD iterations, BaB splits.

`compute_bounds(constraints=None, objective=None, interm_bounds=None, return_linear_bounds=False)`
returns `BoundsResult` with:
- `lower`, `upper` — sound + tight lower/upper bound tensors for the objective expression(s).
  When you pass `K` objectives the result is a length-`K` tensor; when you pass a single
  `LinearExpr` you get a length-1 tensor.
- `success` — `False` if any bound could not be extracted before verifier termination.
- `stats` — metadata (per-objective statuses, source input indices, the effective
  BaB timeout, and the sort-domain interval).
- `linear_bounds` — `None` by default. When `return_linear_bounds=True`, contains
  a structured affine relaxation object with keys `lower_A`, `lower_bias`,
  `upper_A`, `upper_bias`, and `subdomains`. The first four entries describe
  the global input-side CROWN affine relaxation. `subdomains` contains the
  current per-subdomain affine records when input-BaB is active.

**Soundness + tightness guarantee.** `compute_bounds` lays each objective out as an
independent input batch and uses a per-c-group hook around the single parallel BaB run
to expose a per-objective tightened ``lb - rhs`` back to the API. The returned bound is
guaranteed to satisfy `lb_i <= min_x obj_i(x)` and `ub_i >= max_x obj_i(x)` for every
`x` in the input region (soundness). It is also as tight as BaB's current frontier
allows (tightness — verified by the regression test
``test_compute_bounds_sound_multi_objective`` in `examples_abcrown/api_tests.py`,
which Monte-Carlo + corner-samples the model output and refuses to certify any
bound that's punctured by a sampled value).

**Default-tuning that `compute_bounds` applies internally.** PGD attack is skipped,
BaB attack is disabled, domain sorting is enabled when the config leaves it off
(`bab.sort_domain_interval <= 0`), and a 30-second per-subproblem BaB timeout is
used when the config still has the project-wide default of 360 seconds. If you
explicitly set `bab/timeout` or `bab/override_timeout`, your setting is preserved.

`minimize(objective=None, constraints=None, return_bound_history=False)` /
`maximize(objective=None, constraints=None, return_bound_history=False)`
return `OptimizeResult` with `status`, `success`, `primal_value`, `x_best`, `solver`,
`stats`, and optionally `bound_history`. By default they drive the local PGD-based
optimizer. When `return_bound_history=True`, the API also runs an input-BaB refinement
pass and fills `result.bound_history` with returned `primal_values` and `dual_bounds`
so you can plot convergence directly from the solver output.

`SolveResult` (returned by `verify`) exposes `status`, `success`, `reference`, and
`stats` as attributes.

---

## 3. Typical workflow recap

1. Declare symbols: `x = input_vars(shape)`, `y = output_vars(dim)`.
2. Write input/output constraints using the DSL.
3. Build constraints via `IOConstraints(...)`.
4. (Optional) tweak a config with `ConfigBuilder` or `default_config()`.
5. Instantiate `ABCrownSolver(model, x, y, config)` and call one of:
   - `.verify(constraints=...)`
   - `.compute_bounds(constraints=..., objective=...)`
  - `.minimize(objective=..., constraints=..., return_bound_history=False)`
  - `.maximize(objective=..., constraints=..., return_bound_history=False)`

`result.status` indicates which stage produced the answer for verification:
- `unsafe-pgd`: PGD attack already found a counterexample.
- `unsafe-bab`: branching-and-bound found a counterexample.
- `safe`: property proven.
- `safe-incomplete`: incomplete verification sufficed.
- `unknown`: search stopped early (timeout / resource limit).

---

## 4. Reference snippets

All four snippets use the same five lines of plumbing: build constraints with
`IOConstraints`, construct `ABCrownSolver(model, x, y)`, call one method.

### Image classification — logit ordering
```python
x = input_vars((1, IMG_H, IMG_W))
y = output_vars(NUM_CLASSES)
input_constraint  = (x >= base - eps) & (x <= base + eps)
output_constraint = y[label] > y[other_0]
for i in other_classes:
    output_constraint &= (y[label] > y[i])
constraints = IOConstraints(
    input_vars=x, output_vars=y,
    input_constraint=input_constraint,
    output_constraint=output_constraint,
)
result = ABCrownSolver(model, x, y).verify(constraints=constraints)
print(result.status, result.success)
```

### Lyapunov controller check
```python
x = input_vars(2)
y = output_vars(2)                         # y[0] = V(x), y[1] = V_dot(x)
constraints = IOConstraints(
    input_vars=x, output_vars=y,
    input_constraint=(x >= [-4.8, -10.8]) & (x <= [4.8, 10.8]),
    output_constraint=(y[0] < v_min) | (y[0] > v_max) | (y[1] < 0.0),  # falsification predicate
)

# Lyapunov graphs that produce V_dot via Jacobian need one config knob.
cfg = ConfigBuilder.from_defaults().set("model/with_jacobian", True)
result = ABCrownSolver(model, x, y, config=cfg).verify(constraints=constraints)
print(result.status, result.success)
```

### compute_bounds — sound + tight range of every objective
```python
class ReachabilityGraph(torch.nn.Module):
    def forward(self, x):
        y0 = x[:, 0] + 0.5 * x[:, 1]
        y1 = torch.relu(x[:, 2]) - x[:, 3]
        y2 = x[:, :4].sum(dim=1)
        return torch.stack([y0, y1, y2], dim=1)

x = input_vars(4)
y = output_vars(3)
constraints = IOConstraints(
    input_vars=x,
    input_constraint=(x >= [-1.0, -0.5, -0.25, -0.75])
                     & (x <= [ 0.8,  0.6,  0.9 ,  1.1 ]),
)
result = ABCrownSolver(ReachabilityGraph(), x, y).compute_bounds(
    constraints=constraints,
    objective=[y[0], y[1] - y[2], 0.5 * y[0] + y[2]],
    return_linear_bounds=True,
)
print(result.lower)            # length-3 tensor of certified lower bounds
print(result.upper)             # length-3 tensor of certified upper bounds
print(result.linear_bounds)    # dict with lower_A/lower_bias/upper_A/upper_bias/subdomains
```

### Optimization toy — minimize / maximize
For a scalar control input, the optimization problem can be written as

$$
\min_{u \in [-1, 1]^6} \; y_0(u) + 0.2 y_1(u),
\qquad
\max_{u \in [-1, 1]^6} \; y_2(u) - 0.1 y_1(u).
$$

```python
u = input_vars(6); y = output_vars(3)
constraints = IOConstraints(
    input_vars=u,
    input_constraint=(u >= -1.0) & (u <= 1.0),
)
solver = ABCrownSolver(model, u, y)

min_r = solver.minimize(
  objective=y[0] + 0.2 * y[1],
  constraints=constraints,
  return_bound_history=True,
)
max_r = solver.maximize(objective=y[2] - 0.1 * y[1], constraints=constraints)
print(min_r.primal_value, min_r.x_best)
print(min_r.bound_history)
print(max_r.primal_value, max_r.x_best)
```

Swap in your own model — the plumbing stays identical.

## 5. Examples and Colab demos

End-to-end runnable scripts and a richly-visualized Colab notebook live in
`complete_verifier/examples_abcrown/`:

| File | Demo |
|---|---|
| `demo.ipynb` | **Colab walk-through** of all 5 workflows, with matplotlib / plotly visualizations |
| `image_classification_example.py` | tiny CNN, L∞ robustness check on an 8×8 image |
| `neural_lyapunov_example.py` | Van der Pol controller + Lyapunov NN, with state-box verification |
| `compute_bounds_example.py` | residual MLP, sound + tight bounds for 5 output objectives |
| `optimization_example.py` | `minimize` / `maximize` on a small tanh-ReLU "process control" graph |
| `vnnlib_example.py` | load a VNNLIB property + ONNX model and verify it directly |
| `api_tests.py` | smoke tests including the soundness regression for `compute_bounds` |
