# du

Numerical study of two-point correlators in **perturbed dual-unitary brickwork
circuits**, plus benchmarks of two error-mitigation techniques (ZNE and TEM)
under noise models calibrated to `ibm_miami`.

The codebase is organised around four threads.

## Threads

**1. Physics foundation** — `src/du/simulation.py`

DU gate, perturbation, Choi-trick 2N-qubit brickwork, light-cone correlators
via exact statevector and AER MPS. ZNE-style local gate folding lives here too
since it is a circuit transformation.

**2. Error mitigation** — `src/du/tem.py`, `tem_mpo.py`, `tem_julia.py`
(`tem_propagate.jl`)

The TEM identity `M(O) = (Φ*)⁻¹ ∘ Φ_id*(O)` implemented three ways:

- `tem.py` — dense `SparsePauliOp` Heisenberg sweep (reference; small N).
- `tem_mpo.py` — Pauli-basis MPO with SVD bond truncation.
- `tem_julia.py` + `tem_propagate.jl` — PauliPropagation.jl backend (sparse
  Pauli propagation with weight + coefficient truncation).

`gamma_diagnostics` (γ₁, γ₂², #terms, max Pauli weight) drives shot-budget
forecasting.

**3. Noise modeling** — `src/du/noise.py`

Equal-rate Pauli-Lindblad ansatz (λ = ε / 12) and matched depolarizing rate
derived from `FakeMiami` calibration. Builds an AER `NoiseModel` ready to feed
the density-matrix backend.

**4. Hardware integration** — `experiments/learn_noise_miami.py`

Submits a `NoiseLearner` job on `ibm_miami` to fit a sparse Pauli-Lindblad
generator dict per brickwork-offset layer; pickles to `data/`. Greedy
longest-path layout picker on the device coupling map.

## Layout

```
src/du/        library (importable)
experiments/        runnable analyses
data/               generated artifacts (CSVs, pickled noise results)
tests/              pytest-style smoke tests
utils/              one-off credential / access helpers
```

## Typical pipeline

```
python experiments/eps_sweep.py            # verify ε² off-cone leakage
python experiments/noise_zne.py            # ZNE benchmark under matched depolarizing noise
python -m du.tem                      # dense TEM demo
python -m du.noise                    # FakeMiami → TEM under PL noise
python experiments/learn_noise_miami.py    # real device noise learning (queue + cost)
```
