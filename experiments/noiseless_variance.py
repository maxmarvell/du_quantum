"""Noiseless variance of the fixed all-X-basis estimator.

All XX-type correlators commute, so a single measurement setting -- rotate
every qubit into X and read Z -- estimates every <X_c X_t> along the light
cone at once. Because the setting is fixed, the whole run is ONE circuit
sampled n_shots times (Aer evolves once, samples n_shots), so it is
memory-light and fast even at larger T.

The single-shot sigma (std * sqrt(n), flat across shot counts) should match
sqrt(1 - <XX>^2): ~1 in the light-cone interior and exactly 0 at the edge,
where <XX> = 1 is deterministic.
"""

from __future__ import annotations

import numpy as np

from qiskit.quantum_info import Operator

from du.simulation import (
    build_circuit,
    expect_x,
    get_target_qubits,
    get_control_qubit,
    kicked_ising_gate,
    is_dual_unitary,
    x_basis_measurement,
)


T = 3
MIN_X = 1
ZZ_COUPLING = np.pi / 4
TRANSVERSE_FIELD = np.pi / 4
LONGITUDINAL_FIELD = 0
N_SHOTS = [500, 1000, 2000]
N_REPS = 20
SEED = 0


def main() -> None:

    gate = kicked_ising_gate(LONGITUDINAL_FIELD, TRANSVERSE_FIELD)
    assert is_dual_unitary(Operator(gate).data), (
        f"gate is not dual-unitary (h={LONGITUDINAL_FIELD}, b={TRANSVERSE_FIELD}); "
        f"need transverse field b = pi/4"
    )

    qc = build_circuit(T, MIN_X, h=LONGITUDINAL_FIELD, b=TRANSVERSE_FIELD)

    targets = get_target_qubits(T, MIN_X)
    control = get_control_qubit(T, MIN_X)

    x_range = [round(float(x), 1) for x in np.arange(MIN_X, T + 0.5, 0.5)]
    print(
        f"=== Noiseless X-basis variance ===\n"
        f"  t                : {T}\n"
        f"  x range          : {x_range}\n"
        f"  longitudinal (h) : {LONGITUDINAL_FIELD}\n"
        f"  transverse   (b) : {TRANSVERSE_FIELD}\n"
        f"  qubits           : {qc.num_qubits}\n"
        f"  control qubit    : {control}\n"
        f"  target qubits    : {targets}\n"
    )

    xx_corr = np.full((len(N_SHOTS), N_REPS, len(targets)), np.nan)

    for i, n in enumerate(N_SHOTS):
        print(f"[shot-count {i + 1}/{len(N_SHOTS)}] n_shots={n}: ", end="", flush=True)
        for j in range(N_REPS):
            outcomes = x_basis_measurement(
                qc, n_shots=n, backend=None, seed=SEED + i * N_REPS + j
            )
            for k, qt in enumerate(targets):
                xx_corr[i, j, k] = expect_x(outcomes, np.array([control, qt]))
            print(f"{j + 1}", end=" ", flush=True)  # rep progress
        print("done", flush=True)

    xx_mean = xx_corr.mean(axis=1)
    xx_std = xx_corr.std(axis=1)

    for i, n in enumerate(N_SHOTS):
        print(f"\n--- n_shots = {n}   (X-basis ideal sigma <= 1/sqrt(n) = "
              f"{1 / np.sqrt(n):.4f}) ---")
        print(f"{'x':>4}  {'mean<XX>':>10}  "
              f"{'std':>8}  {'std*sqrt(n)':>11}")

        for k, qt in enumerate(targets):
            x = MIN_X + 0.5 * k
            print(f"{x:>4.1f}  {xx_mean[i, k]:>10.4f}  "
                  f"{xx_std[i, k]:>8.4f}  {xx_std[i, k] * np.sqrt(n):>11.4f}")


if __name__ == "__main__":
    main()
