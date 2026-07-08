from __future__ import annotations

import numpy as np

from du.simulation import build_circuit, get_n_qubits, estimate_correlator


T = 3
MIN_X = 1
ZZ_COUPLING = np.pi / 4
TRANSVERSE_FIELD = np.pi / 4
LONGITUDINAL_FIELD = 0
N_SHOTS = [500, 1000, 2000]
N_REPS = 20
SEED = 0

XX = np.array([0, 0])

def main() -> None:

    x_range = [round(float(x), 1) for x in np.arange(MIN_X, T + 0.5, 0.5)]
    n_qubits = get_n_qubits(T, T)
    print(
        f"=== Noiseless shadow variance ===\n"
        f"  t                : {T}\n"
        f"  x range          : {x_range}\n"
        f"  longitudinal (h) : {LONGITUDINAL_FIELD}\n"
        f"  transverse   (b) : {TRANSVERSE_FIELD}\n"
        f"  qubits           : {n_qubits}\n"
    )

    xx_corr = np.full((len(N_SHOTS), N_REPS, len(x_range)), np.nan)

    for i, n in enumerate(N_SHOTS):
        print(f"[shot-count {i + 1}/{len(N_SHOTS)}] n_shots={n}: ", end="", flush=True)
        for j in range(N_REPS):

            for k, x in enumerate(x_range):
                qc = build_circuit(T, x, h=LONGITUDINAL_FIELD, b=TRANSVERSE_FIELD)
                xx_corr[i, j, k] = estimate_correlator(qc, n, seed=SEED + i * N_REPS + j)

            print(f"{j + 1}", end=" ", flush=True)  # rep progress
        print("done", flush=True)

    xx_mean = xx_corr.mean(axis=1)
    xx_std = xx_corr.std(axis=1)

    for i, n in enumerate(N_SHOTS):
        print(
            f"\n--- n_shots = {n}   (weight-2 ideal sigma ~ 3/sqrt(n) = "
            f"{3 / np.sqrt(n):.4f}) ---"
        )
        print(f"{'x':>4}  {'mean<XX>':>10}  {'std':>8}  {'std*sqrt(n)':>11}")

        for k, x in enumerate(x_range):
            print(
                f"{x:>4.1f}  {xx_mean[i, k]:>10.4f}  "
                f"{xx_std[i, k]:>8.4f}  {xx_std[i, k] * np.sqrt(n):>11.4f}"
            )


if __name__ == "__main__":
    main()
