from __future__ import annotations

import numpy as np

from du.simulation import (
    correlator_noisy,
    correlator_statevector,
    make_depolarizing_noise,
)


def zne_extrapolate(scales: list[int], values: list[float],
                    degree: int = 1) -> float:
    """Extrapolate `values` at scale=0 by polynomial fit of given degree."""
    return float(np.polyval(np.polyfit(scales, values, degree), 0))


def main() -> None:
    N, depth, J, eps = 6, 4, 0.7, 0.15
    print(f"=== Noise + ZNE: N={N}, depth={depth}, J={J:.2f}, eps={eps:.2f} ===\n")

    # Noiseless reference
    ref = {x: correlator_statevector(N, depth, J, eps, x) for x in range(N)}

    noise_levels = {
        "low  (cx err 1e-3)":  make_depolarizing_noise(two_qubit_err=1e-3),
        "med  (cx err 5e-3)":  make_depolarizing_noise(two_qubit_err=5e-3),
        "high (cx err 1e-2)":  make_depolarizing_noise(two_qubit_err=1e-2),
    }
    scales = [1, 3, 5]

    for label, nm in noise_levels.items():
        print(f"\n--- Noise level: {label} ---")
        print(f"{'x':>3}  {'noiseless':>11}  "
              + "  ".join(f"{'s='+str(s):>11}" for s in scales)
              + f"  {'ZNE lin':>11}  {'ZNE quad':>11}  {'lin err':>10}  "
              + f"{'quad err':>10}")
        print("-" * 110)

        for x in range(N):
            vals = [correlator_noisy(N, depth, J, eps, x,
                                     noise_model=nm, scale_factor=s)
                    for s in scales]
            zne_lin = zne_extrapolate(scales, vals, degree=1)
            zne_quad = zne_extrapolate(scales, vals, degree=2)
            lin_err = zne_lin - ref[x]
            quad_err = zne_quad - ref[x]
            print(f"{x:>3}  {ref[x]:>11.4e}  "
                  + "  ".join(f"{v:>11.4e}" for v in vals)
                  + f"  {zne_lin:>11.4e}  {zne_quad:>11.4e}  "
                  + f"{lin_err:>10.2e}  {quad_err:>10.2e}")


if __name__ == "__main__":
    main()
