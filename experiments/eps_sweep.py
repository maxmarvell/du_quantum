from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from du.simulation import correlator_statevector


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"


def sweep(N: int, depth: int, J: float, x: int,
          eps_values: np.ndarray) -> np.ndarray:
    return np.array([
        correlator_statevector(N, depth, J, float(eps), x)
        for eps in eps_values
    ])


def main() -> None:
    N, depth, J = 6, 4, 0.7
    positions = list(range(N))                       # x = 0 .. N-1
    on_cone_x = depth                                # only x = depth is on-cone

    print(f"=== Perturbative sweep: N={N}, depth={depth}, J={J:.3f} ===\n")

    # ---- Linear sweep ----------------------------------------------------
    eps_lin = np.linspace(0.0, 0.30, 16)
    print(f"--- Polynomial fit  C(eps) = sum_k c_k eps^k  (degree 4) ---")
    print(f"{'x':>3}  {'cone?':>5}  {'c_0':>12}  {'c_1':>12}  {'c_2':>12}  "
          f"{'c_3':>12}  {'c_4':>12}  {'RMS resid':>11}")
    print("-" * 96)

    raw = {}
    for x in positions:
        C = sweep(N, depth, J, x, eps_lin)
        raw[x] = C
        coeffs_desc = np.polyfit(eps_lin, C, 4)        # highest degree first
        coeffs_asc = coeffs_desc[::-1]                 # c_0, c_1, ...
        resid = np.sqrt(np.mean((np.polyval(coeffs_desc, eps_lin) - C) ** 2))
        tag = "ON " if x == on_cone_x else "off"
        print(f"{x:>3}  {tag:>5}  "
              + "  ".join(f"{c:>12.4e}" for c in coeffs_asc)
              + f"  {resid:>11.3e}")

    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / "eps_sweep_data.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eps"] + [f"C_x{x}" for x in positions])
        for i, eps in enumerate(eps_lin):
            w.writerow([eps] + [raw[x][i] for x in positions])
    print(f"\nRaw linear-sweep data: {csv_path.relative_to(HERE.parent)}\n")

    # ---- Leading exponent: ratio test at very small eps ------------------
    # For C(eps) ~ A eps^p, the ratio C/eps^p is constant at small eps for
    # the correct p. Scan integer p and pick the one with smallest relative
    # variation of C/eps^p across the small-eps window.
    eps_small = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    print(f"--- Leading exponent: ratio test  (eps in {{{eps_small[0]:.0e}, ..., "
          f"{eps_small[-1]:.0e}}}, integer p in 1..5) ---")
    print(f"{'x':>3}  {'cone?':>5}  {'p*':>3}  {'A = C/eps^p*':>14}  "
          f"{'rel. spread':>12}")
    print("-" * 50)

    for x in positions:
        if x == on_cone_x:
            print(f"{x:>3}  {'ON ':>5}   (skip)")
            continue
        C = sweep(N, depth, J, x, eps_small)
        if np.max(np.abs(C)) < 1e-12:
            print(f"{x:>3}  {'off':>5}   (numerical zero at all eps)")
            continue
        best_p, best_spread, best_A = None, np.inf, None
        for p in range(1, 6):
            ratios = C / eps_small ** p
            if np.max(np.abs(ratios)) < 1e-10:
                continue
            spread = (np.max(ratios) - np.min(ratios)) / np.mean(np.abs(ratios))
            if abs(spread) < best_spread:
                best_p, best_spread, best_A = p, abs(spread), float(np.mean(ratios))
        print(f"{x:>3}  {'off':>5}  {best_p:>3}  {best_A:>14.4e}  {best_spread:>12.2e}")


if __name__ == "__main__":
    main()
