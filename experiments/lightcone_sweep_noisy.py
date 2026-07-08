"""Light-cone correlator sweep across the full (x, t) grid under device noise.

For each light-cone depth t (integer) and spatial offset x (half-integer steps
0, 0.5, ..., t), evaluate the ZZ (or XX) correlator on the causal-block circuit
using a realistic IBM-device noise model (FakeMiami). On the exact DU line the
signal is sharply peaked on the light-cone edge x = t; noise broadens / decays
it, and this sweep maps that decay over the whole (x, t) plane.

Outputs (written to ../data):
  * lightcone_sweep_noisy.npz  -- x, t, and the 2D correlator grid C[t, x]
  * lightcone_sweep_noisy.{png,pdf,svg}  -- heatmap of C over (x, t)

Run:  python experiments/lightcone_sweep_noisy.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeMiami

from du.simulation import correlator_noisy


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"

# ---- Sweep parameters -----------------------------------------------------
J = 0.7
EPS = 0.0
O = "ZZ"
SHOTS = 4096
T_VALUES = [0.5, 1.0, 1.5, 2.0]        # can extend to 2.5/3.0 with statevector method
X_STEP = 0.5
OUT_STEM = "lightcone_sweep_noisy"


def build_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return (t_axis, x_axis) for the sweep.

    x runs 0, 0.5, ..., max(t); points with x > t are off the causal cone and
    left as NaN in the result grid.
    """
    t_axis = np.array(T_VALUES, dtype=float)
    x_max = max(T_VALUES)
    x_axis = np.arange(0.0, x_max + X_STEP / 2, X_STEP)
    return t_axis, x_axis


def sweep(backend) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_axis, x_axis = build_grid()
    grid = np.full((t_axis.size, x_axis.size), np.nan)

    print(f"=== Light-cone noisy sweep: J={J}, eps={EPS}, O={O}, "
          f"shots={SHOTS}, backend={backend} ===\n")
    header = "  t \\ x " + "".join(f"{x:>9.1f}" for x in x_axis)
    print(header)
    print("-" * len(header))

    for i, t in enumerate(t_axis):
        row = []
        for j, x in enumerate(x_axis):
            if x > t:                       # outside the causal cone
                row.append("        .")
                continue
            c = correlator_noisy(x, t, J, EPS, O=O,
                                 backend=backend, shots=SHOTS)
            grid[i, j] = c
            row.append(f"{c:>9.3f}")
        print(f"{t:>5.0f} " + "".join(row))

    return t_axis, x_axis, grid


def plot(t_axis: np.ndarray, x_axis: np.ndarray, grid: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[x_axis[0] - X_STEP / 2, x_axis[-1] + X_STEP / 2,
                t_axis[0] - 0.5, t_axis[-1] + 0.5],
        vmin=-1.0, vmax=1.0,
    )
    # light-cone edge x = t
    ax.plot(t_axis, t_axis, "r--", lw=1.2, label="light-cone edge $x=t$")
    ax.set_xlabel("spatial offset $x$")
    ax.set_ylabel("depth $t$")
    ax.set_title(f"Noisy light-cone correlator $\\langle {O} \\rangle$  "
                 f"(J={J}, $\\epsilon$={EPS}, FakeMiami)")
    ax.set_yticks(t_axis)
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=ax, label=f"$\\langle {O} \\rangle$")
    fig.tight_layout()

    for ext in ("png", "pdf", "svg"):
        path = DATA_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(path, dpi=150)
        print(f"  wrote {path.relative_to(HERE.parent)}")
    plt.close(fig)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    # FakeMiami device noise, but forced onto the statevector method so noisy
    # circuits use quantum-trajectory sampling (cost ~ shots * 2^n) instead of
    # the automatic-method density matrix (cost ~ 4^n), which blows up past t=2.
    backend = AerSimulator.from_backend(FakeMiami(), method="statevector")

    t_axis, x_axis, grid = sweep(backend)

    npz_path = DATA_DIR / f"{OUT_STEM}.npz"
    np.savez(npz_path, t=t_axis, x=x_axis, C=grid,
             J=J, eps=EPS, O=O, shots=SHOTS)
    print(f"\n  wrote {npz_path.relative_to(HERE.parent)}")

    plot(t_axis, x_axis, grid)


if __name__ == "__main__":
    main()
