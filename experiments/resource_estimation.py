"""Qubit-resource estimate: classical-shadow (causal cone) vs full simulation
(causal block), swept over Floquet time and light-cone depth.

For every Floquet cycle t (0 .. MAX_T) and light-cone depth x (0 .. MIN_LC_DEPTH,
half-integer steps), we build both circuits down to x_min = max(t - x, 0) and read
off their qubit width:

  * classical shadow  -> ``_build_causal_cone_circuit``  (needs the full 2t-wide
    light cone, so width grows with both t and the depth x)
  * full simulation   -> ``_build_causal_block_circuit`` (a single causal block;
    its width ~ 2(2t+1) is set by t alone, so the depth-x lines coincide)

The figure has two panels (one per approach) sharing axes: number of qubits on x,
Floquet cycles on y, one line per light-cone depth (coloured by depth). A dashed
vertical line marks the target device's qubit count; the region to its right is
shaded as physically out of reach on that device.

Outputs (written to ../data/resource_estimation):
  * resource_estimation.npz            -- axes + the two 2D qubit-count grids
  * resource_estimation.{pdf,svg,png}  -- the two-panel figure

Run:  python experiments/resource_estimation.py
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library import UnitaryGate

from du.simulation import (
    kicked_ising_gate,
    _build_causal_cone_circuit,
    _build_causal_block_circuit,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data" / "resource_estimation"
STYLE = HERE.parent / "wong.mplstyle"

MAX_T = 40
MIN_LC_DEPTH = 5
TARGET = "ibm_miami"


def save_figure(fig, output, *, dpi=300):
    """Save `fig` as PDF + SVG + PNG using `output` as the filename stem."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        path = output.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        print(f"  wrote {path.relative_to(HERE.parent)}")


def get_target_qubits(target: str) -> int:
    """Qubit count of the target device, falling back to FakeMiami offline."""
    try:
        return QiskitRuntimeService().backend(target).num_qubits
    except Exception as exc:  # no credentials / offline -> use the fake backend
        from qiskit_ibm_runtime.fake_provider import FakeMiami

        n = FakeMiami().num_qubits
        print(f"  (live backend '{target}' unavailable: {type(exc).__name__}; "
              f"using FakeMiami = {n} qubits)")
        return n


def plot(floquet_cycles, lc_depth, num_qubits, num_qubits_cs,
         num_qubits_avail, target):
    plt.style.use(str(STYLE))

    fig, (ax_cs, ax_blk) = plt.subplots(
        1, 2, figsize=(9, 4.2), sharex=True, sharey=True
    )

    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=lc_depth.min(), vmax=lc_depth.max())
    x_max = max(num_qubits.max(), num_qubits_cs.max())

    def mark_ceiling(ax):
        # hardware ceiling: shade the unreachable region and mark the edge
        ax.axvspan(num_qubits_avail, x_max * 1.05, color="#D55E00", alpha=0.08,
                   linewidth=0, zorder=0)
        ax.axvline(num_qubits_avail, color="#D55E00", ls="--", lw=1.2, zorder=1)

    # ---- classical shadow: one line per (half-step) light-cone depth --------
    for j, x in enumerate(lc_depth):
        ax_cs.plot(num_qubits_cs[j], floquet_cycles, color=cmap(norm(x)),
                   lw=1.0, label=rf"$x = {x:.1f}$")
    mark_ceiling(ax_cs)
    ax_cs.set_title(r"Classical shadow" + "\n" + r"(causal cone)", fontsize=10)
    ax_cs.set_ylabel(r"Floquet cycles $t$")
    ax_cs.legend(title=r"light-cone depth $x$", loc="upper left", ncol=2,
                 fontsize=7, title_fontsize=8, labelspacing=0.3,
                 columnspacing=1.0, handlelength=1.4)

    # ---- full simulation: block width is depth-independent -> a single line --
    ax_blk.plot(num_qubits[0], floquet_cycles, color="#0072B2", lw=1.4,
                label=r"all depths $x$ (coincident)")
    mark_ceiling(ax_blk)
    ax_blk.set_title(r"Full simulation" + "\n" + r"(causal block)", fontsize=10)
    ax_blk.legend(loc="upper left", fontsize=8)

    for ax in (ax_cs, ax_blk):
        ax.set_xlabel(r"number of qubits")
        ax.set_xlim(0, x_max * 1.05)
        ax.set_ylim(floquet_cycles.min(), floquet_cycles.max())
        # device annotation next to the dashed line
        ax.annotate(
            rf"\texttt{{{target.replace('_', r'\_')}}}"
            + f"\n({num_qubits_avail} qubits)",
            xy=(num_qubits_avail, floquet_cycles.max() * 0.5),
            xytext=(-6, 0), textcoords="offset points",
            rotation=90, va="center", ha="right", fontsize=8, color="#D55E00",
        )

    fig.tight_layout()
    save_figure(fig, DATA_DIR / "resource_estimation")
    plt.close(fig)



def main():

    gate = UnitaryGate(kicked_ising_gate(0))

    floquet_cycles = np.arange(0, MAX_T + 0.5, 0.5)
    lc_depth = np.arange(0, MIN_LC_DEPTH + 0.5)
    num_qubits = np.empty((len(lc_depth), len(floquet_cycles)), dtype=np.int64)
    num_qubits_cs = np.empty((len(lc_depth), len(floquet_cycles)), dtype=np.int64)

    for i, t in enumerate(floquet_cycles):
        for j, x in enumerate(lc_depth):

            # estimate number of qubits for classical shadow approach
            qc_cs = _build_causal_cone_circuit(
                max((t - x, 0)),
                t, gate
            )
            num_qubits_cs[j, i] = qc_cs.num_qubits

            # estimate number of qubits for regular approach
            qc = _build_causal_block_circuit(
                max((t - x, 0)),
                t, gate
            )
            num_qubits[j, i] = qc.num_qubits

    num_qubits_avail = get_target_qubits(TARGET)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = DATA_DIR / "resource_estimation.npz"
    np.savez(
        npz_path,
        floquet_cycles=floquet_cycles,
        lc_depth=lc_depth,
        num_qubits=num_qubits,
        num_qubits_cs=num_qubits_cs,
        num_qubits_avail=num_qubits_avail,
        target=TARGET,
    )
    print(f"  wrote {npz_path.relative_to(HERE.parent)}")

    # vertical line at x=num_qubits_avail (device limit); one line per lc depth,
    # separate subplots for the classical-shadow and full-simulation approaches.
    plot(floquet_cycles, lc_depth, num_qubits, num_qubits_cs,
         num_qubits_avail, TARGET)


if __name__ == "__main__":
    main()
