"""Two-panel figure comparing the DU and perturbed hardware pilot runs.

Panel (a): full spatial <X_c X_x> profiles per T -- the light cone itself
(interior ~ 0, edge peak at x = T) for both runs.
Panel (b): light-cone edge <XX> vs T with the a-priori predictions: F_est for
the DU run, F_est x exact noiseless decay for eps = 0.1.

Picks the newest completed (non-local-test) run per epsilon case under
data/hardware_pilot/.

Run:  python plotting/plot_hardware_pilot.py
Outputs: data/hardware_pilot/plots/hardware_lightcone.{pdf,svg,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "hardware_pilot"
OUT = DATA / "plots" / "hardware_lightcone"

# Exact noiseless edge <XX> at eps = 0.1 (statevector, T = 2..4; T >= 5
# extrapolated with the constant per-cycle ratio 0.9226 the exact values obey).
EXACT_EDGE_EPS01 = {2: 0.8512, 3: 0.7854, 4: 0.7246, 5: 0.6685, 6: 0.6168}

WONG = {"blue": "#0072B2", "vermilion": "#D55E00", "green": "#009E73",
        "orange": "#E69F00", "sky": "#56B4E9", "pink": "#CC79A7"}
T_COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]  # T=2..6


def load_runs() -> dict[float, dict]:
    """Newest completed hardware run per epsilon: {eps: {T: (x, evs, stds), ...}}."""
    cases: dict[float, dict] = {}
    for meta_path in sorted(DATA.glob("run_*/metadata.json")):
        m = json.loads(meta_path.read_text())
        p = m.get("params", {})
        if m.get("status") != "completed" or p.get("local_test"):
            continue
        eps = float(p.get("epsilon", 0.0))
        run = {"per_t": {}, "f_est": {}, "folder": meta_path.parent.name}
        for T in p["t_values"]:
            d = np.load(meta_path.parent / "per_t" / f"T_{T}.npz")
            x = p["min_x"] + 0.5 * np.arange(len(d["evs"]))
            run["per_t"][T] = (x, d["evs"], d["stds"])
            run["f_est"][T] = p["per_t"][str(T)]["f_est"]
        cases[eps] = run  # glob is sorted -> later run numbers overwrite
    return cases


def main() -> None:
    plt.style.use(ROOT / "wong.mplstyle")
    cases = load_runs()
    du, pert = cases[0.0], cases[0.1]
    print("plotting:", du["folder"], "+", pert["folder"])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)

    # --- (a) spatial profiles ------------------------------------------------
    for i, (T, (x, evs, stds)) in enumerate(sorted(du["per_t"].items())):
        c = T_COLORS[i % len(T_COLORS)]
        ax_a.errorbar(x, evs, stds, color=c, marker="o", markersize=4,
                      linewidth=0.8, capsize=2, label=f"$T={T}$")
        if T in pert["per_t"]:
            xp, ep, sp = pert["per_t"][T]
            ax_a.errorbar(xp, ep, sp, color=c, marker="^", mfc="none",
                          markersize=4.5, linewidth=0.8, linestyle="--", capsize=2)
    ax_a.axhline(0.0, color="black", linewidth=0.6, zorder=1)
    ax_a.set_xlabel("spatial offset $x$")
    ax_a.set_ylabel(r"$\langle X_c X_x \rangle$")
    ax_a.set_title(r"(a) light-cone profiles", fontsize=10)
    ax_a.legend(fontsize=8, ncols=2, title=r"$\bullet$ DU, $\triangle$ $\epsilon=0.1$",
                title_fontsize=8)

    # --- (b) edge vs T with predictions --------------------------------------
    for eps, run, color, marker, label in (
        (0.0, du, WONG["blue"], "o", r"DU ($\epsilon=0$)"),
        (0.1, pert, WONG["vermilion"], "^", r"$\epsilon=0.1$"),
    ):
        Ts = sorted(run["per_t"])
        edge = np.array([run["per_t"][T][1][-1] for T in Ts])
        err = np.array([run["per_t"][T][2][-1] for T in Ts])
        pred = np.array([run["f_est"][T] * (1.0 if eps == 0 else EXACT_EDGE_EPS01[T])
                         for T in Ts])
        ax_b.errorbar(Ts, edge, err, color=color, marker=marker, markersize=5,
                      mfc="none" if eps else None, linestyle="none", capsize=3,
                      label=label)
        ax_b.plot(Ts, pred, color=color, linestyle=":", linewidth=1.0,
                  label=("DU" if eps == 0 else r"$\epsilon=0.1$") + " prediction")
    ax_b.set_xlabel("Floquet cycles $T$")
    ax_b.set_xticks(sorted(du["per_t"]))
    ax_b.set_title(r"(b) edge correlator, ibm\_miami", fontsize=10)
    ax_b.legend(fontsize=8)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        path = OUT.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
