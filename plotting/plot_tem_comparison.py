"""Estimator pilot vs TEM probe, side by side.

Panel (a): spatial <X_c X_x> profile at the T values both runs share --
the estimator pilot (TREX readout mitigation only), the TEM run's raw
(unmitigated) values, and the TEM-mitigated values against the noiseless
DU expectation (edge = 1, interior = 0).
Panel (b): the light-cone edge value per method, with F_est and the ideal
value for reference.

Picks the newest completed non-local run per folder (epsilon = 0 case).

Run:  python plotting/plot_tem_comparison.py
Outputs: data/tem_pilot/plots/tem_comparison.{pdf,svg,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "data" / "tem_pilot" / "plots" / "tem_comparison"

WONG = {"blue": "#0072B2", "vermilion": "#D55E00", "green": "#009E73",
        "orange": "#E69F00", "sky": "#56B4E9", "pink": "#CC79A7"}


def newest_completed(folder: Path) -> Path:
    latest = None
    for meta_path in sorted(folder.glob("run_*/metadata.json")):
        m = json.loads(meta_path.read_text())
        if m.get("params", {}).get("local_test") or m.get("params", {}).get("epsilon"):
            continue
        if m.get("status") != "completed":
            continue
        latest = meta_path.parent
    if latest is None:
        sys.exit(f"no completed epsilon=0 run under {folder}"
                 + ("\n(fetch the TEM job first: python experiments/tem_pilot.py"
                    " --fetch <job-id>)" if "tem" in folder.name else ""))
    return latest


def load_per_t(run: Path) -> tuple[dict, dict]:
    params = json.loads((run / "metadata.json").read_text())["params"]
    per_t = {}
    for T in params["t_values"]:
        d = np.load(run / "per_t" / f"T_{T}.npz")
        x = params["min_x"] + 0.5 * np.arange(len(d["evs"]))
        per_t[T] = {"x": x, **{k: d[k] for k in d.files}}
    f_est = {T: params["per_t"][str(T)]["f_est"] for T in params["t_values"]}
    return per_t, f_est


def main() -> None:
    plt.style.use(ROOT / "wong.mplstyle")
    pilot_run = newest_completed(ROOT / "data" / "hardware_pilot")
    tem_run = newest_completed(ROOT / "data" / "tem_pilot")
    print("plotting:", pilot_run.name, "vs", tem_run.name)

    pilot, pilot_f = load_per_t(pilot_run)
    tem, tem_f = load_per_t(tem_run)
    shared = sorted(set(pilot) & set(tem))
    if not shared:
        sys.exit(f"runs share no T values (pilot {sorted(pilot)}, TEM {sorted(tem)})")

    series = [  # (label, evs key, source, color, marker)
        ("estimator pilot (TREX)", "evs", pilot, WONG["blue"], "o"),
        ("TEM raw", "raw_evs", tem, WONG["orange"], "s"),
        ("TEM mitigated", "evs", tem, WONG["green"], "D"),
    ]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.6, 3.4), width_ratios=(2, 1))

    # --- (a) spatial profiles -------------------------------------------------
    for T in shared:
        for j, (label, key, src, color, marker) in enumerate(series):
            d = src[T]
            std_key = "raw_stds" if key == "raw_evs" else "stds"
            ax_a.errorbar(d["x"] + 0.03 * (j - 1), d[key], d[std_key], color=color,
                          marker=marker, markersize=4.5, mfc="none", linewidth=0.8,
                          linestyle="none", capsize=2,
                          label=label if T == shared[0] else None)
        # noiseless DU expectation: 0 in the interior, 1 on the edge
        x = pilot[T]["x"]
        ideal = np.where(x == x[-1], 1.0, 0.0)
        ax_a.plot(x, ideal, color="black", linestyle="none", marker="_",
                  markersize=11, label="noiseless DU" if T == shared[0] else None,
                  zorder=1)
    ax_a.axhline(0.0, color="black", linewidth=0.6, zorder=1)
    ax_a.set_xlabel("spatial offset $x$")
    ax_a.set_ylabel(r"$\langle X_c X_x \rangle$")
    t_str = ", ".join(f"$T={T}$" for T in shared)
    ax_a.set_title(f"(a) profiles at {t_str}", fontsize=10)
    ax_a.legend(fontsize=8)

    # --- (b) edge value per method ---------------------------------------------
    T = shared[-1]
    labels, vals, errs, colors = [], [], [], []
    for label, key, src, color, _ in series:
        std_key = "raw_stds" if key == "raw_evs" else "stds"
        labels.append(label.replace(" (TREX)", "\n(TREX)").replace("TEM ", "TEM\n"))
        vals.append(src[T][key][-1])
        errs.append(src[T][std_key][-1])
        colors.append(color)
    xs = np.arange(len(vals))
    ax_b.bar(xs, vals, yerr=errs, capsize=4, color=colors, alpha=0.75,
             edgecolor="black", linewidth=0.6)
    ax_b.axhline(1.0, color="black", linestyle=":", linewidth=1.0, label="noiseless")
    ax_b.axhline(pilot_f[T], color=WONG["pink"], linestyle="--", linewidth=1.0,
                 label=r"$F_\mathrm{est}$")
    ax_b.set_xticks(xs, labels, fontsize=7)
    ax_b.set_title(f"(b) edge $\\langle XX \\rangle$, $T={T}$", fontsize=10)
    ax_b.legend(fontsize=8)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        path = OUT.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
