"""Batch campaign driver for the hand-rolled (Choi-native) TEM experiment.

Everything shares ONE frozen circuit layout (du.circuit_layout) and runs in
ONE Qiskit Runtime Batch (one calibration window):

  stage a  noise-learner   full even/odd Floquet layers on the system chain
  stage b  estimator       X-basis production PUBs, gate-twirled + TREX
  stage c  sampler         Choi-shadow circuits (X-biased random local bases)
  stage d  estimator       ZNE comparison arm (resilience_level=2)

Sub-commands (dry-run by default; nothing is submitted without --submit):

  python hardware_batch.py layout -T 16 --lc-depth 4 --save --plot
  python hardware_batch.py submit           # build + plan + cost estimate
  python hardware_batch.py submit --submit  # actually send the batch
  python hardware_batch.py fetch            # re-attach + save all results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from du.circuit_layout import (CircuitLayout, find_circuit_layout,
                               n_system_qubits, x_min_for_depth)
from du.experiments import Run, create_run, latest_run
from du.floquet import (brickwork_phase, full_brickwork_layers,
                        layer_parity_classes)
from du.plotting import save_figure, use_style
from du.shadows import sample_settings, shadow_circuits
from du.simulation import build_circuit, get_control_qubit, get_target_qubits
from du.utils import xx_observables

BACKEND_NAME = "ibm_miami"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

LONGITUDINAL_FIELD = 0          # h
TRANSVERSE_FIELD = np.pi / 4    # b = pi/4 - eps

# -- campaign schedule (agreed plan) ----------------------------------------
SHOTS = 30_000
DU_T = (2, 4, 8, 12, 16)                    # eps = 0 horizon sweep
EPS = 0.1
EPS_T = (2, 3, 4, 5, 8)                     # perturbed physics sweep
SHADOW_CONFIGS = ((EPS, 3), (EPS, 5), (0.0, 8))   # (eps, T)
N_SETTINGS = 128
SHADOW_SHOTS = 256
ZNE_CONFIGS = ((0.0, 8), (0.0, 12), (EPS, 4), (EPS, 8))
LEARNER_OPTIONS = {                          # NoiseLearnerOptions semantics
    "max_layers_to_learn": 2,                # full even + full odd layer
    "num_randomizations": 32,
    "shots_per_randomization": 128,
    "layer_pair_depths": [0, 1, 2, 4, 16, 32],
}
N_TWIRL_RANDOMIZATIONS = 64

# validated miami cost model: shots x 4.03 ms, x1.2 twirled, x9 learner bases
CYCLE_S = 4.03e-3
TWIRL_FACTOR = 1.2


def get_backend(fake: bool):
    if fake:
        from qiskit_ibm_runtime.fake_provider import FakeMiami
        return FakeMiami()
    from qiskit_ibm_runtime import QiskitRuntimeService
    try:
        service = QiskitRuntimeService()
    except Exception as exc:
        sys.exit(f"could not connect to IBM Runtime ({exc}); try --fake")
    return service.backend(BACKEND_NAME)


def load_frozen_layout() -> tuple[CircuitLayout, dict]:
    run = latest_run("hardware_batch", data_root=DATA_ROOT)
    while run is not None and "layout" not in run.metadata.get("params", {}):
        run = None  # only layout runs carry params.layout; scan below
    if run is None:
        base = DATA_ROOT / "hardware_batch"
        candidates = sorted(base.glob("run_*/metadata.json"),
                            reverse=True) if base.is_dir() else []
        for meta_path in candidates:
            import json
            params = json.loads(meta_path.read_text()).get("params", {})
            if "layout" in params:
                return CircuitLayout.from_dict(params["layout"]), params
        sys.exit("no frozen layout found -- run the layout sub-command "
                 "with --save first")
    params = run.metadata["params"]
    return CircuitLayout.from_dict(params["layout"]), params


def pin(qc, layout: CircuitLayout, backend, doubled: bool = True):
    from qiskit import generate_preset_pass_manager
    n = qc.num_qubits // 2 if doubled else qc.num_qubits
    physical = (layout.system[:n] + layout.partners[:n] if doubled
                else layout.system[:n])
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend,
                                      initial_layout=physical,
                                      routing_method="none")
    return pm.run(qc), physical


def production_pub(eps: float, T: float, x_min: float, layout, backend):
    b = TRANSVERSE_FIELD - eps
    qc = build_circuit(T, x_min, h=LONGITUDINAL_FIELD, b=b)
    # every Floquet layer must be single-parity (a subset of one of the two
    # learned full layers); raises on any mixed layer. The class sequence is
    # recorded so the inversion pipeline matches learned layers by CLASS,
    # never by order (the first-layer class varies with the block geometry).
    classes = layer_parity_classes(qc)
    assert all(a != b_ for a, b_ in zip(classes, classes[1:])), \
        f"non-alternating layers at T={T}, x_min={x_min}"
    tqc, physical = pin(qc, layout, backend)
    control = get_control_qubit(T, x_min)
    targets = get_target_qubits(T, x_min)
    obs = [o.apply_layout(tqc.layout) for o in
           xx_observables(qc.num_qubits, control, targets)]
    return (tqc, obs), {"eps": eps, "T": T, "x_min": x_min,
                        "control": control, "targets": targets,
                        "layer_classes": classes,
                        "physical_qubits": [int(q) for q in physical]}


def build_batch(layout: CircuitLayout, layout_params: dict, backend) -> dict:
    lc_depth = layout_params.get("lc_depth")
    xm = (lambda T: 1) if lc_depth is None else \
         (lambda T: x_min_for_depth(T, lc_depth))

    # stage a: learning template (system chain only, partners idle),
    # anchored on the point charge of the campaign geometry
    phase = brickwork_phase(layout_params["T"], layout_params["x_min"])
    layers = full_brickwork_layers(layout.n_system, h=LONGITUDINAL_FIELD,
                                   b=TRANSVERSE_FIELD, phase=phase)
    tqc_learn, _ = pin(layers, layout, backend, doubled=False)

    # stage b: X-basis production PUBs
    xbasis, xbasis_info = [], []
    for eps, t_values in ((0.0, DU_T), (EPS, EPS_T)):
        for T in t_values:
            pub, info = production_pub(eps, T, xm(T), layout, backend)
            xbasis.append(pub)
            xbasis_info.append(info)

    # stage c: Choi-shadow circuits (reuse the pinned production circuits)
    shadows, shadow_info = [], []
    for i, (eps, T) in enumerate(SHADOW_CONFIGS):
        pub, info = production_pub(eps, T, xm(T), layout, backend)
        settings = sample_settings(len(info["physical_qubits"]), N_SETTINGS,
                                   seed=hash((eps, T)) % 2**31)
        circuits = shadow_circuits(pub[0], info["physical_qubits"], settings)
        shadows.extend(circuits)
        shadow_info.append({**info, "settings": settings,
                            "n_settings": N_SETTINGS,
                            "shots_per_setting": SHADOW_SHOTS})

    # stage d: ZNE arm (subset of the production PUBs)
    zne, zne_info = [], []
    for eps, T in ZNE_CONFIGS:
        pub, info = production_pub(eps, T, xm(T), layout, backend)
        zne.append(pub)
        zne_info.append(info)

    return {"tqc_learn": tqc_learn, "xbasis": xbasis,
            "xbasis_info": xbasis_info, "shadows": shadows,
            "shadow_info": shadow_info, "zne": zne, "zne_info": zne_info}


def estimate(built: dict) -> float:
    lo = LEARNER_OPTIONS
    learn_shots = (lo["max_layers_to_learn"] * len(lo["layer_pair_depths"])
                   * 9 * lo["num_randomizations"]
                   * lo["shots_per_randomization"])
    learn = learn_shots * CYCLE_S
    xbasis = len(built["xbasis"]) * SHOTS * CYCLE_S * TWIRL_FACTOR
    shadow = len(built["shadows"]) * SHADOW_SHOTS * CYCLE_S
    zne = len(built["zne"]) * SHOTS * 3 * CYCLE_S * TWIRL_FACTOR
    total = learn + xbasis + shadow + zne
    print(f"\nQPU estimate (validated miami model, {CYCLE_S * 1e3:.2f} ms/shot):")
    print(f"  a noise-learner : {learn:7.0f} s  ({learn_shots} shots, x9 bases)")
    print(f"  b x-basis       : {xbasis:7.0f} s  ({len(built['xbasis'])} PUBs x "
          f"{SHOTS} twirled shots)")
    print(f"  c shadows       : {shadow:7.0f} s  ({len(built['shadows'])} circuits "
          f"x {SHADOW_SHOTS} shots)")
    print(f"  d ZNE           : {zne:7.0f} s  ({len(built['zne'])} PUBs x "
          f"{SHOTS} shots x ~3 noise factors)")
    print(f"  total           : {total:7.0f} s  (~{total / 60:.0f} min)")
    return total


def cmd_submit(args) -> None:
    backend = get_backend(args.fake)
    layout, layout_params = load_frozen_layout()
    if layout.backend_name != backend.name:
        print(f"WARNING: layout frozen for {layout.backend_name}, "
              f"running on {backend.name}")
    drift = layout.score(backend)
    print(f"layout: {layout.n_system} system qubits, re-scored cost "
          f"{drift:.3f} (frozen at {layout_params['layout'].get('cost'):.3f})")

    built = build_batch(layout, layout_params, backend)
    print(f"built: learner template ({built['tqc_learn'].num_qubits}q), "
          f"{len(built['xbasis'])} x-basis PUBs, "
          f"{len(built['shadows'])} shadow circuits, "
          f"{len(built['zne'])} ZNE PUBs")
    estimate(built)

    if not args.submit:
        print("\nDRY RUN ONLY -- re-run with --submit to send the batch.")
        return
    if args.fake:
        sys.exit("refusing to --submit against a fake backend")

    from qiskit_ibm_runtime import Batch, EstimatorV2, SamplerV2
    from qiskit_ibm_runtime.noise_learner import NoiseLearner
    from qiskit_ibm_runtime.options import NoiseLearnerOptions

    # run folder + full provenance BEFORE submitting (orphan-proof)
    run = create_run(
        "hardware_batch", f"batch_{backend.name}", data_root=DATA_ROOT,
        params={"backend": backend.name, "shots": SHOTS,
                "layout": layout.to_dict(), "layout_cost_at_submit": drift,
                "learner_options": LEARNER_OPTIONS,
                "xbasis": built["xbasis_info"], "zne": built["zne_info"],
                "shadow_configs": [
                    {k: v for k, v in info.items() if k != "settings"}
                    for info in built["shadow_info"]],
                "jobs": {}},
    )
    for i, info in enumerate(built["shadow_info"]):
        run.save_array("shadows", f"settings_{i}", settings=info["settings"],
                       physical_qubits=np.array(info["physical_qubits"]))

    with Batch(backend=backend) as batch:
        learner = NoiseLearner(mode=batch,
                               options=NoiseLearnerOptions(**LEARNER_OPTIONS))
        job_a = learner.run([built["tqc_learn"]])
        run.save_metadata({"params": {"jobs": {"learner": job_a.job_id()}}})

        est = EstimatorV2(mode=batch)
        est.options.default_shots = SHOTS
        est.options.resilience_level = 1                  # TREX readout
        est.options.twirling.enable_gates = True
        est.options.twirling.num_randomizations = N_TWIRL_RANDOMIZATIONS
        est.options.twirling.shots_per_randomization = \
            SHOTS // N_TWIRL_RANDOMIZATIONS
        job_b = est.run(built["xbasis"])
        run.save_metadata({"params": {"jobs": {"xbasis": job_b.job_id()}}})

        sam = SamplerV2(mode=batch)
        sam.options.default_shots = SHADOW_SHOTS
        job_c = sam.run(built["shadows"])
        run.save_metadata({"params": {"jobs": {"shadows": job_c.job_id()}}})

        zne = EstimatorV2(mode=batch)
        zne.options.default_shots = SHOTS
        zne.options.resilience_level = 2                  # ZNE + twirling
        job_d = zne.run(built["zne"])
        run.save_metadata({"params": {"jobs": {"zne": job_d.job_id()}}})

    print(f"submitted batch -- jobs recorded in {run.path}")
    print("collect results later with:  python experiments/hardware_batch.py fetch")


def cmd_fetch(args) -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService

    run = latest_run("hardware_batch", data_root=DATA_ROOT)
    if run is None or "jobs" not in run.metadata.get("params", {}):
        sys.exit("no submitted batch run found under data/hardware_batch")
    jobs = run.metadata["params"]["jobs"]
    print(f"fetching into {run.path}: {jobs}")
    service = QiskitRuntimeService()

    if "learner" in jobs:
        res = service.job(jobs["learner"]).result()
        for i, layer in enumerate(res):
            run.save_array("noise_model", f"layer_{i}",
                           rates=np.asarray(layer.error.rates),
                           qubits=np.asarray(layer.qubits))
            (run.subdir("noise_model") / f"layer_{i}_generators.txt").write_text(
                "\n".join(str(g) for g in layer.error.generators.to_labels()))
        print(f"  learner: {len(res)} layers saved")

    for stage, infos in (("xbasis", run.metadata["params"]["xbasis"]),
                         ("zne", run.metadata["params"]["zne"])):
        if stage not in jobs:
            continue
        res = service.job(jobs[stage]).result()
        for i, info in enumerate(infos):
            run.save_array(stage, f"eps{info['eps']:g}_T{info['T']:g}",
                           evs=np.asarray(res[i].data.evs, dtype=float),
                           stds=np.asarray(res[i].data.stds, dtype=float),
                           targets=np.asarray(info["targets"]))
        print(f"  {stage}: {len(infos)} PUBs saved")

    if "shadows" in jobs:
        res = service.job(jobs["shadows"]).result()
        configs = run.metadata["params"]["shadow_configs"]
        for i, cfg in enumerate(configs):
            lo, hi = i * N_SETTINGS, (i + 1) * N_SETTINGS
            bits = np.stack([res[j].data.shadow.array
                             for j in range(lo, hi)])
            run.save_array("shadows", f"bits_{i}", bits=bits)
        print(f"  shadows: {len(configs)} configs saved")

    run.save_metadata({"status": "completed"})
    print(f"saved to {run.path}")


def cmd_layout(args) -> None:
    backend = get_backend(args.fake)
    x_min = 1 if args.lc_depth is None else x_min_for_depth(args.T, args.lc_depth)
    n = n_system_qubits(args.T, x_min)
    print(f"sizing: T_max={args.T:g}, x_min={x_min:g}"
          + (f" (lc_depth={args.lc_depth:g})" if args.lc_depth is not None else "")
          + f" -> n_system={n}")
    layout = find_circuit_layout(backend, n, seed=args.seed)
    if layout is None:
        sys.exit("no layout found at all -- check calibration data")
    h = layout.health(backend)
    print(f"backend {backend.name}: requested n_system={n}, "
          f"found {layout.n_system} "
          f"({'complete' if layout.complete else 'INCOMPLETE'})")
    print(f"  system:   {layout.system}")
    print(f"  partners: {layout.partners}")
    print(f"  chain e2q:    median {np.median(h['chain_e2q']):.2e}, "
          f"worst {max(h['chain_e2q']):.2e}")
    print(f"  partner e2q:  median {np.median(h['partner_e2q']):.2e}, "
          f"worst {max(h['partner_e2q']):.2e}")
    print(f"  readout err:  median {np.median(h['readout']):.2e}, "
          f"worst {max(h['readout']):.2e}")
    print(f"  T2:           median {np.median(h['t2']) * 1e6:.0f} us, "
          f"min {min(h['t2']) * 1e6:.0f} us")
    if h["bad_qubits"]:
        print(f"  WARNING: layout members now below par: {h['bad_qubits']}")
    print(f"  layout -log fidelity (prep+wiring): {layout.cost:.3f}")

    validation = None
    if layout.complete:
        try:
            layout.validate(backend, args.T, x_min)
            validation = "passed"
            print(f"  transpiler validation: PASSED (zero-swap at T={args.T}, "
                  f"x_min={x_min:g})")
        except ValueError as exc:
            validation = str(exc)
            print(f"  transpiler validation: FAILED -- {exc}")
    else:
        print(f"\nlayout supports at most T ~ {layout.n_system / 3:.1f}")

    if not (args.save or args.plot):
        print("\n(dry run: --save to freeze this layout, --plot for the figure)")
        return

    run = create_run(
        "hardware_batch", f"layout_{backend.name}_T{args.T:g}",
        data_root=DATA_ROOT,
        params={"backend": backend.name, "T": args.T,
                "lc_depth": args.lc_depth, "x_min": x_min, "n_system": n,
                "seed": args.seed, "layout": layout.to_dict(),
                "validation": validation,
                "health": {k: (list(map(float, v)) if k != "bad_qubits" else v)
                           for k, v in h.items()}},
    )
    if args.plot:
        use_style()
        fig = layout.plot(backend)
        save_figure(fig, run.plots / "circuit_layout")
    print(f"\nsaved to {run.path}")


def cmd_preview(args) -> None:
    from du.shadows import X_BIAS

    T = args.T
    x_min = 1 if args.lc_depth is None else x_min_for_depth(T, args.lc_depth)
    n = n_system_qubits(T, x_min)

    phase = brickwork_phase(T, x_min)
    print(f"=== stage a: noise-learner template (full Floquet layers, "
          f"shown at n_system={n}, phase={phase} anchored on the point "
          f"charge) ===")
    layers = full_brickwork_layers(n, h=LONGITUDINAL_FIELD, b=TRANSVERSE_FIELD,
                                   phase=phase)
    print(layers.draw(output="text", fold=120))
    qc_check = build_circuit(T, x_min, h=LONGITUDINAL_FIELD, b=TRANSVERSE_FIELD)
    classes = layer_parity_classes(qc_check)
    print(f"parity check (T={T}, x_min={x_min}): circuit layers {classes}; "
          f"template first layer class = "
          f"{'odd' if phase else 'even'} -- matches circuit first layer")

    print(f"\n=== stage b: X-basis production circuit "
          f"(shown at T={T}, x_min={x_min}: {2 * n} qubits) ===")
    qc = build_circuit(T, x_min, h=LONGITUDINAL_FIELD, b=TRANSVERSE_FIELD)
    print(qc.draw(output="text", fold=120))
    print(f"observables: <X_c X_x>, control={get_control_qubit(T, x_min)}, "
          f"targets={get_target_qubits(T, x_min)}")

    print(f"\n=== stage c: shadow circuit (one random setting, bias {X_BIAS}) ===")
    settings = sample_settings(qc.num_qubits, 1, seed=args.seed)
    shadow = shadow_circuits(qc, list(range(qc.num_qubits)), settings)[0]
    print(shadow.draw(output="text", fold=120))
    print(f"setting (0=X 1=Y 2=Z): {settings[0].tolist()}")

    print("\n=== stage d: ZNE ===")
    print("same circuits/observables as stage b; noise amplification is an "
          "EstimatorV2 option (resilience_level=2), not a distinct circuit")

    if args.mpl:
        use_style()
        out = DATA_ROOT / "hardware_batch" / "previews"
        for name, circ in (("learner_layers", layers),
                           ("production", qc), ("shadow", shadow)):
            fig = circ.draw(output="mpl", fold=40)
            save_figure(fig, out / name, formats=("png",))
        print(f"\nmpl figures saved under {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_layout = sub.add_parser("layout", help="find + freeze the circuit layout")
    p_layout.add_argument("--fake", action="store_true")
    p_layout.add_argument("-T", type=float, default=5,
                          help="maximum Floquet depth of the campaign")
    p_layout.add_argument("--lc-depth", type=float, default=None,
                          help="fixed depth into the light cone "
                               "(x_min = max(T - LC_DEPTH, 1))")
    p_layout.add_argument("--seed", type=int, default=0)
    p_layout.add_argument("--plot", action="store_true")
    p_layout.add_argument("--save", action="store_true")
    p_layout.set_defaults(func=cmd_layout)

    p_submit = sub.add_parser("submit", help="build + submit the batch")
    p_submit.add_argument("--fake", action="store_true")
    p_submit.add_argument("--submit", action="store_true",
                          help="actually submit (default: dry-run plan)")
    p_submit.set_defaults(func=cmd_submit)

    p_fetch = sub.add_parser("fetch", help="re-attach + save all results")
    p_fetch.set_defaults(func=cmd_fetch)

    p_prev = sub.add_parser("preview", help="print a readable small instance "
                                            "of each stage's circuit")
    p_prev.add_argument("-T", type=float, default=2,
                        help="Floquet depth to render (default 2: small and "
                             "readable; larger T = wider diagrams)")
    p_prev.add_argument("--lc-depth", type=float, default=None,
                        help="fixed light-cone depth (as in the layout "
                             "sub-command); sets x_min = max(T-LC_DEPTH, 1)")
    p_prev.add_argument("--seed", type=int, default=0)
    p_prev.add_argument("--mpl", action="store_true",
                        help="also save matplotlib drawings under "
                             "data/hardware_batch/previews/")
    p_prev.set_defaults(func=cmd_preview)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
