"""X-basis light-cone pilot on IBM hardware (ibm_miami / Nighthawk).

All <X_c X_t> correlators along the DU light cone commute, so EstimatorV2
groups them into ONE measurement setting per T -- a single circuit repeated
SHOTS times, the cheapest thing a device can run. Estimator also buys TREX
readout mitigation (twirled readout -> calibrated attenuation, divided out)
and dynamical decoupling on idle qubits.

Modes (default is a DRY RUN -- nothing is ever submitted without --submit):
  python hardware_pilot.py             # live-calibration preflight: transpile
                                       # best-of-N Sabre seeds, report width /
                                       # 2q count / duration / F_est / QPU time
  python hardware_pilot.py --fake      # same but against FakeMiami snapshot
                                       # (no credentials / network needed)
  python hardware_pilot.py --submit    # preflight + actually submit the job,
                                       # wait for results, save under data/
  python hardware_pilot.py --fractional  # live preflight with native rzz enabled
                                       # (use_fractional_gates); compare the 2q /
                                       # F_est table against the standard target
  python hardware_pilot.py --fetch ID  # re-attach to a submitted job (e.g. after
                                       # a crash while waiting) and save results

Expected signal: edge <XX> ~ F_est (0.8-0.95), interior ~ 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, generate_preset_pass_manager, qpy
from qiskit.quantum_info import SparsePauliOp

from du.simulation import build_circuit, get_control_qubit, get_target_qubits
from experiment_io import Run, create_run, open_run

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"  # cwd-independent


BACKEND_NAME = "ibm_miami"
T_VALUES = (2, 3, 4, 5)
MIN_X = 1
LONGITUDINAL_FIELD = 0          # h
TRANSVERSE_FIELD = np.pi / 4    # b = pi/4 -> self-dual (dual-unitary) point
SHOTS = 30_000
N_TRANSPILE_SEEDS = 6           # best-of-N Sabre; never pin a linear chain
SHOT_CYCLE_S = 260e-6           # ~circuit + readout + default rep_delay
# QPU cost is dominated by per-circuit overhead, not shots (measured on
# run_0005: 735 s for ~2400 circuits ~ 0.3 s each; raw shot time was 39 s).
# The server default twirling (64 shots/randomization -> 469 circuits/PUB at
# 30k shots) is what made run_0005 cost 6x the naive estimate, so we cap it.
N_TWIRL_RANDOMIZATIONS = 64     # per PUB; standard twirl count, 7x fewer circuits
TREX_CIRCUITS_PER_PUB = 32      # approx measurement-noise-learning circuits
CIRCUIT_OVERHEAD_S = 0.30       # per-circuit load/switch cost (run_0005 measured)


def xx_observables(n_qubits: int, control: int, targets: list[int]) -> list[SparsePauliOp]:
    """<X_control X_target> for every target."""
    return [
        SparsePauliOp.from_sparse_list([("XX", [control, qt], 1.0)], num_qubits=n_qubits)
        for qt in targets
    ]


def best_transpile(qc: QuantumCircuit, backend) -> QuantumCircuit:
    """Best-of-N Sabre seeds by two-qubit gate count, preferring placements
    that avoid flagged qubits (T2 < 20% of median). Wide circuits get more
    seeds -- large embeddings fail more often (cf. decoherence-horizon study).
    Falls back to the lowest-2q dirty placement if no clean one is found,
    which the preflight then flags via the bad_T2 column."""
    target = backend.target
    t2s = np.array([target.qubit_properties[q].t2 or 0.0
                    for q in range(target.num_qubits)])
    t2_bad = 0.2 * float(np.median(t2s[t2s > 0]))

    n_seeds = N_TRANSPILE_SEEDS if qc.num_qubits <= 24 else 4 * N_TRANSPILE_SEEDS
    best_clean, best_any = None, None
    for seed in range(n_seeds):
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                          seed_transpiler=seed)
        tqc = pm.run(qc)
        n2q = sum(1 for inst in tqc.data if inst.operation.num_qubits == 2)
        used = tqc.layout.final_index_layout()
        clean = not (t2s[used] < t2_bad).any()
        if best_any is None or n2q < best_any[0]:
            best_any = (n2q, tqc)
        if clean and (best_clean is None or n2q < best_clean[0]):
            best_clean = (n2q, tqc)
    return (best_clean or best_any)[1]


def preflight(backend, t_values: tuple = T_VALUES, shots: int = SHOTS,
              epsilon: float = 0.0) -> dict:
    """Transpile every T, report the resource table, return per-T artifacts.

    epsilon > 0 perturbs the transverse field to b = pi/4 - epsilon, breaking
    dual-unitarity: the noiseless edge signal itself decays (e.g. T=2 edge =
    0.851 / 0.518 at eps = 0.1 / 0.2), on top of the hardware F_est factor.

    At epsilon = 0 the circuit is Clifford, so a stabilizer noisy simulation
    (runtime's Neat debug tool, cliffordize=True) predicts the edge <XX> under
    the backend's Pauli gate errors -- the XX_sim column. It complements F_est:
    Neat sees the actual error placement but no T1/T2 decay, F_est folds in T2
    but treats errors as uniform attenuation. Skipped for epsilon > 0, where
    cliffordizing would silently round the perturbed angles back to the DU point.
    """
    from qiskit.quantum_info import Operator
    from du.simulation import is_dual_unitary, kicked_ising_gate

    b = TRANSVERSE_FIELD - epsilon
    du = is_dual_unitary(Operator(kicked_ising_gate(LONGITUDINAL_FIELD, b)).data)
    print(f"gate: h={LONGITUDINAL_FIELD}, b=pi/4-{epsilon} -> "
          f"dual-unitary: {du}")

    target = backend.target
    gname_2q = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
    e2q = float(np.median([props.error for q in target[gname_2q]
                           if (props := target[gname_2q][q]) and props.error is not None]))
    t2s = np.array([target.qubit_properties[q].t2 for q in range(target.num_qubits)
                    if target.qubit_properties[q].t2])
    t2_med = float(np.median(t2s))

    neat = None
    if epsilon == 0:
        try:
            from qiskit_ibm_runtime.debug_tools import Neat
            neat = Neat(backend)
        except Exception as exc:
            print(f"(Neat stabilizer check unavailable: {exc})")

    print(f"backend {backend.name}: {target.num_qubits}q, "
          f"median e2q={e2q:.2e} ({gname_2q}), median T2={t2_med * 1e6:.0f} us")
    print(f"{'T':>3} {'width':>6} {'2q':>5} {'dur_us':>8} {'F_est':>7} {'XX_sim':>7} "
          f"{'bad_T2?':>8}")

    jobs = {}
    for T in t_values:
        qc = build_circuit(T, MIN_X, h=LONGITUDINAL_FIELD, b=b)
        targets_ = get_target_qubits(T, MIN_X)
        control = get_control_qubit(T, MIN_X)

        tqc = best_transpile(qc, backend)
        n2q = sum(1 for inst in tqc.data if inst.operation.num_qubits == 2)
        dur = tqc.estimate_duration(target, unit="s")
        F = (1 - e2q) ** n2q * np.exp(-dur / t2_med)

        # flag if routing grazed a dead qubit (e.g. miami's T2 ~ 10us outlier)
        used = tqc.layout.final_index_layout()
        used_t2 = np.array([target.qubit_properties[q].t2 or 0.0 for q in used])
        bad = bool((used_t2 < 0.2 * t2_med).any())

        obs = [o.apply_layout(tqc.layout) for o in
               xx_observables(qc.num_qubits, control, targets_)]

        xx_sim = None
        if neat is not None:
            try:
                vals = neat.noisy_sim([(tqc, obs)], cliffordize=True,
                                      seed_simulator=0)[0].vals
                xx_sim = float(np.asarray(vals)[-1])   # light-cone edge
            except Exception as exc:  # e.g. fractional-rx ISA the cliffordizer can't map
                print(f"(Neat stabilizer check failed, skipping: {exc})")
                neat = None

        jobs[T] = {"tqc": tqc, "observables": obs, "targets": targets_,
                   "control": control, "n2q": n2q, "duration_s": float(dur),
                   "f_est": float(F), "xx_sim_edge": xx_sim,
                   "layout": [int(q) for q in used], "bad_t2": bad}
        xx_str = f"{xx_sim:>7.3f}" if xx_sim is not None else f"{'-':>7}"
        print(f"{T:>3} {qc.num_qubits:>6} {n2q:>5} {dur * 1e6:>8.1f} {F:>7.3f} "
              f"{xx_str} {'YES' if bad else 'no':>8}")

    shots_per_rand = max(1, shots // N_TWIRL_RANDOMIZATIONS)
    n_circuits = len(t_values) * (N_TWIRL_RANDOMIZATIONS + TREX_CIRCUITS_PER_PUB)
    qpu_s = n_circuits * (CIRCUIT_OVERHEAD_S + shots_per_rand * SHOT_CYCLE_S)
    print(f"\nestimated QPU time: ~{qpu_s:.0f} s "
          f"({n_circuits} circuits x ~{CIRCUIT_OVERHEAD_S:.2f} s overhead + "
          f"{shots} shots/PUB; validated against run_0005's 735 s)")
    return jobs


def collect_results(run: Run, job, t_values: list, local: bool = False) -> None:
    """Block on the job, save evs/stds per T, record usage + final status.

    Reads targets/f_est back from the run's metadata so it works both inline
    after submit() and when re-attaching later via --fetch.
    """
    result = job.result()
    per_t = run.metadata["params"]["per_t"]
    for i, T in enumerate(t_values):
        info = per_t[str(T)]
        evs = np.asarray(result[i].data.evs, dtype=float)
        stds = np.asarray(result[i].data.stds, dtype=float)
        run.save_array("per_t", f"T_{T}", evs=evs, stds=stds,
                       targets=np.asarray(info["targets"]))
        print(f"\n--- T = {T}  (F_est = {info['f_est']:.3f}) ---")
        print(f"{'x':>4}  {'<XX>':>8}  {'std':>7}")
        for k, _ in enumerate(info["targets"]):
            x = MIN_X + 0.5 * k
            print(f"{x:>4.1f}  {evs[k]:>8.4f}  {stds[k]:>7.4f}")

    meta: dict = {"status": "local_test" if local else "completed"}
    if hasattr(job, "usage"):  # RuntimeJobV2 only; local PrimitiveJob has no usage
        try:
            meta["qpu_usage_s"] = float(job.usage())
            print(f"\nactual QPU usage: {meta['qpu_usage_s']:.1f} s")
        except Exception as exc:
            print(f"\ncould not fetch job usage: {exc}")
    run.save_metadata(meta)
    print(f"\nsaved to {run.path}")


def submit(backend, jobs: dict, t_values: tuple = T_VALUES,
           shots: int = SHOTS, local: bool = False,
           epsilon: float = 0.0) -> None:
    from qiskit_ibm_runtime import EstimatorV2 as Estimator

    estimator = Estimator(mode=backend)
    estimator.options.default_shots = shots
    if not local:  # hardware-only options; local testing mode ignores/rejects them
        estimator.options.resilience_level = 1          # TREX readout mitigation
        estimator.options.dynamical_decoupling.enable = True
        estimator.options.dynamical_decoupling.sequence_type = "XY4"
        estimator.options.twirling.enable_gates = True  # Pauli-twirl 2q gates
        # Cap randomizations: server 'auto' is 64 shots/randomization, which
        # at 30k shots meant ~469 circuits/PUB and 735 s billed on run_0005.
        estimator.options.twirling.num_randomizations = N_TWIRL_RANDOMIZATIONS
        estimator.options.twirling.shots_per_randomization = max(
            1, shots // N_TWIRL_RANDOMIZATIONS)

    # Run folder + full provenance BEFORE submitting, so a paid job can never
    # be orphaned by a save-side failure.
    case = "du" if epsilon == 0 else f"eps{epsilon:g}"
    run = create_run(
        "hardware_pilot",
        "local_test" if local else f"{backend.name}_{case}",
        data_root=DATA_ROOT,
        params={
            "backend": backend.name, "t_values": list(t_values), "min_x": MIN_X,
            "h": LONGITUDINAL_FIELD, "b": float(TRANSVERSE_FIELD - epsilon),
            "epsilon": float(epsilon), "shots": shots,
            "local_test": local,
            "resilience_level": 0 if local else 1, "dd": None if local else "XY4",
            "gate_twirling": not local,
            "per_t": {str(T): {k: jobs[T][k] for k in
                               ("targets", "control", "n2q", "duration_s",
                                "f_est", "xx_sim_edge", "layout", "bad_t2")}
                      for T in t_values},
        },
    )
    with open(run.path / "circuits.qpy", "wb") as f:
        qpy.dump([jobs[T]["tqc"] for T in t_values], f)

    pubs = [(jobs[T]["tqc"], jobs[T]["observables"]) for T in t_values]
    job = estimator.run(pubs)
    run.save_metadata({"status": "submitted", "params": {"job_id": job.job_id()}})
    print(f"submitted job {job.job_id()} -- waiting for results...")

    try:
        collect_results(run, job, list(t_values), local=local)
    except BaseException:  # incl. Ctrl-C: results are recoverable, say how
        if not local:
            print(f"\nresults NOT saved -- recover once the job finishes with:\n"
                  f"  python experiments/hardware_pilot.py --fetch {job.job_id()}",
                  file=sys.stderr)
        raise


def fetch(job_id: str) -> None:
    """Re-attach to a submitted job and save its results into its run folder."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    base = DATA_ROOT / "hardware_pilot"
    run = None
    for p in sorted(base.iterdir()) if base.is_dir() else []:
        meta_path = p / "metadata.json"
        if meta_path.exists():
            candidate = open_run(p)
            if candidate.metadata.get("params", {}).get("job_id") == job_id:
                run = candidate
    if run is None:
        sys.exit(f"no run under {base} has job_id {job_id}")
    print(f"found {run.path} (status: {run.metadata.get('status', '?')})")

    service = QiskitRuntimeService()
    job = service.job(job_id)
    t_values = run.metadata["params"]["t_values"]
    collect_results(run, job, t_values, local=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true",
                        help="use FakeMiami snapshot (no credentials)")
    parser.add_argument("--submit", action="store_true",
                        help="actually submit to hardware (default: dry run)")
    parser.add_argument("--test-save", action="store_true",
                        help="end-to-end save test: run T=2 locally on FakeMiami "
                             "(Aer + noise model) and exercise the full result "
                             "parsing + data/ saving path; nothing leaves the machine")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="perturbation strength: b = pi/4 - epsilon "
                             "(0 = dual-unitary case)")
    parser.add_argument("--fractional", action="store_true",
                        help="load the live backend with use_fractional_gates=True "
                             "so the transpiler can target native rzz; preflight "
                             "comparison only -- submit is blocked until twirling "
                             "compatibility is verified")
    parser.add_argument("--fetch", metavar="JOB_ID",
                        help="re-attach to a previously submitted job and save "
                             "its results into the matching run folder")
    args = parser.parse_args()

    if args.fetch:
        fetch(args.fetch)
        return

    if args.fractional and (args.fake or args.test_save):
        sys.exit("--fractional needs the live backend "
                 "(the FakeMiami snapshot exposes no fractional gates)")
    if args.fractional and args.submit:
        sys.exit("refusing to --submit with --fractional: submit enables gate "
                 "twirling, whose compatibility with fractional rzz must be "
                 "verified first (compare preflight tables, then lift this guard)")

    if args.test_save:
        from qiskit_ibm_runtime.fake_provider import FakeMiami
        backend = FakeMiami()
        t_values = (2,)
        jobs = preflight(backend, t_values=t_values, shots=1000, epsilon=args.epsilon)
        submit(backend, jobs, t_values=t_values, shots=1000, local=True,
               epsilon=args.epsilon)
        return

    if args.fake:
        from qiskit_ibm_runtime.fake_provider import FakeMiami
        backend = FakeMiami()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        try:
            service = QiskitRuntimeService()
        except Exception as exc:  # no saved credentials
            sys.exit(f"could not connect to IBM Runtime ({exc}); try --fake")
        backend = service.backend(BACKEND_NAME,
                                  use_fractional_gates=args.fractional)
        if args.fractional:
            has_rzz = "rzz" in backend.target.operation_names
            print(f"fractional gates: native rzz "
                  f"{'AVAILABLE' if has_rzz else 'NOT available'} on {backend.name}")

    jobs = preflight(backend, epsilon=args.epsilon)

    if not args.submit:
        print("\nDRY RUN ONLY -- re-run with --submit to send to the device.")
        return
    if args.fake:
        sys.exit("refusing to --submit against a fake backend")
    submit(backend, jobs, epsilon=args.epsilon)


if __name__ == "__main__":
    main()
