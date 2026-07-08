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

Expected signal: edge <XX> ~ F_est (0.8-0.95), interior ~ 0.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

from du.simulation import build_circuit_cs, get_cs_control, get_cs_targets
from du.experiment_io import create_run


BACKEND_NAME = "ibm_miami"
T_VALUES = (2, 3, 4, 5, 6)
MIN_X = 1
LONGITUDINAL_FIELD = 0          # h
TRANSVERSE_FIELD = np.pi / 4    # b = pi/4 -> self-dual (dual-unitary) point
SHOTS = 30_000
N_TRANSPILE_SEEDS = 6           # best-of-N Sabre; never pin a linear chain
SHOT_CYCLE_S = 260e-6           # ~circuit + readout + default rep_delay
TREX_OVERHEAD = 3.0             # rough factor for readout-twirl calibration


def xx_observables(n_qubits: int, control: int, targets: list[int]) -> list[SparsePauliOp]:
    """<X_control X_target> for every target (little-endian label build)."""
    obs = []
    for qt in targets:
        label = ["I"] * n_qubits
        label[control] = "X"
        label[qt] = "X"
        obs.append(SparsePauliOp("".join(reversed(label))))
    return obs


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
        tqc = transpile(qc, backend, optimization_level=3, seed_transpiler=seed)
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
    """
    from qiskit.quantum_info import Operator
    from du.simulation import is_dual_unitary, kicked_ising_gate

    b = TRANSVERSE_FIELD - epsilon
    du = is_dual_unitary(Operator(kicked_ising_gate(LONGITUDINAL_FIELD, b)).data)
    print(f"gate: h={LONGITUDINAL_FIELD}, b=pi/4-{epsilon} -> "
          f"dual-unitary: {du}")

    target = backend.target
    e2q = float(np.median([props.error for q in target["cz"]
                           if (props := target["cz"][q]) and props.error is not None]))
    t2s = np.array([target.qubit_properties[q].t2 for q in range(target.num_qubits)
                    if target.qubit_properties[q].t2])
    t2_med = float(np.median(t2s))

    print(f"backend {backend.name}: {target.num_qubits}q, "
          f"median e2q={e2q:.2e}, median T2={t2_med * 1e6:.0f} us")
    print(f"{'T':>3} {'width':>6} {'2q':>5} {'dur_us':>8} {'F_est':>7} {'bad_T2?':>8}")

    jobs = {}
    for T in t_values:
        qc = build_circuit_cs(T, MIN_X, h=LONGITUDINAL_FIELD, b=b)
        targets_ = get_cs_targets(T, MIN_X)
        control = get_cs_control(T, MIN_X)

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
        jobs[T] = {"tqc": tqc, "observables": obs, "targets": targets_,
                   "control": control, "n2q": n2q, "duration_s": float(dur),
                   "f_est": float(F)}
        print(f"{T:>3} {qc.num_qubits:>6} {n2q:>5} {dur * 1e6:>8.1f} {F:>7.3f} "
              f"{'YES' if bad else 'no':>8}")

    qpu_s = len(t_values) * shots * SHOT_CYCLE_S * TREX_OVERHEAD
    print(f"\nestimated QPU time: ~{qpu_s:.0f} s "
          f"({len(t_values)} PUBs x {shots} shots, ~x{TREX_OVERHEAD:.0f} TREX overhead)")
    return jobs


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

    pubs = [(jobs[T]["tqc"], jobs[T]["observables"]) for T in t_values]
    job = estimator.run(pubs)
    print(f"submitted job {job.job_id()} -- waiting for results...")

    case = "du" if epsilon == 0 else f"eps{epsilon:g}"
    run = create_run(
        "x_basis_hardware_pilot",
        "local_test" if local else f"{backend.name}_{case}",
        params={
            "backend": backend.name, "t_values": list(t_values), "min_x": MIN_X,
            "h": LONGITUDINAL_FIELD, "b": float(TRANSVERSE_FIELD - epsilon),
            "epsilon": float(epsilon), "shots": shots,
            "local_test": local,
            "resilience_level": 0 if local else 1, "dd": None if local else "XY4",
            "gate_twirling": not local,
            "job_id": job.job_id(),
            "per_t": {str(T): {k: jobs[T][k] for k in
                               ("targets", "control", "n2q", "duration_s", "f_est")}
                      for T in t_values},
        },
    )

    result = job.result()
    for i, T in enumerate(t_values):
        evs = np.asarray(result[i].data.evs, dtype=float)
        stds = np.asarray(result[i].data.stds, dtype=float)
        run.save_array("per_t", f"T_{T}", evs=evs, stds=stds,
                       targets=np.asarray(jobs[T]["targets"]))
        print(f"\n--- T = {T}  (F_est = {jobs[T]['f_est']:.3f}) ---")
        print(f"{'x':>4}  {'<XX>':>8}  {'std':>7}")
        for k, _ in enumerate(jobs[T]["targets"]):
            x = MIN_X + 0.5 * k
            print(f"{x:>4.1f}  {evs[k]:>8.4f}  {stds[k]:>7.4f}")

    run.save_metadata({"status": "local_test" if local else "completed"})
    print(f"\nsaved to {run.path}")


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
    args = parser.parse_args()

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
        backend = service.backend(BACKEND_NAME)

    jobs = preflight(backend, epsilon=args.epsilon)

    if not args.submit:
        print("\nDRY RUN ONLY -- re-run with --submit to send to the device.")
        return
    if args.fake:
        sys.exit("refusing to --submit against a fake backend")
    submit(backend, jobs, epsilon=args.epsilon)


if __name__ == "__main__":
    main()
