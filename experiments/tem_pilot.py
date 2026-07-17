from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from qiskit import qpy

from du.simulation import build_circuit, get_control_qubit, get_target_qubits
from du.utils import best_transpile, xx_observables
from du.experiments import Run, create_run, open_run

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"  # cwd-independent

BACKEND_NAME = "ibm_miami"
CATALOG_CHANNEL = "ibm_cloud"
T_VALUES = (2,)
MIN_X = 1
LONGITUDINAL_FIELD = 0  # h
TRANSVERSE_FIELD = np.pi / 4  # b = pi/4 -> self-dual (dual-unitary) point
SHOTS = 5_000
N_TRANSPILE_SEEDS = 6

TEM_OPTIONS = {
    "default_shots": SHOTS,
    "tem_max_bond_dimension": 256,
    "compute_shadows_bias_from_observable": True,
    "num_randomizations": 32,
    "shots_per_randomization": 128,
    "max_layers_to_learn": 2 * len(T_VALUES),  # even/odd brickwork per transpiled PUB
    "layer_pair_depths": [0, 1, 2, 4, 16, 32],
    "private": False,
}


def load_tem():
    from qiskit_ibm_catalog import QiskitFunctionsCatalog

    return QiskitFunctionsCatalog(channel=CATALOG_CHANNEL).load("algorithmiq/tem")


def preflight(
    backend, t_values: tuple = T_VALUES, shots: int = SHOTS, epsilon: float = 0.0
) -> dict:
    from qiskit.quantum_info import Operator
    from du.simulation import is_dual_unitary, kicked_ising_gate

    b = TRANSVERSE_FIELD - epsilon
    du = is_dual_unitary(Operator(kicked_ising_gate(LONGITUDINAL_FIELD, b)).data)
    print(f"gate: h={LONGITUDINAL_FIELD}, b=pi/4-{epsilon} -> dual-unitary: {du}")

    target = backend.target
    gname_2q = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
    e2q = float(
        np.median(
            [
                props.error
                for q in target[gname_2q]
                if (props := target[gname_2q][q]) and props.error is not None
            ]
        )
    )
    t2s = np.array(
        [
            target.qubit_properties[q].t2
            for q in range(target.num_qubits)
            if target.qubit_properties[q].t2
        ]
    )
    t2_med = float(np.median(t2s))

    print(
        f"backend {backend.name}: {target.num_qubits}q, "
        f"median e2q={e2q:.2e} ({gname_2q}), median T2={t2_med * 1e6:.0f} us"
    )
    print(f"{'T':>3} {'width':>6} {'2q':>5} {'dur_us':>8} {'F_est':>7} {'bad_T2?':>8}")

    jobs = {}
    for T in t_values:
        qc = build_circuit(T, MIN_X, h=LONGITUDINAL_FIELD, b=b)
        targets_ = get_target_qubits(T, MIN_X)
        control = get_control_qubit(T, MIN_X)

        tqc = best_transpile(qc, backend, n_seeds=N_TRANSPILE_SEEDS)
        n2q = sum(1 for inst in tqc.data if inst.operation.num_qubits == 2)
        dur = tqc.estimate_duration(target, unit="s")
        F = (1 - e2q) ** n2q * np.exp(-dur / t2_med)  # predicts the RAW edge

        # flag if routing grazed a dead qubit (e.g. miami's T2 ~ 10us outlier)
        used = tqc.layout.final_index_layout()
        used_t2 = np.array([target.qubit_properties[q].t2 or 0.0 for q in used])
        bad = bool((used_t2 < 0.2 * t2_med).any())

        obs = [
            o.apply_layout(tqc.layout)
            for o in xx_observables(qc.num_qubits, control, targets_)
        ]

        jobs[T] = {
            "tqc": tqc,
            "observables": obs,
            "targets": targets_,
            "control": control,
            "n2q": n2q,
            "duration_s": float(dur),
            "f_est": float(F),
            "layout": [int(q) for q in used],
            "bad_t2": bad,
        }
        print(
            f"{T:>3} {qc.num_qubits:>6} {n2q:>5} {dur * 1e6:>8.1f} {F:>7.3f} "
            f"{'YES' if bad else 'no':>8}"
        )

    cycle = getattr(backend, "default_rep_delay", None) or 250e-6
    # x9: Pauli-basis settings per (depth, randomization) -- see the note in
    # qiskit_ibm_runtime NoiseLearnerOptions. Validated on the T=2 probe:
    # predicted 1769 s learning vs 1782 s billed (IBM's dashboard estimate
    # omits this factor and came out ~9x low).
    nl_per_layer = (
        TEM_OPTIONS["num_randomizations"]
        * TEM_OPTIONS["shots_per_randomization"]
        * len(TEM_OPTIONS["layer_pair_depths"])
        * 9
    )
    meas_s = len(t_values) * shots * cycle
    nl_lo = 2 * nl_per_layer * cycle  # best case: PUB layouts coincide
    nl_hi = TEM_OPTIONS["max_layers_to_learn"] * nl_per_layer * cycle  # all distinct
    print(
        f"\ntrue QPU runtime estimate: ~{meas_s + nl_lo:.0f}-{meas_s + nl_hi:.0f} s "
        f"({(meas_s + nl_lo) / 60:.1f}-{(meas_s + nl_hi) / 60:.1f} min)\n"
        f"  noise learning {nl_lo:.0f}-{nl_hi:.0f} s "
        f"(2-{TEM_OPTIONS['max_layers_to_learn']} unique layers x "
        f"{nl_per_layer} shots x {cycle * 1e3:.1f} ms/shot, "
        f"validated to <1% on the T=2 probe)\n"
        f"  measurement {meas_s:.0f} s "
        f"({len(t_values)} PUBs x {shots} shots x {cycle * 1e3:.1f} ms/shot)\n"
        f"  (ignore IBM's dashboard estimate -- it omits the x9 learning bases "
        f"and miami's 4 ms rep_delay)"
    )
    return jobs


def collect_results(run: Run, job, t_values: list) -> None:

    result = job.result()
    per_t = run.metadata["params"]["per_t"]
    for i, T in enumerate(t_values):
        info = per_t[str(T)]
        pub_res = result[i]
        evs = np.asarray(pub_res.data.evs, dtype=float)
        stds = np.asarray(pub_res.data.stds, dtype=float)
        raw_evs = np.asarray(pub_res.metadata["evs_non_mitigated"], dtype=float)
        raw_stds = np.asarray(pub_res.metadata["stds_non_mitigated"], dtype=float)
        run.save_array(
            "per_t",
            f"T_{T}",
            evs=evs,
            stds=stds,
            raw_evs=raw_evs,
            raw_stds=raw_stds,
            targets=np.asarray(info["targets"]),
        )
        print(f"\n--- T = {T}  (F_est = {info['f_est']:.3f}) ---")
        print(f"{'x':>4}  {'<XX>_tem':>9} {'std':>7}  {'<XX>_raw':>9} {'std':>7}")
        for k, _ in enumerate(info["targets"]):
            x = MIN_X + 0.5 * k
            print(
                f"{x:>4.1f}  {evs[k]:>9.4f} {stds[k]:>7.4f}  "
                f"{raw_evs[k]:>9.4f} {raw_stds[k]:>7.4f}"
            )

    meta: dict = {"status": "completed"}
    try:
        usage = result.metadata["resource_usage"]
        meta["qpu_usage_s"] = float(usage["RUNNING: EXECUTING_QPU"]["QPU_TIME"])
        print(f"\nactual QPU usage: {meta['qpu_usage_s']:.1f} s")
        cpu = sum(v.get("CPU_TIME", 0.0) for v in usage.values())
        if cpu:
            meta["classical_usage_s"] = float(cpu)
            print(f"classical (cloud) usage: {cpu:.1f} s")
    except Exception as exc:
        print(f"\ncould not read resource usage: {exc}")
    run.save_metadata(meta)
    print(f"\nsaved to {run.path}")


def submit(
    backend, jobs: dict, t_values: tuple = T_VALUES, epsilon: float = 0.0
) -> None:
    tem = load_tem()

    case = "du" if epsilon == 0 else f"eps{epsilon:g}"
    run = create_run(
        "tem_pilot",
        f"{backend.name}_{case}",
        data_root=DATA_ROOT,
        params={
            "backend": backend.name,
            "t_values": list(t_values),
            "min_x": MIN_X,
            "h": LONGITUDINAL_FIELD,
            "b": float(TRANSVERSE_FIELD - epsilon),
            "epsilon": float(epsilon),
            "shots": SHOTS,
            "mitigation": "tem",
            "tem_options": TEM_OPTIONS,
            "per_t": {
                str(T): {
                    k: jobs[T][k]
                    for k in (
                        "targets",
                        "control",
                        "n2q",
                        "duration_s",
                        "f_est",
                        "layout",
                        "bad_t2",
                    )
                }
                for T in t_values
            },
        },
    )
    with open(run.path / "circuits.qpy", "wb") as f:
        qpy.dump([jobs[T]["tqc"] for T in t_values], f)

    pubs = [(jobs[T]["tqc"], jobs[T]["observables"]) for T in t_values]
    job = tem.run(pubs=pubs, backend_name=backend.name, options=TEM_OPTIONS)
    run.save_metadata({"status": "submitted", "params": {"job_id": job.job_id}})
    print(f"submitted TEM job {job.job_id} -- waiting for results...")

    try:
        collect_results(run, job, list(t_values))
    except BaseException:  # incl. Ctrl-C: results are recoverable, say how
        print(
            f"\nresults NOT saved -- recover once the job finishes with:\n"
            f"  python experiments/tem_pilot.py --fetch {job.job_id}",
            file=sys.stderr,
        )
        raise


def fetch(job_id: str) -> None:
    from qiskit_ibm_catalog import QiskitFunctionsCatalog

    base = DATA_ROOT / "tem_pilot"
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

    catalog = QiskitFunctionsCatalog(channel=CATALOG_CHANNEL)
    job = catalog.get_job_by_id(job_id)
    if job is None:
        sys.exit(f"catalog has no job {job_id}")
    collect_results(run, job, run.metadata["params"]["t_values"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fake", action="store_true", help="use FakeMiami snapshot (no credentials)"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="actually submit through the TEM function (default: dry run)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="perturbation strength: b = pi/4 - epsilon (0 = dual-unitary case)",
    )
    parser.add_argument(
        "--fetch",
        metavar="JOB_ID",
        help="re-attach to a previously submitted TEM job and "
        "save its results into the matching run folder",
    )
    args = parser.parse_args()

    if args.fetch:
        fetch(args.fetch)
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
