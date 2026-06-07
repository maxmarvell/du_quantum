from __future__ import annotations

import pickle
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.noise_learner import NoiseLearner
from qiskit_ibm_runtime.options import NoiseLearnerOptions

from du.simulation import perturbed_du_gate


TARGET = "ibm_miami"
N = 6                    # A-register size; also chain length on hardware
J, EPS = 0.7, 0.15       # gate parameters; noise depends on the actual gate executed
OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def linear_chain(backend, length: int) -> list[int]:

    cm = backend.coupling_map
    if cm is None:
        raise RuntimeError(f"{backend.name} has no coupling map")
    neighbors: dict[int, set[int]] = {}
    for a, b in cm.get_edges():
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)

    # DFS for a longest simple path starting from each vertex (heuristic)
    best: list[int] = []
    for start in neighbors:
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) > len(best):
                best = path
            if len(best) >= length:
                return best[:length]
            for nb in neighbors[node]:
                if nb not in path:
                    stack.append((nb, path + [nb]))
    if len(best) < length:
        raise RuntimeError(f"could not find a chain of length {length} on {backend.name}")
    return best[:length]

def brickwork_layer(N_logical: int, offset: int, gate_4x4) -> QuantumCircuit:
    qc = QuantumCircuit(N_logical, name=f"brickwork_offset_{offset}")
    for i in range(offset, N_logical - 1, 2):
        qc.unitary(gate_4x4, [i, i + 1])
    return qc


def main() -> None:
    service = QiskitRuntimeService()
    backend = service.backend(TARGET)
    print(f"Backend  : {backend.name}")
    print(f"basis    : {backend.operation_names}")
    print(f"# qubits : {backend.num_qubits}")

    chain = linear_chain(backend, N)
    print(f"Chain    : {chain}")

    gate = perturbed_du_gate(J, EPS)
    layer_even_logical = brickwork_layer(N, offset=0, gate_4x4=gate)
    layer_odd_logical  = brickwork_layer(N, offset=1, gate_4x4=gate)

    # Transpile each layer into native gates on the chosen physical qubits.
    layers_isa = transpile(
        [layer_even_logical, layer_odd_logical],
        backend=backend,
        initial_layout=chain,
        optimization_level=1,
    )
    for q, qc in zip(("even", "odd"), layers_isa):
        print(f"\nLayer {q} after transpile (ops): "
              f"{dict(qc.count_ops())}")

    options = NoiseLearnerOptions()
    options.num_randomizations    = 32
    options.shots_per_randomization = 128
    options.layer_pair_depths     = [0, 1, 2, 4, 16, 32]
    options.twirling_strategy     = "active"
    options.max_execution_time    = 60 * 60   # 1 h ceiling

    learner = NoiseLearner(mode=backend, options=options)
    job = learner.run(layers_isa)
    print(f"\nSubmitted job: {job.job_id()}   status: {job.status()}")
    print("Waiting for result (this can take a while in queue)...")
    result = job.result()

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"noise_miami_{job.job_id()}.pkl"
    with out_path.open("wb") as f:
        pickle.dump({
            "backend": backend.name,
            "chain": chain,
            "J": J, "eps": EPS,
            "layer_errors": list(result),     # iterable of LayerError
        }, f)
    print(f"Saved: {out_path.name}")

    for i, layer_err in enumerate(result):
        print(f"\n=== Layer {i} ({['even','odd'][i]} brickwork) ===")
        print(f"  physical qubits used: {layer_err.qubits}")
        err = layer_err.error           # PauliLindbladError-like
        # Generators and rates layout: list-aligned Pauli strings + lambda values
        gens  = list(err.generators)
        rates = list(err.rates)
        print(f"  # generators learned: {len(gens)}")
        print(f"  {'generator':>16}  {'rate (lambda)':>14}")
        for g, r in sorted(zip(gens, rates), key=lambda x: -abs(x[1]))[:20]:
            print(f"  {str(g):>16}  {r:>14.4e}")


if __name__ == "__main__":
    main()
