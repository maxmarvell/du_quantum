from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp


def xx_observables(n_qubits: int, control: int, targets: list[int]) -> list[SparsePauliOp]:
    return [
        SparsePauliOp.from_sparse_list([("XX", [control, qt], 1.0)], num_qubits=n_qubits)
        for qt in targets
    ]


def best_transpile(qc: QuantumCircuit, backend, n_seeds: int = 6) -> QuantumCircuit:
    target = backend.target
    t2s = np.array([target.qubit_properties[q].t2 or 0.0
                    for q in range(target.num_qubits)])
    t2_bad = 0.2 * float(np.median(t2s[t2s > 0]))

    n_seeds = n_seeds if qc.num_qubits <= 24 else 4 * n_seeds
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
