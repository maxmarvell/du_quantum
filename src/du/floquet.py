from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit

from du.simulation import kicked_ising_gate


def brickwork_phase(T: float, x_min: float) -> int:
    # the causal block's first Floquet layer contains the point charge's
    # first bond (origin, origin+1), origin = u - 1: anchor the template there
    origin = math.floor(T + 1 - x_min) - 1
    return origin % 2


def full_brickwork_layers(n_system: int, h: float = 0, b: float = np.pi / 4,
                          phase: int = 0) -> QuantumCircuit:
    # phase = origin % 2 (see brickwork_phase): the FIRST template layer is
    # the class containing the point charge's first bond
    gate = kicked_ising_gate(h, b)
    qc = QuantumCircuit(n_system)
    for i in range(phase, n_system - 1, 2):
        qc.append(gate, [i, i + 1])
    qc.barrier()
    for i in range(1 - phase, n_system - 1, 2):
        qc.append(gate, [i, i + 1])
    return qc


def layer_parity_classes(qc: QuantumCircuit) -> list[str]:
    # walk the circuit's time-layers and classify each Floquet layer by the
    # parity of its bonds' start indices; raises if any layer mixes parities
    # (which would mean the full even/odd learning layers cannot represent it)
    from qiskit.converters import circuit_to_dag

    classes = []
    for layer in circuit_to_dag(qc).layers():
        bonds = [sorted(qc.find_bit(q).index for q in node.qargs)
                 for node in layer["graph"].op_nodes()
                 if node.op.name == "KI"]
        if not bonds:
            continue
        parities = {b[0] % 2 for b in bonds}
        if len(parities) > 1:
            raise ValueError(f"mixed-parity Floquet layer: bonds {bonds}")
        classes.append("even" if parities.pop() == 0 else "odd")
    return classes
