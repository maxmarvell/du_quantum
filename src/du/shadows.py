from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister

X_BIAS = (0.8, 0.1, 0.1)   # (p_X, p_Y, p_Z)


def sample_settings(n_qubits: int, n_settings: int,
                    bias: tuple = X_BIAS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(3, size=(n_settings, n_qubits), p=bias).astype(np.int8)


def _apply_basis(qc: QuantumCircuit, qubit: int, basis: int) -> None:
    if basis == 0:      # X
        qc.rz(np.pi / 2, qubit)
        qc.sx(qubit)
        qc.rz(np.pi / 2, qubit)
    elif basis == 1:    # Y
        qc.rz(np.pi / 2, qubit)
        qc.sx(qubit)


def shadow_circuits(tqc: QuantumCircuit, physical_qubits: list[int],
                    settings: np.ndarray) -> list[QuantumCircuit]:
    # tqc: ISA circuit already pinned to the layout (no measurements);
    # physical_qubits: which device qubits carry the logical register, in
    # register order -- the classical bit order of every returned circuit.
    circuits = []
    for setting in settings:
        qc = tqc.copy()
        creg = ClassicalRegister(len(physical_qubits), "shadow")
        qc.add_register(creg)
        for k, (phys, basis) in enumerate(zip(physical_qubits, setting)):
            _apply_basis(qc, phys, int(basis))
            qc.measure(phys, creg[k])
        circuits.append(qc)
    return circuits
