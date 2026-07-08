from __future__ import annotations

import numpy as np
import math

from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit.circuit import Gate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def kicked_ising_gate(h: float, b: float) -> Gate:
    sub = QuantumCircuit(2, name="KI")
    sub.rz(2 * h, 1)
    sub.rzz(np.pi / 2, 0, 1)  # J = pi/4
    sub.rx(2 * b, 0)
    sub.rx(2 * b, 1)
    sub.rzz(np.pi / 2, 0, 1)  # J = pi/4
    sub.rz(2 * h, 1)
    return sub.to_gate()


def is_unitary(U, atol=1e-12):
    U = np.asarray(U)
    n = U.shape[0]
    return np.allclose(U.conj().T @ U, np.eye(n), atol=atol)


def reshuffle(U):
    return U.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)


def is_dual_unitary(U, atol=1e-12):
    return is_unitary(U, atol) and is_unitary(reshuffle(U), atol)


def _bell_pairs(qc: QuantumCircuit, N: int) -> None:
    for i in range(N):
        qc.h(i)
        qc.cx(i, i + N)


def _get_causal_block_dims(x: float, t: float) -> tuple[int, int]:
    return math.floor(t + 1 - x), math.ceil(t + x)


def get_n_qubits(t: float, x: float) -> int:
    u, v = _get_causal_block_dims(x, t)
    return int(u + v) * 2


def build_circuit(
    t: float, x: float, h: float = 0, b: float = np.pi / 4, O: str = "XX"
) -> QuantumCircuit:

    u, v = _get_causal_block_dims(x, t)

    n_qubits = u + v
    qc = QuantumCircuit(2 * n_qubits)
    _bell_pairs(qc, n_qubits)

    gate = kicked_ising_gate(h, b)

    for i in range(v):
        for j in range(u + i, i, -1):
            qc.append(gate, [j - 1, j])

    target_qubit_0 = u - 1
    target_qubit_1 = u + int(2 * x) - 1

    if O == "XX":
        qc.h(target_qubit_0 + n_qubits)
        qc.h(target_qubit_1)

    qc.cx(target_qubit_0 + n_qubits, target_qubit_1)

    creg = ClassicalRegister(1, name="m")
    qc.add_register(creg)
    qc.measure(target_qubit_1, creg[0])

    return qc

def estimate_correlator(
    qc: QuantumCircuit,
    n_shots: int,
    backend: AerSimulator | None = None,
    seed: int | None = None,
) -> np.float64:

    backend = backend or AerSimulator()
    tqc = transpile(qc, backend)
    counts = backend.run(tqc, shots=n_shots, seed_simulator=seed).result().get_counts()
    p1 = counts.get('1', 0) / n_shots
    return 1 - 2 * p1


### Classical shadows

def build_circuit_cs(
    t: float,
    x_min: float,
    x_max: float | None = None,
    h: float = 0,
    b: float = np.pi / 4,
) -> QuantumCircuit:

    if x_max is not None:
        raise NotImplementedError("Need to implement differnt causal blocks")

    u_max, v_min = _get_causal_block_dims(x_min, t)
    _, v_max = _get_causal_block_dims(t, t)

    n_qubits = u_max + v_max
    qc = QuantumCircuit(2 * n_qubits)
    _bell_pairs(qc, n_qubits)

    gate = kicked_ising_gate(h, b)

    for i in range(v_min):
        for j in range(u_max + i, i, -1):
            qc.append(gate, [j - 1, j])

    for _i, i in enumerate(range(v_min, v_max)):
        for j in range(u_max + v_min, i + 1, -1):
            qc.append(gate, [j + _i - 1, j + _i])

    return qc


def _append_pauli_measure(qc: QuantumCircuit, bases: np.ndarray) -> QuantumCircuit:
    n = qc.num_qubits
    creg = ClassicalRegister(n, "shadow")
    qc = qc.copy()
    qc.add_register(creg)
    for q, b in enumerate(bases):
        if b == 0:  # X: H
            qc.h(q)
        elif b == 1:  # Y: S† H
            qc.sdg(q)
            qc.h(q)
        # b == 2 is Z, no rotation
        qc.measure(q, creg[q])
    return qc


def _is_noiseless_aer(backend: AerSimulator | None) -> bool:
    if backend is None:
        return True
    if not isinstance(backend, AerSimulator):
        return False
    return getattr(backend.options, "noise_model", None) is None


def run_classical_shadow(
    qc: QuantumCircuit,
    n_shots: int,
    backend: AerSimulator | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)
    n = qc.num_qubits
    bases = rng.integers(0, 3, size=(n_shots, n))
    backend = backend or AerSimulator()

    if _is_noiseless_aer(backend):
        source = QuantumCircuit(n)
        source.set_statevector(Statevector(qc))
    else:
        source = qc

    outcomes = np.empty((n_shots, n), dtype=np.int8)
    for s in range(n_shots):
        circuit = transpile(_append_pauli_measure(source, bases[s]), backend)
        shot_seed = None if seed is None else seed + s
        result = backend.run(circuit, shots=1, seed_simulator=shot_seed).result()
        bitstr = next(iter(result.get_counts()))
        outcomes[s] = np.fromiter(reversed(bitstr), dtype=np.int8)  # little-endian
    return bases, outcomes


def run_x_basis_shadow(
    qc: QuantumCircuit,
    n_shots: int,
    backend: AerSimulator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    
    n = qc.num_qubits
    backend = backend or AerSimulator()

    if _is_noiseless_aer(backend):
        meas = QuantumCircuit(n)
        meas.set_statevector(Statevector(qc))
    else:
        meas = qc.copy()

    creg = ClassicalRegister(n, "x")
    meas.add_register(creg)
    for q in range(n):
        meas.h(q)
        meas.measure(q, creg[q])

    meas = transpile(meas, backend)
    counts = backend.run(meas, shots=n_shots, seed_simulator=seed).result().get_counts()

    outcomes = np.empty((n_shots, n), dtype=np.int8)
    s = 0
    for bitstr, c in counts.items():
        outcomes[s:s + c] = np.fromiter(reversed(bitstr), dtype=np.int8)
        s += c
    return outcomes


def expect_x(outcomes: np.ndarray, support: np.ndarray) -> float:
    signs = np.prod(1 - 2 * outcomes[:, support], axis=1)
    return float(np.mean(signs))


def get_cs_targets(t: float, x_min: float, x_max: float | None = None) -> list[int]:
    x_max = x_max or t

    u_max, v_min = _get_causal_block_dims(x_min, t)
    u_min, v_max = _get_causal_block_dims(x_max, t)

    parity = (t - x_min) % 1 != 0

    return [i for i in range(u_min + v_min - parity - 1, u_max + v_max)]


def get_cs_control(t: float, x_min: float, x_max: float | None = None) -> int:
    x_max = x_max or t

    u, _ = _get_causal_block_dims(x_min, t)
    _, v = _get_causal_block_dims(x_max, t)

    return 2 * u + v - 1


def expect_pauli(
    bases: np.ndarray, outcomes: np.ndarray, support: np.ndarray, paulis: np.ndarray
) -> float:
    hit = np.all(bases[:, support] == paulis, axis=1)
    signs = np.prod(1 - 2 * outcomes[:, support], axis=1)
    return (3 ** len(support)) * np.mean(hit * signs)


if __name__ == "__main__":
    x_min = 2
    t = 5.5
    x_max = None

    x_max = x_max or t

    u_max, v_min = _get_causal_block_dims(x_min, t)
    u_min, v_max = _get_causal_block_dims(x_max, t)

    parity = (t - x_min) % 1 == 0

    print(u_max, v_max, u_min, v_min, parity)

    print(get_cs_targets(t, x_min))

    qc = build_circuit_cs(t, x_min)
    qc.draw("mpl", filename="circuit.png")

    print(get_cs_control(t, x_min))
