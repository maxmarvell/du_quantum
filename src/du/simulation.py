from __future__ import annotations

import numpy as np
import math
from scipy.linalg import expm

from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_XX = np.kron(_X, _X)
_YY = np.kron(_Y, _Y)
_ZZ = np.kron(_Z, _Z)


def du_gate(J: float) -> np.ndarray:
    H = (np.pi / 4) * (_XX + _YY) + J * _ZZ
    return expm(-1j * H)


def perturbed_du_gate(J: float, eps: float, P: np.ndarray) -> np.ndarray:
    return expm(-1j * eps * P) @ du_gate(J)


def _bell_pairs(qc: QuantumCircuit, N: int) -> None:
    for i in range(N):
        qc.h(i)
        qc.cx(i, i + N)


def _brickwork_layer(qc: QuantumCircuit, N: int,
                     gate: UnitaryGate, offset: int) -> None:
    for i in range(offset, N - 1, 2):
        qc.append(gate, [i, i + 1])


def get_causal_block_dims(x: float, t: float) -> tuple[int, int]:
    return math.floor(t + 1 - x), math.ceil(t + x)


def build_circuit(N: int, depth: int, J: float, eps: float,
                  P: np.ndarray | None = None) -> QuantumCircuit:
    """2N-qubit Choi circuit: Bell-pair prep + `depth` brickwork layers on A."""
    qc = QuantumCircuit(2 * N)
    _bell_pairs(qc, N)
    gate = UnitaryGate(perturbed_du_gate(J, eps, P))
    for t in range(depth):
        _brickwork_layer(qc, N, gate, offset=t % 2)
    return qc


def _build_causal_block_circuit(x: float, t: float, gate: UnitaryGate) -> QuantumCircuit:

    u, v = get_causal_block_dims(x, t)

    n_qubits = (u + v)
    qc = QuantumCircuit(2 * n_qubits)
    _bell_pairs(qc, n_qubits)

    for i in range(v):
        for j in range(u + i, i, -1):
            qc.append(gate, [j-1, j])

    return qc


def _build_causal_cone_circuit(x_min: float, t: float, gate: UnitaryGate) -> QuantumCircuit:

    u_min, v_min = get_causal_block_dims(x_min, t)
    _, v_max = get_causal_block_dims(t, t)

    n_qubits = u_min + v_max
    qc = QuantumCircuit(2 * n_qubits)
    _bell_pairs(qc, n_qubits)

    for i in range(v_min):
        for j in range(u_min + i, i, -1):
            qc.append(gate, [j-1, j])

    for _i, i in enumerate(range(v_min, v_max)):
        for j in range(u_min + v_min, i + 1, -1):
            qc.append(gate, [j + _i - 1, j + _i])

    return qc


def _append_pauli_measure(qc: QuantumCircuit, bases: np.ndarray) -> QuantumCircuit:
    """Rotate each qubit into the chosen Pauli basis, then measure all-to-all."""
    n = qc.num_qubits
    creg = ClassicalRegister(n, "shadow")
    qc = qc.copy()
    qc.add_register(creg)
    for q, b in enumerate(bases):
        if b == 0:        # X: H
            qc.h(q)
        elif b == 1:      # Y: S† H
            qc.sdg(q); qc.h(q)
        # b == 2 is Z, no rotation
        qc.measure(q, creg[q])
    return qc


def classical_shadow(qc: QuantumCircuit, n_shots: int,
                     backend: AerSimulator | None = None,
                     seed: int | None = None,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Random-Pauli classical shadow of `qc`.

    Returns (bases, outcomes), each of shape (n_shots, n_qubits), with
    bases ∈ {0:X, 1:Y, 2:Z} and outcomes ∈ {0,1}.
    """
    rng = np.random.default_rng(seed)
    n = qc.num_qubits
    bases = rng.integers(0, 3, size=(n_shots, n))

    circuits = [_append_pauli_measure(qc, bases[s]) for s in range(n_shots)]
    backend = backend or AerSimulator()
    circuits = transpile(circuits, backend)
    result = backend.run(circuits, shots=1).result()

    outcomes = np.empty((n_shots, n), dtype=np.int8)
    for s in range(n_shots):
        bitstr = next(iter(result.get_counts(s)))   # one shot per circuit
        outcomes[s] = np.fromiter(reversed(bitstr), dtype=np.int8)  # little-endian
    return bases, outcomes


def expect_pauli(bases: np.ndarray, outcomes: np.ndarray,
                 support: np.ndarray, paulis: np.ndarray) -> float:
    """Shadow estimate of <P> for a Pauli P with non-identity Paulis `paulis`
    ({0:X, 1:Y, 2:Z}) on sites `support`."""
    hit = np.all(bases[:, support] == paulis, axis=1)
    signs = np.prod(1 - 2 * outcomes[:, support], axis=1)
    return (3 ** len(support)) * np.mean(hit * signs)


def build_and_measure(x: float, t: float, gate: UnitaryGate, O: str = 'ZZ') -> QuantumCircuit:

    qc = _build_causal_block_circuit(x, t, gate)

    u, v = get_causal_block_dims(x, t)
    n_qubits = u + v

    target_qubit_0 = u - 1
    target_qubit_1 = u + int(2 * x) - 1

    if O == 'XX':
        qc.h(target_qubit_0 + n_qubits)
        qc.h(target_qubit_1)

    qc.cx(target_qubit_0 + n_qubits, target_qubit_1)

    creg = ClassicalRegister(1, name="m")
    qc.add_register(creg)
    qc.measure(target_qubit_1, creg[0])

    return qc

_AER_BASIS = ["cx", "rz", "sx", "x", "h", "id"]


def make_depolarizing_noise(two_qubit_err: float = 5e-3,
                            one_qubit_err: float = 1e-4) -> NoiseModel:
    nm = NoiseModel(basis_gates=_AER_BASIS)
    err1 = depolarizing_error(one_qubit_err, 1)
    err2 = depolarizing_error(two_qubit_err, 2)
    for g in ("h", "rz", "sx", "x"):
        nm.add_all_qubit_quantum_error(err1, [g])
    nm.add_all_qubit_quantum_error(err2, ["cx"])
    return nm


def make_du_gate_noise(p: float) -> NoiseModel:
    """Depolarizing noise on each DU UnitaryGate (the 'unitary' instruction)."""
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["unitary"])
    return nm


def make_miami_backend() -> AerSimulator:
    from qiskit_ibm_runtime.fake_provider import FakeMiami
    return AerSimulator.from_backend(FakeMiami())


def fold_gates(qc: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError("scale_factor must be an odd positive integer")
    n_extra = (scale_factor - 1) // 2
    out = qc.copy_empty_like()
    for instr in qc.data:
        out.append(instr)
        if n_extra and instr.operation.name in {"barrier", "measure"}:
            continue
        for _ in range(n_extra):
            out.append(instr.operation.inverse(), instr.qubits)
            out.append(instr.operation, instr.qubits)
    return out


def _counts_to_z_expectation(counts: dict[str, int], shots: int) -> float:
    p0 = counts.get("0", 0) / shots
    p1 = counts.get("1", 0) / shots
    return p0 - p1


def correlator_noisy(x: float, t: float, J: float, eps: float,
                     O: str = "ZZ",
                     noise_model: NoiseModel | None = None,
                     backend: AerSimulator | None = None,
                     scale_factor: int = 1,
                     shots: int = 4096,
                     P: np.ndarray | None = None) -> float:
    
    U = du_gate(J) if P is None else perturbed_du_gate(J, eps, P)
    gate = UnitaryGate(U)
    qc = build_and_measure(x, t, gate, O=O)

    if backend is None:
        if noise_model is not None:
            backend = AerSimulator(noise_model=noise_model)
        else:
            backend = make_miami_backend()

    qc_t = transpile(qc, backend=backend, optimization_level=0)
    if scale_factor != 1:
        qc_t = fold_gates(qc_t, scale_factor)

    counts = backend.run(qc_t, shots=shots).result().get_counts()
    return _counts_to_z_expectation(counts, shots)


if __name__ == "__main__":

    J = 0.5
    DU = du_gate(J)
    gate = UnitaryGate(DU)
    qc = build_and_measure(4, 4, gate)
    qc.draw('mpl', filename='circuit.png')