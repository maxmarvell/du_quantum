"""DU light-cone circuits + X-basis measurement (hardware-pilot core).

The kicked-Ising gate at the self-dual point (b = pi/4) is dual-unitary; on
the Bell-pair Choi state the operator transported to the light-cone edge is X,
so the edge correlator <X_control X_edge> = 1 while the strict interior
vanishes. All X-type correlators commute, so one fixed all-X measurement
setting estimates every <X_c X_t> at once with no shadow-style 3^k penalty --
`x_basis_measurement` is the default measurement everywhere (simulation and
hardware; on hardware EstimatorV2 realises the same setting via observable
grouping).
"""

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


def _is_noiseless_aer(backend: AerSimulator | None) -> bool:
    if backend is None:
        return True
    if not isinstance(backend, AerSimulator):
        return False
    return getattr(backend.options, "noise_model", None) is None


def x_basis_measurement(
    qc: QuantumCircuit,
    n_shots: int,
    backend: AerSimulator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Fixed all-X-basis measurement: rotate every qubit into X and read Z.

    All X-only Pauli strings commute, so this single setting estimates every
    XX-type correlator at once. The setting is fixed, so it is ONE circuit
    sampled n_shots times (Aer evolves the state once). Returns outcomes of
    shape (n_shots, n); feed to `expect_x`.
    """
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
    """<X_{support[0]} X_{support[1]} ...> from all-X-basis outcomes."""
    signs = np.prod(1 - 2 * outcomes[:, support], axis=1)
    return float(np.mean(signs))
