"""Classical-shadow estimate of the DU light-cone correlator.

`build_and_measure` reads out the two-point parity by applying a CX and
measuring the target in Z — equivalent to measuring the Pauli string
    O = CX†(I⊗Z)CX = Z⊗Z.
Dual-unitarity then predicts, on the Bell-pair Choi state produced by
`_build_causal_cone_circuit`,
    <Z_{N+0} Z_{q_t}> = 1   at the LC-edge target q_t (x = t)
                      = 0   in the strict interior (x_min ≤ x < t).
The companion test_du_lightcone covers the edge case via direct CX +
projective measurement; here we recover the *same* prediction from one
random-Pauli shadow run, scanning every A-side target position off the
same (bases, outcomes) dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import SparsePauliOp, Statevector

from du.simulation import (
    _build_causal_cone_circuit,
    classical_shadow,
    du_gate,
    expect_pauli,
)


SHOTS = 4000
TOL = 0.15  # weight-2 Pauli std ≈ 3/sqrt(SHOTS) ≈ 0.05; 3σ margin
PAULI_Z = 2


def _zz(bases: np.ndarray, outcomes: np.ndarray, q_c: int, q_t: int) -> float:
    return expect_pauli(
        bases, outcomes, np.array([q_c, q_t]), np.array([PAULI_Z, PAULI_Z])
    )


def _find_lc_edge(sv: Statevector, q_c: int, n_side: int, n_total: int) -> int:
    """A-side qubit q_t with exact <Z_{q_c} Z_{q_t}> = 1."""
    for q_t in range(n_side):
        zz = sv.expectation_value(
            SparsePauliOp.from_sparse_list(
                [("ZZ", [q_c, q_t], 1.0)], num_qubits=n_total
            )
        ).real
        if abs(zz - 1.0) < 1e-9:
            return q_t
    raise AssertionError("no LC-edge qubit found")


@pytest.mark.parametrize("x_min, t, J", [(1, 2, 0.0), (1, 3, 0.0), (1, 3, 0.5)])
def test_shadow_recovers_lightcone(x_min: int, t: int, J: float) -> None:
    gate = UnitaryGate(du_gate(J))
    qc = _build_causal_cone_circuit(x_min, t, gate)
    n_side = qc.num_qubits // 2
    q_c = n_side + 0  # B-side qubit Bell-paired with A-side seed (qubit 0)

    sv = Statevector(qc)
    edge_q_t = _find_lc_edge(sv, q_c, n_side, qc.num_qubits)
    interior_q_ts = [q for q in range(n_side) if q != edge_q_t]

    bases, outcomes = classical_shadow(qc, n_shots=SHOTS, seed=0)

    zz_edge = _zz(bases, outcomes, q_c, edge_q_t)
    assert abs(zz_edge - 1.0) < TOL, (
        f"LC-edge <ZZ>={zz_edge:.3f} not near 1 "
        f"(x_min={x_min}, t={t}, J={J}, q_t={edge_q_t})"
    )
    for q_t in interior_q_ts:
        zz = _zz(bases, outcomes, q_c, q_t)
        assert abs(zz) < TOL, (
            f"Interior <ZZ>={zz:.3f} not near 0 "
            f"(x_min={x_min}, t={t}, J={J}, q_t={q_t})"
        )
