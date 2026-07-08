"""DU light-cone tests via the default x_basis_measurement.

At the self-dual point (b = pi/4) the kicked-Ising gate is dual-unitary and
the Choi-state light-cone edge correlator <X_c X_edge> = 1 exactly, with the
strict interior = 0. Perturbing the transverse field (b = pi/4 - eps) breaks
dual-unitarity, so the edge signal decays -- monotonically in eps, and faster
at larger T (exact values: T=2 edge = 1.000 / 0.851 / 0.518 / 0.215 at
eps = 0 / 0.1 / 0.2 / 0.3).
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

from du.simulation import (
    build_circuit,
    expect_x,
    get_control_qubit,
    get_target_qubits,
    is_dual_unitary,
    kicked_ising_gate,
    x_basis_measurement,
)

SHOTS = 4000
TOL = 0.15


def _exact_profile(t: float, x_min: float, eps: float) -> np.ndarray:
    """Exact <X_c X_qt> for every target, via statevector."""
    qc = build_circuit(t, x_min, h=0, b=np.pi / 4 - eps)
    n = qc.num_qubits
    control = get_control_qubit(t, x_min)
    sv = Statevector(qc)
    vals = []
    for qt in get_target_qubits(t, x_min):
        obs = SparsePauliOp.from_sparse_list([("XX", [control, qt], 1.0)], num_qubits=n)
        vals.append(np.real(sv.expectation_value(obs)))
    return np.asarray(vals)


def test_self_dual_gate_is_dual_unitary() -> None:
    # b = pi/4 is the self-dual point for any longitudinal field h ...
    for h in (0.0, 0.3, 1.0):
        assert is_dual_unitary(Operator(kicked_ising_gate(h, np.pi / 4)).data)
    # ... and perturbing b away from it breaks dual-unitarity
    for eps in (0.1, 0.2, 0.3):
        assert not is_dual_unitary(Operator(kicked_ising_gate(0, np.pi / 4 - eps)).data)


@pytest.mark.parametrize("t, x_min", [(2, 1), (3, 1)])
def test_x_basis_measurement_recovers_du_lightcone(t: float, x_min: float) -> None:
    """One fixed all-X setting recovers edge = 1, interior = 0 (DU case)."""
    qc = build_circuit(t, x_min)  # defaults h=0, b=pi/4 (self-dual)
    targets = get_target_qubits(t, x_min)
    control = get_control_qubit(t, x_min)

    outcomes = x_basis_measurement(qc, n_shots=SHOTS, seed=0)

    for qt in targets:
        xx = expect_x(outcomes, np.array([control, qt]))
        if qt == targets[-1]:  # light-cone edge (x = t)
            assert abs(xx - 1.0) < TOL, f"edge <XX>={xx:.3f} not near 1 (qt={qt})"
        else:
            assert abs(xx) < TOL, f"interior <XX>={xx:.3f} not near 0 (qt={qt})"


def test_perturbed_lightcone_signal_loss_exact() -> None:
    """Breaking dual-unitarity (eps > 0) monotonically degrades the edge."""
    edges = [_exact_profile(2, 1, eps)[-1] for eps in (0.0, 0.1, 0.2, 0.3)]

    assert edges[0] == pytest.approx(1.0, abs=1e-9)      # DU: exactly protected
    for stronger, weaker in zip(edges, edges[1:]):
        assert weaker < stronger - 0.05                  # strictly decaying in eps
    assert edges[2] < 0.6                                # eps=0.2 well off unity

    # decay is ballistic: same eps hurts more at larger T
    assert _exact_profile(3, 1, 0.1)[-1] < _exact_profile(2, 1, 0.1)[-1]


def test_perturbed_lightcone_via_x_basis_measurement() -> None:
    """The default measurement sees the signal loss too (eps=0.2, T=2)."""
    qc = build_circuit(2, 1, h=0, b=np.pi / 4 - 0.2)
    targets = get_target_qubits(2, 1)
    control = get_control_qubit(2, 1)

    outcomes = x_basis_measurement(qc, n_shots=SHOTS, seed=0)
    edge = expect_x(outcomes, np.array([control, targets[-1]]))

    assert 0.3 < edge < 0.7, f"perturbed edge <XX>={edge:.3f}, expected ~0.52"
