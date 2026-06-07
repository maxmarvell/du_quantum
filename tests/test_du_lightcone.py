"""Light-cone edge expectation test for the unperturbed dual-unitary causal
block.

For the exact DU gate U(J) = exp(-i((pi/4)(XX+YY) + J*ZZ)), the single-site Z
operator propagates ballistically: J*ZZ commutes with Z(x)I and with the
XX+YY transport, so U(J)^dagger (Z x I) U(J) = I x Z for any J. The causal-
block circuit built by `build_and_measure(t, t, gate, O='ZZ')` is a 2t-long
gate ladder that transports the seed Z exactly across the register, so the
Bell-pair-encoded ZZ correlator collapses onto P(0) = 1: every shot returns
'0' and <Z> on the parity bit equals +1.
"""

from __future__ import annotations

import pytest
from qiskit import transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

from du.simulation import build_and_measure, du_gate


SHOTS = 256


@pytest.fixture(scope="module")
def backend() -> AerSimulator:
    return AerSimulator()


@pytest.mark.parametrize("t", [1, 2, 3, 4])
@pytest.mark.parametrize("J", [0.0, 0.3, 0.7, 1.5])
def test_zz_lc_edge_is_deterministic_zero(backend: AerSimulator, t: int, J: float) -> None:
    """Exact DU + LC edge (x=t) => ZZ correlator = 1 => every shot is '0'."""
    gate = UnitaryGate(du_gate(J))
    qc = build_and_measure(t, t, gate, O="ZZ")

    result = backend.run(transpile(qc, backend), shots=SHOTS).result()
    counts = result.get_counts()

    assert counts.get("1", 0) == 0, (
        f"DU LC-edge ZZ correlator must be +1, but saw '1' outcomes. "
        f"J={J}, t={t}, counts={counts}"
    )
    assert counts.get("0", 0) == SHOTS
