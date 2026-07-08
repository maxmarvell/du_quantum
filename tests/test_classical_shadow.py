"""Classical-shadow estimate of the DU light-cone correlator.

The circuit (`build_circuit_cs`) evolves the A-side of a Bell-pair Choi
state with the self-dual kicked-Ising gate, whose transverse-field kick is
along X (`rx`). The operator that propagates coherently to the light-cone
edge is therefore X, not Z, so on the Choi state dual-unitarity predicts
    <X_{q_c} X_{q_t}> = 1   at the LC-edge target q_t (x = t)
                      = 0   in the strict interior (x_min ≤ x < t),
where q_c is the B-side Bell partner of the source site. (The Z–Z
correlator vanishes identically here — Z is not transported by this gate.)
We recover this prediction from one random-Pauli shadow run, reading the
edge correlator straight off the (bases, outcomes) dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from du.simulation import (
    run_classical_shadow,
    build_circuit_cs,
    expect_pauli,
    get_cs_control,
    get_cs_targets
)


SHOTS = 2000
TOL = 0.15
PAULI_X = 0


def _xx(bases: np.ndarray, outcomes: np.ndarray, q_c: int, q_t: int) -> float:
    return expect_pauli(
        bases, outcomes, np.array([q_c, q_t]), np.array([PAULI_X, PAULI_X])
    )


@pytest.mark.parametrize("x_min, t, h", [(1, 2, 0.0), (1, 3, 0.0), (np.pi/8, np.pi/4, np.pi/2)])
def test_shadow_recovers_lightcone(x_min: int, t: int, h: float) -> None:

    qc = build_circuit_cs(t, x_min, h=h)

    control_qubit = get_cs_control(t, x_min)
    target_qubits = get_cs_targets(t, x_min)

    bases, outcomes = run_classical_shadow(qc, n_shots=SHOTS, seed=0)

    for target in target_qubits:

        xx = _xx(bases, outcomes, control_qubit, target)

        if target == qc.num_qubits // 2 - 1:
            assert abs(xx - 1.0) < TOL, (
                f"LC-edge <XX>={xx:.3f} not near 1 "
                f"(x_min={x_min}, t={t}, h={h}, q_t={target})"
            )
        else:
            assert abs(xx) < TOL, (
                f"LC-edge <XX>={xx:.3f} not near 1 "
                f"(x_min={x_min}, t={t}, h={h}, q_t={target})"
            )



if __name__ == "__main__":
    pass