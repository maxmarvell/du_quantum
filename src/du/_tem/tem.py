"""TEM sketch: Heisenberg-picture inverse-noise correction for the perturbed-DU
correlator.

Setup
-----
Forward noisy channel per layer t:  N_t o U_t   (noise after gate).
Full circuit:    Phi      = prod_{t=1..T} (N_t o U_t)
Ideal circuit:   Phi_id   = prod_{t=1..T} U_t

The TEM identity:    <O>_ideal  =  Tr[ M(O) * rho_noisy(T) ]
with the modified observable

    M(O) = (Phi^*)^{-1}  o  Phi_id^*  (O).

Expanding the composition and cancelling the leftmost (U_1^*)^{-1} U_1^* gives a
two-phase algorithm:

  Phase A (backward Heisenberg through ideal circuit):
    O <- U_T^* o U_{T-1}^* o ... o U_1^*  (O)

  Phase B (forward Heisenberg + inverse noise, layer by layer):
    for t = 1..T:
        O <- (U_t^*)^{-1} (O)         # i.e. U_t (.) U_t^dagger,  Heisenberg forward
        O <- (N_t^*)^{-1} (O)         # inverse-noise rescaling on gate qubits

For Pauli noise (depolarizing here) (N^*)^{-1} multiplies each non-II Pauli
component on the gate's 2-qubit support by 1/(1-p).

At p = 0 the two phases cancel exactly (M(O) = O) so the consistency check is
that TEM equals raw noisy when p -> 0.

The dense Heisenberg evolution below is conceptually identical to an MPO local
update; for larger N replace `heisenberg_*_through_gate` with an MPO-style
contraction and the rest stays put.
"""

from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit_aer import AerSimulator

from du.simulation import (
    build_circuit,
    correlator_statevector,
    fold_gates,
    make_du_gate_noise,
    perturbed_du_gate,
    zz_correlator_observable,
)


# ---------------------------------------------------------------------------
# Layered circuit description (matches build_circuit's brickwork)
# ---------------------------------------------------------------------------

def gate_layers(N: int, depth: int) -> list[list[tuple[int, int]]]:
    """List of layers. Layer t = list of (q1, q2) for the brickwork gates."""
    return [
        [(i, i + 1) for i in range(t % 2, N - 1, 2)]
        for t in range(depth)
    ]


# ---------------------------------------------------------------------------
# TEM building blocks
# ---------------------------------------------------------------------------

def apply_inverse_depolarizing(O: SparsePauliOp, q1: int, q2: int,
                                p: float) -> SparsePauliOp:
    """(N_dep^{-1})^dagger on qubits (q1, q2): scale non-II components by 1/(1-p)."""
    coeffs = np.array(O.coeffs, dtype=complex, copy=True)
    paulis = O.paulis
    inv = 1.0 / (1.0 - p)
    for i in range(len(O)):
        non_id_q1 = bool(paulis[i].x[q1]) or bool(paulis[i].z[q1])
        non_id_q2 = bool(paulis[i].x[q2]) or bool(paulis[i].z[q2])
        if non_id_q1 or non_id_q2:
            coeffs[i] *= inv
    return SparsePauliOp(paulis, coeffs).simplify()


def _lifted_gate(gate_4x4: np.ndarray, q1: int, q2: int, n_total: int) -> np.ndarray:
    qc = QuantumCircuit(n_total)
    qc.unitary(gate_4x4, [q1, q2])
    return Operator(qc).data


def heisenberg_backward_through_gate(O: SparsePauliOp, gate_4x4: np.ndarray,
                                       q1: int, q2: int, n_total: int) -> SparsePauliOp:
    """O -> U^dagger O U on (q1, q2). Ideal-evolution adjoint."""
    U = _lifted_gate(gate_4x4, q1, q2, n_total)
    return SparsePauliOp.from_operator(Operator(U.conj().T @ O.to_matrix() @ U)).simplify()


def heisenberg_forward_through_gate(O: SparsePauliOp, gate_4x4: np.ndarray,
                                      q1: int, q2: int, n_total: int) -> SparsePauliOp:
    """O -> U O U^dagger on (q1, q2). Inverse of the adjoint channel."""
    U = _lifted_gate(gate_4x4, q1, q2, n_total)
    return SparsePauliOp.from_operator(Operator(U @ O.to_matrix() @ U.conj().T)).simplify()


def tem_modified_observable(O_initial: SparsePauliOp, N: int, depth: int,
                            J: float, eps: float, p_noise: float,
                            pert_H: np.ndarray | None = None) -> SparsePauliOp:
    """Compute M(O) for the perturbed-DU brickwork.

    Phase A: backward Heisenberg through all layers (ideal evolution).
    Phase B: forward Heisenberg + inverse noise, layer by layer.
    """
    n_total = 2 * N
    gate = perturbed_du_gate(J, eps, pert_H)
    layers = gate_layers(N, depth)
    T = len(layers)
    O = O_initial

    # Phase A
    for t in range(T - 1, -1, -1):
        for q1, q2 in layers[t]:
            O = heisenberg_backward_through_gate(O, gate, q1, q2, n_total)

    # Phase B
    for t in range(T):
        for q1, q2 in layers[t]:
            O = heisenberg_forward_through_gate(O, gate, q1, q2, n_total)
        for q1, q2 in layers[t]:
            O = apply_inverse_depolarizing(O, q1, q2, p_noise)

    return O


# ---------------------------------------------------------------------------
# Density-matrix execution
# ---------------------------------------------------------------------------

def noisy_density_matrix(N: int, depth: int, J: float, eps: float,
                          p_noise: float, scale_factor: int = 1,
                          pert_H: np.ndarray | None = None) -> np.ndarray:
    qc = build_circuit(N, depth, J, eps, pert_H)
    if scale_factor != 1:
        qc = fold_gates(qc, scale_factor)
    qc.save_density_matrix(label="rho")
    backend = AerSimulator(noise_model=make_du_gate_noise(p_noise),
                           method="density_matrix")
    return np.asarray(backend.run(qc, shots=1).result().data(0)["rho"])


def correlator_raw_and_tem(N: int, depth: int, J: float, eps: float, x: int,
                            p_noise: float, pauli: str = "Z") -> tuple[float, float]:
    """Return (raw noisy correlator, TEM-corrected correlator)."""
    O = zz_correlator_observable(N, x, pauli)
    rho = noisy_density_matrix(N, depth, J, eps, p_noise)
    raw = float(np.real(np.trace(O.to_matrix() @ rho)))
    M_O = tem_modified_observable(O, N, depth, J, eps, p_noise)
    tem = float(np.real(np.trace(M_O.to_matrix() @ rho)))
    return raw, tem


def correlator_zne_quadratic(N: int, depth: int, J: float, eps: float, x: int,
                              p_noise: float, pauli: str = "Z") -> float:
    """ZNE at the DU-gate level (scales 1, 3, 5; quadratic extrapolation)."""
    scales = [1, 3, 5]
    O = zz_correlator_observable(N, x, pauli)
    vals = []
    for s in scales:
        rho = noisy_density_matrix(N, depth, J, eps, p_noise, scale_factor=s)
        vals.append(float(np.real(np.trace(O.to_matrix() @ rho))))
    return float(np.polyval(np.polyfit(scales, vals, 2), 0))


# ---------------------------------------------------------------------------
# Pauli-Lindblad TEM (matched to noise_estimate_miami.make_pl_noise_model)
# ---------------------------------------------------------------------------

def apply_inverse_pauli_lindblad(
    O: SparsePauliOp,
    q1: int,
    q2: int,
    generators: list,
    rates: list[float],
) -> SparsePauliOp:
    """(Λ_PL*)^{-1} on qubits (q1, q2).

    The 2-qubit Pauli-Lindblad channel's Heisenberg action on a Pauli Q whose
    restriction to (q1, q2) is Q_loc has eigenvalue
        η(Q_loc) = exp(-2 Σ_{k: {Q_loc, G_k} = 0} λ_k).
    The inverse multiplies each Pauli coefficient by 1/η(Q_loc).
    """
    from qiskit.quantum_info import Pauli
    coeffs = np.array(O.coeffs, dtype=complex, copy=True)
    paulis = O.paulis
    gen_paulis = [g if isinstance(g, Pauli) else Pauli(g) for g in generators]
    for i in range(len(O)):
        z_loc = np.array([paulis[i].z[q1], paulis[i].z[q2]], dtype=bool)
        x_loc = np.array([paulis[i].x[q1], paulis[i].x[q2]], dtype=bool)
        local = Pauli((z_loc, x_loc))
        anti_sum = 0.0
        for g_pauli, lam in zip(gen_paulis, rates):
            if not local.commutes(g_pauli):
                anti_sum += lam
        if anti_sum > 0:
            coeffs[i] *= np.exp(2.0 * anti_sum)
    return SparsePauliOp(paulis, coeffs).simplify()


def tem_modified_observable_pl(
    O_initial: SparsePauliOp,
    N: int,
    depth: int,
    J: float,
    eps: float,
    pl_generators: list,
    pl_rates: list[float],
    pert_H: np.ndarray | None = None,
) -> SparsePauliOp:
    """M(O) for the perturbed-DU brickwork under Pauli-Lindblad noise."""
    n_total = 2 * N
    gate = perturbed_du_gate(J, eps, pert_H)
    layers = gate_layers(N, depth)
    T = len(layers)
    O = O_initial

    # Phase A
    for t in range(T - 1, -1, -1):
        for q1, q2 in layers[t]:
            O = heisenberg_backward_through_gate(O, gate, q1, q2, n_total)

    # Phase B
    for t in range(T):
        for q1, q2 in layers[t]:
            O = heisenberg_forward_through_gate(O, gate, q1, q2, n_total)
        for q1, q2 in layers[t]:
            O = apply_inverse_pauli_lindblad(O, q1, q2, pl_generators, pl_rates)

    return O


def gamma_diagnostics(M_O: SparsePauliOp) -> dict:
    """ℓ1, ℓ2², #terms, max Pauli weight of the modified observable.

      γ₁ = Σ_P |c_P|         (shot-overhead bound: Var ≤ γ₁² / N)
      γ₂² = Σ_P |c_P|²       (variance with uniform shot allocation)
      n_terms                (number of non-zero Pauli strings)
      max_weight             (max number of non-I qubit sites in any term)
    """
    coeffs_abs = np.abs(M_O.coeffs)
    return {
        "gamma_1": float(coeffs_abs.sum()),
        "gamma_2_sq": float((coeffs_abs ** 2).sum()),
        "n_terms": len(M_O),
        "max_weight": int(max(
            (int((p.x | p.z).sum()) for p in M_O.paulis),
            default=0,
        )),
    }


def correlator_raw_and_tem_pl(
    N: int, depth: int, J: float, eps: float, x: int,
    pl_generators: list, pl_rates: list[float],
    pauli: str = "Z",
) -> tuple[float, float]:
    """Raw noisy + TEM-corrected correlator under matched Pauli-Lindblad noise."""
    from du.noise import make_pl_noise_model
    O = zz_correlator_observable(N, x, pauli)
    qc = build_circuit(N, depth, J, eps)
    qc.save_density_matrix(label="rho")
    backend = AerSimulator(
        noise_model=make_pl_noise_model(pl_generators, pl_rates),
        method="density_matrix",
    )
    rho = np.asarray(backend.run(qc, shots=1).result().data(0)["rho"])
    raw = float(np.real(np.trace(O.to_matrix() @ rho)))
    M_O = tem_modified_observable_pl(O, N, depth, J, eps, pl_generators, pl_rates)
    tem = float(np.real(np.trace(M_O.to_matrix() @ rho)))
    return raw, tem


def variance_forecast_demo(target_bias: float = 0.01) -> None:
    """Sweep depth, report γ-based shot-budget forecast for TEM post-processing.

    For target bias δ on the TEM-corrected expectation:
        shots ≥ γ₁² / δ²       (worst-case ℓ1 bound)
        shots ≥ γ₂² / δ²       (uniform-allocation bound; tighter for sparse M(O))
    """
    from du.noise import equal_rate_pl, miami_pauli_lindblad

    gens, rates, p_du = miami_pauli_lindblad()
    lam = rates[0]

    N, J, eps = 4, 0.7, 0.15
    print(f"=== TEM variance forecast: N={N}, J={J:.2f}, eps={eps:.2f} ===")
    print(f"FakeMiami-derived ε_2q = {p_du:.3e}  →  λ = {lam:.3e} per PL generator")
    print(f"Target bias δ = {target_bias:g}\n")
    print(f"{'depth':>5}  {'x':>3}  {'γ_1':>10}  {'γ_2²':>10}  "
          f"{'#terms':>7}  {'max_w':>5}  {'shots(γ_1²/δ²)':>15}  "
          f"{'shots(γ_2²/δ²)':>15}")
    print("-" * 84)

    for depth in range(2, 7):
        for x in range(N):
            O = zz_correlator_observable(N, x)
            M_O = tem_modified_observable_pl(O, N, depth, J, eps, gens, rates)
            d = gamma_diagnostics(M_O)
            shots_l1 = int(np.ceil(d["gamma_1"] ** 2 / target_bias ** 2))
            shots_l2 = int(np.ceil(d["gamma_2_sq"] / target_bias ** 2))
            print(f"{depth:>5}  {x:>3}  {d['gamma_1']:>10.3e}  "
                  f"{d['gamma_2_sq']:>10.3e}  {d['n_terms']:>7}  "
                  f"{d['max_weight']:>5}  {shots_l1:>15,d}  {shots_l2:>15,d}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    N, depth, J, eps = 4, 4, 0.7, 0.15
    p_levels = [1e-3, 5e-3, 1e-2, 3e-2]

    print(f"=== TEM sketch: N={N}, depth={depth}, J={J:.2f}, eps={eps:.2f} ===\n")

    ref = {x: correlator_statevector(N, depth, J, eps, x) for x in range(N)}

    for p in p_levels:
        print(f"\n--- 2-qubit depolarizing rate p = {p:g} ---")
        print(f"{'x':>3}  {'ideal':>12}  {'raw noisy':>12}  {'ZNE quad':>12}  "
              f"{'TEM':>12}  {'raw err':>10}  {'ZNE err':>10}  {'TEM err':>10}")
        print("-" * 100)

        for x in range(N):
            raw, tem = correlator_raw_and_tem(N, depth, J, eps, x, p)
            zne = correlator_zne_quadratic(N, depth, J, eps, x, p)
            print(f"{x:>3}  {ref[x]:>12.4e}  {raw:>12.4e}  {zne:>12.4e}  "
                  f"{tem:>12.4e}  {raw-ref[x]:>10.2e}  {zne-ref[x]:>10.2e}  "
                  f"{tem-ref[x]:>10.2e}")


if __name__ == "__main__":
    main()
    print()
    variance_forecast_demo()
