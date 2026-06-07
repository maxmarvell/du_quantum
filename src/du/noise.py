from __future__ import annotations

import numpy as np
import statistics

from qiskit.quantum_info import Pauli
from qiskit_aer.noise import NoiseModel, PauliLindbladError
from qiskit_ibm_runtime.fake_provider import FakeMiami


def median_2q_error(backend) -> float:
    """Median 2-qubit gate error from a V2 backend's target.

    Filters out broken-qubit entries (error == 1.0 sentinel).
    """
    target = backend.target
    for gname in ("cz", "ecr", "cx"):
        if gname in target.operation_names:
            errs = [p.error for _, p in target[gname].items()
                    if p and p.error is not None and p.error < 0.5]
            if errs:
                return float(statistics.median(errs))
    raise RuntimeError("No 2-qubit gate error data on backend.")


def effective_du_gate_error(backend, cz_per_du: int = 3) -> float:
    """A DU 2-qubit gate transpiles to ~cz_per_du CZ + single-qubit gates.
    Single-qubit error is ~1e-4 << CZ error, so approximate by CZ error compounded."""
    p_cz = median_2q_error(backend)
    return 1.0 - (1.0 - p_cz) ** cz_per_du


def two_qubit_generators() -> list[Pauli]:
    """The 15 non-identity 2-qubit Pauli operators."""
    gens: list[Pauli] = []
    for z0 in (False, True):
        for x0 in (False, True):
            for z1 in (False, True):
                for x1 in (False, True):
                    if not (z0 or x0 or z1 or x1):
                        continue                          # skip II
                    z = np.array([z0, z1], dtype=bool)
                    x = np.array([x0, x1], dtype=bool)
                    gens.append(Pauli((z, x)))
    return gens


def equal_rate_pl(avg_2q_err: float) -> tuple[list[Pauli], list[float]]:
    """Equal-rate Pauli-Lindblad model from a scalar average 2q error.

    Derivation: a 2-qubit PL channel with all 15 non-I generators at rate λ
    has Heisenberg eigenvalue exp(-16 λ) on every non-I Pauli. Process
    fidelity F_pro = (1 + 15 e^{-16λ}) / 16. Average gate fidelity
    F_avg = (4 F_pro + 1) / 5, giving ε = 1 - F_avg ≈ 12 λ at small λ.
    Hence λ ≈ ε / 12.
    """
    gens = two_qubit_generators()
    lam = float(avg_2q_err) / 12.0
    return gens, [lam] * len(gens)


def make_pl_noise_model(
    generators: list[Pauli],
    rates: list[float],
    gate_name: str = "unitary",
) -> NoiseModel:
    nm = NoiseModel()
    ple = PauliLindbladError(generators, rates)
    nm.add_all_qubit_quantum_error(ple, [gate_name])
    return nm


def miami_pauli_lindblad(cz_per_du: int = 3) -> tuple[list[Pauli], list[float], float]:
    fake = FakeMiami()
    p_du = effective_du_gate_error(fake, cz_per_du)
    gens, rates = equal_rate_pl(p_du)
    return gens, rates, p_du


def main() -> None:
    fake = FakeMiami()
    p_du = effective_du_gate_error(fake)
    print(f"=== FakeMiami calibration snapshot ===")
    print(f"Backend     : {fake.name} ({fake.num_qubits} qubits)")
    print(f"Median CZ err: {median_2q_error(fake):.3e}")
    print(f"Effective per-DU-gate depolarizing rate: {p_du:.3e}")

    gens, rates = equal_rate_pl(p_du)
    lam = rates[0]
    print(f"Pauli-Lindblad equal-rate ansatz:")
    print(f"  λ per generator   = ε / 12 = {lam:.4e}")
    print(f"  2q Pauli eigval   = exp(-16λ) = {np.exp(-16 * lam):.6f}")
    print(f"  generators ({len(gens)}): {[str(g) for g in gens]}\n")

    # Lazy imports: tem depends on this file at import time.
    from du.simulation import correlator_statevector
    from du._tem.tem import correlator_raw_and_tem, correlator_zne_quadratic

    N, depth, J, eps = 4, 4, 0.7, 0.15

    print(f"=== (A) Matched depolarizing model at p = {p_du:.3e} ===")
    print(f"{'x':>3}  {'ideal':>12}  {'raw noisy':>12}  {'ZNE quad':>12}  "
          f"{'TEM':>12}  {'raw err':>10}  {'ZNE err':>10}  {'TEM err':>10}")
    print("-" * 105)
    for x in range(N):
        ideal = correlator_statevector(N, depth, J, eps, x)
        raw, tem = correlator_raw_and_tem(N, depth, J, eps, x, p_du)
        zne = correlator_zne_quadratic(N, depth, J, eps, x, p_du)
        print(f"{x:>3}  {ideal:>12.4e}  {raw:>12.4e}  {zne:>12.4e}  "
              f"{tem:>12.4e}  {raw-ideal:>10.2e}  {zne-ideal:>10.2e}  "
              f"{tem-ideal:>10.2e}")


if __name__ == "__main__":
    main()
