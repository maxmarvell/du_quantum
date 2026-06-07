from __future__ import annotations

import numpy as np
from scipy.linalg import expm


# ---------------------------------------------------------------------------
# Pauli basis + perturbed-DU gate
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS_1Q = (_I, _X, _Y, _Z)
_PAULIS_2Q = tuple(np.kron(a, b) for a in _PAULIS_1Q for b in _PAULIS_1Q)


def perturbed_du_gate(J: float, eps: float) -> np.ndarray:
    """U(eps) = exp(-i eps (XX+ZZ)) @ exp(-i ((pi/4)(XX+YY) + J ZZ))."""
    XX = np.kron(_X, _X)
    YY = np.kron(_Y, _Y)
    ZZ = np.kron(_Z, _Z)
    H_du = (np.pi / 4) * (XX + YY) + J * ZZ
    H_pert = XX + ZZ
    return expm(-1j * eps * H_pert) @ expm(-1j * H_du)


# ---------------------------------------------------------------------------
# Heisenberg PTM and inverse-PL diagonal in Pauli basis
# ---------------------------------------------------------------------------

def heisenberg_ptm_2q(U: np.ndarray) -> np.ndarray:
    """16×16 Heisenberg PTM in Pauli basis.

    PTM[k, l] = (1/4) Tr(P_k U† P_l U), so c'_k = Σ_l PTM[k, l] c_l represents
    the action O = Σ c_l P_l  ->  U† O U = Σ c'_k P_k.
    Index convention: k = 4·a + b for the 2-qubit Pauli (P_a ⊗ P_b).
    """
    Udag = U.conj().T
    ptm = np.zeros((16, 16), dtype=complex)
    for k in range(16):
        Pk = _PAULIS_2Q[k]
        for l in range(16):
            ptm[k, l] = np.trace(Pk @ Udag @ _PAULIS_2Q[l] @ U) / 4.0
    return ptm


def inverse_pl_diagonal(lam: float) -> np.ndarray:
    """Equal-rate PL inverse: diagonal entries on joint 2-qubit Pauli index.

    All 15 non-identity 2-Paulis get exp(16 λ); identity stays at 1.
    """
    diag = np.full(16, np.exp(16.0 * lam), dtype=complex)
    diag[0] = 1.0
    return diag


# ---------------------------------------------------------------------------
# PauliMPO
# ---------------------------------------------------------------------------

class PauliMPO:
    """Pauli-basis MPO. tensors[i] has shape (D_left, 4, D_right)."""

    def __init__(self, tensors: list[np.ndarray]):
        self.tensors = tensors
        self.n_sites = len(tensors)

    @classmethod
    def from_pauli_string(cls, n_sites: int, pauli_idx: dict[int, int]) -> "PauliMPO":
        """Bond-dim-1 MPO with given Pauli at each site (default I)."""
        tensors = []
        for i in range(n_sites):
            t = np.zeros((1, 4, 1), dtype=complex)
            t[0, pauli_idx.get(i, 0), 0] = 1.0
            tensors.append(t)
        return cls(tensors)

    def apply_two_site_op(
        self,
        i: int,
        op16: np.ndarray,
        chi_max: int = 256,
        tol: float = 1e-12,
    ) -> None:
        """Apply a 2-site PTM (16×16) or diagonal (length-16) on the joint
        physical index of sites i, i+1. SVD across the bond, truncate to χ_max.
        """
        A, B = self.tensors[i], self.tensors[i + 1]
        D_l, _, D_m = A.shape
        D_m2, _, D_r = B.shape
        assert D_m == D_m2

        # Join physical indices: T[l, a, b, r]
        T = np.einsum("lam,mbr->labr", A, B)
        T = T.reshape(D_l, 16, D_r)
        if op16.ndim == 1:
            T = T * op16[None, :, None]
        else:
            T = np.einsum("xy,lyr->lxr", op16, T)
        T = T.reshape(D_l, 4, 4, D_r)

        # SVD across the bond between site i and i+1
        M = T.reshape(D_l * 4, 4 * D_r)
        Uu, S, Vt = np.linalg.svd(M, full_matrices=False)

        # Truncate by absolute tol and chi_max
        if S[0] > 0:
            keep = int(np.sum(S > tol * S[0]))
        else:
            keep = 1
        keep = max(1, min(keep, chi_max))

        sqrtS = np.sqrt(S[:keep])
        A_new = (Uu[:, :keep] * sqrtS).reshape(D_l, 4, keep)
        B_new = (sqrtS[:, None] * Vt[:keep, :]).reshape(keep, 4, D_r)
        self.tensors[i] = A_new
        self.tensors[i + 1] = B_new

    def gamma_2_squared(self) -> float:
        """γ_2² = Σ_P |c_P|² via left-to-right contraction.

        For a Pauli-basis MPO, this is the Hilbert-Schmidt norm of the operator
        divided by 2^N (because Tr(P_k P_l) = 2^N δ_kl with unnormalised
        Pauli basis). Here we use unit-normalised Paulis as the physical index
        basis, so the contraction gives exactly Σ |c_P|².
        """
        L = np.array([[1.0 + 0j]])
        for A in self.tensors:
            L = np.einsum("ab,apr,bps->rs", L, A, A.conj())
        return float(np.real(L[0, 0]))

    def max_bond_dim(self) -> int:
        return max((A.shape[2] for A in self.tensors[:-1]), default=1)

    def bond_dims(self) -> list[int]:
        return [A.shape[2] for A in self.tensors[:-1]]

    def to_sparse_pauli_op(self):
        """Convert to a Qiskit SparsePauliOp. Only feasible for small N."""
        from qiskit.quantum_info import SparsePauliOp
        n = self.n_sites
        if 4 ** n > 5_000_000:
            raise ValueError(f"to_sparse_pauli_op: {n} sites would enumerate "
                             f"{4**n} Pauli strings (too many). Use γ_2² instead.")
        # symbols ordered so leftmost char in label = highest qubit index
        sym = ("I", "X", "Y", "Z")
        labels = []
        coeffs = []
        # Iterate physical indices via Cartesian product; build coefficient by
        # contracting bond dimensions for each Pauli configuration.
        from itertools import product
        for cfg in product(range(4), repeat=n):
            # contract MPO with this configuration
            v = np.array([1.0 + 0j])
            for site, p in enumerate(cfg):
                v = v @ self.tensors[site][:, p, :]
            c = complex(v[0])
            if abs(c) > 1e-14:
                # qiskit label: rightmost = qubit 0 → reverse the cfg
                label = "".join(sym[p] for p in reversed(cfg))
                labels.append(label)
                coeffs.append(c)
        if not labels:
            return SparsePauliOp("I" * n, coeffs=[0.0])
        return SparsePauliOp(labels, coeffs=coeffs)


# ---------------------------------------------------------------------------
# Main TEM operator construction (MPO version)
# ---------------------------------------------------------------------------

def tem_modified_observable_mpo(
    N: int,
    depth: int,
    J: float,
    eps: float,
    lam: float,
    x: int,
    *,
    chi_max: int = 256,
    tol: float = 1e-12,
) -> PauliMPO:
    """M(O) MPO for the Z_{A,x} Z_{B,0} correlator on the 2N-qubit Choi layout.

    Phase A: ideal backward Heisenberg (PTM of U) on each brickwork gate,
    layers iterated in reverse time order.
    Phase B: per layer in physical time order, apply forward Heisenberg
    (PTM of U†) on each gate, then inverse PL diagonal.
    """
    n_total = 2 * N
    # Initial observable: Z at A_x (Python site x) and Z at B_0 (Python site N)
    mpo = PauliMPO.from_pauli_string(n_total, {x: 3, N: 3})

    U = perturbed_du_gate(J, eps)
    ptm_U = heisenberg_ptm_2q(U)               # backward Heisenberg PTM of U
    ptm_Udag = ptm_U.T                          # forward Heisenberg via U
    diag_invPL = inverse_pl_diagonal(lam)

    def gate_pairs(t: int) -> list[tuple[int, int]]:
        offset = t % 2
        return [(i, i + 1) for i in range(offset, N - 1, 2)]

    # Phase A: backward Heisenberg through ideal circuit, reverse time order
    for t in range(depth - 1, -1, -1):
        for i, j in gate_pairs(t):
            mpo.apply_two_site_op(i, ptm_U, chi_max=chi_max, tol=tol)

    # Phase B: forward Heisenberg + inverse PL, physical time order
    for t in range(depth):
        for i, j in gate_pairs(t):
            mpo.apply_two_site_op(i, ptm_Udag, chi_max=chi_max, tol=tol)
        for i, j in gate_pairs(t):
            mpo.apply_two_site_op(i, diag_invPL, chi_max=chi_max, tol=tol)

    return mpo


def gamma_diagnostics_mpo(
    N: int,
    depth: int,
    J: float,
    eps: float,
    lam: float,
    x: int,
    *,
    chi_max: int = 256,
    tol: float = 1e-12,
) -> dict:
    """γ_2², max χ, bond dims at all cuts. (γ_1 is not directly computable
    from an MPO without enumeration; use γ_2 = √γ_2² as a lower bound, or
    `to_sparse_pauli_op()` for small N.)"""
    mpo = tem_modified_observable_mpo(
        N, depth, J, eps, lam, x, chi_max=chi_max, tol=tol,
    )
    return {
        "gamma_2_sq": mpo.gamma_2_squared(),
        "max_chi": mpo.max_bond_dim(),
        "bond_dims": mpo.bond_dims(),
    }
