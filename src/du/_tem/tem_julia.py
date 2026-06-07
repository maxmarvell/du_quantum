from __future__ import annotations

from pathlib import Path

from qiskit.quantum_info import SparsePauliOp


_HERE = Path(__file__).resolve().parent
_JL = None


def _get_julia():
    global _JL
    if _JL is not None:
        return _JL
    from juliacall import Main as jl
    jl.seval("""
        using Pkg
        if !haskey(Pkg.project().dependencies, "PauliPropagation")
            Pkg.add("PauliPropagation")
        end
        using PauliPropagation
    """)
    jl.include(str(_HERE / "tem_propagate.jl"))
    _JL = jl
    return _JL


def tem_modified_observable_jl(
    N: int,
    depth: int,
    J: float,
    eps: float,
    lambda_pl: float,
    x: int,
    *,
    max_weight: int | None = None,
    min_abs_coeff: float = 1e-12,
) -> SparsePauliOp:
    """M(O) for the Z_{A,x} Z_{B,0} correlator, via PauliPropagation.jl."""
    jl = _get_julia()
    if max_weight is None:
        max_weight = 2 * N
    psum_jl = jl.tem_modified_observable(
        N, depth, float(J), float(eps), float(lambda_pl), int(x),
        max_weight=int(max_weight),
        min_abs_coeff=float(min_abs_coeff),
    )
    n_total = 2 * N
    pairs = jl.paulisum_to_pairs(psum_jl, n_total)
    labels: list[str] = []
    coeffs: list[complex] = []
    for pair in pairs:
        labels.append(str(pair[0]))
        coeffs.append(complex(pair[1]))
    if not labels:
        return SparsePauliOp("I" * n_total, coeffs=[0.0])
    return SparsePauliOp(labels, coeffs=coeffs)


def gamma_diagnostics_jl(
    N: int,
    depth: int,
    J: float,
    eps: float,
    lambda_pl: float,
    x: int,
    *,
    max_weight: int | None = None,
    min_abs_coeff: float = 1e-12,
) -> dict:
    """γ-diagnostics for M(O) computed in Julia without round-tripping the full sum.

    Returns {'gamma_1', 'gamma_2_sq', 'n_terms', 'max_weight'}.
    """
    jl = _get_julia()
    if max_weight is None:
        max_weight = 2 * N
    psum_jl = jl.tem_modified_observable(
        N, depth, float(J), float(eps), float(lambda_pl), int(x),
        max_weight=int(max_weight),
        min_abs_coeff=float(min_abs_coeff),
    )
    d = jl.gamma_diagnostics_jl(psum_jl)
    return {
        "gamma_1": float(d.gamma_1),
        "gamma_2_sq": float(d.gamma_2_sq),
        "n_terms": int(d.n_terms),
        "max_weight": int(d.max_weight),
    }
