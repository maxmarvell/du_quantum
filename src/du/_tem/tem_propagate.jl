using PauliPropagation
using LinearAlgebra

struct InverseEqualRatePL <: StaticGate
    qinds::Tuple{Int,Int}
    lambda::Float64
end

function PauliPropagation.apply(gate::InverseEqualRatePL, pstr, coeff; kwargs...)
    local_pauli = getpauli(pstr, [gate.qinds[1], gate.qinds[2]])
    if local_pauli == 0
        return ((pstr, coeff),)
    end
    return ((pstr, coeff * exp(16 * gate.lambda)),)
end

const _X = ComplexF64[0 1; 1 0]
const _Y = ComplexF64[0 -im; im 0]
const _Z = ComplexF64[1 0; 0 -1]
const _XX = kron(_X, _X)
const _YY = kron(_Y, _Y)
const _ZZ = kron(_Z, _Z)

function perturbed_du_gate(J::Float64, eps::Float64)
    H_du = (π / 4) * (_XX + _YY) + J * _ZZ
    H_pert = _XX + _ZZ
    return exp(-im * eps * H_pert) * exp(-im * H_du)
end

function _layer_pairs(N::Int, t::Int)
    offset = (t - 1) % 2
    return [(i, i + 1) for i in (offset + 1):2:(N - 1)]
end

function tem_modified_observable(N::Int, depth::Int, J::Float64, eps::Float64,
                                  lambda::Float64, x::Int;
                                  max_weight::Int = 2 * N,
                                  min_abs_coeff::Float64 = 1.0e-12)
    n_total = 2 * N
    U = perturbed_du_gate(J, eps)
    Udag = adjoint(U)

    qa = x + 1                  # Julia qubit for A_x  (Python qubit x)
    qb = N + 1                  # Julia qubit for B_0  (Python qubit N)
    obs = PauliString(n_total, [:Z, :Z], [qa, qb])
    psum = PauliSum(obs)

    # Phase A: ideal backward Heisenberg through layers 1..depth (in time order;
    # `propagate` reverses internally so layer `depth` is applied first to obs).
    phase_A = Gate[]
    for t in 1:depth
        for (i, j) in _layer_pairs(N, t)
            push!(phase_A, TransferMapGate(U, (i, j)))
        end
    end
    psum_A = propagate(phase_A, psum; max_weight = max_weight,
                       min_abs_coeff = min_abs_coeff)

    # Phase B: layer by layer in physical time order.
    psum_B = psum_A
    for t in 1:depth
        pairs = _layer_pairs(N, t)
        # Forward Heisenberg via U†: propagate applies (U†)* (.) which is U (.) U†.
        layer_Udag = Gate[TransferMapGate(Udag, (i, j)) for (i, j) in pairs]
        psum_B = propagate(layer_Udag, psum_B; max_weight = max_weight,
                            min_abs_coeff = min_abs_coeff)
        # Inverse PL: multiply non-identity 2-Pauli coefficients on the support.
        layer_invpl = Gate[InverseEqualRatePL((i, j), lambda) for (i, j) in pairs]
        psum_B = propagate(layer_invpl, psum_B; max_weight = max_weight,
                            min_abs_coeff = min_abs_coeff)
    end

    return psum_B
end

function paulisum_to_pairs(psum, n_total::Int)
    out = Vector{Tuple{String, ComplexF64}}()
    sizehint!(out, length(psum.terms))
    for (intpstr, coeff) in psum.terms
        syms = [string(inttosymbol(getpauli(intpstr, ii))) for ii in 1:n_total]
        label = join(reverse(syms))            # qiskit: rightmost = qubit 1 (= Python qubit 0)
        push!(out, (label, ComplexF64(coeff)))
    end
    return out
end

function gamma_diagnostics_jl(psum)
    g1 = 0.0
    g2sq = 0.0
    n = 0
    mw = 0
    for (intpstr, c) in psum.terms
        ac = abs(c)
        g1 += ac
        g2sq += ac * ac
        n += 1
        w = countweight(intpstr)
        if w > mw
            mw = w
        end
    end
    return (gamma_1 = g1, gamma_2_sq = g2sq, n_terms = n, max_weight = mw)
end
