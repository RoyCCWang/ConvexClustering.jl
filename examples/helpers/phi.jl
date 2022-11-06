
function evalϕUcost(U, X::Matrix{T}, Z::Matrix{T}, J, σ::T, γ::T) where T

    N, N_edges = size(J)
    D = size(X,1)
    @assert size(X,2) == N
    @assert size(Z) == size(U) == (D, N_edges)

    tmp = U- X*J - 1/σ .* Z
    p_U = evalp(U, W, γ)
    cost = p_U + σ/2 * norm(tmp, 2)
    #cost = 1.0

    return cost
end

function evalp(U, W::Vector{T}, γ::T) where T
    @assert length(W) == size(U,2)

    out = γ*sum( W[l]*norm(U[:,l], 2) for l = 1:size(U,2) )

    return out
end

function evalϕ1term(X, Z::Matrix{T}, w::Vector{T}, J, σ::T, γ::T) where T

    λ = 1/σ

    V = X*J + λ .* Z

    Y = ConvexClustering.proximaltp(V, w, γ, λ)
    p_Y = evalp(Y, w, γ)

    P2 = ConvexClustering.proximaltpconj(V, w, γ, λ)

    term11 = p_Y
    term12 = (σ/2) * norm(P2, 2)^2

    return term11+term12, Y, P2
end

function evalϕ(X, Z, w, J, σ, γ::T, A) where T

    term0 = 0.5 * norm(X-A, 2)^2

    term1, _ = evalϕ1term(X, Z, w, J, σ, γ)

    term2 = -norm(Z, 2)^2/(2*σ)

    return term0 + term1 + term2
end
