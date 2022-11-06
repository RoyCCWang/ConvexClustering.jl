# Equation 17 and related discussions in section 5.3. (Sun, JMLR 2011).
# We focus on ϕ and dϕ_dx in our work.

function evalϕ(X, Z, w, J, σ, γ::T, A)::T where T <: AbstractFloat

    term_X = evalϕX(X, Z, w, J, σ, γ)

    term_rest = -dot(Z,Z)/(2*σ)

    return term_X + term_rest
end

"""
Evaluates the terms of ϕ(X) in sec 5.3 of (Sun, JMLR 2021).
1/2 * norm(X-A,2)^2 + inf_U{p(U) + σ/2 * norm(U - XJ - Z ./ σ)}
"""
function evalϕX!(Y::Matrix{T}, P2::Matrix{T}, V::Matrix{T},
    X::Matrix{T}, Z::Matrix{T}, w::Vector{T}, edge_pairs, γ::T, σ::T, A::Matrix{T})::T where T <: AbstractFloat

    term1_x = (dot(X,X) + dot(A,A) - 2*dot(X,A))/2 # norm(X-A,2)^2

    term2_x = evalϕmoreauenvelope!(Y, P2, V,
        X, Z, w, edge_pairs, γ, σ)

    return term1_x + term2_x
end

function evalϕX(X::Matrix{T}, Z::Matrix{T}, w::Vector{T},
    edge_pairs, γ::T, σ::T, A::Matrix{T})::T where T <: AbstractFloat

    Y = similar(Z)
    P2 = similar(Z)
    V = similar(Z)

    return evalϕX!(Y, P2, V, X, Z, w, edge_pairs, γ, σ, A)
end

function evalϕmoreauenvelope!(Y::Matrix{T}, proxconj_V::Matrix{T}, V::Matrix{T},
    X::Matrix{T}, Z::Matrix{T}, w::Vector{T}, edge_pairs, γ::T, σ::T)::T where T <: AbstractFloat

    λ = 1/σ

    computeV!(V, X, Z, λ, edge_pairs)

    proximaltp!(Y, V, w, γ, λ)
    p_Y = evalp(Y, w, γ)

    ##proximaltpconj!(proxconj_V, V, w, γ, λ)
    #proximaltpconjgivenproximaltp!(proxconj_V, V, Y)
    #term2 = (σ/2) * norm(proxconj_V, 2)^2
    term2 = (σ/2)*(dot(V,V) + dot(Y, Y) - 2*dot(Y,V))

    return p_Y + term2
end

function evalpdirect(U::Matrix{T}, w::Vector{T}, γ::T) where T
    @assert length(w) == size(U,2)

    out = γ*sum( w[l]*norm(U[:,l], 2) for l = 1:size(U,2) )

    return out
end

function evalp(U::Matrix{T}, w::Vector{T}, γ::T) where T
    @assert length(w) == size(U,2)

    out = zero(T)
    for l in axes(U,2)

        out += w[l]*sqrt(sum( U[d,l]*U[d,l] for d in axes(U,1)))
    end

    out = out*γ

    return out
end

## dϕ
function computedϕdirect(X::Matrix{T}, proxconj_V::Matrix{T}, J, A::Matrix{T}, σ::T)::Matrix{T} where T
    #
    #proximaltpconjgivenproximaltp!(proxconj_V, V, Y)
    return X.-A .+ σ .* proxconj_V*J'
end

function computedϕ(X::Matrix{T}, Z::Matrix{T}, J, w::Vector{T}, A::Matrix{T},
    γ::T, σ::T, edge_pairs)::Matrix{T} where T

    λ = 1/σ

    V = similar(Z)
    prox_V = similar(Z)
    proxconj_V = similar(Z)

    computeV!(V, X, Z, λ, edge_pairs)
    proximaltp!(prox_V, V, w, γ, λ)
    proximaltpconjgivenproximaltp!(proxconj_V, V, prox_V)

    return X.-A .+ σ .* proxconj_V*J'
end


function computedϕ!(out::Matrix{T},
    X::Matrix{T}, P2::Matrix{T}, A::Matrix{T},
    src_nodes::Vector{Int}, dest_nodes::Vector{Int}, σ::T) where T

    @assert size(X) == size(A) == size(out)

    # X - A
    for i in eachindex(X)
        out[i] = X[i] - A[i]
    end

    # σ*proxconj_V*J'. see applyJt!(), but no fill!(out, 0).
    for l in eachindex(src_nodes)
        src = src_nodes[l]

        for d in axes(out,1)
            out[d,src] += P2[d,l]*σ
        end
    end

    for l in eachindex(dest_nodes)
        dest = dest_nodes[l]

        for d in axes(out,1)
            out[d,dest] -= P2[d,l]*σ
        end
    end

    return nothing
end

function computedϕgivenproximaltp!(out::Matrix{T},
    X::Matrix{T}, V::Matrix{T}, prox_V::Matrix{T}, A::Matrix{T},
    src_nodes::Vector{Int}, dest_nodes::Vector{Int}, σ::T) where T

    @assert size(X) == size(A) == size(out)

    # X - A
    for i in eachindex(X)
        out[i] = X[i] - A[i]
    end

    # σ*proxconj_V*J'. see applyJt!(), but no fill!(out, 0).
    for l in eachindex(src_nodes)
        src = src_nodes[l]

        for d in axes(out,1)
            out[d,src] += (V[d,l]-prox_V[d,l])*σ
        end
    end

    for l in eachindex(dest_nodes)
        dest = dest_nodes[l]

        for d in axes(out,1)
            out[d,dest] -= (V[d,l]-prox_V[d,l])*σ
        end
    end

    return nothing
end

## Mutates V, BX.
# All arrays must use the same indexing scheme.
function computeV!(V::Matrix{T}, BX::Matrix{T},
    X::Matrix{T},
    Z::Matrix{T},
    λ::T,
    edge_pairs::Vector{Tuple{Int,Int}}) where T <: AbstractFloat

    evalB!(BX, X, edge_pairs)

    for k in eachindex(V)
        V[k] = BX[k] + λ*Z[k]
    end

    return nothing
end

function computeV!(V::Matrix{T},
    X::Matrix{T},
    Z::Matrix{T},
    λ::T,
    edge_pairs::Vector{Tuple{Int,Int}}) where T <: AbstractFloat

    D, N = size(X)
    N_edges = length(edge_pairs)

    @assert size(V,1) == D
    @assert size(V,2) == N_edges

    for j in axes(V,2)
        a, b = edge_pairs[j]

        for d in axes(V,1)
            V[d,j] = X[d,a] - X[d,b] + λ*Z[d,j]
        end
    end

    return nothing
end

function computeV(X::Matrix{T}, Z, λ::T, edge_pairs) where T <: AbstractFloat

    V = Matrix{T}(undef, size(Z))

    computeV!(V, X, Z, λ, edge_pairs)

    return V
end

# step 2 of algorithm 1 in (Sun, JMLR 2021)
function computeU!(U::Matrix{T}, # mutated output
    BX::Matrix{T}, V::Matrix{T}, # buffers
    X::Matrix{T}, Z::Matrix{T}, # inputs
    w::Vector{T},
    γ::T,
    λ::T,
    edge_pairs::Vector{Tuple{Int,Int}}) where T <: AbstractFloat

    D, N = size(X)
    N_edges = length(edge_pairs)

    @assert size(V) == (D,N_edges) == size(Z) == size(BX)

    # update V, BX given X, Z.
    computeV!(V, BX,
        X, Z, λ, edge_pairs)

    # the update for U is prox_V. See text near quation 20 in (Sun, JMLR 2021).
    proximaltp!(U, V, w, γ, λ)

    return nothing
end


##### for optim


# input buffers are on the first line of function inputs.
# σ_in can be a Base.RefValue{T} or σ::T itself. Either way `σ::T = convert(T, σ_in[])` would work.
function compteϕXoptim!(V::Matrix{T}, prox_V::Matrix{T}, proxconj_V::Matrix{T},
    X::Matrix{T}, Z::Matrix{T}, σ_in,
    A::Matrix{T}, w::Vector{T}, γ::T, edge_pairs::Vector{Tuple{Int,Int}})::T where T <: AbstractFloat

    σ::T = convert(T, σ_in[])
    λ::T = one(T)/σ

    #
    #evalB!(BX, X, edge_pairs)
    computeV!(V, X, Z, λ, edge_pairs)
    proximaltp!(prox_V, V, w, γ, λ)
    p_prox_V = evalp(prox_V, w, γ)

    # proximaltpconjgivenproximaltp!(proxconj_V, V, prox_V)
    # term3 = (σ/2) * norm(proxconj_V, 2)^2
    term3 = (σ/2)*(dot(V,V) + dot(prox_V,prox_V) - 2*dot(prox_V,V))
    #term3 = (σ/2)*norm(V-prox_V,2)^2

    term1 = (dot(X,X) + dot(A,A) - 2*dot(X,A))/2 # norm(X-A,2)^2

    return term1 + p_prox_V + term3
end

# input buffers (gets mutated) on the second line of function inputs.
function computedϕoptim!(grad::Vector{T},
    V::Matrix{T}, prox_V::Matrix{T}, proxconj_V::Matrix{T}, grad_mat::Matrix{T},
    X::Matrix{T}, Z::Matrix{T}, σ_in,
    A::Matrix{T}, w::Vector{T}, γ::T, edge_pairs::Vector{Tuple{Int,Int}},
    src_nodes::Vector{Int}, dest_nodes::Vector{Int}) where T <: AbstractFloat

    σ::T = convert(T, σ_in[])
    λ::T = one(T)/σ

    ## set up.
    #evalB!(BX, X, edge_pairs)
    computeV!(V, X, Z, λ, edge_pairs)
    proximaltp!(prox_V, V, w, γ, λ)

    ## gradient.
    # proximaltpconjgivenproximaltp!(proxconj_V, V, prox_V)
    # computedϕ!(grad_mat, X, proxconj_V, A, src_nodes, dest_nodes, σ)
    computedϕgivenproximaltp!(grad_mat, X, V, prox_V, A,
        src_nodes, dest_nodes, σ) # faster.

    # parse.
    for i in eachindex(grad_mat)
        grad[i] = grad_mat[i]
    end

    return nothing
end

# # see https://julianlsolvers.github.io/Optim.jl/ under "Getting Better Performance" for usage case.
# # input buffers (gets mutated) on the second line of function inputs.
# function computeϕanddϕoptim!(grad::Vector{T},
#     BX::Matrix{T}, V::Matrix{T}, prox_V::Matrix{T}, X::Matrix{T}, proxconj_V::Matrix{T}, grad_mat::Matrix{T},
#     X_in::Vector{T}, Z::Matrix{T}, σ::T,
#     A::Matrix{T}, w::Vector{T}, γ::T, edge_pairs::Vector{Tuple{Int,Int}},
#     src_nodes::Vector{Int}, dest_nodes::Vector{Int}) where T <: AbstractFloat
#
#     # parse.
#     for i in eachindex(X_in)
#         X[i] = X_in[i]
#     end
#     λ::T = one(T)/σ
#
#     # prepare.
#     evalB!(BX, X, edge_pairs)
#     computeV!(V, BX, X, Z, λ, edge_pairs)
#     proximaltp!(prox_V, V, w, γ, λ)
#
#     ## gradient.
#     proximaltpconjgivenproximaltp!(proxconj_V, V, prox_V)
#     computedϕ!(grad_mat, X, proxconj_V, A, src_nodes, dest_nodes, σ)
#
#     # parse.
#     for i in eachindex(grad_mat)
#         grad[i] = grad_mat[i]
#     end
#
#     ## objective.
#     p_prox_V = evalp(prox_V, w, γ)
#
#     term3 = (σ/2)*(dot(V,V) + dot(prox_V,prox_V) - 2*dot(prox_V,V))
#     #term3 = (σ/2)*norm(V-prox_V,2)^2
#
#     term1 = (dot(X,X) + dot(A,A) - 2*dot(X,A))/2 # norm(X-A,2)^2
#
#     return term1 + p_prox_V + term3
# end
