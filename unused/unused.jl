################### for ϕ, dϕ! function preparation for Optim. Do simultaneous versions later.
function preparefuncs(X::Matrix{T}, Z::Matrix{T}, σ, A, w, γ, edges) where T <: AbstractFloat

    V_buf::Matrix{T} = Matrix{T}(undef, size(Z))
    prox_V_buf::Matrix{T} = Matrix{T}(undef, size(Z))
    proxconj_V_buf::Matrix{T} = Matrix{T}(undef, size(Z))

    D, N = size(X)

    # function under test.
    h = xx->compteϕXoptim!(
        #V_buf,
        #prox_V_buf,
        ##proxconj_V_buf,
        reg,
        reshape(xx,D,N),
        #Z,
        σ,
        A,
        #w,
        γ,
        edge_set,
    )

    # oracles.
    f = xx->evalϕX(reshape(xx,D,N), Z, w, edges, γ, σ, A)
    f2 = xx->evalϕX!(prox_V_buf, proxconj_V_buf, V_buf,
        reshape(xx,D,N), Z, w, edges, γ, σ, A)

    return h, f, f2
end

function preparedfuncs(
    X::Matrix{T},
    Z::Matrix{T},
    σ,
    A,
    w,
    γ,
    info::EdgeFormulation,
    ) where T <: AbstractFloat

    V_buf::Matrix{T} = Matrix{T}(undef, size(Z))
    prox_V_buf::Matrix{T} = Matrix{T}(undef, size(Z))
    proxconj_V_buf::Matrix{T} = Matrix{T}(undef, size(Z))
    grad_mat_buf::Matrix{T} = Matrix{T}(undef, size(X))

    #dh_eval = zeros(length(X))

    D, N = size(X)

    # src_nodes = collect( edge[1] for edge in edges)
    # dest_nodes = collect( edge[2] for edge in edges)

    # function under test.
    dh = (gg,xx)->computedϕoptim!(
        gg,
        V_buf,
        prox_V_buf,
        #proxconj_V_buf,
        grad_mat_buf,
        reshape(xx, D,N), Z, σ, A, w, γ,
        info.edges,
        #src_nodes, dest_nodes,
    )

    # oracles.
    df = xx->computedϕ(reshape(xx,D,N), Z, J, w, A,
        γ, σ, info.edges)

    return dh, df#, dh_eval
end


##### for optim


# function compteϕXoptim!(
#     #V::Matrix{T},
#     #prox_V::Matrix{T},
#     #proxconj_V::Matrix{T},
#     reg::RegularizationBuffer,
#     X::Matrix{T},
#     #Z::Matrix{T},
#     σ_buffer::Vector{T},
#     #A::Matrix{T},
#     #w::Vector{T},
#     #γ::T,
#     #edge_pairs::Vector{Tuple{Int,Int}},
#     #edge_set::EdgeFormulation,
#     problem::ProblemType,
#     )::T where T <: AbstractFloat

#     A, γ, E = unpackspecs(problem)

#     V, prox_V = reg.V, reg.prox_V
#     σ = σ_buffer[begin]
#     λ = one(T)/σ

#     #
#     ##computeV!(V, X, Z, λ, edge_pairs)
#     #computeV!(V, X, Z, λ, info)
#     #proximaltp!(prox_V, V, w, γ, λ)
#     #p_prox_V = evalp(prox_V, w, γ)
#     computeV!(reg, X, λ, E)
#     proximaltp!(prox_V, V, E.w, γ, λ)
#     p_prox_V = evalp(prox_V, w, γ)

#     # proximaltpconjgivenproximaltp!(proxconj_V, V, prox_V)
#     # term3 = (σ/2) * norm(proxconj_V, 2)^2
#     term3 = (σ/2)*(dot(V,V) + dot(prox_V,prox_V) - 2*dot(prox_V,V))
#     #term3 = (σ/2)*norm(V-prox_V,2)^2

#     term1 = (dot(X,X) + dot(A,A) - 2*dot(X,A))/2 # norm(X-A,2)^2

#     # the dot(Z,Z) term is a constant with respect to X. It is not computed.

#     return term1 + p_prox_V + term3
# end

# input buffers (gets mutated) on the second line of function inputs.
# function computedϕoptim!(
#     grad::Vector{T},
#     #V::Matrix{T},
#     #prox_V::Matrix{T},
#     ##proxconj_V::Matrix{T},
#     reg::RegularizationBuffer,
#     grad_mat::Matrix{T},
#     X::Matrix{T},
#     #Z::Matrix{T},
#     σ_buffer::Vector{T},
#     #A::Matrix{T}, w::Vector{T}, γ::T,
#     problem::ProblemType,
#     #edge_pairs::Vector{Tuple{Int,Int}},
#     #E::EdgeFormulation,
#     ) where T <: AbstractFloat

#     A, γ, E = unpackspecs(problem)

#     Z, V, prox_V = reg.Z, reg.V, reg.prox_V
#     σ = σ_buffer[begin]
#     λ = one(T)/σ

#     ## set up.
#     #computeV!(V, X, Z, λ, edge_pairs)
#     computeV!(reg, X, λ, E)
#     proximaltp!(prox_V, V, w, γ, λ)

#     ## gradient.
#     # proximaltpconjgivenproximaltp!(proxconj_V, V, prox_V)
#     # computedϕ!(grad_mat, X, proxconj_V, A, src_nodes, dest_nodes, σ)
#     computedϕgivenproximaltp!(
#         grad_mat,
#         X,
#         V,
#         prox_V,
#         A,
#         E.edges,
#         #edge_pairs,
#         σ,
#     ) # faster.

#     # parse.
#     for i in eachindex(grad_mat)
#         grad[i] = grad_mat[i]
#     end

#     return nothing
# end

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




