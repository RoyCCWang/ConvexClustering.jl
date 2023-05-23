# Equation 17 and related discussions in section 5.3. (Sun, JMLR 2011).
# We focus on ϕ and dϕ_dx in our work.

# function evalϕ(X, Z, w, J, σ, γ::T, A)::T where T <: AbstractFloat

#     term_X = evalϕX(X, Z, w, J, σ, γ)

#     term_rest = -dot(Z,Z)/(2*σ)

#     return term_X + term_rest
# end

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

########### dϕ

# not used in runALM()
function computedϕdirect(X::Matrix{T}, proxconj_V::Matrix{T}, J, A::Matrix{T}, σ::T)::Matrix{T} where T
    #
    #proximaltpconjgivenproximaltp!(proxconj_V, V, Y)
    return X.-A .+ σ .* proxconj_V*J'
end

# not used in runALM()
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

# not used in runALM()
function computedϕ!(
    out::Matrix{T},
    X::Matrix{T},
    P2::Matrix{T},
    A::Matrix{T},
    # src_nodes::Vector{Int},
    # dest_nodes::Vector{Int},
    edges::Vector{Tuple{Int,Int}},
    σ::T,
    ) where T

    @assert size(X) == size(A) == size(out)

    # X - A
    for i in eachindex(X)
        out[i] = X[i] - A[i]
    end

    # σ*proxconj_V*J'. see applyJt!(), but no fill!(out, 0).
    #for l in eachindex(src_nodes)
        #src = src_nodes[l]
    for l in eachindex(edges)
        src, dest = edges[l]

        for d in axes(out,1)
            out[d,src] += P2[d,l]*σ
        end
    #end

    #for l in eachindex(dest_nodes)
        #dest = dest_nodes[l]

        for d in axes(out,1)
            out[d,dest] -= P2[d,l]*σ
        end
    end

    return nothing
end

## Mutates V, BX.
# All arrays must use the same indexing scheme.
function computeV!(
    V::Matrix{T},
    BX::Matrix{T},
    op_trait::MatrixOperationTrait,
    X::AbstractMatrix{T},
    Z::Matrix{T},
    λ::T,
    edge_pairs::Vector{Tuple{Int,Int}},
    ) where T <: AbstractFloat

    evalB!(BX, op_trait, X, edge_pairs)

    for k in eachindex(V)
        V[k] = BX[k] + λ*Z[k]
    end

    return nothing
end

function computeV!(
    V::Matrix{T},
    ::ColumnWise,
    X::Matrix{T},
    Z::Matrix{T},
    λ::T,
    edge_pairs::Vector{Tuple{Int,Int}},
    ) where T <: AbstractFloat

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

function computeV!(
    V::Matrix{T},
    ::RowWise,
    X::Matrix{T},
    Z::Matrix{T},
    λ::T,
    edge_pairs::Vector{Tuple{Int,Int}},
    ) where T <: AbstractFloat

    D, N = size(X)
    N_edges = length(edge_pairs)

    @assert size(V,1) == N
    @assert size(V,2) == N_edges

    for j in axes(V,2)
        a, b = edge_pairs[j]

        for d in axes(V,1)
            V[d,j] = X[a,d] - X[b,d] + λ*Z[d,j]
        end
    end

    return nothing
end

# function computeV(X::Matrix{T}, Z, λ::T, edge_pairs) where T <: AbstractFloat

#     V = Matrix{T}(undef, size(Z))

#     computeV!(V, X, Z, λ, edge_pairs)

#     return V
# end

################################### compatible with co-clustering

function computeV!(
    Q::BMapBuffer{T},
    X::Matrix{T},
    λ::T,
    E::EdgeSet,
    ) where T <: AbstractFloat

    return computeV!(Q.V, ColumnWise(), X, Q.Z, λ, E.edges)
end
 
function computeV!(
    Q::CoBMapBuffer{T},
    X::Matrix{T},
    λ::T,
    E::CoEdgeSet,
    ) where T <: AbstractFloat

    computeV!(Q.col.V, ColumnWise(), X, Q.col.Z, λ, E.col.edges)
    computeV!(Q.row.V, RowWise(), X, Q.row.Z, λ, E.row.edges)

    return nothing
end


##########

function compteϕXoptimterms!(
    reg::BMapBuffer,
    op_trait::MatrixOperationTrait,
    X::Matrix{T},
    σ_buffer::Vector{T},
    problem::ProblemType,
    )::Tuple{T,T} where T <: AbstractFloat
    
    _, γ, E = unpackspecs(problem)
    Z, V, prox_V = reg.Z, reg.V, reg.prox_V
    
    σ =  σ_buffer[begin]
    λ = one(T)/σ

    w = getw(E, op_trait)
    edges = getedges(E, op_trait)
    
    computeV!(V, op_trait, X, Z, λ, edges)
    proximaltp!(prox_V, V, w, γ, λ)
    p_prox_V = evalp(prox_V, w, γ)

    #this is (σ/2)*norm(V-prox_V,2)^2
    term3 = (σ/2)*(dot(V,V) + dot(prox_V,prox_V) - 2*dot(prox_V,V))

    return p_prox_V, term3
end

function computeϕXoptim!(
    reg::BMapBuffer,
    X::Matrix{T},
    σ_buffer::Vector{T},
    problem::ProblemType{T,EdgeSet{T}},
    )::T where T <: AbstractFloat
    
    term2, term3 = compteϕXoptimterms!(reg, ColumnWise(), X, σ_buffer, problem)

    A = problem.A
    term1 = (dot(X,X) + dot(A,A) - 2*dot(X,A))/2 # thi is norm(X-A,2)^2

    return term1 + term2 + term3
end

function computeϕXoptim!(
    reg::CoBMapBuffer,
    X::Matrix{T},
    σ_buffer::Vector{T},
    problem::ProblemType{T,CoEdgeSet{T}},
    )::T where T <: AbstractFloat
    
    term2_col, term3_col = compteϕXoptimterms!(reg.col, ColumnWise(), X, σ_buffer, problem)
    term2_row, term3_row = compteϕXoptimterms!(reg.row, RowWise(), X, σ_buffer, problem)

    A = problem.A
    term1 = (dot(X,X) + dot(A,A) - 2*dot(X,A))/2 # thi is norm(X-A,2)^2

    return term1 + term2_col + term2_row + term3_col + term3_row
end

############

function computedϕoptim!(
    grad::Vector{T},
    reg::BMapBuffer,
    grad_mat::Matrix{T},
    X::Matrix{T},
    σ_buffer::Vector{T},
    problem::ProblemType{T,EdgeSet{T}},
    ) where T <: AbstractFloat

    γ = problem.γ
    σ = σ_buffer[begin]
    λ = one(T)/σ

    ## update pre-requisites.
    computeV!(reg.V, ColumnWise(), X, reg.Z, λ, problem.edge_set.edges)
    proximaltp!(reg.prox_V, reg.V, problem.edge_set.w, γ, λ)

    ## gradient.
    computedϕgivenproximaltp!(grad_mat, X, reg, problem, σ) # faster.

    # parse into output vector.
    for i in eachindex(grad_mat)
        grad[i] = grad_mat[i]
    end

    return nothing
end

function computedϕoptim!(
    grad::Vector{T},
    reg::CoBMapBuffer,
    grad_mat::Matrix{T},
    X::Matrix{T},
    σ_buffer::Vector{T},
    problem::ProblemType{T,CoEdgeSet{T}},
    ) where T <: AbstractFloat

    γ = problem.γ
    σ = σ_buffer[begin]
    λ = one(T)/σ

    ## update pre-requisites.
    # column edges.
    computeV!(reg.col.V, ColumnWise(), X, reg.col.Z, λ, problem.edge_set.col.edges)
    proximaltp!(reg.col.prox_V, reg.col.V, problem.edge_set.col.w, γ, λ)

    # row edges.
    #@show size(reg.row.V), size(reg.row.Z), size(problem.edge_set.row.edges) # debug
    computeV!(reg.row.V, RowWise(), X, reg.row.Z, λ, problem.edge_set.row.edges)
    proximaltp!(reg.row.prox_V, reg.row.V, problem.edge_set.row.w, γ, λ)
    
    ## gradient.
    computedϕgivenproximaltp!(grad_mat, X, reg, problem, σ) # faster.

    # parse into output vector.
    for i in eachindex(grad_mat)
        grad[i] = grad_mat[i]
    end

    return nothing
end

function computedϕgivenproximaltp!(
    out::Matrix{T},
    X::Matrix{T},
    reg::BMapBuffer,
    problem::ProblemType{T,EdgeSet{T}},
    σ::T,
    ) where T
    
    A = problem.A
    @assert size(X) == size(A) == size(out)

    # contribution from the data fidelity terms.
    for i in eachindex(X)
        out[i] = X[i] - A[i]
    end

    # contribution from regularization terms.
    applyproxcontribution!(out, ColumnWise(), reg.V, reg.prox_V, problem.edge_set.edges, σ)

    return nothing
end

function computedϕgivenproximaltp!(
    out::Matrix{T},
    X::Matrix{T},
    reg::CoBMapBuffer,
    problem::ProblemType{T,CoEdgeSet{T}},
    σ::T,
    ) where T
    
    A = problem.A
    @assert size(X) == size(A) == size(out)

    # contribution from the data fidelity terms.
    for i in eachindex(X)
        out[i] = X[i] - A[i]
    end

    # contribution from regularization terms.
    applyproxcontribution!(out, ColumnWise(), reg.col.V, reg.col.prox_V, problem.edge_set.col.edges, σ)
    applyproxcontribution!(out, RowWise(), reg.row.V, reg.row.prox_V, problem.edge_set.row.edges, σ)

    return nothing
end

# the computation of one proximal conjugation term for dϕ.
# updates the output where the nodes in `edges` are columns of out.
function applyproxcontribution!(
    out::Matrix{T},
    ::ColumnWise,
    V::Matrix{T},
    prox_V::Matrix{T},
    edges::Vector{Tuple{Int,Int}},
    σ::T,
    ) where T

    # σ*proxconj_V*J'. see applyJt!(), but no fill!(out, 0).
    for l in eachindex(edges)
        src, dest = edges[l]

        for d in axes(out,1)
            out[d,src] += (V[d,l]-prox_V[d,l])*σ
        end

        for d in axes(out,1)
            out[d,dest] -= (V[d,l]-prox_V[d,l])*σ
        end
    end

    return nothing
end

# updates the output where the nodes in `edges` are rows of out.
function applyproxcontribution!(
    out::Matrix{T},
    ::RowWise,
    V::Matrix{T},
    prox_V::Matrix{T},
    edges::Vector{Tuple{Int,Int}},
    σ::T,
    ) where T

    # σ*proxconj_V*J'. see applyJt!(), but no fill!(out, 0).
    for l in eachindex(edges)
        src, dest = edges[l]

        for d in axes(out,2)
            out[src,d] += (V[d,l]-prox_V[d,l])*σ
        end

        for d in axes(out,2)
            out[dest,d] -= (V[d,l]-prox_V[d,l])*σ
        end
    end

    return nothing
end