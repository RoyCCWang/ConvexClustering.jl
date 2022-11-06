function computeD(X::Matrix{T}, J, σ::T, Z::Matrix{T}) where T
    @assert size(X,2) == size(J,1)
    @assert size(X,1) == size(Z,1)
    @assert size(J,2) == size(Z,2)

    D = X*J + (1/σ) .* Z

    return D
end

"""
D is where the derivative is taking place, uses X_^{j} and tilde{Z}.
X is the input to the linear map B_adjoint ∘ P ∘ B.
B is specified by the graph and weight setup, given by the quantities:
W, edge_pairs, J.
γ is the clustering problem regularization hyperparameter.
σ is the augmented Lagragian hyperparameter.
"""
function computeBadjPB(X::Matrix{T}, D::Matrix{T},
    W::Vector{T}, edge_pairs::Vector{Tuple{Int,Int}}, J,
    γ::T, σ::T) where T

    # setup.
    α = Vector{T}(undef, 0)
    ρ = Vector{T}(undef, 0)
    edge_inds_α_less_1 = Vector{Int}(undef, 0)

    # update from D.
    updateαρ!(α, ρ, edge_inds_α_less_1, D, W, γ, σ, X, edge_pairs)

    #
    part2 = D*diagm(ρ)*J'

    part1 = computeY(α, edge_inds_α_less_1, X, edge_pairs)*J'

    return part1, part2
end

# proposition 10a.
function computeY(α::Vector{T}, edge_inds_α_less_1::Vector{Int}, X::Matrix{T},
    edge_pairs::Vector{Tuple{Int,Int}}) where T

    Y = zeros(T, size(X,1), length(α))
    for l in edge_inds_α_less_1

        i,j = edge_pairs[l]
        Y[:,l] = (1-α[l]) .* (X[:,i]-X[:,j])
    end

    return Y
end

# size(D) is d x ne(h).
function updateαρ!(α::Vector{T}, ρ::Vector{T}, edge_inds_α_less_1::Vector{Int},
    D::Matrix{T}, W::Vector{T},
    γ::T, σ::T, X::Matrix{T}, edge_pairs::Vector{Tuple{Int,Int}}) where T

    @assert size(D,1) == size(X,1)
    @assert size(D,2) == length(edge_pairs)

    N_edges = size(D,2)
    @assert length(edge_pairs) == N_edges

    # reset and initialize.
    resize!(α, N_edges)
    fill!(α, Inf)

    resize!(ρ, N_edges)
    fill!(ρ, zero(T))

    resize!(edge_inds_α_less_1, N_edges)

    # compute α.
    k = 0
    for l = 1:N_edges
        D_l = D[:,l]

        i,j = edge_pairs[l]

        norm_D_edge = norm(D_l)
        if norm_D_edge > zero(T)

            α[l] = γ*W[l]/(norm_D_edge*σ)
        end

        if α[l] < one(T)
            k += 1
            edge_inds_α_less_1[k] = l

            ρ[l] = α[l]*dot(D_l, X[:,i]-X[:,j])/norm_D_edge^2
        end
    end
    resize!(edge_inds_α_less_1, k)

    return nothing
end
