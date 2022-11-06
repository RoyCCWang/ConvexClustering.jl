


# using paper's convention for incidence matrix.
function computeBX(X::Vector{Vector{T}}, edge_pairs) where T
    return collect(X[edge[1]] - X[edge[2]] for edge in edge_pairs)
end

function computeBX(X::Matrix{T}, h) where T
    return collect(X[:,edge[1]] - X[:,edge[2]] for edge in edge_pairs)
end

# using Graphs.jl's convention for incidence matrix.
function computeBXrev(X::Vector{Vector{T}}, h) where T
    return collect(X[edge[2]] - X[edge[1]] for edge in edge_pairs)
end


function computeBXedgeinds(X::Vector{Vector{T}}, edge_inds) where T
    return collect(X[a[1]] - X[a[2]] for a in edge_inds)
end

function computePU(α::Vector{T}, D::Matrix{T}, edge_inds_α_less_1::Vector{Int},
    U::Matrix{T}) where T

    @assert size(U) == size(D)

    out1 = zeros(T, size(U))
    out2 = zeros(T, size(U))
    for l in edge_inds_α_less_1

        D_l = D[:,l]
        U_l = U[:,l]

        out1[:,l] = (1-α[l]) .* U_l
        out2[:,l] = α[l]*dot(D_l,U_l)/norm(D_l)^2 .* D_l
    end

    return out1, out2
end
