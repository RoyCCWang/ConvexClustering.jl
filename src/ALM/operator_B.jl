


##### operator B.

# implements X*J. J described by `edge_pairs`.
# faster than X*J, X is dense matrix, J is sparse matrix.
#function evalB!(BX::Matrix{T}, X::Matrix{T}, J) where T <: AbstractFloat
# each `edge_pairs` entry contains a pair of indices. Each is a column index of X.
function evalB!(BX::Matrix{T}, ::ColumnWise, X::Matrix{T}, edge_pairs::Vector{Tuple{Int,Int}}) where T <: AbstractFloat

    D, N = size(X)
    N_edges = length(edge_pairs)

    @assert size(BX,1) == D
    @assert size(BX,2) == N_edges

    for j in axes(BX,2)
        a, b = edge_pairs[j]

        for d in axes(BX,1)
            BX[d,j] = X[d,a] - X[d,b]
        end
    end

    return nothing
end

function evalB!(BX::Matrix{T}, X::Matrix{T}, edge_pairs::Vector{Tuple{Int,Int}}) where T <: AbstractFloat
    evalB!(BX, ColumnWise(), X, edge_pairs)
end

function evalB!(BX::Matrix{T}, ::RowWise, X::Matrix{T}, edge_pairs::Vector{Tuple{Int,Int}}) where T <: AbstractFloat

    D, N = size(X)
    N_edges = length(edge_pairs)

    @assert size(BX,1) == N
    @assert size(BX,2) == N_edges

    for j in axes(BX,2)
        a, b = edge_pairs[j]

        for d in axes(BX,1)
            BX[d,j] = X[a,d] - X[b,d]
        end
    end

    return nothing
end

# slower the matrix BX version of evalB!().
# Each element of BX should be of length D.
# BX is for internal use. No error-handling via resize!() for it, for speed.
function evalB!(BX::Vector{Vector{T}}, X::Matrix{T}, edge_pairs::Vector{Tuple{Int,Int}}) where T <: AbstractFloat

    D, N = size(X)
    N_edges = length(edge_pairs)

    @assert length(BX) == N_edges

    for j in eachindex(BX)
        #@assert size(BX[j]) == D
        #resize(BX[j], D)
        a, b = edge_pairs[j]

        for d in eachindex(BX[j])
            BX[j][d] = X[d,a] - X[d,b]
        end
    end

    return nothing
end

# biclustering
function evalB!(
    BX::Matrix{T},
    X::Matrix{T},
    edge_pairs::Vector{Vector{Tuple{Int,Int}}},
    ) where T <: AbstractFloat

    D, N = size(X)
    N_edges = length(edge_pairs)

    @assert size(BX,1) == D
    @assert size(BX,2) == N_edges

    for j in axes(BX,2)
        a, b = edge_pairs[j]

        for d in axes(BX,1)
            BX[d,j] = X[d,a] - X[d,b]
        end
    end

    return nothing
end

# no error-checking on the indices in positives and negatives! They should be column indices of BadjZ.
function evalBadj!(BadjZ::Matrix{T}, Z::Matrix{T}, positives, negatives) where T <: AbstractFloat

    D, N_edges = size(Z)
    N = size(BadjZ,2)

    @assert size(BadjZ,1) == D
    @assert length(positives) == length(negatives) == N

    fill!(BadjZ, zero(T))

    ## very slow.
    # for n in axes(BadjZ,2)
    #     # for d in axes(BadjZ,1)
    #     #     for i in positives[n]
    #     #         BadjZ[d,n] += Z[d,i]
    #     #     end
    #     #
    #     #     for i in negatives[n]
    #     #         BadjZ[d,n] -= Z[d,i]
    #     #     end
    #     # end
    # end

    # faster.
    for n in axes(BadjZ,2)

        for i in positives[n]
            for d in axes(BadjZ,1)
                BadjZ[d,n] += Z[d,i]
            end
        end

        for i in negatives[n]
            for d in axes(BadjZ,1)
                BadjZ[d,n] -= Z[d,i]
            end
        end
    end

    return nothing
end

# src_nodes = collect( edge[1] for edge in edge_pairs)
# dest_nodes = collect( edge[2] for edge in edge_pairs)
# # here, consider going through the src_nodes and add their d to Badj.
# # then the dest_nodes and subtract their entries to Badj.
# # instead of iterating over Badj.
function applyJt(
    Z::Matrix{T},
    edges::Vector{Tuple{Int,Int}},
    N::Int) where T <: AbstractFloat

    @assert length(edges) == size(Z,2) # this is N_edges.

    out = Matrix{T}(undef, size(Z,1), N)
    applyJt!(out, Z, edges)

    return out
end

# different data structure. faster according to examples/timing.jl
function applyJt!(
    out::Matrix{T},
    Z::Matrix{T},
    edges::Vector{Tuple{Int,Int}},
    ) where T <: AbstractFloat

    #N = size(out,2) # no error-checking on this.
    @assert length(edges) == size(Z,2) # this is N_edges.
    @assert size(Z,1) == size(out,1) #  this is D.

    fill!(out, zero(T))

    for l in eachindex(edges)
        src, dest = edges[l]

        for d in axes(out,1)
            out[d,src] += Z[d,l]
        end

        for d in axes(out,1)
            out[d,dest] -= Z[d,l]
        end
    end

    return out
end

function applyJt!(
    out::Matrix{T},
    Z::Matrix{T},
    src_nodes::Vector{Int},
    dest_nodes::Vector{Int},
    ) where T <: AbstractFloat

    #N = size(out,2) # no error-checking on this.
    @assert length(src_nodes) == length(dest_nodes) == size(Z,2) # this is N_edges.
    @assert size(Z,1) == size(out,1) #  this is D.

    fill!(out, zero(T))

    for l in eachindex(src_nodes)
        src = src_nodes[l]

        for d in axes(out,1)
            out[d,src] += Z[d,l]
        end
    end

    for l in eachindex(dest_nodes)
        dest = dest_nodes[l]

        for d in axes(out,1)
            out[d,dest] -= Z[d,l]
        end
    end

    return out
end

"""
A data point is a node in the graph, indexed from 1 to the number of data points, denoted `N` here.
The output here contains the information of the positive 1 entries and negative 1 entries of transpose(J) in (Sun JMLR 2021).
"""
function decomposeedgepairs(edge_pairs, N::Int)

    # outputs.
    positives = Vector{Vector{Int}}(undef, N)
    negatives = Vector{Vector{Int}}(undef, N)
    for i in eachindex(positives)
        positives[i] = Vector{Int}(undef,0)
        negatives[i] = Vector{Int}(undef,0)
    end

    #
    for j in eachindex(edge_pairs)
        src_node, dest_node = edge_pairs[j]

        push!(positives[src_node], j)
        push!(negatives[dest_node], j)
    end

    return positives, negatives
end


function setupB(X::Matrix{T}, edge_pairs::Vector{Tuple{Int,Int}}) where T <: AbstractFloat

    D, N = size(X)
    N_edges = length(edge_pairs)

    BX = zeros(T, D, N_edges)
    updateBXfunc = xx->evalB!(BX, xx, edge_pairs)

    positives, negatives = decomposeedgepairs(edge_pairs, N)
    BadjZ = zeros(T, D, N)
    updateBadjZfunc = zz->evalBadj!(BadjZ, zz, positives, negatives)

    return updateBXfunc, BX, updateBadjZfunc, BadjZ
end

function computepdgap(
    reg::BMapBuffer{T}, # mutates tmp buffers.
    X::AbstractMatrix{T},
    A::AbstractMatrix{T},
    γ::T,
    edge_set::EdgeSet,
    norm_A_p_1::T,
    norm_U_factor::T,
    )::T where T

    U, BX, prox_U_plus_Z, U_plus_Z, BadjZ = unpackbuffer(reg.residual)
    w = edge_set.w

    ## primal-dual.

    # pd1, first term in numerator.
    #   (a+b-c)^2 = a^2 + 2ab - 2ac + b^2 - 2bc + c^2
    #applyJt!(BadjZ, Z, src_nodes, dest_nodes)
    applyJt!(BadjZ, Z, edge_set.edges)
    # a = BadjZ
    # b = X
    # c = A
    #norm_c = abs(norm_A_p_1 - one(T)) # abs() in case of numerical precision issues.
    #norm_a_plus_b_minus_c = dot(a,a) + 2*dot(a,b) - 2*dot(a,c) + dot(b,b) - 2*dot(b,c) + norm_c*norm_c
    #pd1::T = sqrt(norm_a_plus_b_minus_c)
    pd1 = norm(BadjZ .+ X .- A, 2)

    # pd2, second term in numberator.
    pd2 = computepd2gap(reg, γ, edge_set)

    primal_dual_gap = (pd1+pd2)/(norm_A_p_1 + norm_U_factor)

    return primal_dual_gap
end

function computepdgap(
    reg::CoBMapBuffer{T}, # mutates tmp buffers.
    X::AbstractMatrix{T},
    A::AbstractMatrix{T},
    γ::T,
    edge_set::CoEdgeSet,
    norm_A_p_1::T,
    norm_U_factor::T,
    )::T where T

    _, _, _, _, BadjZ_col = unpackbuffer(reg.col.residual)
    _, _, _, _, BadjZ_row = unpackbuffer(reg.row.residual)
    #w = edge_set.w

    ## primal-dual.

    # pd1, first term in numerator.
    #   (a+b-c)^2 = a^2 + 2ab - 2ac + b^2 - 2bc + c^2
    #applyJt!(BadjZ, Z, src_nodes, dest_nodes)
    applyJt!(BadjZ_col, reg.col.Z, edge_set.col.edges)
    applyJt!(BadjZ_row, reg.row.Z, edge_set.row.edges)

    pd1 = norm(BadjZ_col .+ BadjZ_row' .+ X .- A, 2)

    # pd2, second term in numberator.
    pd2_col = computepd2gap(reg.col, γ, edge_set.col)
    pd2_row = computepd2gap(reg.row, γ, edge_set.row)

    primal_dual_gap = (pd1 + pd2_col + pd2_row)/(norm_A_p_1 + norm_U_factor)

    return primal_dual_gap
end

# assumes U has been updated!
function computepd2gap(
    reg::BMapBuffer{T}, # mutates tmp buffers.
    γ::T,
    edge_set::EdgeSet,
    )::T where T

    U, BX, prox_U_plus_Z, U_plus_Z, BadjZ = unpackbuffer(reg.residual)
    w = edge_set.w
    Z = reg.Z

    ## primal-dual

    # pd2, second term in numberator.
    # overwrite the temp. buffer `U_plus_Z`.
    for i in eachindex(U_plus_Z)
        U_plus_Z[i] = U[i] + Z[i]
    end
    proximaltp!(prox_U_plus_Z, U_plus_Z, w, γ, one(T))
    pd2 = sqrt(evalnorm2sq(U, prox_U_plus_Z))

    return pd2
end