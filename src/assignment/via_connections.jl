"""
Uses the graph of BX to assign partitions. Chi 2015's method.
Only points that have non-negative weights are checked to see if in the same partition.
"""
function assignviaBX(X::Matrix{T},
    edge_pairs::Vector{Tuple{Int,Int}};
    zero_tol = 1e-6)::Vector{Vector{Int}} where T <: AbstractFloat

    D, N = size(X)
    N_edges = length(edge_pairs)

    BX = Matrix{T}(undef, D, N_edges)
    ConvexClustering.evalB!(BX, X, edge_pairs)

    # if BX[:l] is near zero, add an edge between node i and node j,
    #   i,j = edge_pairs[l].
    h = Graphs.SimpleGraph(N)

    for l in axes(BX,2)

        #if evalcolnorm(BX,l) < zero_tol
        if norm(BX[:,l],2) < zero_tol

            i, j = edge_pairs[l]
            Graphs.add_edge!(h, i, j)
        end
    end

    G = Graphs.connected_components(h)

    return G
end

"""
assignviaX(
    X::AbstractMatrix{T},
    metric::Distances.Metric;
    zero_tol = 1e-6,
)::Tuple{Vector{Vector{Int}}, Vector{Vector{Int}}} where T <: AbstractFloat

Assumes that each column of X as a point in the set to the partitioned.
"""
function assignviaX(
    X::AbstractMatrix{T},
    metric::Distances.Metric;
    zero_tol::T = convert(T,1e-6),
    )::Tuple{Vector{Vector{Int}}, Vector{Vector{Int}}} where T <: AbstractFloat

    connectivity = RadiusType(zero_tol)
    g, g_neighbourhoods = constructgraph(X, metric, connectivity)
    G = Graphs.connected_components(g)

    return G, g_neighbourhoods
end

#
"""
applyassignment(connected_components::Vector{Vector{Int}},
    A::Vector{Vector{T}})::Vector{Vector{Vector{T}}} where T <: AbstractFloat

does not create a copy of A.
"""
function applyassignment(connected_components::Vector{Vector{Int}},
    A::Vector{Vector{T}})::Vector{Vector{Vector{T}}} where T <: AbstractFloat

    N_parts = length(connected_components)

    # partioning as a 1D array.
    P = Vector{Vector{Vector{T}}}(undef, N_parts)
    for k in eachindex(P)
        P[k] = A[connected_components[k]]
    end

    return P
end
