
"""
setupproblem(
    A_vecs::Vector{Vector{T}},
    θ,
    connectivity::AbstractConnectivityType;
    kernelfunc::Function = (xx,zz,tt)->convert(T, exp(-tt*norm(xx-zz)^2)),
    metric = Distances.Euclidean(),
    ) where T <: AbstractFloat

"""
function setupproblem(
    A_vecs::Vector{Vector{T}},
    θ,
    connectivity::AbstractConnectivityType;
    kernelfunc::Function = (xx,zz,tt)->convert(T, exp(-tt*norm(xx-zz)^2)),
    metric = Distances.Euclidean(),
    ) where T <: AbstractFloat

    A = array2matrix(A_vecs)
    #D, N = size(A)

    h, neighbourhoods = constructgraph(A, metric, connectivity)
    most_connections = maximum(length.(neighbourhoods))
    N_edges = Graphs.ne(h)

    #eds = collect(edge for edge in Graphs.edges(h))
    edge_pairs::Vector{Tuple{Int64, Int64}} = collect(
        (
            Graphs.src(edge),
            Graphs.dst(edge)
        )
        for edge in Graphs.edges(h)
    )
    @assert length(edge_pairs) == N_edges

    #J0 = Graphs.incidence_matrix(h; oriented = true)
    #L_G = Graphs.laplacian_matrix(h)

    #J = computeJ(edge_pairs, N, N_edges) # slip flipped from J0, but this is (Sun, JMLR 2011)'s convention.
    #norm(J*J' - L_G) # should be zero.

    wfunc = (xx,zz)->kernelfunc(xx,zz,θ)
    w::Vector{T} = computeweights(edge_pairs, A, wfunc)

    return A, edge_pairs, w, neighbourhoods#, J, h
end

"""
setupproblem(A_vecs::Vector{Vector{T}}, θ, config::WeightedGraphConfigType{ET}) where {T <: AbstractFloat, ET}
"""
function setupproblem(
    A_vecs::Vector{Vector{T}},
    θ,
    config::WeightedGraphConfigType{ET},
    ) where {T <: AbstractFloat, ET}

    return setupproblem(
        A_vecs,
        θ,
        config.connectivity;
        metric = config.metric,
        kernelfunc = config.kernelfunc,
    )
end

function getneighbourhoods(X::Matrix{T}, metric::Distances.Metric,
    connectivity::KNNSearchType)::Vector{Vector{Int}} where T

    neighbourhoods, knn = connectivity.searchfunc(X, metric)
    connectivity.knn = knn # store search result.

    return neighbourhoods
end

function getneighbourhoods(X::Matrix{T}, metric::Distances.Metric,
    connectivity::RadiusSearchType{ET})::Vector{Vector{Int}} where {T,ET}

    neighbourhoods, radius = connectivity.searchfunc(X, metric)
    connectivity.radius = radius # store search result.

    return neighbourhoods
end

function getneighbourhoods(X::Matrix{T}, metric::Distances.Metric,
    connectivity::KNNType{ET})::Vector{Vector{Int}} where {T,ET}
    #
    knn = connectivity.knn

    if knn >= size(X,2)-1
        println("Warning, the supplied knn is larger than the number of points, N ($(size(X,2))). Default to using knn = N-1 instead.")
        knn = size(X,2)-1
    end

    balltree = NearestNeighbors.BallTree(X, metric; reorder = false)
    neighbourhoods, _ = NearestNeighbors.knn(balltree, X, knn+1) # since X is used instead of X without current point, need to add 1.

    return neighbourhoods
end

function getneighbourhoods(X::Matrix{T}, metric::Distances.Metric,
    connectivity::RadiusType{ET})::Vector{Vector{Int}} where {T,ET}

    radius = connectivity.radius

    if radius < 0
        println("Warning, the supplied radius is negative. Default to using norm(X,2)/length(X) ($(norm(X,2))/$(length(X))) instead.")
        radius = convert(ET, norm(X,2)/length(X))
    end

    balltree = NearestNeighbors.BallTree(X, metric; reorder = false)
    neighbourhoods = NearestNeighbors.inrange(balltree, X, radius)

    return neighbourhoods
end

function constructgraph(X::Matrix{T},
    metric::Distances.Metric,
    connectivity::AbstractConnectivityType{ET}) where {T<: AbstractFloat, ET}

    # TODO work on a pre-processing option on reducing the columns of X such that points are merged if they are within ϵ-disance of each other.
    # however, there might be physical meaning of points that have the same coordinate in applications, and one might want to keep them as separate points.

    # neighbourhoods[k] will include k itself.
    neighbourhoods = getneighbourhoods(X, metric, connectivity)

    # create graph from neighbourhoods, avoid self-loop because the same point will be assigned to the same part anyways in the assignment algorithm.
    h = buildgraph(neighbourhoods, X)

    return h, neighbourhoods
end

# create graph from neighbourhoods, avoid self-loop because the same point will be assigned to the same part anyways in the assignment algorithm.
function buildgraph(neighbourhoods::Vector{Vector{Int}}, X::Matrix{T}) where T
    N = size(X,2)
    h = Graphs.SimpleGraph(N)

    for n in eachindex(neighbourhoods)
        for k in eachindex(neighbourhoods[n])
            if n != neighbourhoods[n][k] # avoid self-loops.
                Graphs.add_edge!(h, n, neighbourhoods[n][k])
            end
        end
    end

    return h
end
# no error-checking on the contents of edge_pairs. wfunc must return type T.
"""
computeweights(edge_pairs, X::Matrix{T}, wfunc::Function)::Vector{T} where T
"""
function computeweights(edge_pairs, X::Matrix{T}, wfunc::Function)::Vector{T} where T

    w = Vector{T}(undef, length(edge_pairs))

    for l in eachindex(w)
        edge = edge_pairs[l]

        w[l] = convert(T, wfunc(X[:,edge[1]], X[:,edge[2]]))
    end

    return w
end

# using the SSNL paper's sign convention.
function computeJ(edge_pairs, N::Int, N_edges::Int)

    J = spzeros(Int, N, N_edges)

    for k in axes(J,1)

        c = 1
        for edge in edge_pairs
            if k == edge[1]
                J[k,c] = 1
            elseif k == edge[2]
                J[k,c] = -1
            end

            c += 1
        end
    end

    return J
end
