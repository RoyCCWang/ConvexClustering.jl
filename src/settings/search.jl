##### search frontend for hyperparameters, given some user-specified conditions. Not an optimization formulation.


"""
Search for the first `γ` such that the number of partitions (i.e., `length(G)`) is less than `max_partition_size`, given the search strategy in `getγfunc`: `iters -> γ`.

```
searchγ(X0::Matrix{T},
    Z0::Matrix{T},
    problem::ProblemType{T},
    optim_config::ALMConfigType{T},
    assignment_config::AssignmentConfigType{T},
    search_config::SearchγConfigType;
    store_trace::Bool = true,
    report_cost::Bool = true) where T <: AbstractFloat
```

Outputs:
Gs, ret, iter
"""
function searchγ(X0::Matrix{T},
    Z0::Matrix{T},
    problem::ProblemType{T},
    optim_config::ALMConfigType{T},
    assignment_config::AssignmentConfigType{T},
    search_config::SearchγConfigType;
    store_trace::Bool = true,
    store_trace_assignments::Bool = true,
    report_cost::Bool = true) where T <: AbstractFloat

    max_iters, max_partition_size, getγfunc = search_config.max_iters, search_config.max_partition_size, search_config.getγfunc

    Gs = Vector{Vector{Vector{Int}}}(undef, 1)
    rets = Vector{ALMSolutionType{T}}(undef, 1)

    # first run.
    iter = 1
    problem.γ = getγfunc(iter)
# I am here return rets.
    G, ret = runconvexclustering(X0, Z0,
        problem, optim_config, assignment_config;
        store_trace = store_trace,
        report_cost = report_cost)

    Gs[begin] = G
    rets[begin] = ret

    # keep running if too many parts in the returned partition G.
    while length(G) > max_partition_size && iter <= max_iters

        iter += 1
        problem.γ = getγfunc(iter)

        new_G, new_ret = runconvexclustering(X0, Z0,
        problem, optim_config, assignment_config;
            store_trace = store_trace,
            report_cost = report_cost)

        if store_trace_assignments
            push!(Gs, new_G)
            push!(rets, new_ret)
        else
            Gs[begin] = new_G
            rets[begin] = new_ret
        end

        if length(new_G) < max_partition_size
            return Gs, rets, iter
        end
    end

    return Gs, rets, iter
end


# only positive-valued kernel functions are allowed.
"""
```
searchkernelparameters(
    A_vecs::Vector{Vector{T}},
    config_θ::SearchθConfigType{T},
    graph_config::WeightedGraphConfigType{ET};
    verbose::Bool = false
    ) where {T <: AbstractFloat,ET}
```

Inputs:

Optional inputs:

Outputs:
A::Matrix{T}
edge_pairs::Vector{Tuple{Int,Int}}
w::Vector{T}
A_neighbourhoods::?
iter::Int
"""
function searchkernelparameters(
    A_vecs::Vector{Vector{T}},
    config_θ::SearchθConfigType{T},
    graph_config::WeightedGraphConfigType{ET};
    verbose::Bool = false
    ) where {T <: AbstractFloat,ET}

    max_iters, min_dynamic_range, getθfunc = config_θ.max_iters, config_θ.min_dynamic_range, config_θ.getθfunc
    connectivity, metric, kernelfunc = graph_config.connectivity, graph_config.metric, graph_config.kernelfunc

    @assert zero(T) < min_dynamic_range < one(T)

    # first try.
    iter = 1
    θ::T = getθfunc(iter)

    A, edge_pairs, w, A_neighbourhoods = setupproblem(A_vecs, θ, connectivity;
        metric = metric,
        kernelfunc = kernelfunc)

    if verbose
        @show (iter, θ, minimum(w), maximum(w), min_dynamic_range)
    end

    # repeat if neccessary.
    while maximum(w) - minimum(w) < min_dynamic_range && iter <= max_iters
        iter += 1

        θ = getθfunc(iter)
        wfunc = (xx,zz)->kernelfunc(xx,zz,θ)

        w = computeweights(edge_pairs, A, wfunc)

        if maximum(w) - minimum(w) > min_dynamic_range
            A, edge_pairs, w, A_neighbourhoods = setupproblem(A_vecs, θ, connectivity;
                metric = metric,
                kernelfunc = kernelfunc)

            if verbose
                @show (iter, θ, minimum(w), maximum(w), min_dynamic_range)
            end
            return A, edge_pairs, w, A_neighbourhoods, iter
        end
    end

    if verbose
        @show (iter, minimum(w), maximum(w), min_dynamic_range)
    end
    return A, edge_pairs, w, A_neighbourhoods, iter
end



######### search strategies for knn and radius for building a graph that has a specified maximum number of connected components.

function searchknn(start_knn::Int,
    metric::Distances.Metric,
    A::Matrix{T},
    max_connected_components::Int;
    max_knn::Int = size(A,2)-1,
    verbose::Bool = false)::Tuple{Vector{Vector{Int}},Int} where T

    D, N = size(A)
    @assert 0 < max_knn < N
    @assert 0 < max_connected_components <= N

    start_knn = clamp(start_knn, 1, max_knn)

    # first.
    knn = start_knn

    connectivity = ConvexClustering.KNNType(knn)
    h, neighbourhoods = ConvexClustering.constructgraph(A, metric, connectivity)

    connected_components = Graphs.connected_components(h)

    if verbose
        @show knn, length(connected_components)
    end

    # repeat until stopping condition.
    while knn <= max_knn && length(connected_components) > max_connected_components
        knn += 1

        connectivity = ConvexClustering.KNNType(knn)
        h, neighbourhoods = ConvexClustering.constructgraph(A, metric, connectivity)

        connected_components = Graphs.connected_components(h)

        if verbose
            @show knn, length(connected_components)
        end
    end

    return neighbourhoods, knn
end

function getdistances(A::Matrix{T}, metric)::Vector{T} where T

    R = Distances.pairwise(metric, A, dims=2)

    # upper right triangule of R, without diagonal.
    D, N = size(A)

    M = div(N*(N-1), 2)
    out = Vector{T}(undef, M) # 1-indexing array.

    k = 0
    for i in Iterators.filter(i -> i[1] < i[2], CartesianIndices((size(R))))
        k += 1
        out[k] = R[i]
    end

    return sort(out)
end

function searchradius(metric,
    A::Matrix{T},
    max_connected_components::Int;
    verbose::Bool = false,
    increment_amount::Int = div(div(size(A,2)*(size(A,2)-1), 2),100), # this is div(M, 100), less than 100 + 1 iterations before teminating search. Less because of distance entries of < zero_tol, i.e. presence of repeated points or points very close together in A.
    zero_tol::T = 1e-12)::Tuple{Vector{Vector{Int}},T} where T <: AbstractFloat

    D, N = size(A)
    @assert 0 < max_connected_components <= N

    sorted_distances = getdistances(A, metric)
    max_distance = last(sorted_distances)

    M = div(N*(N-1), 2)
    @assert length(sorted_distances) == M

    # first.
    ind::Int = -1
    radius = NaN
    tmp = findfirst(xx->xx>zero_tol, sorted_distances)
    if typeof(tmp) == Nothing
        radius = zero_tol
    else
        ind = tmp
        radius = sorted_distances[ind]
    end

    connectivity = ConvexClustering.RadiusType(radius)
    h, neighbourhoods::Vector{Vector{Int}} = ConvexClustering.constructgraph(A, metric, connectivity)

    connected_components = Graphs.connected_components(h)

    if verbose
        @show radius, length(connected_components)
    end

    # repeat until stopping condition.
    while radius < max_distance && length(connected_components) > max_connected_components

        ind += increment_amount
        ind = min(ind, M)
        radius = sorted_distances[ind]

        connectivity = ConvexClustering.RadiusType(radius)
        h, neighbourhoods = ConvexClustering.constructgraph(A, metric, connectivity)

        connected_components = Graphs.connected_components(h)

        if verbose
            @show radius, length(connected_components)
        end
    end

    return neighbourhoods, radius
end
