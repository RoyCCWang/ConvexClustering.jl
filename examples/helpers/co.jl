

function preparedatagraph(
    data::Vector{Vector{T}};
    knn = length(data)-1,
    metric = Distances.Euclidean(),
    kernelfunc = evalSqExpkernel,
    ) where T

    #connectivity = ConvexClustering.KNNType(60) # make an edge for this number of nearest neighbours of a given point i, cycle through all i in the point set to be partitioned. Takes a positive integer.
    connectivity = ConvexClustering.KNNType(knn)
    #connectivity = ConvexClustering.RadiusType(1.0) # make an edge for all points within this radius of a given point i, cycle through all i in the point set to be partitioned. Takes a finite floating point number.

    # package up config parameters.
    graph_config = ConvexClustering.WeightedGraphConfigType(
        connectivity,
        metric,
        kernelfunc,
    )

    ## variable hyperparameters

    # weight function hyperparameter search.
    length_scale_base = 10.0
    length_scale_rate = 0.7
    length_scale_max_iters = 1000
    min_dynamic_range = 0.95
    getθfunc = nn->lengthscale2θ(
        evalgeometricsequence(
            nn-1,
            length_scale_base,
            length_scale_rate,
        ),
    )
    config_θ = ConvexClustering.SearchθConfigType(
        length_scale_max_iters,
        min_dynamic_range,
        getθfunc,
    )

    verbose_kernel = true

    ### setup convex clustering problem.
    # NB is neighbourhood of A.
    A, edges, w, neighbourhood, θs = ConvexClustering.searchkernelparameters(
        T,
        data,
        config_θ,
        graph_config;
        verbose = verbose_kernel,
    )
    iter_kernel = length(θs)
    length_scale = θ2lengthscale( getθfunc(iter_kernel) )
    println("Starting length scale: ", getθfunc(1))
    println("resulting length scale: ", length_scale)
    @show min_dynamic_range

    return A, edges, w, neighbourhood, θs 
end

