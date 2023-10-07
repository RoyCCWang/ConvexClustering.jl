

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

function loadresult(
    A_col::Matrix{T},
    γ::T,
    σ_base::T,
    N_edges_col::Integer,
    N_edges_row::Integer,
    project_folder::String,
    ) where T

    # default
    X0 = copy(A_col)

    dual_initial = CC.ALMCoDualVar(
        CC.ALMDualVar(zeros(T, D_col, N_edges_col)),
        CC.ALMDualVar(zeros(T, D_row, N_edges_row)),
    )

    # load if we already have a result saved.
    all_file_paths = readdir(project_folder, join  = true)
    inds = findall(
        xx->(
            occursin("co_gamma_$(γ)_", xx) && occursin(".bson", xx)
        ),
        all_file_paths,
    )

    if !isempty(inds)

        file_paths = all_file_paths[inds]

        date_list = Dates.unix2datetime.(mtime.(file_paths))
        val, ind = findmax(date_list) # maximum of dates means the most recent file.
        load_path = file_paths[ind]
        println("Loading from ", load_path)
        println()

        dic = BSON.load(load_path)
        X0 = dic[:X_star]
        
        Z0_col = dic[:Z_star_col]
        Z0_row = dic[:Z_star_row]

        dual_initial = CC.ALMCoDualVar(
            CC.ALMDualVar(Z0_col),
            CC.ALMDualVar(Z0_row),
        )

        σ_base = dic[:last_sigma]
    end

    return X0, dual_initial, σ_base, length(inds)
end

function standardizedata(X::Matrix{T}) where T

    σ = Statistics.std(X)
    μ = Statistics.mean(X)
    X_std = (X .- μ ) ./ σ
    
    return X_std
end

# need to use bracket search instead of monotone search. Returning θ too small.
function autoθ(
    A_col_vecs::Vector{Vector{T}},
    A_row_vecs::Vector{Vector{T}};
    knn_factor_col = 0.2, # NaN here leads to full connectivity.
    knn_factor_row = 0.2, # NaN here leads to full connectivity.
    )

    # column j of A_col or A_row is the j-th point in the set to be partitioned.
    knn = processknn(knn_factor_col, A_col_vecs)
    A_col, edges_col, w_col, neighbourhood_col, θs_col = preparedatagraph(
        A_col_vecs;
        knn = knn,
    )

    knn = processknn(knn_factor_row, A_row_vecs)
    A_row, edges_row, w_row, neighbourhood_row, θs_row = preparedatagraph(
        A_row_vecs;
        knn = knn,
    )

    return A_col, edges_col, w_col, neighbourhood_col, θs_col,
        A_row, edges_row, w_row, neighbourhood_row, θs_row
end

function processknn(knn_factor_col::T, A_col_vecs) where T <: AbstractFloat
    knn = 0
    if isfinite(knn_factor_col)
        knn = round(Int, length(A_col_vecs)*knn_factor_col)
    else
        knn = length(A_col_vecs)-1
    end
    return knn
end

function manualθ(
    A_col_vecs::Vector{Vector{T}},
    A_row_vecs::Vector{Vector{T}};
    length_scale_col = 22.0,
    length_scale_row = 22.0,
    knn_factor_col = 0.2, # NaN here leads to full connectivity.
    knn_factor_row = 0.2, # NaN here leads to full connectivity.
    )

    knn = processknn(knn_factor_col, A_col_vecs)
    connectivity_col = CC.KNNType(knn)

    θ_col = lengthscale2θ(length_scale_col)
    A_col, edges_col, w_col, neighbourhoods_col = CC.setupproblem(
        A_col_vecs,
        θ_col,
        connectivity_col;
        metric = metric,
        kernelfunc = kernelfunc,
    )
    w_max_col = maximum(w_col)
    w_min_col = minimum(w_col)
    @show w_min_col, w_max_col, abs(w_max_col-w_min_col), knn

    knn = processknn(knn_factor_row, A_row_vecs)
    connectivity_row = CC.KNNType(knn)
    #θ = 0.0005
    #θ = 0.005
    #θ = 0.015 # yields large dynamic range: ~0.8
    #θ = 0.05
    #θ = 0.5

    θ_row = lengthscale2θ(length_scale_row)
    A_row, edges_row, w_row, neighbourhoods_row = CC.setupproblem(
        A_row_vecs,
        θ_row,
        connectivity_row;
        metric = metric,
        kernelfunc = kernelfunc,
    )
    w_max_row = maximum(w_row)
    w_min_row = minimum(w_row)
    @show w_min_row, w_max_row, abs(w_max_row-w_min_row), knn

    return A_col, edges_col, w_col, neighbourhoods_col,
        A_row, edges_row, w_row, neighbourhoods_row 
end