
function runcc(
    A::Matrix{T},
    edges,
    w::Vector{T},
    config_γ,
    optim_config,
    assignment_config;
    #verbose_subproblem = false,
    report_cost = true, # want to see the objective score per θ run or γ run.
    store_trace = false,
    knn_factor = 0.2, # NaN here leads to full connectivity.
    ) where T <: Real
    #

    # initialize γ to NaN since it will be replaced by the search sequence in config_γ = getγfunc
    problem = CC.ProblemType(
        A,
        NaN,
        CC.EdgeSet(w, edges),
    )

    ### initial guess.
    D, N = size(A)
    N_edges = length(edges)

    #X0 = zeros(D, N)
    X0 = copy(A)

    dual_initial = zeros(D, N_edges)

    ### run optimization.
    Gs, rets, γs = ConvexClustering.searchγ(
        X0,
        dual_initial,
        problem,
        optim_config,
        assignment_config,
        config_γ;
        store_trace = store_trace,
        report_cost = report_cost,
    )
   
    return Gs, rets, γs
end

function constructposteriorproxy(
    Xs::Vector{Matrix{T}},
    γs::Vector{T},
    A,
    edges,
    w;
    norm = norm,
    rtol = sqrt(eps(T)), atol = 0, maxevals = typemax(Int), initdiv = 1,
    ) where T
    
    @assert length(γs) == length(Xs)

    
    ys = collect( ConvexClustering.evalprimal(Xs[n], w, edges, γs[n], A) for n in eachindex(Xs) )

    #negative_log_likelihood = Interpolations.cubic_spline_interpolation(γs, ys)
    p_tilde(x) = exp(-negative_log_likelihood(x))

    lb = minimum(γs)
    ub = maximum(γs)

    Z, err_Z = HCubature.hquadrature(p_tilde, lb, ub; norm = norm, rtol = rtol, atol = atol, maxevals = maxevals, initdiv = initdiv)

    p = xx->p_tilde(xx)/Z

    return p, lb, ub, Z, err_Z
end

#= # put on ice for now.
function buildtree!(
    LUT::Dict{Vector{Int}, Vector{Vector{Vector{Int}}}}, # mutates.
    Gs::Vector{Vector{Vector{Int}}};
    starting_level = 1,
    )
    
    Qs = unique(Gs)

    for l in Iterators.drop(eachindex(Qs), starting_level-1)
        for k in eachindex(Qs[l])
            println("Working on (level, part): ($l,$k).")

            part = Qs[l][k]
            
            # run cvx clustering.

        end
    end

    return Qs, LUT
end =#