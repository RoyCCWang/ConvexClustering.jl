"""
runconvexclustering(X0::Matrix{T},
    Z0::Matrix{T},
    problem::ProblemType{T, ET},
    optim_config::ALMConfigType{T},
    assignment_config::AssignmentConfigType{T};
    store_trace::Bool = false,
    report_cost = false) where {T <: AbstractFloat, ET <: EdgeFormulation}
"""
function runconvexclustering(
    X0::Matrix{T},
    Z0::Matrix{T},
    problem::ProblemType{T,EdgeSet{T}},
    optim_config::ALMConfigType{T},
    assignment_config::AssignmentConfigType{T};
    store_trace::Bool = false,
    report_cost = false,
    verbose_ALM = false,
    ) where T <: AbstractFloat

    ret = runALM(
        X0,
        ALMDualVar(Z0),
        problem,
        optim_config;
        store_trace = store_trace,
        verbose = verbose_ALM,
    )

    G, g_neighbourhoods = assignviaX(ret.X_star, assignment_config.metric;
        zero_tol = assignment_config.zero_tol)

    if report_cost
        cost = evalprimal(ret.X_star,problem)
        cost = round.(cost, sigdigits = 4)

        sol_gaps = round.(ret.gaps, sigdigits = 4)

        partition_size = length(G)
        cc_iters = ret.num_iters_ran

        @show (problem.γ, cost, sol_gaps, cc_iters, partition_size)
    end

    return G, ret
end

function runconvexclustering(
    X0::Matrix{T},
    dual0::ALMDualVar,
    problem::ProblemType{T,EdgeSet{T}},
    optim_config::ALMConfigType{T},
    assignment_config::AssignmentConfigType{T};
    store_trace::Bool = false,
    report_cost = false,
    verbose_ALM = false,
    ) where T <: AbstractFloat
    
    return runconvexclustering(X0, dual0.Z, problem, optim_config, assignment_config;
        store_trace = store_trace,
        report_cost = report_cost,
        verbose_ALM = verbose_ALM,
    )
end

# co-clustering case.
function runconvexclustering(
    X0::Matrix{T},
    dual0::ALMCoDualVar{T}, #Z0::Matrix{T},
    problem::ProblemType{T,CoEdgeSet{T}},
    optim_config::ALMConfigType{T},
    ac::CoAssignmentConfigType{T};
    store_trace::Bool = false,
    report_cost = false,
    verbose_ALM = false,
    ) where T <: AbstractFloat

    assignment_config_col, assignment_config_row = ac.col, ac.row
    ret = runALM(
        X0,
        dual0,
        problem,
        optim_config;
        store_trace = store_trace,
        verbose = verbose_ALM,
    )

    G_col, g_neighbourhoods_col = assignviaX(
        ret.X_star,
        assignment_config_col.metric;
        zero_tol = assignment_config_col.zero_tol,
    )
    G_row, g_neighbourhoods_row = assignviaX(
        ret.X_star',
        assignment_config_row.metric;
        zero_tol = assignment_config_col.zero_tol,
    )

    if report_cost
        cost = evalprimal(ret.X_star, problem)
        cost = round.(cost, sigdigits = 4)

        sol_gaps = round.(ret.gaps, sigdigits = 4)

        partition_size_col = length(G_col)
        partition_size_row = length(G_row)
        cc_iters = ret.num_iters_ran

        @show (problem.γ, cost, sol_gaps, cc_iters, partition_size_col, partition_size_row)
    end

    return CoAssignmentResult(G_col, G_row), ret
end
