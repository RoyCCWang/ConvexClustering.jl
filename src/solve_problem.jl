"""
runconvexclustering(X0::Matrix{T},
    Z0::Matrix{T},
    problem::ProblemType{T, ET},
    optim_config::ALMConfigType{T},
    assignment_config::AssignmentConfigType{T};
    store_trace::Bool = false,
    report_cost = false) where {T <: AbstractFloat, ET <: EdgeFormulation}
"""
function runconvexclustering(X0::Matrix{T},
    Z0::Matrix{T},
    problem::ProblemType{T,ET},
    optim_config::ALMConfigType{T},
    assignment_config::AssignmentConfigType{T};
    store_trace::Bool = false,
    report_cost = false) where {T <: AbstractFloat, ET <: EdgeFormulation}

    ret = runALM(X0, Z0, problem, optim_config; store_trace = store_trace)

    G, g_neighbourhoods = assignviaX(ret.X_star, assignment_config.metric;
        zero_tol = assignment_config.assignment_zero_tol)

    if report_cost
        cost = primaldirect(ret.X_star,problem)
        cost = round.(cost, sigdigits = 4)

        sol_gaps = round.(ret.gaps, sigdigits = 4)

        partition_size = length(G)
        cc_iters = ret.num_iters_ran

        @show (problem.γ, cost, sol_gaps, cc_iters, partition_size)
    end

    return G, ret
end
