
function runALM(
    X_initial::Matrix{T},
    Z_initial::Matrix{T},
    problem::ProblemType{T,ET},
    config::ALMConfigType{T};
    store_trace::Bool = false,
    verbose = false,
    ) where {T,ET}

    return runALM(X_initial, ALMDualVar(Z_initial), problem, config; store_trace = store_trace, verbose = verbose)
end

"""
```
function runALM(
    X_initial::Matrix{T},
    dual_initial::AuxiliaryVariable,
    problem::ProblemType{T,ET},
    config::ALMConfigType{T};
    store_trace::Bool = false,
    )::ALMSolutionType{T} where {T <: AbstractFloat, ET <: EdgeFormulation}

```

Inputs:

Optional inputs:

Outputs:
"""
function runALM(
    X_initial::Matrix{T},
    #Z_initial::Matrix{T},
    dual_initial::AuxiliaryVariable,
    problem::ProblemType{T,ET},
    config::ALMConfigType{T};
    store_trace::Bool = false,
    verbose = false,
    ) where {T <: AbstractFloat, ET <: EdgeFormulation}

    # unpack, parse, checks
    A, γ, edge_set = unpackspecs(problem)
    
    runoptimfunc, updateσfunc = config.runoptimfunc, config.updateσfunc
    updateϵfunc, max_iters, gap_tol = config.updateϵfunc, config.max_iters, config.gap_tol

    D, N = size(A)
    @assert size(X_initial) == size(A)

    # # variables.
    reg = getZbuffer(dual_initial, problem)

    # debug
    # @show size(reg.col.V), size(reg.row.V)
    # @show size(reg.col.Z), size(reg.row.Z)
    # @show size(dual_initial.col.Z), size(dual_initial.row.Z)    
    # @assert 1==2
    # end debug

    #Z::Matrix{T} = copy(Z_initial)
    x::Vector{T} = vec(copy(X_initial))
    grad_mat_buf::Matrix{T} = Matrix{T}(undef, size(A))

    # pre-allocate, initialize.
    σ_base::T = updateσfunc(1)
    σ_buffer::Vector{T} = [σ_base;]
    
    # setup subproblem cost function and derivative.
    ϕ = xx->computeϕXoptim!(
        reg,
        reshape(xx,D,N),
        σ_buffer,
        problem,
    )

    dϕ! = (gg,xx)->computedϕoptim!(
        gg,
        reg,
        grad_mat_buf,
        reshape(xx, D,N),
        σ_buffer,
        problem,
    )

    # # Run ALM

    # pre-allocate.
    gaps::Vector{T} = ones(T, 3) .* Inf
    x_star = Vector{T}(undef, length(x))
    #Z_prev = copy(Z_initial)
    reg_prev = getZbuffer(dual_initial, problem)
    trace = TraceType(traitof(edge_set), T, 0)
    if store_trace
        resizetrace!(trace, traitof(edge_set), max_iters)
    end

    # pre-compute constants.
    norm_A_p_1 = convert(T, norm(A,2) + 1)

    # run ALM.
    for iter = 1:max_iters

        # update σ
        σ = convert(T, updateσfunc(iter))
        σ_buffer[begin] = σ
        #λ = one(T)/σ

        # subproblem gradient tolerance.
        ϵ::T = convert(T, updateϵfunc(iter))
        g_tol::T = ϵ/max(1,sqrt(σ))

        # update x, Z.
        x_star, gaps = solvesubproblem17!(
            x,
            reg,
            ϕ, dϕ!,
            σ,
            problem,
            g_tol,
            runoptimfunc,
            norm_A_p_1,
        )

        # store trace diagnostics. 
        if store_trace
            trace.gaps[iter] = gaps
            trace.problem_cost[iter] = evalprimal(
                reshape(x_star,size(A)),
                problem,
            )
            trace.diff_x[iter] = evalnorm2sq(x, x_star) #norm(x_prev-x_star,2)^2

            # trace.diff_Z[iter] = evalnorm2sq(Z_prev, Z)
            # Z_prev[:] = Z
            storetrace!(trace, reg, reg_prev, iter)
        end

        if verbose
            @show iter, gaps
        end

        # update.
        x = x_star

        # stopping criterion.
        if maximum(gaps) < gap_tol

            resizetrace!(trace, traitof(edge_set), iter)

            return ALMSolutionType(
                reshape(x_star, D, N),
                reg,
                iter,
                gaps,
                trace,
            )
        end
    end

    return ALMSolutionType(reshape(x_star, D, N), reg, max_iters, gaps, trace)
end

function ALMSolutionType(
    X_star::Matrix{T},
    reg::BMapBuffer{T}, #Z::Matrix{T},
    iter::Int,
    gaps::Vector{T},
    trace::TraceType{T},
    )::ALMSolutionType{T,ALMDualVar{T}} where T

    return ALMSolutionType(X_star, ALMDualVar(reg.Z), iter, gaps, trace)
end

function ALMSolutionType(
    X_star::Matrix{T},
    reg::CoBMapBuffer{T}, #Z::Matrix{T},
    iter::Int,
    gaps::Vector{T},
    trace::TraceType{T},
    )::ALMSolutionType{T,ALMCoDualVar{T}} where T

    return ALMSolutionType(
        X_star,
        ALMCoDualVar(
            ALMDualVar(reg.col.Z),
            ALMDualVar(reg.row.Z),
        ),
        iter, gaps, trace,
    )
end

############

function solvesubproblem17!(
    x0::Vector{T}, # an optim variable.
    reg::RegularizationBuffer,
    ϕ::Function,
    dϕ!::Function,
    σ::T,
    problem::ProblemType,
    g_tol::T,
    runoptimfunc::Function,
    norm_A_p_1::T, # a pre-computed constant.
    ) where T <: AbstractFloat

    A = problem.A

    # D, N_edges = size(Z)
    # @assert size(A) == size(BadjZ)
    # @assert size(V) == (D,N_edges) == size(Z) == size(BX) == size(U)
    # @assert N_edges == getNedges(info) #length(edge_pairs)

    # minimize ϕ(X).
    x_star::Vector{T}, norm_dϕ_x::T = runoptimfunc(x0, ϕ, dϕ!, g_tol)

    # update U and compute primal, dual, primal_dual gaps.
    primal_gap, dual_gap, primal_dual_gap = computeteALMKKTgaps!(
        reshape(x_star, size(A)),
        reg,
        problem,
        σ,
        norm_A_p_1,
    )

    gaps::Vector{T} = [primal_gap; dual_gap; primal_dual_gap]

    return x_star, gaps
end


# mutates Z, U, V, BX
function computeteALMKKTgaps!(
    X::Matrix{T},
    reg::RegularizationBuffer,
    problem::ProblemType,
    σ::T,
    norm_A_p_1::T,
    )::Tuple{T,T,T} where T <: AbstractFloat

    # Z, V = reg.Z, reg.V
    # U, BX, _, _, BadjZ = unpackbuffer(reg.residual)

    # @assert norm_A_p_1 > one(T)
    # @assert size(X) == size(BadjZ)
    # @assert size(Z) == size(U) == size(BX) == size(V)

    ## updates/mutates Z, U, BX, V.
    #updateZ!(reg, X, γ, σ, edge_set)
    computeU!(reg, X, problem.γ, one(T)/σ, problem.edge_set)
    updateZ!(reg, σ)

    return computeKKTresiduals!(reg, X, problem, norm_A_p_1)
end
