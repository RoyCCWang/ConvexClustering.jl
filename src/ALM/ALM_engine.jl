


"""
```
function runALM(
    X_initial::Matrix{T},
    Z_initial::Matrix{T},
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
    Z_initial::Matrix{T},
    problem::ProblemType{T,ET},
    config::ALMConfigType{T};
    store_trace::Bool = false,
    )::ALMSolutionType{T} where {T <: AbstractFloat, ET <: EdgeFormulation}

    # unpack, checks
    A, γ, edge_set = unpackspecs(problem)
    
    runoptimfunc, updateσfunc = config.runoptimfunc, config.updateσfunc
    updateϵfunc, max_iters, gap_tol = config.updateϵfunc, config.max_iters, config.gap_tol

    D, N = size(A)
    @assert size(X_initial) == size(A)

    #N_edges = getNedges(edge_set)
    #@assert length(edge_pairs) == N_edges

    # # variables.
    reg = getZbuffer(traitof(edge_set), Z_initial, N)

    #Z::Matrix{T} = copy(Z_initial)
    x::Vector{T} = vec(copy(X_initial))
    grad_mat_buf::Matrix{T} = Matrix{T}(undef, size(A))

    # # buffer setup. taken care of by `reg`.
    # V_buf::Matrix{T} = Matrix{T}(undef, D, N_edges)
    # prox_V_buf::Matrix{T} = Matrix{T}(undef, D, N_edges)
    # ##proxconj_V_buf::Matrix{T} = Matrix{T}(undef, D, N_edges)
    # # reuse buffers from ϕ or dϕ for primal, dual, primal_dual gap calculation.
    # # this is possible because no ϕ and dϕ evaluations are occuring in the gap calculations.
    # BadjZ_buf::Matrix{T} = similar(grad_mat_buf)
    # U_buf::Matrix{T} = similar(V_buf)
    # BX_buf::Matrix{T} = similar(V_buf)
    # prox_U_plus_Z_buf::Matrix{T} = similar(V_buf)
    # U_plus_Z_buf::Matrix{T} = similar(V_buf)


    # then incorporate the use of SubproblemRegVars if time permits.
    # reg = RegularizationVars(T, ColumnFormulation(), edge_pairs, D, N)
    # x::Vector{T} = vec(copy(X_initial))
    # grad_mat_buf::Matrix{T} = Matrix{T}(undef, size(A))

    # pre-allocate, initialize.
    σ_base::T = updateσfunc(1)
    σ_buffer::Vector{T} = [σ_base;]
    
    # setup subproblem cost function and derivative.
    ϕ = xx->compteϕXoptim!(
        #V_buf,
        #prox_V_buf,
        ##proxconj_V_buf,
        reg,
        reshape(xx,D,N),
        #Z,
        σ_buffer,
        #A,
        #w,
        #γ,
        #edge_set,
        problem,
    )

    dϕ! = (gg,xx)->computedϕoptim!(
        gg,
        #V_buf,
        #prox_V_buf,
        ##proxconj_V_buf,
        reg,
        grad_mat_buf,
        reshape(xx, D,N),
        #Z,
        σ_buffer,
        #A,
        #w,
        #γ,
        #edge_set,
        problem,
    )

    # # Run ALM

    # pre-allocate.
    gaps::Vector{T} = ones(T, 3) .* Inf
    x_star = Vector{T}(undef, length(x))
    #Z_prev = copy(Z_initial)
    reg_prev = getZbuffer(traitof(edge_set), Z_initial, N)
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
            x, #Z,
            #V_buf, U_buf, BX_buf, BadjZ_buf, prox_U_plus_Z_buf, U_plus_Z_buf,
            reg,
            ϕ, dϕ!,
            σ,
            #w,
            #γ,
            #edge_set,
            #A,
            problem,
            g_tol,
            runoptimfunc,
            norm_A_p_1,
        )

        # store trace diagnostics. 
        if store_trace
            trace.gaps[iter] = gaps
            trace.problem_cost[iter] = primaldirect(
                reshape(x_star,size(A)),
                problem,
            )
            trace.diff_x[iter] = evalnorm2sq(x, x_star) #norm(x_prev-x_star,2)^2

            # trace.diff_Z[iter] = evalnorm2sq(Z_prev, Z)
            # Z_prev[:] = Z
            storetrace!(trace, reg, reg_prev, iter)
        end

        # update.
        x = x_star

        # stopping criterion.
        if maximum(gaps) < gap_tol
            return ALMreturnroutine(
                reshape(x_star, D, N),
                reg,
                iter,
                gaps,
                trace,
            )
        end
    end

    return ALMreturnroutine(reshape(x_star, D, N), reg, max_iters, gaps, trace)
end


function ALMreturnroutine(
    X_star::Matrix{T},
    reg::BMapBuffer{T}, #Z::Matrix{T},
    iter::Int,
    gaps::Vector{T},
    trace::TraceType{T},
    ) where T <: AbstractFloat

    return ALMSolutionType(
        X_star, collect( reg.Z for _ = 1:1 ), iter, gaps, trace,
    )
end

function ALMreturnroutine(
    X_star::Matrix{T},
    reg::CoBMapBuffer{T}, #Z::Matrix{T},
    iter::Int,
    gaps::Vector{T},
    trace::TraceType{T},
    ) where T <: AbstractFloat

    Z_star = Vector{Matrix{T}}(undef, 2)
    Z_star[begin] = reg.col.Z
    Z_star[begin+1] = reg.row.Z

    return ALMSolutionType(
        X_star, Z_star, iter, gaps, trace,
    )
end

############

function solvesubproblem17!(
    x0::Vector{T}, # an optim variable.
    #Z::Matrix{T}, # fixed for this subproblem.
    #V::Matrix{T}, U::Matrix{T}, BX::Matrix{T}, BadjZ::Matrix{T}, # buffers.
    #prox_U_plus_Z, U_plus_Z, #buffers
    reg::RegularizationBuffer,
    ϕ::Function,
    dϕ!::Function,
    σ::T,
    #w::Vector{T},
    #γ::T,
    #edge_pairs::Vector{Tuple{Int,Int}},
    #edge_set::EdgeFormulation,
    #A::Matrix{T},
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
        #Z,
        #U, V, BX, BadjZ, # buffers.
        #prox_U_plus_Z, U_plus_Z, # buffers
        #edge_pairs,
        reg,
        #edge_set,
        #w,
        problem, #γ, A,
        σ,
        norm_A_p_1,
    )

    gaps::Vector{T} = [primal_gap; dual_gap; primal_dual_gap]

    return x_star, gaps
end


# mutates Z, U, V, BX
function computeteALMKKTgaps!(
    X::Matrix{T},
    # Z::Matrix{T},
    # U::Matrix{T},
    # V::Matrix{T},
    # BX::Matrix{T},
    # BadjZ::Matrix{T}, # buffers.
    # prox_U_plus_Z, U_plus_Z, #buffers
    reg::BMapBuffer{T},
    #edge_set::EdgeSet,
    problem::ProblemType{T,EdgeSet{T}},
    # src_nodes::Vector{Int}, dest_nodes::Vector{Int},
    σ::T,
    norm_A_p_1::T,
    )::Tuple{T,T,T} where T <: AbstractFloat

    Z, V = reg.Z, reg.V
    U, BX, _, _, BadjZ = unpackbuffer(reg.residual)

    # @assert norm_A_p_1 > one(T)
    @assert size(X) == size(BadjZ)
    @assert size(Z) == size(U) == size(BX) == size(V)

    ## updates/mutates Z, U, BX, V.
    #updateZ!(reg, X, γ, σ, edge_set)
    computeU!(reg, X, problem.γ, one(T)/σ, problem.edge_set)
    updateZ!(reg, σ)

    return computeKKTresiduals!(reg, X, problem, norm_A_p_1)
end
