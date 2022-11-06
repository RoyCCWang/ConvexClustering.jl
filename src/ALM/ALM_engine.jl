


"""
```
runALM(X_initial::Matrix{T},
    Z_initial::Matrix{T},
    problem::ProblemType{T},
    config::ALMConfigType{T};
    store_trace::Bool = false)::ALMSolutionType{T} where T <: AbstractFloat
```

Inputs:

Optional inputs:

Outputs:



"""
function runALM(X_initial::Matrix{T},
    Z_initial::Matrix{T},
    problem::ProblemType{T},
    config::ALMConfigType{T};
    store_trace::Bool = false)::ALMSolutionType{T} where T <: AbstractFloat

    # unpack, checks
    A, γ, w, edge_pairs = problem.A, problem.γ, problem.w, problem.edge_pairs
    runoptimfunc, updateσfunc = config.runoptimfunc, config.updateσfunc
    updateϵfunc, max_iters, gap_tol = config.updateϵfunc, config.max_iters, config.gap_tol

    D, N = size(A)
    @assert size(X_initial) == size(A)

    N_edges = length(w)
    @assert length(edge_pairs) == N_edges

    # variables.
    Z::Matrix{T} = copy(Z_initial)
    x::Vector{T} = vec(copy(X_initial))

    # buffer setup.
    V_buf::Matrix{T} = Matrix{T}(undef, D, N_edges)
    prox_V_buf::Matrix{T} = Matrix{T}(undef, D, N_edges)
    proxconj_V_buf::Matrix{T} = Matrix{T}(undef, D, N_edges)
    grad_mat_buf::Matrix{T} = Matrix{T}(undef, size(A))

    # reuse buffers from ϕ or dϕ for primal, dual, primal_dual gap calculation.
    # this is possible because no ϕ and dϕ evaluations are occuring in the gap calculations.
    BadjZ_buf::Matrix{T} = similar(grad_mat_buf)
    U_buf::Matrix{T} = similar(V_buf)
    BX_buf::Matrix{T} = similar(V_buf)
    prox_U_plus_Z_buf::Matrix{T} = similar(V_buf)
    U_plus_Z_buf::Matrix{T} = similar(V_buf)

    # for dϕ.
    src_nodes::Vector{Int} = collect( edge[1] for edge in edge_pairs)
    dest_nodes::Vector{Int} = collect( edge[2] for edge in edge_pairs)
    #dϕ_eval::Vector{T} = zeros(T, N*D)

    # setup subproblem cost function and derivative.
    ϕ = xx->compteϕXoptim!(V_buf, prox_V_buf, proxconj_V_buf,
        reshape(xx,D,N), Z, σ, A, w, γ, edge_pairs)

    dϕ! = (gg,xx)->computedϕoptim!(gg,
        V_buf, prox_V_buf, proxconj_V_buf, grad_mat_buf,
        reshape(xx, D,N), Z, σ, A, w, γ, edge_pairs, src_nodes, dest_nodes)

    ## run ALM

    # pre-allocate, initialize.
    σ_base::T = updateσfunc(1)
    σ = σ_base


    # pre-allocate.
    gaps::Vector{T} = ones(T, 3) .* Inf
    x_star = Vector{T}(undef, length(x))
    Z_prev = copy(Z_initial)
    trace = TraceType(max_iters, one(T))

    # pre-compute constants.
    norm_A_p_1 = convert(T, norm(A,2) + 1)

    for iter = 1:max_iters

        # update σ
        σ = convert(T, updateσfunc(iter))
        #λ = one(T)/σ

        # subproblem gradient tolerance.
        ϵ::T = convert(T, updateϵfunc(iter))
        g_tol::T = ϵ/max(1,sqrt(σ))

        # update x, Z.
        x_star, gaps = solvesubproblem17!(x, Z,
            V_buf, U_buf, BX_buf, BadjZ_buf, prox_U_plus_Z_buf, U_plus_Z_buf,
            ϕ, dϕ!, σ, w, γ, edge_pairs, src_nodes, dest_nodes, A,
            g_tol, runoptimfunc;
            norm_A_p_1 = norm_A_p_1)

        # store trace diagnostics.
        if store_trace
            trace.gaps[iter] = gaps
            trace.problem_cost[iter] = primaldirect(reshape(x_star,size(A)), w, edge_pairs, γ, A)

            trace.diff_x[iter] = evalnorm2sq(x, x_star) #norm(x_prev-x_star,2)^2

            trace.diff_Z[iter] = evalnorm2sq(Z_prev, Z)
            Z_prev[:] = Z
        end

        # update.
        x = x_star

        # stopping criterion.
        if maximum(gaps) < gap_tol
            return ALMreturnroutine(reshape(x_star,D,N), Z, iter, gaps, trace; store_trace = store_trace)
        end
    end

    return ALMreturnroutine(reshape(x_star,D,N), Z, max_iters, gaps, trace; store_trace = store_trace)
end

function ALMreturnroutine(X_star::Matrix{T}, Z::Matrix{T}, iter::Int,
    gaps::Vector{T},
    trace::TraceType{T};
    store_trace = false) where T <: AbstractFloat

    resizetrace!(trace, iter)

    return ALMSolutionType(X_star, Z, iter, gaps, trace)
end

function solvesubproblem17!(x0::Vector{T}, Z::Matrix{T}, # optim variables.
    V::Matrix{T}, U::Matrix{T}, BX::Matrix{T}, BadjZ::Matrix{T}, # buffers.
    prox_U_plus_Z, U_plus_Z, #buffers
    ϕ::Function,
    dϕ!::Function,
    σ::T,
    w::Vector{T},
    γ::T,
    edge_pairs::Vector{Tuple{Int,Int}}, src_nodes::Vector{Int}, dest_nodes::Vector{Int},
    A::Matrix{T},
    g_tol::T,
    runoptimfunc::Function;
    norm_A_p_1::T = convert(T, 1+norm(A,2))) where T <: AbstractFloat

    D, N_edges = size(Z)

    @assert size(A) == size(BadjZ)
    @assert size(V) == (D,N_edges) == size(Z) == size(BX) == size(U)
    @assert N_edges == length(edge_pairs)

    # minimize ϕ(X).
    x_star::Vector{T}, norm_dϕ_x::T = runoptimfunc(x0, ϕ, dϕ!, g_tol)

    # update U and compute primal, dual, primal_dual gaps.
    primal_gap, dual_gap, primal_dual_gap = computeteALMKKTgaps!(reshape(x_star, size(A)),
        Z,
        U, V, BX, BadjZ, # buffers.
        prox_U_plus_Z, U_plus_Z, # buffers
        edge_pairs, w, γ, A,
        src_nodes, dest_nodes, σ;
        norm_A_p_1 = norm_A_p_1)

    gaps::Vector{T} = [primal_gap; dual_gap; primal_dual_gap]

    return x_star, gaps
end

function updateZ!(Z::Matrix{T}, # mutates output.
    U::Matrix{T}, BX::Matrix{T}, V::Matrix{T}, # buffers.
    X::Matrix{T}, #inputs.
    w::Vector{T}, γ::T, σ::T, edge_pairs::Vector{Tuple{Int,Int}}) where T <: AbstractFloat

    computeU!(U, BX, V,
        X, Z, w, γ, one(T)/σ, edge_pairs)

    # update Z in-place. See step 2 of Algorithm 1 in (Sun, JMLR 2011).
    for i in eachindex(Z)
        Z[i] = Z[i] + σ*(BX[i]-U[i])
    end

    return nothing
end

# mutates Z, U, V, BX
function computeteALMKKTgaps!(X::Matrix{T}, Z::Matrix{T},
    U::Matrix{T}, V::Matrix{T}, BX::Matrix{T}, BadjZ::Matrix{T}, # buffers.
    prox_U_plus_Z, U_plus_Z, #buffers
    edge_pairs::Vector{Tuple{Int,Int}}, w::Vector{T}, γ::T, A::Matrix{T},
    src_nodes::Vector{Int}, dest_nodes::Vector{Int}, σ::T;
    norm_A_p_1::T = convert(T, 1+norm(A,2)))::Tuple{T,T,T} where T <: AbstractFloat

    @assert norm_A_p_1 > one(T)
    @assert length(w) == length(edge_pairs)
    @assert size(A) == size(X) == size(BadjZ)
    @assert size(Z) == size(U) == size(BX) == size(V)

    ## updates/mutates Z, U, BX, V.
    updateZ!(Z, U, BX, V, X, w, γ, σ, edge_pairs)

    ### KKT gaps. page 22 and section 5.1 in (Sun, JMLR 2021).
    norm_U::T = norm(U,2)

    ## primal.
    numerator_p = sqrt(evalnorm2sq(BX,U)) # norm(BX-U,2)
    primal_gap = numerator_p/(one(T)+norm_U)

    ## dual.
    numerator_d = zero(T)
    norm_Z_l = zero(T)
    for l in eachindex(w)

        # allocation-less version of norm(Z[:,l],2)
        norm_Z_l = zero(T)
        for d in axes(Z,1)
            norm_Z_l += Z[d,l]*Z[d,l]
        end
        norm_Z_l = sqrt(norm_Z_l)

        # η_D on page 22 in (Sun, JMLR 2021).
        # KKT condition V - Badj(Z) = 0, V is the input to the norm portion
        #   of the primal problem.
        numerator_d += max(zero(T), norm_Z_l-γ*w[l])
    end
    dual_gap = numerator_d/norm_A_p_1

    ## primal-dual.

    # pd1, first term in numerator.
    #   (a+b-c)^2 = a^2 + 2ab - 2ac + b^2 - 2bc + c^2
    applyJt!(BadjZ, Z, src_nodes, dest_nodes)
    a = BadjZ
    b = X
    c = A
    norm_c = abs(norm_A_p_1 - one(T)) # abs() in case of numerical precision issues.
    #norm_a_plus_b_minus_c = dot(a,a) + 2*dot(a,b) - 2*dot(a,c) + dot(b,b) - 2*dot(b,c) + norm_c*norm_c
    #pd1::T = sqrt(norm_a_plus_b_minus_c)
    pd1 = norm(BadjZ .+ X .- A, 2)

    # pd2, second term in numberator.
    for i in eachindex(U_plus_Z)
        U_plus_Z[i] = U[i] + Z[i]
    end
    proximaltp!(prox_U_plus_Z, U_plus_Z, w, γ, one(T))
    pd2::T = sqrt(evalnorm2sq(U, prox_U_plus_Z))

    primal_dual_gap::T = (pd1+pd2)/(norm_A_p_1+norm_U)

    return primal_gap, dual_gap, primal_dual_gap
end

################### for ϕ, dϕ! function preparation for Optim. Do simultaneous versions later.
function preparefuncs(X::Matrix{T}, Z::Matrix{T}, σ, A, w, γ, edge_pairs) where T <: AbstractFloat

    V_buf::Matrix{T} = Matrix{T}(undef, size(Z))
    prox_V_buf::Matrix{T} = Matrix{T}(undef, size(Z))
    proxconj_V_buf::Matrix{T} = Matrix{T}(undef, size(Z))

    D, N = size(X)

    # function under test.
    h = xx->compteϕXoptim!(V_buf, prox_V_buf, proxconj_V_buf,
        reshape(xx,D,N), Z, σ, A, w, γ, edge_pairs)

    # oracles.
    f = xx->evalϕX(reshape(xx,D,N), Z, w, edge_pairs, γ, σ, A)
    f2 = xx->evalϕX!(prox_V_buf, proxconj_V_buf, V_buf,
        reshape(xx,D,N), Z, w, edge_pairs, γ, σ, A)

    return h, f, f2
end

function preparedfuncs(X::Matrix{T}, Z::Matrix{T}, σ, A, w, γ, edge_pairs) where T <: AbstractFloat

    V_buf::Matrix{T} = Matrix{T}(undef, size(Z))
    prox_V_buf::Matrix{T} = Matrix{T}(undef, size(Z))
    proxconj_V_buf::Matrix{T} = Matrix{T}(undef, size(Z))
    grad_mat_buf::Matrix{T} = Matrix{T}(undef, size(X))

    #dh_eval = zeros(length(X))

    D, N = size(X)

    src_nodes = collect( edge[1] for edge in edge_pairs)
    dest_nodes = collect( edge[2] for edge in edge_pairs)

    # function under test.
    dh = (gg,xx)->computedϕoptim!(gg,
        V_buf, prox_V_buf, proxconj_V_buf, grad_mat_buf,
        reshape(xx, D,N), Z, σ, A, w, γ, edge_pairs, src_nodes, dest_nodes)


    # oracles.
    df = xx->computedϕ(reshape(xx,D,N), Z, J, w, A,
        γ, σ, edge_pairs)

    return dh, df#, dh_eval
end
