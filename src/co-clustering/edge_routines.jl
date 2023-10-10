# routines that use edges for runALM().


########## for computing the duality gap.

# step 2 of algorithm 1 in (Sun, JMLR 2021)
# mutates U.
function computeU!(reg::BMapBuffer, op_trait::MatrixOperationTrait, X::Matrix{T}, γ::T, λ::T, edge_set::EdgeSet) where T <: AbstractFloat

    # the update for U is prox_V. See text near quation 20 in (Sun, JMLR 2021).
    residual = reg.residual
    computeV!(reg.V, residual.BX, op_trait, X, reg.Z, λ, edge_set.edges) # update V, BX given X, Z.
    proximaltp!(residual.U, reg.V, edge_set.w, γ, λ)

    return nothing
end

function computeU!(reg::BMapBuffer, X::Matrix{T}, γ::T, λ::T, edge_set::EdgeSet) where T <: AbstractFloat
    return computeU!(reg, ColumnWise(), X, γ, λ, edge_set)
end

function computeU!(reg::CoBMapBuffer, X::Matrix{T}, γ::T, λ::T, edge_set::CoEdgeSet) where T <: AbstractFloat

    computeU!(reg.col, ColumnWise(), X, γ, λ, edge_set.col)
    computeU!(reg.row, RowWise(), X, γ, λ, edge_set.row)

    return nothing
end

# update Z in-place. See step 2 of Algorithm 1 in (Sun, JMLR 2011).
function updateZ!(reg::BMapBuffer{T}, σ::T) where T

    Z, BX, U = reg.Z, reg.residual.BX, reg.residual.U

    for i in eachindex(Z)
        Z[i] = Z[i] + σ*(BX[i]-U[i])
    end

    return nothing
end

function updateZ!(reg::CoBMapBuffer{T}, σ::T) where T

    updateZ!(reg.col, σ)
    updateZ!(reg.row, σ)

    return nothing
end

function computeKKTresiduals!(
    reg::BMapBuffer{T}, # mutates tmp buffers.
    X::Matrix{T},
    problem::ProblemType{T,EdgeSet{T}},
    norm_A_p_1::T,
    )::Tuple{T,T,T} where T

    A, γ, edge_set = unpackspecs(problem)

    p, d, norm_U = computeKKTresiduals!(reg, γ, edge_set, norm_A_p_1)
    pd = computepdgap(reg, X, A, γ, edge_set, norm_A_p_1, norm_U)

    return p, d, pd
end

function computeKKTresiduals!(
    reg::CoBMapBuffer{T}, # mutates tmp buffers.
    X::Matrix{T},
    problem::ProblemType{T,CoEdgeSet{T}},
    norm_A_p_1::T,
    )::Tuple{T,T,T} where T

    A, γ, edge_set = unpackspecs(problem)
    p_col, d_col, norm_U_col = computeKKTresiduals!(
        reg.col,
        γ,
        edge_set.col,
        norm_A_p_1,
    )
    p_row, d_row, norm_U_row = computeKKTresiduals!(
        reg.row,
        γ,
        edge_set.row,
        norm_A_p_1,
    )

    # # debug.
    # U, BX, prox_U_plus_Z, U_plus_Z, BadjZ_col = unpackbuffer(reg.col.residual)
    # U, BX, prox_U_plus_Z, U_plus_Z, BadjZ_row = unpackbuffer(reg.row.residual)
    # pd1_new = norm(BadjZ_col .+ BadjZ_row' .+ X .- A, 2)
    # #pd1_row = norm( .+ X' .- A', 2)
    # @show pd1_new
    # # end debug
    pd = computepdgap(reg, X, A, γ, edge_set, norm_A_p_1, norm_U_col + norm_U_row)

    return p_col + p_row, d_col + d_row, pd
end

# For a single regularizer.
function computeKKTresiduals!(
    reg::BMapBuffer{T}, # mutates tmp buffers.
    γ::T,
    edge_set::EdgeSet,
    norm_A_p_1::T,
    )::Tuple{T,T,T} where T
    
    Z = reg.Z
    U, BX, prox_U_plus_Z, U_plus_Z, BadjZ = unpackbuffer(reg.residual)
    w = edge_set.w

    #@assert size(A) == size(X)
    #@assert size(Z,1) == size(X,1)
    @assert size(Z,2) == length(w)

    ### KKT gaps. page 22 and section 5.1 in (Sun, JMLR 2021).
    norm_U::T = norm(U,2)

    ## primal.
    numerator_p = sqrt(evalnorm2sq(BX, U)) # norm(BX-U,2)
    primal_gap = numerator_p/(one(T) + norm_U)

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
        numerator_d += max( zero(T), norm_Z_l - γ*w[l] )
    end
    dual_gap = numerator_d/norm_A_p_1

    return primal_gap, dual_gap, norm_U
end


function computepdgap(
    reg::BMapBuffer{T}, # mutates tmp buffers.
    X::AbstractMatrix{T},
    A::AbstractMatrix{T},
    γ::T,
    edge_set::EdgeSet,
    norm_A_p_1::T,
    norm_U_factor::T,
    )::T where T

    U, BX, prox_U_plus_Z, U_plus_Z, BadjZ = unpackbuffer(reg.residual)
    w = edge_set.w
    Z = reg.Z

    ## primal-dual.

    # pd1, first term in numerator.
    #   (a+b-c)^2 = a^2 + 2ab - 2ac + b^2 - 2bc + c^2
    #applyJt!(BadjZ, Z, src_nodes, dest_nodes)
    applyJt!(BadjZ, Z, edge_set.edges)
    # a = BadjZ
    # b = X
    # c = A
    #norm_c = abs(norm_A_p_1 - one(T)) # abs() in case of numerical precision issues.
    #norm_a_plus_b_minus_c = dot(a,a) + 2*dot(a,b) - 2*dot(a,c) + dot(b,b) - 2*dot(b,c) + norm_c*norm_c
    #pd1::T = sqrt(norm_a_plus_b_minus_c)
    pd1 = norm(BadjZ .+ X .- A, 2)

    # pd2, second term in numberator.
    pd2 = computepd2gap(reg, γ, edge_set)

    primal_dual_gap = (pd1+pd2)/(norm_A_p_1 + norm_U_factor)

    return primal_dual_gap
end

function computepdgap(
    reg::CoBMapBuffer{T}, # mutates tmp buffers.
    X::AbstractMatrix{T},
    A::AbstractMatrix{T},
    γ::T,
    edge_set::CoEdgeSet,
    norm_A_p_1::T,
    norm_U_factor::T,
    )::T where T

    _, _, _, _, BadjZ_col = unpackbuffer(reg.col.residual)
    _, _, _, _, BadjZ_row = unpackbuffer(reg.row.residual)
    #w = edge_set.w

    ## primal-dual.

    # pd1, first term in numerator.
    #   (a+b-c)^2 = a^2 + 2ab - 2ac + b^2 - 2bc + c^2
    #applyJt!(BadjZ, Z, src_nodes, dest_nodes)
    applyJt!(BadjZ_col, reg.col.Z, edge_set.col.edges)
    applyJt!(BadjZ_row, reg.row.Z, edge_set.row.edges)

    pd1 = norm(BadjZ_col .+ BadjZ_row' .+ X .- A, 2)

    # pd2, second term in numberator.
    pd2_col = computepd2gap(reg.col, γ, edge_set.col)
    pd2_row = computepd2gap(reg.row, γ, edge_set.row)

    primal_dual_gap = (pd1 + pd2_col + pd2_row)/(norm_A_p_1 + norm_U_factor)

    return primal_dual_gap
end

# assumes U has been updated!
function computepd2gap(
    reg::BMapBuffer{T}, # mutates tmp buffers.
    γ::T,
    edge_set::EdgeSet,
    )::T where T

    U, BX, prox_U_plus_Z, U_plus_Z, BadjZ = unpackbuffer(reg.residual)
    w = edge_set.w
    Z = reg.Z

    ## primal-dual

    # pd2, second term in numberator.
    # overwrite the temp. buffer `U_plus_Z`.
    for i in eachindex(U_plus_Z)
        U_plus_Z[i] = U[i] + Z[i]
    end
    proximaltp!(prox_U_plus_Z, U_plus_Z, w, γ, one(T))
    pd2 = sqrt(evalnorm2sq(U, prox_U_plus_Z))

    return pd2
end

##########

function evalprimal(X::Matrix{T}, problem::ProblemType{T,EdgeSet{T}})::T where T <: AbstractFloat
    
    A, γ, E = unpackspecs(problem)

    return evalprimal(X, E.w, E.edges, γ, A)
end

# co-clustering
function evalprimal(X::Matrix{T}, problem::ProblemType{T,CoEdgeSet{T}})::T where T <: AbstractFloat

    A, γ, E = unpackspecs(problem)
    col_edges, row_edges = E.col.edges, E.row.edges
    col_w, row_w = E.col.w, E.row.w

    term1 = (dot(X,X) + dot(A,A) - 2*dot(X,A))/2 # norm(X-A,2)^2

    running_sum = zero(T)

    # columns of X are points.
    term2 = zero(T)
    for l in eachindex(col_edges)
        a, b = col_edges[l]

        # this is norm(X[:,a] - X[:,b]).
        running_sum = sum(
            (X[d,a]- X[d,b])^2 for d in axes(X,1)
        )
        term2 += col_w[l]*sqrt(running_sum)
    end

    # rows of X are points.
    term3 = zero(T)
    for l in eachindex(row_edges)
        a, b = row_edges[l]

        # this is norm(X[:,a] - X[:,b]).
        running_sum = sum(
            (X[a,n]- X[b,n])^2 for n in axes(X,2)
        )
        term3 += row_w[l]*sqrt(running_sum)
    end

    return term1 + γ*(term2 + term3)
end

