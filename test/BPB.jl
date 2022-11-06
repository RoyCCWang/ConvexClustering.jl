# randomize the Linear system evaluation H(X), left-hand side of eqn 23 of Sun et. al's JMLR 2021 paper.
function runtrialsBadjPB(D_mat::Matrix{T},
    W::Vector{T}, edge_pairs::Vector{Tuple{Int,Int}}, J,
    γ::T, σ::T,
    N_linear_sys_eval_trials::Int) where T

    f = XX->ConvexClustering.computeBadjPB(XX, D_mat, W, edge_pairs, J, γ, σ)

    N, N_edges = size(J)
    D = size(D_mat,1)

    discrepancies = zeros(T, N_linear_sys_eval_trials)
    fill!(discrepancies, Inf)

    for n = 1:N_linear_sys_eval_trials
        X1 = randn(D, N)
        X2 = randn(D, N)
        a1 = randn()
        a2 = randn()
        Y_sys_inputs = f(a1 .* X1+ a2 .* X2)
        Y_sys_outputs = a1 .* f(X1) .+ a2 .* f(X2)

        #println("linearity test. Should be zero")
        #@show norm(sum(Y_sys_inputs)-sum(Y_sys_outputs))
        discrepancies[n] = norm(sum(Y_sys_inputs)-sum(Y_sys_outputs))
    end

    return maximum(discrepancies)
end

# randomize the Hessian evaluation location (X,Z).
function runtrialsBadjPB(W::Vector{T}, edge_pairs::Vector{Tuple{Int,Int}}, J, γ::T, σ::T,
    N_linear_sys_eval_trials::Int, N_derivative_eval_trials::Int, D::Int) where T

    N, N_edges = size(J)

    discrepancies = zeros(T, N_derivative_eval_trials)
    fill!(discrepancies, Inf)

    for n = 1:N_derivative_eval_trials
        Z = randn(D, N_edges)
        X = randn(D, N)

        D_mat = ConvexClustering.computeD(X, J, σ, Z)
        discrepancies[n] = runtrialsBadjPB(D_mat,
            W, edge_pairs, J, γ, σ, N_linear_sys_eval_trials)
    end

    return maximum(discrepancies)
end

# randomize the data to be clustered, for a fixed dimension D and number of data instances N.
# clustering radius affects the graph of connections and weights W.
function runtrialsBadjPB(D::Int, N::Int,
    N_linear_sys_eval_trials::Int,
    N_derivative_eval_trials::Int,
    N_data_trials::Int,
    γ::T, σ::T;
    clustering_radius = 1.5,
    wfunc = (xx,zz)->exp(-norm(xx-zz)^2)) where T <: AbstractFloat

    discrepancies = zeros(T, N_data_trials)
    fill!(discrepancies, Inf)

    for n = 1:N_data_trials
        A = randn(T, D, N)

        discrepancies[n] = runtrialsBadjPB(W, edge_pairs, J, γ, σ, N_linear_sys_eval_trials, N_derivative_eval_trials, D)
    end

    return maximum(discrepancies)
end
