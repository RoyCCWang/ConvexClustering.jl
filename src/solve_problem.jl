"""
```
runconvexclustering(X0::Matrix{T},
    Z0::Matrix{T},
    problem::ProblemType{T},
    optim_config::ALMConfigType{T},
    assignment_config::AssignmentConfigType{T};
    store_trace::Bool = false,
    report_cost = false) where T <: AbstractFloat
```
Nomenclature:
- `D` is the dimension of a point in the set to be partitioned.
- `N` is the number of points in the set.
- `N_edges` is the number of edges in the neighbourhood graph of `A` and some external weight function.
    This graph has `N` nodes, which are numbered from `1` to `N`.
    There are `N_edge` number of edges, with weights given by `w`.
- `runconvexclustering` calls a subroutine that does an outer optimization (augmented Lagragian method), and a once-differentiable unconstrained convex subproblem. The current impelemntation allows the user to to specify how to solve the subproblem via the `runoptimfunc` input.
- A partition `P` of a finite set of points, `S`, is a set of disjoint subsets of `S` where their collectively union must recover `S`. A subset in `P` is called a part of the partition `P`.
- The size of a partition is the number of subsets in the partition. If a partition is interpreted as a clustering assignment, then the size of the partition is the number of distinct clusters.

Inputs:
- `A::Matrix{T}`: `D x N` matrix containing the points to be partitioned. One should consider rescaling the values of `A` if it is ill-conditioned or have very large magnitude of values relative to the floating-point precision of datatype `T`.
- `edge_pairs::Vector{Tuple{Int,Int}}`: `edge_pairs[k]` contains the node indices of the `k`-th edge.
- `w::Vector{T}`: non-negative weights associated with the graph.
- `X0::Matrix{T}`: `X0[:,n]` is the initial guess for the partition center for point `n` (i.e. the point in `A[:,n]`).
- `Z0::Matrix{T}`: initial guess of the dual variable.
- `γ`: the regularization parameter.
- `metric::Distances.Metric`: a distance datatype from Distances.jl. An example is Distances.Euclidean().
    Used to do partition assignment.
- `runoptimfunc::Function`: a function that serves as the numerical optimizer for the solving the subproblem.
    See the example scripts on the ConvexClustering.jl repository webpage for an example, in particular, /examples/helpers/optim.jl
    
    For example, suppose we have an user-supplied optimization solve function `runoptimlib` that has the function definition:
    ```
    runexternalsolver(
        x_initial::Vector{T},           # input slot: initial guess.
        f::Function,                    # input slot: objective function.
        df!::Function,                  # input slot: in-place gradident.
        g_tol::T;                       # input slot: termination condition based on gradient norm.
        x_tol::T = zero(T),             # tunining slot: early-termination condition on optimization variable x.
        f_tol::T = zero(T),             # tuning slot: early-termination condition on objective function value.
        max_time::T = convert(T, Inf),  # tuning slot: early-termination condition on time.
        max_iters::Int = 100000,        # tuning slot: early-termination condition on maximum iterations.
        lp::Int = 2,                    # tuning slot: the type of l-p norm used to measure gradient norm for comparson with `g_tol`.
        verbose::Bool = false,          # tuning slot: verbose mode for enabling diagonstics of the external optimizer in this function, if any.
        ) where T <: AbstractFloat
    ```
    
    The `runoptimfunc` would then be declared as the following.
    ```
    runoptimfunc = (xx,ff,dff,gg_tol)->runexternalsolver(xx, ff, dff, gg_tol; verbose = true)
    ```
    
    The subproblem is equation 17 from `Sun, D., Toh, K. C., & Yuan, Y. (2021). Convex Clustering: Model, Theoretical Guarantee and Efficient Algorithm. J. Mach. Learn. Res., 22(9), 1-32.`

Optional inputs:
- `assignment_zero_tol::T`: The proximity tolerance with respect to `metric` for deciding whether two partition centers should be considered the same one.
    See `assignviaX()` for information on the assignment algorithm implemented.
- `updateσfunc::Function`: `updateσfunc(iter)` returns `σ_k`, the penalty parameter for the `k`-th outer optimization. Should return an increasing finite non-negative number.
- `updateϵfunc::Function`: `updateϵfunc(iter)` returns `ϵ_k`, the allowed discrepancy parameter for solving the subproblem associated with the `k`-th iteration for the outer optimization. Should be a summable non-negative sequence.
- `gap_tol::T`: stopping condition for the optimization. Should be a small number.
- `cc_max_iters::Int`: the maximum number of iterations to run the convex clustering algorithm. One should adjust `updateσfunc` and `cc_max_iters` such that `updateσfunc(cc_max_iters)` should not exceed a desired magnitude for the floating-point precision of type `T`. This is to avoid desired numerical stability issues.
- `store_trace::Bool`: Store diagnostic information of the optimization. See `runALM()`.
- `report_cost::Bool`: displays the convex clustering optimization cost of the solution to the REPL at the end of the optimization. This is not the same as entries in the output `gaps`.

Outputs:
- `G::Vector{Vector{Int}}`: contains the indices of the assigned partition. G[k] contains the indices for points that belong in the `k`-th part of the partition.
- See `runALM()` for information on `X_star`, `Z_star`, `num_iters_ran`, `gaps`, `trace`.
"""
function runconvexclustering(X0::Matrix{T},
    Z0::Matrix{T},
    problem::ProblemType{T},
    optim_config::ALMConfigType{T},
    assignment_config::AssignmentConfigType{T};
    store_trace::Bool = false,
    report_cost = false) where T <: AbstractFloat

    A, γ, w, edge_pairs = problem.A, problem.γ, problem.w, problem.edge_pairs


    ret = runALM(X0, Z0, problem, optim_config; store_trace = store_trace)

    #
    G, g_neighbourhoods = assignviaX(ret.X_star, assignment_config.metric;
        zero_tol = assignment_config.assignment_zero_tol)

    if report_cost
        cost = primaldirect(ret.X_star, w, edge_pairs, γ, A)
        cost = round.(cost, sigdigits = 4)

        sol_gaps = round.(ret.gaps, sigdigits = 4)

        partition_size = length(G)
        cc_iters = ret.num_iters_ran

        @show (γ, cost, sol_gaps, cc_iters, partition_size)
    end

    return G, ret
end
