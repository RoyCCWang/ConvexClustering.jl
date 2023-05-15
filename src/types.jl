
### graph connectivity specification.

abstract type AbstractConnectivityType{T<:Real} end

struct KNNType{T<:Int} <: AbstractConnectivityType{T}
    knn::T
end

#
struct RadiusType{T<:AbstractFloat} <: AbstractConnectivityType{T}
    radius::T
end

#
abstract type AbstractSearchConnectivityType{T<:Real} <: AbstractConnectivityType{T} end

### graph creation for setting up the convex clustering problem.
struct WeightedGraphConfigType{CT<:AbstractConnectivityType}
    connectivity::CT
    metric::Distances.Metric
    kernelfunc::Function
end

mutable struct KNNSearchType <: AbstractSearchConnectivityType{Int}
    knn::Int # search result.
    const searchfunc::Function # (X::Matrix{T2<:AbstractFloat}, metric::Distances.Metric) ↦ neighbourhoods::Vector{Vector{int}}
end

# max_connected_components set to size(X,1) = N.
function KNNSearchType(knn::T;
    verbose::Bool = false,
    start_knn::T = 30,
    searchfunc = (xx,mm)->searchknn(start_knn, mm, xx, size(xx,1);
        verbose = verbose)) where T <: Int

    return KNNSearchType(knn, searchfunc)
end

function KNNSearchType(knn::T,
    max_connected_components::Int;
    verbose::Bool = false,
    start_knn::T = 30,
    searchfunc = (xx,mm)->searchknn(start_knn, mm, xx, max_connected_components;
        verbose = verbose)) where T <: Int

    return KNNSearchType(knn, searchfunc)
end

#
mutable struct RadiusSearchType{T<:AbstractFloat} <: AbstractSearchConnectivityType{T}
    radius::T # search result.
    const searchfunc::Function # (X::Matrix{T2<:AbstractFloat}, metric::Distances.Metric) ↦ neighbourhoods::Vector{Vector{int}}
end

# max_connected_components set to size(X,1) = N.
function RadiusSearchType(radius::T;
    verbose = false,
    max_iters::Int = 100,
    searchfunc = (xx,mm)->searchradius(mm, xx, size(xx,1);
        increment_amount = div(div(size(xx,2)*(size(xx,2)-1), 2),max_iters),
        verbose = verbose)) where T <: AbstractFloat

    return RadiusSearchType(radius, searchfunc)
end

function RadiusSearchType(radius::T,
    max_connected_components::Int;
    verbose = false,
    max_iters::Int = 100,
    searchfunc = (xx,mm)->searchradius(mm, xx, max_connected_components;
        increment_amount = div(div(size(xx,2)*(size(xx,2)-1), 2),max_iters),
        verbose = verbose)) where T <: AbstractFloat

    return RadiusSearchType(radius, searchfunc)
end


### assignment algorithm config.
struct AssignmentConfigType{T}
    metric::Distances.Metric
    assignment_zero_tol::T
end

### ALM optimization algorithm config type.
struct ALMConfigType{T}
    runoptimfunc::Function
    updateσfunc::Function
    updateϵfunc::Function
    max_iters::Int
    gap_tol::T
end

"""
```
ALMConfigType(gap_tol::T,
    runoptimfunc::Function;
    max_iters::Int = 200,
    updateσfunc::Function = nn::Int->convert(T, 0.4*1.05^nn),
    updateϵfunc::Function = nn::Int->convert(T, 1/(nn)^2)) where T <: AbstractFloat
```

Optional inputs:
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
"""
function ALMConfigType(gap_tol::T;
    #runoptimfunc::Function;
    verbose_subproblem::Bool = false,
    runoptimfunc::Function = (xx, ff, dff, gg_tol)->runOptimjl(xx, ff, dff, gg_tol; verbose = verbose_subproblem),
    max_iters::Int = 200,
    updateσfunc::Function = nn::Int->convert(T, 0.4*1.05^(nn-1)), # -1 because we start on iteration 1.
    updateϵfunc::Function = nn::Int->convert(T, 1/(nn-1)^2)) where T <: AbstractFloat

    return ALMConfigType(runoptimfunc, updateσfunc, updateϵfunc,
        max_iters, gap_tol)
end

### convex clustering problem config.
mutable struct ProblemType{T}
    A::Matrix{T}
    γ::T
    w::Vector{T}
    edge_pairs::Vector{Tuple{Int,Int}}
end


### trace diagonistics to the ALM.

"""
Summary
≡≡≡≡≡≡≡≡≡

struct TraceType{T}


Fields
≡≡≡≡≡≡≡≡

gaps            :: Vector{Vector{T}}
problem_cost    :: Vector{T}
diff_x          :: Vector{T}
diff_Z          :: Vector{T}

Structure fields, in order of definition:

- `gaps::Vector{Vector{T}}`: The residual gaps of each iteration.
    gaps[1] is the primal gap.
    gaps[2] is the dual gap.
    gaps[3] is the primal-dual gap.
    These are related to the KKT conditions of the optimization problem. See the discussion on `η_P`, `η_D`, and `η` in section 6 from (Sun, 2021):
    `Sun, D., Toh, K. C., & Yuan, Y. (2021). Convex Clustering: Model, Theoretical Guarantee and Efficient Algorithm. J. Mach. Learn. Res., 22(9), 1-32.`

- `problem_cost::Vector{T}`: The cost of the standard formulation of the convex clustering problem. See equation 2 from (Sun, 2021).

- `diff_x::Vector{T}`: The change in the solution iterates for the primal variable.

- `diff_Z::Vector{T}`: The change in the solution iterates for the dual variable.
"""
struct TraceType{T}
    gaps::Vector{Vector{T}}
    problem_cost::Vector{T}
    diff_x::Vector{T}
    diff_Z::Vector{T}
end

function TraceType(N::Int, val::T) where T <: AbstractFloat

    gaps = Vector{Vector{T}}(undef, N)
    poblem_cost = Vector{T}(undef, N)
    diff_x = Vector{T}(undef, N)
    diff_Z = Vector{T}(undef, N)

    return TraceType(gaps, poblem_cost, diff_x, diff_Z)
end

function resizetrace!(trace::TraceType{T}, iter::Int) where T <: AbstractFloat
    @assert length(trace.gaps) == length(trace.problem_cost)

    resize!(trace.gaps, iter)
    resize!(trace.problem_cost, iter)
    resize!(trace.diff_x, iter)
    resize!(trace.diff_Z, iter)

    return nothing
end

### solution data structure for the ALM algorithm.
"""
Summary
≡≡≡≡≡≡≡≡≡

struct ALMSolutionType{T}


Fields
≡≡≡≡≡≡≡≡

X_star          :: Matrix{T}
Z_star          :: Matrix{T}
num_iters_ran   :: Int
gaps            :: Vector{T}
trace           :: TraceType{T}

Structure fields, in order of definition:
- `X_star::Matrix{T}`: optimization solution for the primal variable. The `n`-th column of `X_star` is the partition center of the `n`-th column of `A`.
- `Z_star::Matrix{T}`: optimization solution for the dual variable.
- 'iter::Int`: the number of iterations ran in the outer optimization.
- `gaps::Vector{T}`:
    gaps[1] is the primal gap of the solution
- `trace`: diagnostic information. See `TraceType{T}`.
X_star, Z, iter, gaps, ϕ, dϕ!, trace
"""
struct ALMSolutionType{T}
    X_star::Matrix{T}
    Z_star::Matrix{T}
    num_iters_ran::Int
    gaps::Vector{T}
    trace::TraceType{T}
end


### search sequence regularizer γ, config.
struct SearchγConfigType
    max_iters::Int
    max_partition_size::Int
    getγfunc::Function # iteration_number::Int ↦ γ::T
end

#= """
```
makegeometricγconfig(γ_base::T,
    max_partition_size::Int;
    γ_rate::T = 1.05,
    max_iters::Int = 100)::SearchγConfigType where T
```
"""
function makegeometricγconfig(γ_base::T,
    max_partition_size::Int;
    γ_rate::T = 1.05,
    max_iters::Int = 100)::SearchγConfigType where T

    getγfunc = nn->evalgeometricsequence(nn, γ_base, γ_rate)

    return SearchγConfigType(max_iters, max_partition_size, getγfunc)
end =#

### search sequence for kernel parameter, config.
struct SearchθConfigType{T}
    max_iters::Int
    min_dynamic_range::T
    getθfunc::Function # # iteration_number::Int ↦ θ, whatever datatype θ is.
end

#= """
```
makeθlengthscaleconfig(length_scale_base::T,
    min_dynamic_range::T; # takes a positive, finite, non-zero real number. Must be in the range of the weight function.
    length_scale_rate::T = 1.05,
    max_iters::Int = 100)::SearchθConfigType{T} where T
```
"""
function makeθlengthscaleconfig(length_scale_base::T,
    min_dynamic_range::T; # takes a positive, finite, non-zero real number. Must be in the range of the weight function.
    length_scale_rate::T = 1.05,
    max_iters::Int = 10000)::SearchθConfigType{T} where T

    getθfunc = nn->lengthscale2θ(evalgeometricsequence(nn, length_scale_base, length_scale_rate))

    return SearchθConfigType(max_iters, min_dynamic_range, getθfunc)
end =#
