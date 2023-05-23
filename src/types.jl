


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

abstract type AbstractAssignmentConfigType end

struct AssignmentConfigType{T} <: AbstractAssignmentConfigType
    metric::Distances.Metric
    zero_tol::T
end

struct CoAssignmentConfigType{T} <: AbstractAssignmentConfigType
    col::AssignmentConfigType{T}
    row::AssignmentConfigType{T}
end


### problem hyperparameters

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



#### for use with subroutines called by runALM().

abstract type MatrixOperationTrait end
struct ColumnWise <: MatrixOperationTrait end
struct RowWise <: MatrixOperationTrait end

abstract type OperationTrait end
struct Conventional <: OperationTrait end
struct CoClustering <: OperationTrait end

abstract type EdgeFormulation end

struct EdgeSet{T} <: EdgeFormulation
    w::Vector{T}
    edges::Vector{Tuple{Int,Int}}
end

function getNedges(A::EdgeSet)::Int
    return length(A.edges)
end

function getw(A::EdgeSet{T})::Vector{T} where T
    return A.w
end

function getedges(A::EdgeSet)::Vector{Tuple{Int,Int}}
    return A.edges
end

struct CoEdgeSet{T} <: EdgeFormulation
    col::EdgeSet{T}
    row::EdgeSet{T}
end

function getNedges(A::CoEdgeSet)::Int
    return getNedges(A.col) + getNedges(A.row)
end

function getw(A::CoEdgeSet{T}, ::RowWise)::Vector{T} where T
    return A.row.w
end

function getw(A::CoEdgeSet{T}, ::ColumnWise)::Vector{T} where T
    return A.col.w
end

function getedges(A::CoEdgeSet, ::RowWise)::Vector{Tuple{Int,Int}}
    return A.row.edges
end

function getedges(A::CoEdgeSet, ::ColumnWise)::Vector{Tuple{Int,Int}}
    return A.col.edges
end

# traits.
function traitof(::EdgeSet)
    return Conventional()
end

function traitof(::CoEdgeSet)
    return CoClustering()
end

### dual variable for ALM.

# traits.
function getdualtype(::EdgeSet{T}) where T
    return ALMDualVar{T}
end

function getdualtype(::CoEdgeSet{T}) where T
    return ALMCoDualVar{T}
end

# the different variables for each operation option.
abstract type AuxiliaryVariable end

struct ALMDualVar{T} <: AuxiliaryVariable
    Z::Matrix{T}
end

struct ALMCoDualVar{T} <: AuxiliaryVariable
    col::ALMDualVar{T}
    row::ALMDualVar{T}
end


### convex clustering problem config. User-facing.
struct ProblemType{T, ET <: EdgeFormulation} # basic partition problem: no co-clustering.
    A::Matrix{T}
    γ::T
    edge_set::ET # diagnose why this is so slow.
end

# utilities.
function copywithγ(p::ProblemType{T,ET}, γ::T)::ProblemType{T,ET} where {T,ET}
    return ProblemType(p.A, γ, p.edge_set)
end

function unpackspecs(p::ProblemType{T,ET})::Tuple{Matrix{T},T,ET} where {T,ET}
    return p.A, p.γ, p.edge_set
end

# conventional clustering.
function ProblemType(
    A::Matrix{T},
    γ::T,
    w::Vector{T},
    edges::Vector{Tuple{Int,Int}},
    )::ProblemType{T,EdgeSet{T}} where T

    return ProblemType(A, γ, EdgeSet(w, edges))
end

# co-clustering.
function ProblemType(
    A::Matrix{T},
    γ::T,
    w_col::Vector{T},
    w_row::Vector{T},
    col_edges::Vector{Tuple{Int,Int}},
    row_edges::Vector{Tuple{Int,Int}},
    )::ProblemType{T,CoEdgeSet{T}} where T

    return ProblemType(
        A,
        γ,
        CoEdgeSet(
            EdgeSet(w_col, col_edges),
            EdgeSet(w_row, row_edges),
        ),
    )
end


### duality gap-related buffers.
struct DualityGapBuffer{T}
    # reuse buffers from ϕ or dϕ for primal, dual, primal_dual gap calculation.
    # this is possible because no ϕ and dϕ evaluations are occuring in the gap calculations.
    U::Matrix{T} # D by N_edges
    BX::Matrix{T} # D by N_edges
    prox_U_plus_Z::Matrix{T} # D by N_edges
    U_plus_Z::Matrix{T} # D by N_edges

    BadjZ::Matrix{T} # D by N.
end

function DualityGapBuffer(::Type{T}, D::Integer, N::Integer, N_edges::Integer)::DualityGapBuffer{T} where T
    DualityGapBuffer(
        Matrix{T}(undef, D, N_edges),
        Matrix{T}(undef, D, N_edges),
        Matrix{T}(undef, D, N_edges),
        Matrix{T}(undef, D, N_edges),
        Matrix{T}(undef, D, N),
    )
end

function unpackbuffer(A::DualityGapBuffer)
    #
    return A.U, A.BX, A.prox_U_plus_Z, A.U_plus_Z, A.BadjZ
end

### edge-related buffers

abstract type RegularizationBuffer end

# the B map is from Eq. 7 from (Sun, 2021).
struct BMapBuffer{T} <: RegularizationBuffer

    # variables.
    Z::Matrix{T} # the set of dual vairables. size D by N_edges
    U::Matrix{T} # a subset of the primal variables. size D by N_edges

    # buffers for evaluating the subproblem.
    V::Matrix{T} # size D by N_edges
    prox_V::Matrix{T} # size D by N_edges
    prox_conj_V::Matrix{T} # size D by N_edges. use only in fdf!()

    # reuse buffers from ϕ or dϕ for primal, dual, primal_dual gap calculation.
    # this is possible because no ϕ and dϕ evaluations are occuring in the gap calculations.
    residual::DualityGapBuffer{T}
end

# creates a copy of Z0.
function BMapBuffer(Z0::Matrix{T}, N::Integer)::BMapBuffer{T} where T

    D, N_edges = size(Z0)

    return BMapBuffer(
        copy(Z0),
        Matrix{T}(undef, D, N_edges),
        Matrix{T}(undef, D, N_edges),
        Matrix{T}(undef, D, N_edges),
        Matrix{T}(undef, D, N_edges),
        DualityGapBuffer(T, D, N, N_edges)
    )
end

struct CoBMapBuffer{T} <: RegularizationBuffer
    col::BMapBuffer{T}
    row::BMapBuffer{T}
end

function getZbuffer(dual::ALMDualVar{T}, problem::ProblemType)::BMapBuffer{T} where T
    N = size(problem.A, 2)
    return BMapBuffer(dual.Z, N)
end

function getZbuffer(dual::ALMCoDualVar{T}, problem::ProblemType)::CoBMapBuffer{T} where T

    N_col = size(problem.A, 2)
    N_row = size(problem.A, 1)

    return CoBMapBuffer(
        BMapBuffer(dual.col.Z, N_col),
        BMapBuffer(dual.row.Z, N_row),
    )
end

### trace diagonistics to the ALM.

struct TraceType{T}#, VT <: TraceVariableContainer}
    gaps::Vector{Vector{T}}
    problem_cost::Vector{T}
    diff_x::Vector{T}
    #dual_var::VT
    diff_Z::Vector{Vector{T}}
end

function TraceType(::Conventional, ::Type{T}, N::Int)::TraceType{T} where T <: AbstractFloat

    gaps = Vector{Vector{T}}(undef, N)
    poblem_cost = Vector{T}(undef, N)
    diff_x = Vector{T}(undef, N)
    diff_Z = collect( Vector{T}(undef, N) for _ = 1:1 )

    return TraceType(gaps, poblem_cost, diff_x, diff_Z)
end

function TraceType(::CoClustering, ::Type{T}, N::Int)::TraceType{T} where T <: AbstractFloat

    gaps = Vector{Vector{T}}(undef, N)
    poblem_cost = Vector{T}(undef, N)
    diff_x = Vector{T}(undef, N)
    diff_Z = collect( Vector{T}(undef, N) for _ = 1:2 )

    return TraceType(gaps, poblem_cost, diff_x, diff_Z)
end

function resizetrace!(trace::TraceType{T}, ::Conventional, iter::Int) where T <: AbstractFloat
    @assert length(trace.gaps) == length(trace.problem_cost)

    resize!(trace.gaps, iter)
    resize!(trace.problem_cost, iter)
    resize!(trace.diff_x, iter)
    
    resize!(trace.diff_Z[begin], iter)

    return nothing
end

function resizetrace!(trace::TraceType{T}, ::CoClustering, iter::Int) where T <: AbstractFloat
    @assert length(trace.gaps) == length(trace.problem_cost)

    resize!(trace.gaps, iter)
    resize!(trace.problem_cost, iter)
    resize!(trace.diff_x, iter)
    
    resize!(trace.diff_Z[begin], iter)
    resize!(trace.diff_Z[begin+1], iter)

    return nothing
end

function storetrace!(
    trace::TraceType{T},
    current::BMapBuffer,
    prev::BMapBuffer,
    iter::Integer;
    ind_Z::Integer = 1, # The index to save the dual var difference, diff_Z, to. 1 for the column edge constraints, 2 for the row edge constraints.
    ) where T
    
    trace.diff_Z[ind_Z][iter] = evalnorm2sq(prev.Z, current.Z)
    prev.Z[:] = current.Z

    return nothing
end

function storetrace!(
    trace::TraceType{T},
    current::CoBMapBuffer,
    prev::CoBMapBuffer,
    iter::Integer,
    ) where T

    storetrace!(trace, current.col, prev.col, iter; ind_Z = 1)
    storetrace!(trace, current.row, prev.row, iter; ind_Z = 2)

    return nothing
end

### solution data structure for the ALM algorithm.

struct ALMSolutionType{T, DT <: AuxiliaryVariable}
    X_star::Matrix{T}
    #aux_star::DT
    #Z_star::Vector{Matrix{T}}
    dual_star::DT
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

struct SearchCoγConfigType
    max_iters::Int
    col_max_partition_size::Int
    row_max_partition_size::Int
    getγfunc::Function # iteration_number::Int ↦ γ::T
end

### search sequence for kernel parameter, config.
struct SearchθConfigType{T}
    max_iters::Int
    min_dynamic_range::T
    getθfunc::Function # # iteration_number::Int ↦ θ, whatever datatype θ is.
end

### assignment result.


# traits.
function getassignmenttype(::EdgeSet{T}) where T
    return Vector{Vector{Int}} # assignment result type for conventional case.
end

function getassignmenttype(::CoEdgeSet{T}) where T
    return CoAssignmentResult
end

# assignment result type container for co-clustering case.
struct CoAssignmentResult
    col::Vector{Vector{Int}}
    row::Vector{Vector{Int}}
end

