
import Distances

import JLD
using LinearAlgebra

# include("../../src/ConvexClustering.jl")
# import .ConvexClustering
import ConvexClustering

#import Optim
#include("./helpers/optim.jl")

### these specify the square exponential kernel for the weight function, and how to update the length scale hyperparameter.
function lengthscale2θ(l::T)::T where T <: AbstractFloat
    return 1/(2*l^2)
end

function θ2lengthscale(θ::T)::T where T<: AbstractFloat
    return 1/sqrt(2*θ)
end

function evalgeometricsequence(n::Int, a0::T, r::T)::T where T <: AbstractFloat
    return a0*r^n
end

function evalSqExpkernel(x::Vector{T}, z::Vector{T}, θ::T)::T where T <: AbstractFloat
    return exp(-θ*norm(x-z)^2)
end
###

### load data.
#dict = JLD.load("./JLD/nmr_coherence/L-Leucine_700.13_MHz.jld") # WARNING: Slow to run.
dict = JLD.load("./JLD/nmr_coherence/L-Glutamine_700.13_MHz.jld") # with search_θ enabled, non-monotonic decrease in partition size as γ monotonically increases.
#dict = JLD.load("./JLD/nmr_coherence/beta-D-Glucose_700.13_MHz.jld")
Δc_m = dict["Δc_m_molecule"][1] # this is like `A_vecs` in the other example scripts. Select spin group 1.


## constant hyperparameters

metric = Distances.Euclidean()
kernelfunc = evalSqExpkernel # must be a positive-definite RKHS kernel that does not output negative numbers.

connectivity = ConvexClustering.KNNType(60) # make an edge for this number of nearest neighbours of a given point i, cycle through all i in the point set to be partitioned. Takes a positive integer.
#connectivity = ConvexClustering.RadiusType(1.0) # make an edge for all points within this radius of a given point i, cycle through all i in the point set to be partitioned. Takes a finite floating point number.

# package up config parameters.
graph_config = ConvexClustering.WeightedGraphConfigType(connectivity, metric, kernelfunc)




## variable hyperparameters

# weight function hyperparameter search.
length_scale_base = 10.0
length_scale_rate = 0.7
length_scale_max_iters = 1000
min_dynamic_range = 0.95
getθfunc = nn->lengthscale2θ(evalgeometricsequence(nn-1, length_scale_base, length_scale_rate))
config_θ = ConvexClustering.SearchθConfigType(length_scale_max_iters, min_dynamic_range, getθfunc)

# regularization parameter search.
γ_base = 0.1
γ_rate = 1.05
max_partition_size = length(Δc_m[1]) + 2 # stop searching once the size of the returned partition is less than `max_partition_size`.
max_iters_γ = 100
getγfunc = nn->evalgeometricsequence(nn-1, γ_base, γ_rate)
config_γ = ConvexClustering.SearchγConfigType(max_iters_γ, max_partition_size, getγfunc)

# convex clustering optimization algorithm configuration.
σ_base = 0.4
σ_rate = 1.05
updateσfunc = nn->σ_base*σ_rate^(nn-1)
updateϵfunc = nn->1/(nn-1)^2
#runoptimfunc = (xx,ff,dff,gg_tol)->runOptimjl(xx, ff, dff, gg_tol; verbose = verbose_subproblem)
gap_tol = 1e-8
cc_max_iters = 300
optim_config = ConvexClustering.ALMConfigType(
    gap_tol;
    #runoptimfunc;
    max_iters = cc_max_iters,
    updateσfunc = updateσfunc,
    updateϵfunc = updateϵfunc
)

# assignment.
assignment_zero_tol = 1e-3
assignment_config = ConvexClustering.AssignmentConfigType(metric, assignment_zero_tol)

# verbose, trace, and stopping condition configs.
verbose_subproblem = false
report_cost = true # want to see the objective score per θ run or γ run.
store_trace = true

verbose_kernel = true


### setup convex clustering problem.
A, edge_pairs, w, A_neighbourhoods, θs = ConvexClustering.searchkernelparameters(
    T,
    Δc_m,
    config_θ,
    graph_config;
    verbose = verbose_kernel,
)
iter_kernel = length(θs)
length_scale = θ2lengthscale( getθfunc(iter_kernel) )
println("Starting length scale: ", getθfunc(1))
println("resulting length scale: ", length_scale)
@show min_dynamic_range

# initialize γ to NaN since it will be replaced by the search sequence in config_γ = getγfunc
problem = ConvexClustering.ProblemType(A, NaN, w, edge_pairs)


### initial guess.
D, N = size(A)
N_edges = length(edge_pairs)

X0 = zeros(D, N)
Z0 = zeros(D, N_edges)

### optimization algorithm settings.

### run optimization.
Gs, rets, γs = ConvexClustering.searchγ(
    X0, Z0, problem, optim_config, assignment_config, config_γ;
    store_trace = store_trace,
    report_cost = report_cost)
G = Gs[end]
iters_γ = length(γs)

println("Are the returned partitions nested successively? ", all(ConvexClustering.isnestedsuccessively(Gs)))
#=
however, if we had:
```
connectivity = ConvexClustering.KNNType(60)
```
then we wouldn't have nested partitions. The theory in (Chi, SIAM 2019) is for fully-connected graphs, not k-nearest neighbours. Therefore a larger knn number (up to the maximum of N-1) will increase the chance of getting nested partitions.
=#

println("Starting γ: ", config_γ.getγfunc(1))
println("resulting γ: ", config_γ.getγfunc(iters_γ))
@show max_partition_size
