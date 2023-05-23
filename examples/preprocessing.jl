
T = Float64



metricfunc = (xx,yy)->norm(xx-yy)
Δc, referenec_sol_Gs_cc = alphaglucose700()


distance_threshold = 1e-3

# early_stop_distance = 1e-3
# distance_set, partition_set = SingleLinkagePartitions.runsinglelinkage(
#     Δc,
#     metricfunc;
#     early_stop_distance = early_stop_distance,
# )

function getmergedpartition(
    X::Vector{Vector{T}};
    metricfunc = (xx,yy)->norm(xx-yy),
    distance_threshold = 1e-3,
    ) where T

    Y, status_flag, partitioned_set_sorted, h_set_sorted, chosen_ind = SingleLinkagePartitions.mergepointsfull(X, metricfunc; tol = distance_threshold)

    return Y, status_flag, partitioned_set_sorted[chosen_ind]
end

Δc_post_SL, SL_status, SL_partitions, SL_distances, SL_terminal_ind = SL.mergepointsfull(
    Δc,
    metricfunc;
    tol = distance_threshold,
)
SL_terminal_partition = SL_partitions[SL_terminal_ind]

###### search.


## constant hyperparameters

metric = Distances.Euclidean()
kernelfunc = evalSqExpkernel # must be a positive-definite RKHS kernel that does not output negative numbers.

connectivity = ConvexClustering.KNNType(60) # make an edge for this number of nearest neighbours of a given point i, cycle through all i in the point set to be partitioned. Takes a positive integer.
#connectivity = ConvexClustering.KNNType(length(Δc_post_SL)-1)

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
γ_base = 0.01
γ_rate = 1.05
max_partition_size = length(Δc_post_SL[1]) + 2 # stop searching once the size of the returned partition is less than `max_partition_size`.
max_iters_γ = 100
getγfunc = nn->evalgeometricsequence(nn-1, γ_base, γ_rate)
config_γ = ConvexClustering.SearchγConfigType(max_iters_γ, max_partition_size, getγfunc)

# convex clustering optimization algorithm configuration.
σ_base = 0.4
#σ_base = 5000.0
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
    Δc_post_SL,
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
Gs_cc, rets, γs = ConvexClustering.searchγ(
    X0, Z0, problem, optim_config, assignment_config, config_γ;
    store_trace = store_trace,
    report_cost = report_cost)
G_cc_last = Gs_cc[end]
iters_γ = length(γs)

println("Are the returned partitions nested successively? ",
    all(ConvexClustering.isnestedsuccessively(Gs_cc)),
)

function applySL(
    G_cc::Vector{Vector{Int}}, # convex clustering result.
    G_sl::Vector{Vector{Int}}, # SL result.
    )

    out = Vector{Vector{Int}}(undef, length(G_cc))

    for n in eachindex(G_cc)
        CC_part = G_cc[n]

        out[n] = Vector{Int}(undef, 0)
        for i in eachindex(CC_part)

            SL_part_ind = CC_part[i]
            push!(out[n], G_sl[SL_part_ind]...)
        end
    end

    return out
end

G = applySL(Gs_cc[end-1], SL_terminal_partition)
G_flat = collect(Iterators.flatten(G))
@assert length(G_flat) == length(unique(G_flat))


function verifysol(ref_Gs::Vector{Vector{Vector{Int}}}, Gs::Vector{Vector{Vector{Int}}})

    @assert length(Gs) == length(ref_Gs) # same number of tree levels.

    discrepancy = 0
    for l in eachindex(Gs)
        for i in eachindex(Gs[l])
            discrepancy += norm(Gs[l][i]-ref_Gs[l][i])
        end
    end

    return discrepancy
end

# should be zero if Gs_cc is the same as the reference solution.
@show verifysol(referenec_sol_Gs_cc, Gs_cc)

nothing