# explore partition trees.

file_names = [
"Agmatine_700.13_MHz.jld"; # passed. γ_base = 0.01. slow.
"Gamma-Aminobutyric acid_700.13_MHz.jld"; # passed γ_base = 0.001. slow.
"L-Glutamine_700.13_MHz.jld"; # passed. γ_base = 0.01
"L-Glutathione oxidized_700.13_MHz.jld";
"L-Glutathione reduced_700.13_MHz.jld";
"L-Histidine_700.13_MHz.jld";
"L-Isoleucine_700.13_MHz.jld";
"L-Leucine_700.13_MHz.jld";
"L-Phenylalanine_700.13_MHz.jld";
"L-Valine_700.13_MHz.jld";
"alpha-D-Glucose_700.13_MHz.jld";
"beta-D-Glucose_700.13_MHz.jld";
]

γ_base = 0.1
#file_name = file_names[end-1]
file_name = file_names[3] # glutamine.
#file_name = file_names[1] # agmatine.

# file_name = file_names[7] # isoleucine.
# γ_base = 1e-6

#file_name = file_names[8] # leucine.
#file_name = file_names[10] # valine

# file_name = file_names[end] # beta glucose
# γ_base = 0.0001


dict = JLD.load(joinpath("./JLD/nmr_coherence/", file_name))

Δc_molecule = dict["Δc_m_molecule"]

spin_sys_select = 1
Δc_m0 = dict["Δc_m_molecule"][spin_sys_select] # this is like `A_vecs` in the other example scripts. Select spin group 1.
@show file_name, spin_sys_select

# try conditioning the points.
function scalebydim(A::Vector{Vector{T}})::Vector{Vector{T}} where T
    out = similar(A)
    for i in eachindex(A)
        out[i] = A[i] .* length(A[i]) 
    end

    return out
end

Δc_m = scalebydim(Δc_m0)
#Δc_m = Δc_m0

#@assert 2==4

metric = Distances.Euclidean()
kernelfunc = evalSqExpkernel # must be a positive-definite RKHS kernel that does not output negative numbers.

start_knn = 60
#max_knn = max(32 + start_knn, round(Int, length(Δc_m)*0.2))
max_knn = max(32 + start_knn, round(Int, length(Δc_m)*0.05))
knn = max_knn

connectivity = ConvexClustering.KNNType(knn)
#connectivity = ConvexClustering.KNNType(length(Δc_m)-1) # make it fully-connected.
graph_config = ConvexClustering.WeightedGraphConfigType(connectivity, metric, kernelfunc)
@show connectivity

length_scale_base = 10.0
length_scale_rate = 0.7
length_scale_max_iters = 1000
min_dynamic_range = 0.95
getθfunc = nn->lengthscale2θ(evalgeometricsequence(nn-1, length_scale_base, length_scale_rate))
config_θ = ConvexClustering.SearchθConfigType(length_scale_max_iters, min_dynamic_range, getθfunc)

#γ_base = 0.01

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
#gap_tol = 1e-6
#cc_max_iters = 200
gap_tol = 1e-6
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
A, edge_pairs, w, A_neighbourhoods, iter_kernel = ConvexClustering.searchkernelparameters(
    Δc_m, config_θ, graph_config; verbose = verbose_kernel)

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
println("Timing: ")
@time Gs, rets, iters_γ = ConvexClustering.searchγ(
    X0, Z0, problem, optim_config, assignment_config, config_γ;
    store_trace = store_trace,
    store_trace_assignments = true,
    report_cost = report_cost)
G = Gs[end]
ret = rets[end]

successively_nested_partitions = all(ConvexClustering.isnestedsuccessively(Gs))
@show successively_nested_partitions
@show iters_γ
@show γ_base

println("Number of points in set: ", length(Δc_m))

did_not_hit_max_iters = all( rets[n].num_iters_ran <= cc_max_iters for n in eachindex(rets) )
println("Did all runs finish before or on max_iters? ", did_not_hit_max_iters)
println()


@assert successively_nested_partitions
@assert did_not_hit_max_iters

### exploring Glutamine, fully-connected, g_gap 1e-6.
# X3 = rets[3].X_star
# G3 = Gs[3]

# X4 = rets[4].X_star
# G4 = Gs[4]

# println("X3: column 50, 55")
# display( [X3[:, 50] X3[:, 55] ] )
# @show norm(X3[:, 50] - X3[:, 55])
# println()

# println("X4: column 50, 55")
# display( [X4[:, 50] X4[:, 55] ] )
# @show norm(X4[:, 50] - X4[:, 55])
# println()