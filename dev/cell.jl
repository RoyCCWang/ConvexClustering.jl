
T = Float64

include("./helpers/co.jl")
include("./helpers/two_layer.jl")

metricfunc = (xx,yy)->norm(xx-yy)
project_folder = joinpath(homedir(), "convex_clustering/cell_line")

table = CSV.read("./data/CCLE_metabolomics_20190502.csv", TypedTables.Table)

# each row is a data point. column i is the i-th attribute
csv_mat = readdlm("./data/CCLE_metabolomics_20190502.csv", ',', Any)

data_mat_full_original = convert(Matrix{T}, csv_mat[2:end, 3:end])
data_mat_full = standardizedata(data_mat_full_original)
#data_mat_full = collect(data_mat_full')

# debug.
# reduce for determining if gap calculation has a bug.
#data_mat = data_mat_full[1:100,:]
data_mat = data_mat_full
# end debug

cell_line_labels = vec(csv_mat[2:end, 1])
metabolite_labels = vec(csv_mat[1, 2:end])

@assert 1==2

col_headings = vec(csv_mat[1, 3:end])
row_headings = vec(csv_mat[2:end, 1])

metabolite_vecs = collect( vec(data_mat[:,n]) for n in axes(data_mat, 2) )
cell_vecs = collect( vec(data_mat[n,:]) for n in axes(data_mat, 1) )

###### search.


## constant hyperparameters

metric = Distances.Euclidean()
kernelfunc = evalSqExpkernel # must be a positive-definite RKHS kernel that does not output negative numbers.

# regularization parameter search.
#γ_base = 1.6
#γ_base = 1.0 # metabolite.
#

#γ_base = 0.1 # cell line, knn 186.
#γ_rate = 1.05

# # cell line, knn full.
# γ_base = 0.045
#γ_rate = 1.04

# # cell line, knn full.
# γ_base = 0.085
# γ_rate = 1.1


# # metabolite, knn full. testing.
# #γ_base = 0.06 # 225.
# #γ_base = 0.16 # 255
# #γ_base = 1.35 # 255
#γ_base = 6.0 # 22
# 33 is 6.
γ_base = 80.0 # at 2560 we've 2.
γ_rate = 2.0


max_partition_size = 1 #2
max_iters_γ = 5 #15 # 10
getγfunc = nn->evalgeometricsequence(nn-1, γ_base, γ_rate)
config_γ = ConvexClustering.SearchγConfigType(
    max_iters_γ,
    max_partition_size,
    getγfunc,
)

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

# verbose, trace, and stopping condition configs.
verbose_subproblem = false
report_cost = true # want to see the objective score per θ run or γ run.
store_trace = false

# assignment.
assignment_zero_tol = 1e-3
assignment_config = ConvexClustering.AssignmentConfigType(metric, assignment_zero_tol)


########## col.

# knn_factor = 0.2
# A_vecs = metabolite_vecs

# knn_factor = 0.2
# A_vecs = cell_vecs

# knn_factor = NaN
# A_vecs = cell_vecs

knn_factor = NaN
A_vecs = metabolite_vecs

####### run cc.

knn = processknn(knn_factor, A_vecs)
connectivity = CC.KNNType(knn)

# # auto θ
# knn = processknn(knn_factor, A_vecs)
# A, edges, w, neighbourhood, θs = preparedatagraph(
#     A_vecs;
#     knn = knn,
# )

# manual θ
#length_scale = 22.0
length_scale = 18.0
#length_scale = 12.0
θ = lengthscale2θ(length_scale)
A, edges, w, neighbourhoods = CC.setupproblem(
    A_vecs,
    θ,
    connectivity;
    metric = metric,
    kernelfunc = kernelfunc,
)
@show length_scale, minimum(w), maximum(w), abs(maximum(w)-minimum(w))

#@assert 1==2

Gs, rets, γs = runcc(
    A,
    edges,
    w,
    config_γ,
    optim_config,
    assignment_config;
    knn_factor = 0.2,
    report_cost = true, # want to see the objective score per θ run or γ run.
    store_trace = true,
)


# parse.
Xs = collect( rets[n].X_star for n in eachindex(rets) )
Zs = collect( rets[n].dual_star.Z for n in eachindex(rets) )
N_iters = collect( rets[n].num_iters_ran for n in eachindex(rets) )
gaps = collect( rets[n].gaps for n in eachindex(rets) )

# file name options.
time_string = replace(string(Dates.now()), ":"=>"_")
mode_string = "metabolite"
if A_vecs == cell_vecs
    mode_string = "cell_line"
end

BSON.bson(

    joinpath(
        project_folder,
        #"full_$(mode_string)_$(time_string).bson",
        "full_$(mode_string)_$(γ_base)_$(knn).bson",
    ),
    γs = γs,

    knn = knn,
    θ = θ,

    Xs = Xs,
    #Zs = Zs,
    N_iters = N_iters,
    gaps = gaps,

    Gs = Gs,

    γ_base = γ_base,
    γ_rate = γ_rate,

    # include the problem hyperparameters so that this file is self-contained.
    edges = edges,
    w = w,
    A = A,
)



nothing