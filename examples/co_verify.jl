
T = Float64

include("./helpers/co.jl")

metricfunc = (xx,yy)->norm(xx-yy)
project_folder = joinpath(homedir(), "MEGASync/output/convex_clustering/reduced")

table = CSV.read("./data/CCLE_metabolomics_20190502.csv", TypedTables.Table)

# each row is a data point. column i is the i-th attribute
csv_mat = readdlm("./data/CCLE_metabolomics_20190502.csv", ',', Any)

data_mat_full = convert(Matrix{T}, csv_mat[2:end, 3:end])

# debug.
# reduce for determining if gap calculation has a bug.
#data_mat = data_mat_full[1:100,:]
data_mat = data_mat_full
# end debug

col_headings = vec(csv_mat[1, 3:end])
row_headings = vec(csv_mat[2:end, 1])

data_col_vecs = collect( vec(data_mat[:,n]) for n in axes(data_mat, 2) )
data_row_vecs = collect( vec(data_mat[n,:]) for n in axes(data_mat, 1) )

SL_col = data_col_vecs

###### search.


## constant hyperparameters

metric = Distances.Euclidean()
kernelfunc = evalSqExpkernel # must be a positive-definite RKHS kernel that does not output negative numbers.

# regularization parameter search.
#γ = 0.0
#γ = 1e-3
#γ = 1.0
#γ = 10.0 # first process.
γ = 100.0 # second process.
max_iters_γ = 100
getγfunc = nn->evalgeometricsequence(nn-1, γ_base, γ_rate)

# convex clustering optimization algorithm configuration.
σ_base = 0.4
#σ_base = 866284.0
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
    updateϵfunc = updateϵfunc,
    verbose_subproblem = true,
)

# verbose, trace, and stopping condition configs.
verbose_subproblem = false
report_cost = true # want to see the objective score per θ run or γ run.
store_trace = true

# column j of A_col or A_row is the j-th point in the set to be partitioned.
A_col, edges_col, w_col, neighbourhood_col, θs_col = preparedatagraph(
    SL_col;
    #knn = 30,
    knn = length(SL_col)-1,
)

A_row_vecs = collect( vec(A_col[n,:]) for n in axes(A_col,1) )
# A_row, edges_row, w_row, neighbourhood_row, θs_row = preparedatagraph(
#     A_row_vecs;
#     knn = 30,
# )

# debug/
length_scale_base = 10.0
length_scale_rate = 0.7
length_scale_max_iters = 1000
min_dynamic_range = 0.95
getθfunc = nn->lengthscale2θ(
    evalgeometricsequence(
        nn-1,
        length_scale_base,
        length_scale_rate,
    ),
)
# end debug

#knn = 300
knn = length(A_row_vecs)-1
connectivity = CC.KNNType(knn)
#θ = 0.0005
#θ = 0.005
θ = 0.015 # yields large dynamic range: ~0.8
#θ = 0.05
#θ = 0.5
A_row, edges_row, w_row, neighbourhoods_row = CC.setupproblem(
    A_row_vecs,
    θ,
    connectivity;
    metric = metric,
    kernelfunc = kernelfunc,
)
w_max = maximum(w_row)
w_min = minimum(w_row)
@show w_min, w_max, abs(w_max-w_min)

max_val, max_ind = findmax(w_row)
@show edges_row[max_ind], max_val

min_val, min_ind = findmin(w_row)
@show edges_row[min_ind], min_val

@show length(w_col), length(w_row)

#@assert 1==2

@show length(w_col), length(w_row)

# initialize γ to NaN since it will be replaced by the search sequence in config_γ = getγfunc
problem = CC.ProblemType(
    A_col,
    γ,
    CC.CoEdgeSet(
        CC.EdgeSet(w_col, edges_col),
        CC.EdgeSet(w_row, edges_row),
    ),
)

### initial guess.
D_col, N_col = size(A_col)
D_row, N_row = size(A_row)
N_edges_col = length(edges_col)
N_edges_row = length(edges_row)

# default.
#X0 = zeros(D_col, N_col)
X0 = copy(A_col)

dual_initial = CC.ALMCoDualVar(
    CC.ALMDualVar(zeros(D_col, N_edges_col)),
    CC.ALMDualVar(zeros(D_row, N_edges_row)),
)

# # if we've already done a run.
# X0, dual_initial, σ_base, file_counter = loadresult(A_col, γ, σ_base, N_edges_col, N_edges_row, project_folder)

#@assert 1==2

# assignment.
assignment_zero_tol = 1e-3
assignment_config_col = ConvexClustering.AssignmentConfigType(metric, assignment_zero_tol)

assignment_zero_tol = 1e-3
assignment_config_row = ConvexClustering.AssignmentConfigType(metric, assignment_zero_tol)

assignment_config = CC.CoAssignmentConfigType(assignment_config_col, assignment_config_row)

### optimization algorithm settings.

## run optimization.
G, ret = ConvexClustering.runconvexclustering(
    X0,
    dual_initial,
    problem,
    optim_config,
    assignment_config;
    store_trace = store_trace,
    report_cost = report_cost,
    verbose_ALM = true,
)

if isapprox(γ, zero(T))
    @show norm(ret.X_star - A_col)
end

file_counter += 1

BSON.bson(

    joinpath(
        project_folder,
        "co_gamma_$(γ)_$(file_counter).bson",
    ),
    X_star = ret.X_star,
    Z_star_col = ret.dual_star.col.Z,
    Z_star_row = ret.dual_star.row.Z,
    num_iters_ran = ret.num_iters_ran,
    gaps = ret.gaps,
    last_sigma = updateσfunc(ret.num_iters_ran),
)




# #
# objfunc = xx->CC.primaldirect( reshape(xx, size(X0)), problem)

# ret_evo = EVO.optimize(
#     objfunc,
#     vec(X0),
#     EVO.GA(
#         populationSize = 100,
#         selection = EVO.susinv,
#         crossover = EVO.DC,
#         mutation = EVO.PLM(),
#     ),
# )
# X_star_evo = reshape(ret_evo.minimizer, size(X0))

# @show norm(X0-X_star_evo)
# @show objfunc(vec(X0)), objfunc(vec(X_star_evo)), objfunc(vec(A_col))

nothing