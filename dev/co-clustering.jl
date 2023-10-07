
T = Float64

include("./helpers/co.jl")

metricfunc = (xx,yy)->norm(xx-yy)
project_folder = joinpath(homedir(), "convex_clustering/cell_line/reduced")

table = CSV.read("./data/CCLE_metabolomics_20190502.csv", TypedTables.Table)

# each row is a data point. column i is the i-th attribute
csv_mat = readdlm("./data/CCLE_metabolomics_20190502.csv", ',', Any)

data_mat_full_original = convert(Matrix{T}, csv_mat[2:end, 3:end])
data_mat_full = standardizedata(data_mat_full_original)

# debug.
# reduce for determining if gap calculation has a bug.
#data_mat = data_mat_full[1:100,:]
data_mat = data_mat_full
# end debug

col_headings = vec(csv_mat[1, 3:end])
row_headings = vec(csv_mat[2:end, 1])

data_col_vecs = collect( vec(data_mat[:,n]) for n in axes(data_mat, 2) )
data_row_vecs = collect( vec(data_mat[n,:]) for n in axes(data_mat, 1) )
#@assert 1==2

# #distance_threshold = (maximum(data_mat) - minimum(data_mat))/10
# # distance_threshold_col = 6.5
# # distance_threshold_row = 6.3
# distance_threshold = 7.0

# #
# SL_col, SL_colstatus, SL_col_partitions, SL_col_distances,
# SL_col_terminal_ind = SL.mergepointsfull(
#     data_col_vecs,
#     metricfunc;
#     #tol = distance_threshold_col,
#     tol = distance_threshold,
# )
# SL_col_part = SL_col_partitions[SL_col_terminal_ind]
# @show length(SL_col_part)

# # #
# # SL_row, SL_rowstatus, SL_row_partitions, SL_row_distances,
# # SL_row_terminal_ind = SL.mergepointsfull(
# #     data_row_vecs,
# #     metricfunc;
# #     tol = distance_threshold_row,
# # )
# # SL_row_part = SL_row_partitions[SL_row_terminal_ind]
# # @show length(SL_row_part)

#SL_col = data_col_vecs
A_col_vecs = data_col_vecs
A_row_vecs = data_row_vecs

#@assert 1==2

###### search.


## constant hyperparameters

metric = Distances.Euclidean()
kernelfunc = evalSqExpkernel # must be a positive-definite RKHS kernel that does not output negative numbers.

# regularization parameter search.
γ_base = 0.1
#γ_base = 1.0

γ_rate = 1.03
col_max_partition_size = 2
row_max_partition_size = 2
max_iters_γ = 50
getγfunc = nn->evalgeometricsequence(nn-1, γ_base, γ_rate)
config_γ = ConvexClustering.SearchCoγConfigType(
    max_iters_γ,
    col_max_partition_size,
    row_max_partition_size,
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

# column j of A_col or A_row is the j-th point in the set to be partitioned.
# A_col, edges_col, w_col, neighbourhood_col, θs_col,
#     A_row, edges_row, w_row, neighbourhood_row, θs_row = autoθ(
#     A_col_vecs,
#     A_row_vecs;
#     knn_factor_col = 0.2, # NaN here leads to full connectivity.
#     knn_factor_row = 0.2, # NaN here leads to full connectivity.
# )

A_col, edges_col, w_col, neighbourhoods_col,
    A_row, edges_row, w_row, neighbourhoods_row  = manualθ(
    A_col_vecs,
    A_row_vecs;
    length_scale_col = 22.0,
    length_scale_row = 22.0,
    #length_scale_row = 12.0, # comparable with 22.
    knn_factor_col = 0.2, # NaN here leads to full connectivity.
    knn_factor_row = 0.2, # NaN here leads to full connectivity.
)

@show length(w_col), length(w_row), size(data_mat)

#@assert 1==2

# initialize γ to NaN since it will be replaced by the search sequence in config_γ = getγfunc
problem = CC.ProblemType(
    A_col,
    NaN,
    CC.CoEdgeSet(
        CC.EdgeSet(w_col, edges_col),
        CC.EdgeSet(w_row, edges_row),
    ),
)

A_col, edges_col, w_col, neighbourhoods_col,
A_row, edges_row, w_row, neighbourhoods_row = manualθ(A_col_vecs, A_row_vecs;
    length_scale_col = 22.0,
    length_scale_row = 22.0,
    knn_factor_col = 0.2,
    knn_factor_row = 0.2,
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


# assignment.
assignment_zero_tol = 1e-3
assignment_config_col = ConvexClustering.AssignmentConfigType(metric, assignment_zero_tol)

assignment_zero_tol = 1e-3
assignment_config_row = ConvexClustering.AssignmentConfigType(metric, assignment_zero_tol)

assignment_config = CC.CoAssignmentConfigType(assignment_config_col, assignment_config_row)

### optimization algorithm settings.

### run optimization.
Gs, rets, γs = ConvexClustering.searchγ(
    X0,
    dual_initial,
    problem,
    optim_config,
    assignment_config,
    config_γ;
    store_trace = store_trace,
    report_cost = report_cost,
)
G_last = Gs[end]
iters_γ = length(γs)

# parse.
Xs = collect( rets[n].X_star for n in eachindex(rets) )
Zcs = collect( rets[n].dual_star.col.Z for n in eachindex(rets) )
Zrs = collect( rets[n].dual_star.row.Z for n in eachindex(rets) )
N_iters = collect( rets[n].num_iters_ran for n in eachindex(rets) )
gaps = collect( rets[n].gaps for n in eachindex(rets) )

Gs_col = collect( Gs_cc[n].col for n in eachindex(Gs) )
Gs_row = collect( Gs_cc[n].row for n in eachindex(Gs) )

BSON.bson(

    joinpath(
        project_folder,
        "full_$(γ_base).bson",
    ),
    γs = γs,

    Xs = Xs,
    Zcs = Zcs,
    Zrs = Zrs,
    N_iters = N_iters,
    gaps = gaps,

    Gs_col = Gs_col,
    Gs_row = Gs_row,

    γ_base = γ_base,
    γ_rate = γ_rate,
)



@assert 1==2

# I am here. do one γ and one θ solve, verify it solves the co-clustering optim problem.


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

G = applySL(Gs_cc[end-1], SL_col_part)
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