
T = Float64

include("./helpers/co.jl")

metricfunc = (xx,yy)->norm(xx-yy)


table = CSV.read("./data/CCLE_metabolomics_20190502.csv", TypedTables.Table)

# each row is a data point. column i is the i-th attribute
csv_mat = readdlm("./data/CCLE_metabolomics_20190502.csv", ',', Any)

data_mat = convert(Matrix{T}, csv_mat[2:end, 3:end])

col_headings = vec(csv_mat[1, 3:end])
row_headings = vec(csv_mat[2:end, 1])

data_col_vecs = collect( vec(data_mat[n,:]) for n in axes(data_mat, 1) )
data_row_vecs = collect( vec(data_mat[:,n]) for n in axes(data_mat, 2) )


#distance_threshold = (maximum(data_mat) - minimum(data_mat))/10
distance_threshold_col = 6.5
distance_threshold_row = 8.5

#
SL_col, SL_colstatus, SL_col_partitions, SL_col_distances,
SL_col_terminal_ind = SL.mergepointsfull(
    data_col_vecs,
    metricfunc;
    tol = distance_threshold_col,
)
SL_col_part = SL_col_partitions[SL_col_terminal_ind]
@show length(SL_col_part)

#
SL_row, SL_rowstatus, SL_row_partitions, SL_row_distances,
SL_row_terminal_ind = SL.mergepointsfull(
    data_row_vecs,
    metricfunc;
    tol = distance_threshold_row,
)
SL_row_part = SL_row_partitions[SL_row_terminal_ind]
@show length(SL_row_part)


###### search.


## constant hyperparameters

metric = Distances.Euclidean()
kernelfunc = evalSqExpkernel # must be a positive-definite RKHS kernel that does not output negative numbers.

# regularization parameter search.
γ_base = 0.01
γ_rate = 1.05
max_partition_size = 13#length(data[1]) + 2
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

A_col, edges_col, w_col, neighbourhood_col, θs_col = preparedatagraph(SL_col)
A_row, edges_row, w_row, neighbourhood_row, θs_row = preparedatagraph(SL_row)

# initialize γ to NaN since it will be replaced by the search sequence in config_γ = getγfunc
problem = CC.ProblemType(
    A_col,
    NaN,
    CC.CoEdgeSet(
        CC.EdgeSet(w_col, edges_col),
        CC.EdgeSet(w_col, edges_col),
    ),
)

@assert 1==2

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