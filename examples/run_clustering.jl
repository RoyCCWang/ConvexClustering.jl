

import Random
Random.seed!(25)

using LinearAlgebra

#include("../src/ConvexClustering.jl")
import ConvexClustering
#import ConvexClustering

import Distances
#import Optim

import Distributions # for setting up data.
import PyPlot

using Plots; plotlyjs()

include("./helpers/generatesetup.jl")
include("./helpers/utils.jl")
#include("./helpers/optim.jl")

PyPlot.matplotlib["rcParams"][:update](["font.size" => 16, "font.family" => "serif"])

PyPlot.close("all")
fig_num = 1



### use a single gaussian to generate points.
# D = 4
# N = 50
# connectivity = ConvexClustering.KNNType(30) # make an edge for this number of nearest neighbours of a given point i, cycle through all i in the point set to be partitioned. Takes a positive integer.
#
# D = 4
# N = 50
# connectivity = ConvexClustering.RadiusType(1.5) # make an edge for all points within this radius of a given point i, cycle through all i in the point set to be partitioned. Takes a finite floating point number.
#
# D = 10
# N = 6000
# connectivity = ConvexClustering.RadiusType(1.5)
#
# D = 5
# N = 900
# connectivity = ConvexClustering.KNNType(30)
#
# D = 10
# N = 6000
# connectivity = ConvexClustering.KNNType(30)
#
# D = 10
# N = 10000
# connectivity = ConvexClustering.KNNType(30)

### use mixture of gaussians to generate points.
N_mixtures = 3
D = 2
N = 1000

# # manual knn
#connectivity = ConvexClustering.KNNType(30) # make an edge for this number of nearest neighbours of a given point i, cycle through all i in the point set to be partitioned. Takes a positive integer.

# # manual radiu.
#connectivity = ConvexClustering.RadiusType(1.0) # make an edge for all points within this radius of a given point i, cycle through all i in the point set to be partitioned. Takes a finite floating point number.

# # search knn. good. First try at 30 got a connected graph (only 1 connected component) already.
# connectivity = ConvexClustering.KNNSearchType(0, 1; start_knn = 30, verbose = true) # good.

# # search knn. bad. returned knn too small, but still got a connected graph.
# connectivity = ConvexClustering.KNNSearchType(0, 1; start_knn = 2, verbose = true) # bad

# # search radius. good.
connectivity = ConvexClustering.RadiusSearchType(-1.0, 1; max_iters = 100, verbose = true)

Σs = collect( generaterandomposdefmatrix(D) for k = 1:N_mixtures)
μs = collect( randn(D) .+ (k*5) for k = 1:N_mixtures)
dists = collect( Distributions.MvNormal(μs[k], Σs[k]) for k = 1:N_mixtures )
dist = Distributions.MixtureModel(Distributions.MvNormal[dists...])
A_vecs = collect( rand(dist) for n = 1:N )

### setup problem.
θ_w = 0.4
metric = Distances.Euclidean()
kernelfunc = (xx,zz,tt)->exp(-tt*norm(xx-zz)^2)

# package up config parameters.
graph_config = ConvexClustering.WeightedGraphConfigType(connectivity, metric, kernelfunc)

A, edge_pairs, w, A_neighbourhoods = ConvexClustering.setupproblem(A_vecs,
    θ_w, graph_config)

@show maximum(w)
@show minimum(w)
N_edges = length(edge_pairs)
@show N_edges

### the radius works well in this case.
#   perhaps one can approximate the radius-built graph with a knn=built one,
#   - use the norm difference of their graph laplacians as measure of distance.
#   - or the shape (ignore multiplicative factor) to the eigenvalues of the laplacian.
#   - need stoppping condition.
#
# g = ConvexClustering.buildgraph(A_neighbourhoods, A)
# cc_g = Graphs.connected_components(g)
# println("cc_g: knn-built graph connected components")
# display(cc_g)
#
# Q_g = collect( Graphs.dijkstra_shortest_paths(g, cc_g[i]) for i in eachindex(cc_g) )
# dists_g = collect( Q_g[i].dists for i in eachindex(Q_g) )
#
# @show minimum.(dists_g)
# @show maximum.(dists_g)
# @show typemax(Int)
# @show maximum(Graphs.degree(g))
# @show minimum(Graphs.degree(g))
#
# s=Graphs.LinAlg.laplacian_spectrum(g)
# display(s)

#@assert 1==2

# src_nodes = collect( edge[1] for edge in edge_pairs)
# dest_nodes = collect( edge[2] for edge in edge_pairs)
###

# γ = rand() #12.1 # zero for every data point is a cluster.
# σ_base = 1.0
# σ_rate = 1.01

# γ = 12.1
# σ_base = 1.0
# σ_rate = 1.05

# γ = 12.1
# σ_base = 14.0
# σ_rate = 1.05

γ = 3.1
#γ = #14.1
#γ = 38.1
#γ = 21.0 # for D 100 mvnormal case.
σ_base = 0.4
σ_rate = 1.05

updateσfunc = nn->σ_base*σ_rate^(nn-1)
updateϵfunc = nn->1/(nn-1)^2
#runoptimfunc = (xx,ff,dff,gg_tol)->runOptimjl(xx, ff, dff, gg_tol; verbose = true)

# pick cc_max_iters such that updateσfunc(cc_max_iters) does not return a large value for σ.
cc_max_iters = 200
gap_tol = 1e-6 # stopping condition.

# initial guess. Must be finite-valued.
X0 = zeros(Float64, D, N)
Z0 = zeros(Float64, D, N_edges)

# package up the configuration and problem parameters.
config = ConvexClustering.ALMConfigType(
    gap_tol;
    #runoptimfunc;
    max_iters = cc_max_iters,
    updateσfunc = updateσfunc,
    updateϵfunc = updateϵfunc
)

problem = ConvexClustering.ProblemType(A, γ, w, edge_pairs)

println("Timing: runALM")
@time ret = ConvexClustering.runALM(X0, Z0, problem, config; store_trace = true)

X_star, Z_star, num_iters_ran, gaps, trace = ret.X_star, ret.Z_star, ret.num_iters_ran, ret.gaps, ret.trace
@show num_iters_ran



###### cluster assignment.

assignment_zero_tol = 1e-6
# G = ConvexClustering.assignviaBX(X_star, edge_pairs; zero_tol = assignment_zero_tol)
G, g_neighbourhoods = ConvexClustering.assignviaX(X_star, metric;
    zero_tol = assignment_zero_tol)

P = ConvexClustering.applyassignment(G, A_vecs)

d1 = 1
d2 = 2
plot_handle = plot2Dclusters(P, d1, d2)
display(plot_handle)

#@assert 3==4


#### diagnostics on optimization run.

trace_gaps = trace.gaps
trace_evals = trace.problem_cost

max_gap_history = maximum.(trace_gaps)
println("should be zero if maximum of primal, dual, and primal_dual gaps were monotonically decreasing over iterations:")
@show norm(sort(max_gap_history, rev = true) - max_gap_history)
println()

p_gap_history = collect( trace_gaps[i][1] for i in eachindex(trace_gaps) )
d_gap_history = collect( trace_gaps[i][2] for i in eachindex(trace_gaps) )
pd_gap_history = collect( trace_gaps[i][3] for i in eachindex(trace_gaps) )

#### debug ALM increasing cost.

# negative output if decreasing change.
function computeseqchange(x::Vector{T}) where T <: AbstractFloat
    out = zeros(T, length(x))
    for i = 2:length(x)
        out[i] = x[i] - x[i-1]
    end
    return out
end


using DataFrames

diff_x = trace.diff_x
diff_Z = trace.diff_Z
grad_tol = collect( updateϵfunc(i)/max(1,sqrt(updateσfunc(i))) for i = 1:num_iters_ran )
cost = trace_evals
cost_m = computeseqchange(cost)
p_m = computeseqchange(p_gap_history)
d_m = computeseqchange(d_gap_history)
pd_m = computeseqchange(pd_gap_history)
max_gap_m = computeseqchange(max_gap_history)

df = DataFrame(
    diff_x = diff_x,
    diff_Z = diff_Z,
    cost = cost,
    cost_m = cost_m,
    grad_tol = grad_tol,
    p_gap = p_gap_history,
    p_m = p_m,
    d_gap = d_gap_history,
    d_m = d_m,
    pd_gap = pd_gap_history,
    pd_m = pd_m,
    max_gap = max_gap_history,
    max_m = max_gap_m)

display(df)
println("*_m should be negative if * is a decreasing sequence.")
@show maximum(cost_m)
@show maximum(max_gap_m)
@show maximum(p_m)
@show maximum(d_m)
@show maximum(pd_m)

# some issues with convergence:
# - cost_m, max_gap_m, p_m, d_m, pd_m are not strictly decreasing over iters!
#
#@assert 1==2

# next, how to know if we solved problem? page 22.
# do direct naive implementaion of them.

# next, select clusters. need neighest neighbours?

#### visualize diagnostics.
ln_trace_evals = log.(trace_evals)
ln_p_gap_history = log.(p_gap_history)
ln_d_gap_history = log.(d_gap_history)
ln_pd_gap_history = log.(pd_gap_history)

PyPlot.figure(fig_num)
fig_num += 1

#PyPlot.plot(max_gap_history)
PyPlot.plot(ln_trace_evals, label = "ln_trace_evals")
PyPlot.plot(ln_trace_evals, "o")

PyPlot.legend()
PyPlot.xlabel("iter")
PyPlot.ylabel("log-space")
PyPlot.title("primal problem cost")



PyPlot.figure(fig_num)
fig_num += 1

#PyPlot.plot(max_gap_history)
PyPlot.plot(ln_p_gap_history, label = "ln_p_gap_history")
PyPlot.plot(ln_p_gap_history, "o")

PyPlot.legend()
PyPlot.xlabel("iter")
PyPlot.ylabel("log-space")
PyPlot.title("primal gap history")


PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(ln_d_gap_history, label = "ln_d_gap_history")
PyPlot.plot(ln_d_gap_history, "o")

PyPlot.legend()
PyPlot.xlabel("iter")
PyPlot.ylabel("log-space")
PyPlot.title("dual gap history")



PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(ln_pd_gap_history, label = "ln_pd_gap_history")
PyPlot.plot(ln_pd_gap_history, "o")
PyPlot.legend()
PyPlot.xlabel("iter")
PyPlot.ylabel("log-space")
PyPlot.title("primal-dual gap history")
