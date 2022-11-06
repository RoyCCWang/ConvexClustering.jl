# ### package install. comment out if you already have these installed.
# # if you are missing any of the libraries in this script, run the following lines in your Julia REPL.
# using Pkg
# Pkg.add("MAT")
# Pkg.add(url="https://github.com/RoyCCWang/ConvexClustering.jl")
# Pkg.add("Distances")
# Pkg.add("PyPlot")
# ### end of package install

# include("../../src/ConvexClustering.jl")
# import .ConvexClustering
import ConvexClustering
#import Optim

# run the convex clustering algorithm on a data matrix (columns are data point indices, rows are dimensions)
# We use the notation from the convex clustering optimization algorithm article, Sun JMLR 2021, http://jmlr.org/papers/v22/18-694.html . We abbreviate this reference by (Sun 2021) in this example script.
# Although the current impelemntation of ConvexClustering.runALM() uses the problem setup by (Sun 2021), but it does not use their proposed algorithm. In theory, any non-smooth convex optimization algorithm can solve the convex clustering problem.

# The Julia REPL is like the command prompt in MATLAB desktop.
# Use any text editor (notepad, Atom, VS code, etc.), and run Julia commands or scripts from the Julia REPL from an operating system shell (e.g. command prompt, power shell, bash terminal, etc.).
#   - Type long commands into scripts, then run the sciprt by the command
#   include("myscript.jl")
#   in the Julia REPL.
#   -- quick way to run the entire script is to type "include("load_MAT_tutorial.jl")" in the Julia REPL.
#   - This workflow avoids the need to learn open source IDEs with features that might change over time.

# This is an quick and informative introduction to the Julia REPL: https://www.youtube.com/watch?v=C3ro2b5tQws
# You'll want to try out VS Code after the above video:
#   - VS Code setup video and basic visualization: https://www.youtube.com/watch?v=7M8e2Q5BirA&list=PLhQ2JMBcfAsjZTA8_jGhz3BVqYgOeyyeu

# For a comprehensive series on Julia programming and various quantitative sciences topics, see these YouTube channels:
# Informative and practical tutorials: https://www.youtube.com/c/juliafortalentedamateurs/playlists
# Basic tutorials: https://www.youtube.com/c/remivezy/videos

# To call Julia functions (such as the contents of this script or just a command like include("myscript.jl")) from MATLAB, one can use MATDaemon.jl. The user-facing part of the package is actually a MATLAB script that starts a type of server that runs Julia. Variables transfers use .MAT files, like an automated version of this script.
# Tutorial: https://morioh.com/p/cbb5ead00e1e
# Announcement video: https://www.youtube.com/watch?v=ob2eFMfa5YA
# Repository: https://github.com/jondeuce/MATDaemon.jl


using LinearAlgebra

import Random
Random.seed!(25)



# for assignment and weights-graph generation.
import Distances

# for loading MATLAB's .MAT file.
import MAT

# visualize.
import PyPlot


# set up PyPlot plotting preferences.
PyPlot.matplotlib["rcParams"][:update](["font.size" => 16, "font.family" => "serif"])
PyPlot.close("all")
fig_num = 1



# use Optim.jl as optimizer. ConvexClustering will have its own optimizer in the future.
include("./helpers/optim.jl") # replace this line with the contents of this script if you get a path-related error.

### load data from A.mat in this directory.

# The data in A.mat are draws from a mixture of three 2D multivariate normal distributions.
dict_all_MAT_variables = MAT.matread("./files/A.mat")
A_MAT = convert(Matrix{Float64}, dict_all_MAT_variables["A"]) # make explicit that we want the data type of A to be a 2D array of 64-bit floating point numbers.

A_vecs = collect( A_MAT[:,j] for j in axes(A_MAT,2) ) # j iterates through the 2nd dimension (columns) of the multi-dimensional (2D) array A_mat.



### setup problem: guild the graph. Refer to page 3 of (Sun, 2021).
D, N = size(A_MAT)
metric = Distances.Euclidean()

# this is the ϕ hyperparameter in the Gaussian kernel function on page 3 of (Sun 2021).
θ_w = 0.4 # modify to affect how many clusters convex clustering returns.

# The Gaussian kernel function is used, but any kernel function should work.
kernelfunc = (xx,zz,tt)->exp(-tt*norm(xx-zz)^2) # xx, zz are input slots. tt is the hyperparameter slot.

# a different choice of knn, kernelfunc, θ_w (or any
# If those parameters are not changed, then the convex clustering optimization problem will recover an unique global minimum after running long enough, if all data points are dihyperparameters of the chosen kernelfunc), and γ change the convex clustering optimization problem, and can yield different cluster assignments.stinct, due to the optimization problem being convex. Therefore if a different cluster assignment is desired, one should change the parameters that change the optimization problem.

# package up config parameters.
connectivity = ConvexClustering.KNNType(30) # make an edge for this number of nearest neighbours of a given point i, cycle through all i in the point set to be partitioned. Takes a positive integer.
#connectivity = ConvexClustering.RadiusType(1.0) # make an edge for all points within this radius of a given point i, cycle through all i in the point set to be partitioned. Takes a finite floating point number.
graph_config = ConvexClustering.WeightedGraphConfigType(connectivity, metric, kernelfunc)

# compute the graph quantities.
A, edge_pairs, w, A_neighbourhoods = ConvexClustering.setupproblem(A_vecs,
    θ_w, graph_config)

@show norm(A-A_MAT) # should be zero.

# Each edge gets a weight.
# Use these to decide whether to change θ_w if you want to induce a larger/smaller dynamic range of edge weights in the optimization problem.
@show maximum(w) # maximum weight value.
@show minimum(w) # minimum weight value.

N_edges = length(edge_pairs)
@show N_edges # this is |math_caligraphy{E}| on page 3 of (Sun, 2021), the number of edges of the graph that encodes the neighbourhood relationships of the data points in A.

### setup problem: specify the regularization parameter.
# regularization parameter, positive real number.
# This is γ from equation 2 of (Sun JMLR 2021, http://jmlr.org/papers/v22/18-694.html )
#   If γ is 0, every data point (column of A) will be a cluster.
#   If γ approaches infinite, all data points are in a single cluster.
γ = 14.1 # around 2 clusters
# γ = 3.0 # more clusters

### setup the optimization algorithm.
# algorithm convergence parameters.
# Rule of thumb: choose the following such that updateσfunc(cc_max_iters) < a large number like 1e6.
# If your problem doesn't get solved, you can always start another optimization run with the solution (X_star, Z_star) as the initial guess (X0, Z0).
σ_base = 0.4
σ_rate = 1.05
cc_max_iters = 200 # the maximum number of ALM (outer optimization) iterations allowed.
gap_tol = 1e-6 # stopping condition. should be close to zero for attaining close to the global minimum..

updateσfunc = nn->σ_base*σ_rate^nn
updateϵfunc = nn->1/(nn)^2
#runoptimfunc = (xx,ff,dff,gg_tol)->runOptimjl(xx, ff, dff, gg_tol)

# package up.
config = ConvexClustering.ALMConfigType(
    gap_tol;
    #runoptimfunc;
    max_iters = cc_max_iters,
    updateσfunc = updateσfunc,
    updateϵfunc = updateϵfunc
)

problem = ConvexClustering.ProblemType(A, γ, w, edge_pairs)

# initial guess. Must be finite-valued.
X0 = zeros(Float64, D, N) # primal variable. if want to load from file "X_star.mat", do this instead: (MAT.matread("X_star.mat"))["X_star]
Z0 = zeros(Float64, D, N_edges) # dual variable. if want to load from file "Z_star.mat", do this instead: (MAT.matread("Z_star.mat"))["Z_star]

### run the optimization algorithm.
println("Timing: runALM")
@time ret = ConvexClustering.runALM(X0, Z0, problem, config; store_trace = true)

X_star, Z_star, num_iters_ran, gaps, trace = ret.X_star, ret.Z_star, ret.num_iters_ran, ret.gaps, ret.trace
@show num_iters_ran
@show gaps # The returned primal, dual, primal-dual gaps of the solution. should be all less than gap_tol if the optimization was successful. If not, re-run the optimization with X_star, Z_star as X0, Z0.

# save solution to MAT file.
file = MAT.matopen("./files/X_star.mat", "w")
MAT.write(file, "X_star", X_star)
MAT.close(file)

file = MAT.matopen("./files/Z_star.mat", "w")
MAT.write(file, "Z_star", Z_star)
MAT.close(file)

import MAT
file = MAT.matopen("./files/A.mat", "w")
MAT.write(file, "A", A)
MAT.close(file)

###### cluster assignment given optimization solution.

assignment_zero_tol = 1e-6 # shouldn't make this larger. Instead, change the optimization problem parameters γ, θ_w (or even use a different type of kernel function for kernelfunc), knn.

# G is the cluster assignments in terms of data point indices (column indices of A). Type: Vector{Vector{Int}}
# if j == G[k][n], then it means the n-th data point in the k-th cluster is data point label j (i.e. the datapoint corresponding to the j-th column of A)
G, g_neighbourhoods = ConvexClustering.assignviaX(X_star, metric;
    zero_tol = assignment_zero_tol)

# This is like G but contains the actual data point instead of the data point label/index.
#   i.e, if p == P[k][n] and j == G{k}[n], then it means p == A[:,j].
P = ConvexClustering.applyassignment(G, A_vecs)

# save to MAT file. Nested arrays like P and G are stored as cell arrays.
file = MAT.matopen("./files/G.mat", "w")
MAT.write(file, "G", G)
MAT.close(file)

file = MAT.matopen("./files/P.mat", "w")
MAT.write(file, "P", P)
MAT.close(file)


#### diagnostics on optimization run.

trace_gaps = trace.gaps
trace_evals = trace.problem_cost

p_gap_history = collect( trace_gaps[i][1] for i in eachindex(trace_gaps) )
d_gap_history = collect( trace_gaps[i][2] for i in eachindex(trace_gaps) )
pd_gap_history = collect( trace_gaps[i][3] for i in eachindex(trace_gaps) )


#### visualize diagnostics.
ln_trace_evals = log.(trace_evals)
ln_p_gap_history = log.(p_gap_history)
ln_d_gap_history = log.(d_gap_history)
ln_pd_gap_history = log.(pd_gap_history)

# ln of optimization cost.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(ln_trace_evals, label = "ln_trace_evals")
PyPlot.plot(ln_trace_evals, "o")

PyPlot.legend()
PyPlot.xlabel("iter")
PyPlot.ylabel("Natural log space")
PyPlot.title("NOptimization cost")

# ln of primal gap. Should decrease to -Inf. In practice it might fluctuate once small enough due to numerical precision issues.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(ln_p_gap_history, label = "ln_p_gap_history")
PyPlot.plot(ln_p_gap_history, "o")

PyPlot.legend()
PyPlot.xlabel("iter")
PyPlot.ylabel("Natural log space")
PyPlot.title("Primal gap history")

# ln of dual gap. Should decrease to -Inf. Same issue with numerical precision applies.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(ln_d_gap_history, label = "ln_d_gap_history")
PyPlot.plot(ln_d_gap_history, "o")

PyPlot.legend()
PyPlot.xlabel("iter")
PyPlot.ylabel("Natural log space")
PyPlot.title("Dual gap history")


# ln of primal-dual gap. Should decrease to -Inf. Same issue with numerical precision applies.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(ln_pd_gap_history, label = "ln_pd_gap_history")
PyPlot.plot(ln_pd_gap_history, "o")
PyPlot.legend()
PyPlot.xlabel("iter")
PyPlot.ylabel("Natural log space")
PyPlot.title("Primal-dual gap history")
