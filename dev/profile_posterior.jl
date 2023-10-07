# build proxy to profile posterior. Work in progress.

### posterior-related.
#import Interpolations # for building profile posterior
import RKHSRegularization
RKReg = RKHSRegularization

import PyPlot # for visualizing posterior.

import HCubature # for computing the normalizing constant of proxy posterior.
#### end posterior-related

T = Float64
fig_num = 1

include("./helpers/two_layer.jl")

project_folder = joinpath(homedir(), "convex_clustering/assembled_partition_trees")

dic = BSON.load(
    joinpath(project_folder, "full_cell_line.bson"),
)

Gs = convert(Vector{Vector{Vector{Int}}}, dic[:Gs])
Xs = convert(Vector{Matrix{T}}, dic[:Xs])
γs = convert(Vector{T}, dic[:γs])

A = convert(Matrix{T}, dic[:A])
w = convert(Vector{T}, dic[:w])
edges = convert(Vector{Tuple{Int,Int}}, dic[:edges])

LUT = Dict{Vector{Int}, Vector{Vector{Vector{Int}}}}()

# I am here. getting Q from F_A needs a bit of thought. 
#p, lb, ub, Z, err_Z = constructposteriorproxy(Xs, γs, A, edges, w)

# debug
lb = minimum(γs)
ub = maximum(γs)
# end debug.

ys = collect( ConvexClustering.primaldirect(Xs[n], w, edges, γs[n], A) for n in eachindex(Xs) )


##### regression: ignoring prror and using only likelihood (primaldirect costfunc) for now to make sure code runs.
#θ = Spline34KernelType(0.2)
θ = RKReg.BrownianBridge10(1.0)
#θ = BrownianBridge20(1.0)
#θ = BrownianBridge1ϵ(4.5)
#θ = BrownianBridge2ϵ(2.5)

σ² = 1e-5

X = collect( [γs[n]] for n in eachindex(γs) )
y = ys

# check posdef.
K = RKReg.constructkernelmatrix(X, θ)
println("rank(K) = ", rank(K))
println("isposdef = ", isposdef(K))

# fit RKHS.
η = RKReg.RKHSProblemType(
    zeros(T,length(X)),
    X,
    θ,
    σ²,
)
RKReg.fitRKHS!(η, y)

# query.
Nq = 100
xq_range = LinRange(lb, ub, Nq)
xq = collect( [xq_range[n]] for n = 1:Nq )

yq = Vector{T}(undef, Nq)
RKReg.query!(yq,xq,η)

# Visualize regression result.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(X, y, ".", label = "observed")
PyPlot.plot(xq, yq, label = "fit")

#PyPlot.plot(xq, f_xq, label = "true")

title_string = "unnormalized negative log likelihood"
PyPlot.title(title_string)
PyPlot.legend()
##### end of regression


PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(X, exp.(-y), ".", label = "observed")
PyPlot.plot(xq, exp.(-yq), label = "fit")

title_string = "unnormalized posterior"
PyPlot.title(title_string)
PyPlot.legend()

nothing