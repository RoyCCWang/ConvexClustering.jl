
# test tutorial: https://www.matecdev.com/posts/julia-testing.html

using LinearAlgebra
using SparseArrays
include("../src/ConvexClustering.jl")
import .ConvexClustering

#import ConvexClustering
import Graphs
import NearestNeighbors
import Distances

include("../examples/helpers/generatesetup.jl")
include("../examples/helpers/utils.jl")

include("./BPB.jl")

using Test



# # Linear system evaluation: LHS equation 23, Sun et. al JMLR 2021.
# @testset "B ∘ P ∘ B" begin
#     # Write your tests here.
#
#     ZERO_TOL = 1e-12
#
#     # hyperparameters.
#     γ = rand()
#     σ = rand()
#     D = 10
#     N = 6001
#     α_w = 1.0
#     wfunc = (xx,zz)->exp(-α_w*norm(xx-zz)^2)
#
#     # control number of randomized trials to run here.
#     N_linear_sys_eval_trials = 10
#     N_derivative_eval_trials = 9
#     N_data_trials = 3
#
#     # run test.
#     max_descrepancy = runtrialsBadjPB(D, N,
#         N_linear_sys_eval_trials,
#         N_derivative_eval_trials,
#         N_data_trials,
#         γ, σ;
#         clustering_radius = 1.5)
#     @show max_descrepancy
#     @test max_descrepancy < ZERO_TOL
#
# end

@testset "Mordeau identity" begin
    # Write your tests here.

    ZERO_TOL = 1e-12

    # hyperparameters.
    t = rand()

    D = 10
    N_edges = 8001
    W = rand(N_edges)


    # control number of randomized trials to run here.
    N_s_trials = 100
    N_W_trials = 10
    N_U_trials = 10

    discrepancies_s = ones(Float64, N_s_trials)
    fill!(discrepancies_s,Inf)

    for nt = 1:N_s_trials # hyperparameters
        s = rand()

        discrepancies_w = ones(Float64, N_W_trials)
        fill!(discrepancies_w,Inf)

        for nw = 1:N_W_trials

            W = rand(N_edges)
            γ = rand()

            discrepancies = ones(Float64, N_U_trials)
            fill!(discrepancies,Inf)

            for nu = 1:N_U_trials
                U = randn(D, N_edges)

                Mordeau_LHS = ConvexClustering.proximaltp(U, W, γ, s) + s .* ConvexClustering.proximaltpconj(U ./ s, W, γ, 1.0)
                discrepancies[nu] = norm(Mordeau_LHS - U)

                u = U[:,1]
                discrepancies[nu] += norm(s .* ConvexClustering.proximalconjugatetnorm2(u ./ s, 1.0) - ConvexClustering.proximalconjugatetnorm2(u, s))

            end

            discrepancies_w[nw] = maximum(discrepancies)
        end

        discrepancies_s[nt] = maximum(discrepancies_w)
    end
    max_descrepancy = maximum(discrepancies_s)

    # run test.
    @show max_descrepancy
    @test max_descrepancy < ZERO_TOL

end
