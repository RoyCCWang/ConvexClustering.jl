import Distances

import JLD
using LinearAlgebra

import Random
using BenchmarkTools

import CSV
import TypedTables
using DelimitedFiles
import BSON

import Dates
import Statistics

#import Optim
#include("./helpers/optim.jl")

### these specify the square exponential kernel for the weight function, and how to update the length scale hyperparameter.
function lengthscale2θ(l::T)::T where T <: AbstractFloat
    return 1/(2*l^2)
end

function θ2lengthscale(θ::T)::T where T<: AbstractFloat
    return 1/sqrt(2*θ)
end

function evalgeometricsequence(n::Int, a0::T, r::T)::T where T <: AbstractFloat
    return a0*r^n
end

function evalSqExpkernel(x::Vector{T}, z::Vector{T}, θ::T)::T where T <: AbstractFloat
    return exp(-θ*norm(x-z)^2)
end

function unitweight(x::Vector{T}, z::Vector{T}, θ::T)::T where T <: AbstractFloat
    return 1.0
end

###

# reference solver.
# import Evolutionary
# const EVO = Evolutionary

import PythonPlot

import SingleLinkagePartitions
const SL = SingleLinkagePartitions

include("./helpers/data.jl")

using Revise

import ConvexClustering
const CC = ConvexClustering