
# # comment/uncomment to test Revise.
# function bye(x)
#     @show x
# end

# ## this is equivalent to α^N
# # a0 = 1.23
# # rate = 1.01
# # num_iters = 53
# # a0*rate^num_iters
# # a0*applyconstantrate(num_iters,rate)
# function applyconstantrate(N::Int, α::T)::T where T <: AbstractFloat
#     out = one(T)
#     for n = 1:N
#         out = out*α
#     end
#     return out
# end

function array2matrix(X::Array{Vector{T},L})::Matrix{T} where {T,L}

    N = length(X)
    D = length(X[1])

    out = Matrix{T}(undef,D,N)
    for n = 1:N
        out[:,n] = X[n]
    end

    return out
end

# more efficient than norm(X-Y,2)^2 or dot(X-Y,X-Y) for large X, Y.
function evalnorm2sq(X::Array{T,D}, Y::Array{T,D})::T where {T <: AbstractFloat, D}
   @assert length(X) == length(Y)

   # out = zero(T)
   # for i in eachindex(X)
   #    out += (X[i]-Y[i])^2
   # end
   #
   # return out
   out = clamp(convert(T, dot(X,X) + dot(Y,Y) - 2*dot(X,Y)), zero(T), convert(T, Inf))
   return out # faster.
end
# D = 10
# N = 6000
# X = randn(D,N)
# A = randn(D,N)
# @show ConvexClustering.evalnorm2sq(X,A) - norm(X-A,2)^2
# @btime ConvexClustering.evalnorm2sq(X,A)
# @btime norm(X-A,2)^2
# @btime dot(X-A, X-A)
# # got:
# 56.455 μs (1 allocation: 16 bytes)
# 91.552 μs (4 allocations: 468.83 KiB)
# 148.317 μs (5 allocations: 937.61 KiB)

# # allocation-less version of norm(X[:,c],2).
# function evalcolnorm(X::Matrix{T}, c::Int)::T where T <: AbstractFloat
#    out = zero(T)
#    for axes(X,1)
#       out += X[d,c]*X[d,c]
#    end
#
#    out = clamp(out, zero(T), convert(T,Inf))
#    return sqrt(out)
# end


## helpers for choosing γ_base.

function getmultidimσ(σ_1D::T, D::Int)::T where T
    
    # based on:
    # γ^2 = D*σ_1D^2 # strive to get the D-dim distance with the given 1D length.
    
    γ = sqrt(D*σ_1D^2)
    return γ
end