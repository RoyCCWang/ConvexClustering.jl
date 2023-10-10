

#### primal.

function primalproblem!(BX::Matrix{T},
    X::Matrix{T}, w::Vector{T}, edge_pairs, γ::T, A::Matrix{T})::T where T <: AbstractFloat
    #
    term1 = (dot(X,X) + dot(A,A) - 2*dot(X,A))/2 # norm(X-A,2)^2

    #U = B(X) = X*J
    evalB!(BX, X, edge_pairs)
    p_Y = evalp(BX, w, γ)

    return term1 + p_Y
end

function evalprimal(X::Matrix{T}, problem::ProblemType, γ::T)::T where T <: AbstractFloat

    A, _, E = ConvexClustering.unpackspecs(problem)
    return evalprimal(X, E.w, E.edges, γ, A)
end

function evalprimalterms(X::Matrix{T}, problem::ProblemType)::Tuple{T,T} where T <: AbstractFloat

    A, _, E = ConvexClustering.unpackspecs(problem)
    return evalprimalterms(X, E.w, E.edges, A)
end

"""
evalprimal(
    X::Matrix{T},
    w::Vector{T},
    edge_pairs::Vector{Tuple{Int,Int}},
    γ::T,
    A::Matrix{T},
    )::T where T <: AbstractFloat
"""
function evalprimal(
    X::Matrix{T},
    w::Vector{T},
    edge_pairs::Vector{Tuple{Int,Int}},
    γ::T,
    A::Matrix{T},
    )::T where T <: AbstractFloat

    data_fidelity, regularization = evalprimalterms(X, w, edge_pairs, A)

    return data_fidelity + γ*regularization
end

"""
evalprimalterms(
    X::Matrix{T},
    w::Vector{T},
    edge_pairs::Vector{Tuple{Int,Int}},
    A::Matrix{T},
    )::T where T <: AbstractFloat
"""
function evalprimalterms(
    X::Matrix{T},
    w::Vector{T},
    edge_pairs::Vector{Tuple{Int,Int}},
    A::Matrix{T},
    )::Tuple{T,T} where T <: AbstractFloat

    #
    term1 = (dot(X,X) + dot(A,A) - 2*dot(X,A))/2 # norm(X-A,2)^2

    running_sum = NaN
    tmp = NaN

    term2 = zero(T)
    for l in eachindex(edge_pairs)
        i,j = edge_pairs[l]

        #term2 += w[l]*norm(X[:,i]-X[:,j], 2)
        running_sum = zero(T)
        for d in axes(X,1)
            tmp = X[d,i]- X[d,j]
            running_sum += tmp*tmp
        end
        term2 += w[l]*sqrt(running_sum)
    end
    
    return term1, term2
end
