# The semismooth Newton-CG augmented Lagrangian (SSNL) algorithm from  http://jmlr.org/papers/v22/18-694.html


function runSSNL()

end


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

"""
    primaldirect(X::Matrix{T}, w::Vector{T}, edge_pairs, γ::T, A::Matrix{T})::T where T <: AbstractFloat
"""
function primaldirect(X::Matrix{T}, w::Vector{T}, edge_pairs, γ::T, A::Matrix{T})::T where T <: AbstractFloat

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
    term2 = term2*γ

    return term1 + term2
end
