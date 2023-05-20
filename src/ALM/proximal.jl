# can make this faster by update by reference.
# from Table 1, Sun et al, JMLR 2021.
function proximaltnorm2(u::Vector{T}, t::T) where T

    c = 1-t/norm(u,2)
    if c > zero(T)
        return c .* u
    end

    return zeros(T, length(u))
end

# p(.) as defined on pg 15, primal formulation of problem, Sun et. al. JMLR 2021.
# s = γ*σ
# using Euclidean norm.
function proximaltp(U::Matrix{T}, W::Vector{T}, γ::T, s::T) where T
    @assert length(W) == size(U,2)

    a = γ*s

    K = Matrix{T}(undef, size(U))
    for l in axes(U,2)
        K[:,l] = proximaltnorm2(U[:,l], W[l]*a)
        #K[:,l] = s .* proximaltnorm2(U[:,l], W[l]*γ)
    end

    return K
end

# faster version and in-place version of proximaltp().
function proximaltp!(
    K::Matrix{T},
    U::Matrix{T},
    w::Vector{T},
    γ::T,
    s::T,
    ) where T

    @assert length(w) == size(U,2)
    @assert size(K) == size(U)

    a = γ*s
    fill!(K, zero(T))

    for l in axes(U,2)
        #c = one(T)-w[l]*a/norm(U[:,l],2) # t is w[l]*a
        c = one(T)-w[l]*a/sqrt(sum( U[d,l]*U[d,l] for d in axes(U,1))) # t is w[l]*a

        if c > zero(T)
            for d in axes(U,1)
                K[d,l] = c*U[d,l]
            end
        end
    end

    return nothing #K
end

"""
proximalconjugatetnorm2(y::Vector{T}, t::T) where T

Implements:
Prox_{convex conjugate of (t*norm(.,2)) (y).
"""
# from pg 7, Sun et al, JMLR 2021.
function proximalconjugatetnorm2(y::Vector{T}, t::T) where T
    norm_y = norm(y,2)
    if norm_y > t
        return (t/norm_y) .* y
    end

    return y
end

# assumes p uses Euclidean norm.
function proximaltpconj(Y::Matrix{T}, W::Vector{T}, γ::T, s::T) where T
    @assert length(W) == size(Y,2)

    a = γ*s

    K = Matrix{T}(undef, size(Y))
    for l in axes(Y,2)
        K[:,l] = proximalconjugatetnorm2(Y[:,l], W[l]*a)
    end

    return K
end

# faster version and in-place version of proximaltpconj().
function proximaltpconj!(K::Matrix{T},
    Y::Matrix{T}, w::Vector{T}, γ::T, s::T) where T

    @assert length(w) == size(Y,2)
    @assert size(K) == size(Y)

    a = γ*s
    for l in eachindex(Y)
        K[l] = Y[l]
    end

    for l in axes(Y,2)
        # norm_y = norm(Y[:,l],2)
        # t = w[l]*a
        # c = t/norm_y
        #c = w[l]*a/norm(Y[:,l],2)
        c = w[l]*a/sqrt(sum( Y[d,l]*Y[d,l] for d in axes(Y,1)))

        #if norm_y > t
        if one(T) > c
            for d in axes(Y,1)
                K[d,l] = c*Y[d,l]
            end
        end
    end

    return nothing
end

function proximaltpconjgivenproximaltp!(K::Matrix{T},
    V::Matrix{T}, prox_V::Matrix{T}) where T
    @assert size(K) == size(V)

    for i in eachindex(V)
        K[i] = V[i] - prox_V[i]
    end

    return nothing
end
