################### for ϕ, dϕ! function preparation for Optim. Do simultaneous versions later.
function preparefuncs(X::Matrix{T}, Z::Matrix{T}, σ, A, w, γ, edges) where T <: AbstractFloat

    V_buf::Matrix{T} = Matrix{T}(undef, size(Z))
    prox_V_buf::Matrix{T} = Matrix{T}(undef, size(Z))
    proxconj_V_buf::Matrix{T} = Matrix{T}(undef, size(Z))

    D, N = size(X)

    # function under test.
    h = xx->compteϕXoptim!(
        #V_buf,
        #prox_V_buf,
        ##proxconj_V_buf,
        reg,
        reshape(xx,D,N),
        #Z,
        σ,
        A,
        #w,
        γ,
        edge_set,
    )

    # oracles.
    f = xx->evalϕX(reshape(xx,D,N), Z, w, edges, γ, σ, A)
    f2 = xx->evalϕX!(prox_V_buf, proxconj_V_buf, V_buf,
        reshape(xx,D,N), Z, w, edges, γ, σ, A)

    return h, f, f2
end

function preparedfuncs(
    X::Matrix{T},
    Z::Matrix{T},
    σ,
    A,
    w,
    γ,
    info::EdgeFormulation,
    ) where T <: AbstractFloat

    V_buf::Matrix{T} = Matrix{T}(undef, size(Z))
    prox_V_buf::Matrix{T} = Matrix{T}(undef, size(Z))
    proxconj_V_buf::Matrix{T} = Matrix{T}(undef, size(Z))
    grad_mat_buf::Matrix{T} = Matrix{T}(undef, size(X))

    #dh_eval = zeros(length(X))

    D, N = size(X)

    # src_nodes = collect( edge[1] for edge in edges)
    # dest_nodes = collect( edge[2] for edge in edges)

    # function under test.
    dh = (gg,xx)->computedϕoptim!(
        gg,
        V_buf,
        prox_V_buf,
        #proxconj_V_buf,
        grad_mat_buf,
        reshape(xx, D,N), Z, σ, A, w, γ,
        info.edges,
        #src_nodes, dest_nodes,
    )

    # oracles.
    df = xx->computedϕ(reshape(xx,D,N), Z, J, w, A,
        γ, σ, info.edges)

    return dh, df#, dh_eval
end
