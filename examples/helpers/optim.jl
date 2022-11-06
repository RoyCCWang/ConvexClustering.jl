



"""
```
runOptimjl(x_initial::Vector{T},
    f::Function,
    df!::Function, g_tol::T;
    x_tol::T = zero(T),
    f_tol::T = zero(T),
    max_time::T = convert(T, Inf),
    max_iters::Int = 100000,
    lp::Int = 2,
    verbose::Bool = false) where T <: AbstractFloat
```
"""
function runOptimjl(x_initial::Vector{T}, f::Function, df!::Function, g_tol::T;
    x_tol::T = zero(T),
    f_tol::T = zero(T),
    max_time::T = convert(T, Inf),
    max_iters::Int = 100000,
    lp::Int = 2,
    verbose::Bool = false) where T <: AbstractFloat

    @assert g_tol > zero(T)

    optim_config = Optim.Options(x_tol = x_tol,
        f_tol = f_tol,
        g_tol = g_tol,
        iterations = max_iters,
        time_limit = max_time)

    ret = Optim.optimize(f, df!, x_initial, Optim.ConjugateGradient(), optim_config)

    x_star::Vector{T} = convert(Vector{T}, Optim.minimizer(ret))

    df_x_star = similar(x_initial)
    df!(df_x_star, x_star)
    df_norm::T = convert(T, norm(df_x_star, lp))

    if verbose
        @show ret
        println()
    end

    return x_star, df_norm
end