
# randomly generate a convex clustering problem, i.e. equation 2 from (Sun, JMLR 2011).
function generaterandomsetup(D::Int, N::Int, θ;
    radius::Float64 = -Inf,
    knn::Int = 30,
    metric::Distances.Metric = Distances.Euclidean(),
    kernelfunc = (xx,zz,tt)->exp(-tt*norm(xx-zz)^2))

    A_vecs::Vector{Vector{Float64}} = collect( randn(Float64, D) for j = 1:N)

    return ConvexClustering.setupproblem(A_vecs, θ;
        radius = radius, knn = knn, metric = metric, kernelfunc = kernelfunc)
end

function generaterandomposdefmatrix(D::Int)
    C = randn(Float64, D,D)
    return C'*C
end
