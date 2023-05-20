
Random.seed!(25)

D = 250
N = 1000
N_edges = 5000

edges = collect( (rand(1:N), rand(1:N)) for _ = 1:N_edges)
src_nodes::Vector{Int} = collect( edge[1] for edge in edges)
dest_nodes::Vector{Int} = collect( edge[2] for edge in edges)

X = randn(D,N)
Z = randn(D,N_edges)


# timing test.
out = similar(X)
CC.applyJt!(out, Z, edges)

out2 = similar(X)
CC.applyJt!(out2, Z, src_nodes, dest_nodes)

@show norm(out-out2)

@btime CC.applyJt!(out, Z, edges)
@btime CC.applyJt!(out2, Z, src_nodes, dest_nodes)

@btime CC.applyJt!(out2, Z, src_nodes, dest_nodes)
@btime CC.applyJt!(out, Z, edges)

nothing