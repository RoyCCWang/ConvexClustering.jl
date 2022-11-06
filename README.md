# ConvexClustering.jl

See `/examples/run_clustering.jl` for an example of solving the weighted convex clustering optimization problem.

See `/examples/load_MAT_tutorial.jl` for an example that loads a MATLAB data file `A.mat` and saves the solved cluster assignments to MATLAB data files. `A` is a `D x N` matrix, where `D` is the dimension of a data point, and `N` is the number of data points.

See `/examples/gamma_search.jl` for an example of searching for the first regularization parameter `γ` that returns back a partition size that is smaller than the prescribed `max_partition_size`. It also searches for the first weight function hyperparameter `length_scale` such that the dynamic range of the weights `w` in the problem setup is larger than the prescribed `min_dynamic_range`. A user-specified search strategy as a function of the number of iterations should be specified for doing the search for each of these parameters. The example uses the geometric sequence with a user-specified base and rate values for the `γ`, and a similar strategy for searching the `length_scale` hyperparameter.

See `/examples/optim.jl` for an example of the user-supplied external optimizer, or just use the one in that script. It is not part of this package so that users can have the flexibility to use their chosen solver. ConvexClustering.jl will have its own default solver in the future.