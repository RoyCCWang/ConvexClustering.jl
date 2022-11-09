module ConvexClustering

    #using SparseArrays
    using LinearAlgebra
    import NearestNeighbors, Graphs, Distances, Optim

    include("./types.jl")
    include("./utils.jl")

    include("./solvers/optim.jl")

    include("./setup_problem.jl")
    include("./solve_problem.jl")

    include("./assignment/via_connections.jl")

    include("./ALM/ALM_engine.jl")
    include("./ALM/primal.jl")
    include("./ALM/operator_B.jl")
    include("./ALM/proximal.jl")
    include("./ALM/subproblem17.jl")

    #include("./ALM/subproblem/SSNCG/nonlinear_solve.jl")

    include("./settings/search.jl") # search strategies for various hyperparameters given various stopping conditions.

    include("./partition_tree/check_nested.jl")

    export setupproblem,
        computeweights,
        buildgraph,

        runALM, # solves the convex clustering optimization problem
        assignviaX, # get assignment
        applyassignment,
        primaldirect, # evaluate the convex clustering cost function. No thte ALM primal function, even though it and its constraints form an equivalent optimization problem.

        # frontend for search strategies given some stopping conditions for reguliarzation or weight function hyperparameters.
        searchkernelparameters,
        searchγ,
        runconvexclustering,

        # search strategies for knn or radius to build the graph. Use with KNNSearchType, RadiusSearchType.
        searchknn,
        searchradius,

        # data types and constructors.
        ALMConfigType,
        WeightedGraphConfigType,
        ProblemType,
        KNNType,
        RadiusType,
        KNNSearchType,
        RadiusSearchType,

        # check if partition is nested in another partition. Useful when constructing partition trees based on multiple runs of convex clustering.
        isnestedin,
        isnestedsuccessively
end
# The internal routines of this package assumes and is based on 1-indexing arrays. This is done for algorithm index-consistency across buffer objects, ease of implementation, and readability. For example, saving the column indices of data points that have positive weight is easiest done by sticking with a index convention.

# The optimization code in the ALM folder uses the augmented Lagragian method formulation presented in the following article:
# - (Sun, JMLR 2021): Sun, D., Toh, K. C., & Yuan, Y. (2021). Convex Clustering: Model, Theoretical Guarantee and Efficient Algorithm. J. Mach. Learn. Res., 22(9), 1-32.
# However, we allow the user to supply a generic numerical optimizer to solve the subproblem in this package. The conjugate gradient optimizer from Optim.jl is used as the default if the user doesn't supply a method. The actual algorithm used by Sun to solve the subproblem might be implemented in the future.

# The partition assignment algorithm:
# Every data point is a node in an undirected graph. Every data poinrt `i` is checked to see if the returned cluster center (`X_star[:,i]` in the example script `/examples/run_clustering.jl`) is within ϵ-distance (`metricfunc` in the example script) of a the cluster center (`X_star[:,j]`) of another data point `j`. If so, assigned an edge between node i and j, and repeat this for j ≠ i, j ∈ [number of data points], i ∈ [number of data points]. After this graph construction process, a cluster is defined as a connected component of the graph.
# Although the proximity-based comparison is non-transitive and thus is not a mathematical relation between data points, taking the connective components as a cluster would make this cluster assignment method invariant to permutations of data point labels.

# Nomenclature on partition, parts, etc.
# see section 2.1 of Introductory Combinatorics, 5th ed, 2008, Richard A. Brualdi.

# For partition tree-related terms and concepts as we vary γ, see:
# - (Chi, SIAM 2019) Chi, Eric C., and Stefan Steinerberger. "Recovering trees with convex clustering." SIAM Journal on Mathematics of Data Science 1.3 (2019): 383-407.

# We avoid statistical terminologies such as `clustering` in this package, because this package is about solving the "convex clustering" algorithm, which despite its name, is an algorithm that uniquely obtains a partion given unique non-negative weights and a set of points. Any statistical interpretation requires the points to be viewed as data generated by some model, which is out of the scope of the algorithm. One should start another package that builds on top of this one to properly incorporate additional statistical formulation to turn this algorithm into a subroutine of a data point clustering algorithm.
# The size of a partition is the number of subsets in the partition. If a partition is interpreted as a clustering assignment, then the size of the partition is the number of distinct clusters.
