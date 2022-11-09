##### checking routines for whether the partitions generated by a γ search sequence is a partition tree.
# See (Chi, SIAM 2019) for concept.


"""
```
isnestedin(X::Vector{Vector{Int}}, Y::Vector{Vector{Int}})
```
`X` and `Y` are each a partition of some set of points. They each contain sets of indices (1-indexing) to the set of points. Each of these subsets is called a part, an element of a partition.

Ouputs (in order):
- `all(is_nested_flags)`
a boolean indicating whether every part in `X` are contained in or equal to a part in `Y`. If true, `Y` should be the same size or smaller than the size of `X`.

- `is_nested_flags`
a list of booleans that indicate whether each part in `X` is contained in or equal to a part in `Y`. The first output is 
"""
function isnestedin(X::Vector{Vector{Int}}, Y::Vector{Vector{Int}})

    is_nested_flags = falses(length(X))
    
    for i in eachindex(X)
        x = X[i]
        is_nested = collect( all(x[n] in y for n in eachindex(x)) for y in Y )
        is_nested_flags[i] = any(is_nested)
    end
    
    return all(is_nested_flags), is_nested_flags
end

"""
```
isnestedsuccessively(Gs::Vector{Vector{Vector{Int}}})
```

Designed for use with the `Gs` returned output of `searchγ()`, or a similarly ordered list of partitions. If length of `Gs` is `N`, returns a list of `N-1` booleans that indicate whether the `n`-th partition of `Gs` (i.e. the `n`-th element of `Gs`) is a nested partition of the `n+1`-th partition.
"""
function isnestedsuccessively(Gs::Vector{Vector{Vector{Int}}})
    #
    N_searches = length(Gs)
    is_nested_with_next = falses(N_searches-1)

    for n = 1:length(Gs)-1
        is_nested_with_next[begin+n-1], _ = iscontainedin(Gs[begin+n-1], Gs[begin+n-1+1])
    end

    return is_nested_with_next
end