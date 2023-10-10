
# # comment/uncomment to test Revise.
# function hi(x)
#     @show x
# end


function array2matrix(X::Array{Vector{T},L})::Matrix{T} where {T,L}

    N = length(X)
    D = length(X[1])

    out = Matrix{T}(undef,D,N)
    for n = 1:N
        out[:,n] = X[n]
    end

    return out
end

function plot2Dclusters(
    P::Vector{Vector{Vector{T}}},
    d1::Int,
    d2::Int,
    fig_num::Int;
    # legend_position = :outertopright,
    # canvas_size = (1000,1000),
    # line_width = 2,
    # prefix_string = "part",
    x_label = "dimension $d1",
    y_label = "dimension $d2",
    title = "Points labelled with its parts index for dimensions $d2 vs. $d1") where T <: AbstractFloat

    #x1 = collect( first(P)[n][d1] for n in eachindex(first(P)) )
    #x2 = collect( first(P)[n][d2] for n in eachindex(first(P)) )

    x1 = collect( p[d1] for p in Iterators.flatten(P) )
    x2 = collect( p[d2] for p in Iterators.flatten(P) )

    parts_label = collect( ones(Int, length(P[k])) .* k for k in eachindex(P) )
    c = collect( Iterators.flatten(parts_label) )

    PythonPlot.figure(fig_num)
    fig_num += 1
    
    PythonPlot.scatter(
        x = x1,
        y = x2,
        #s = profit_margin_orange * 10, # size.
        c = c, # colour.
    )

    PythonPlot.xlabel(x_label)
    PythonPlot.ylabel(y_label)
    PythonPlot.title(title)

    return fig_num
end

# no error-checking on inputs.
# P must be a standard 1-indexing scheme array.
# function plot2Dclusters(P::Vector{Vector{Vector{T}}},
#     d1::Int, d2::Int;
#     legend_position = :outertopright,
#     canvas_size = (1000,1000),
#     line_width = 2,
#     prefix_string = "part",
#     x_label = "dimension $d1",
#     y_label = "dimension $d2",
#     title = "Points labelled with its parts index for dimensions $d2 vs. $d1") where T <: AbstractFloat

#     x1 = collect( first(P)[n][d1] for n in eachindex(first(P)) )
#     x2 = collect( first(P)[n][d2] for n in eachindex(first(P)) )

#     plot_handle = Plots.plot(x1, x2,
#         seriestype = :scatter,
#         #labels = display_labels,
#         label = "$prefix_string 1",
#         title =  title,
#         xlabel = x_label,
#         ylabel = y_label,
#         linewidth = line_width, legend = legend_position, size = canvas_size)

#     for k = 2:length(P)
#         x1 = collect( P[k][n][d1] for n in eachindex(P[k]) )
#         x2 = collect( P[k][n][d2] for n in eachindex(P[k]) )

#         Plots.plot!(x1, x2, seriestype = :scatter, label = "$prefix_string $k")
#     end

#     return plot_handle
# end
