module Utils


export supertypes, finaleltype, scale_center, colorbar

using Makie, Statistics, Flux.Tracker, LinearAlgebra, LightGraphs

function supertypes(T)
    current = T
    println(current)
    while current != Any
        current = supertype(current)
        println(current)
    end
end

#finaleltype(::Type{<:AbstractArray{T}}) where {T} = T
#finaleltype(x::Type{<:TrackedArray{T}}) where {T<:Real} = T

function scale_center(x; dims = 1)
    x = x .- mean(x, dims = dims)
    norm = sqrt.(sum(x .^ 2, dims = dims))
    x ./ norm
end

logitbinarycrossentropy(logŷ, y; pos_weight = 1) = (1 - y) * logŷ + (1 + (pos_weight - 1) * y) * (log(1 + exp(-abs(logŷ))) + max(-logŷ, 0))

"""Total probability of `y` for the categorical distributions defined by softmax(unormp)"""
function softmaxcategoricallogprob(unormp, y)
    shiftedunormp = unormp .- maximum(unormp, dims = 1)
    sum(y .* (shiftedunormp .- log.(sum(exp.(shiftedunormp), dims = 1))))
end

adjacency_matrix_diag(g) = adjacency_matrix(g) + Matrix(I, size(g)...)

function markersize(xy)
    (xmin, xmax), (ymin, ymax) = extrema(xy, dims = 2)
    0.03 * max(xmax - xmin, ymax - ymin)
end

markersize(xy::TrackedArray) = markersize(xy.data)

#function colorbar(x)
#    a = range(minimum(x), stop = maximum(x), length = 100) |> collect
#    heatmap(reshape(a, 1, :))
#end


end
