module Utils


export mean, scale_center, softmaxcategoricallogprob, logitbinarycrossentropy, adjacency_matrix_diag, randn_like, markersize

using Statistics, Flux.Tracker, LinearAlgebra, LightGraphs
import Statistics.mean


#
# Introspection helpers
#

function supertypes(T)
    current = T
    println(current)
    while current != Any
        current = supertype(current)
        println(current)
    end
end


#
# Neural network helpers
#

function scale_center(x; dims = 1)
    x = x .- mean(x, dims = dims)
    norm = sqrt.(sum(x .^ 2, dims = dims))
    x ./ norm
end

"""Total probability of `y` for the categorical distributions defined by softmax(unormp)"""
function softmaxcategoricallogprob(unormp, y)
    shiftedunormp = unormp .- maximum(unormp, dims = 1)
    sum(y .* (shiftedunormp .- log.(sum(exp.(shiftedunormp), dims = 1))))
end

logitbinarycrossentropy(logŷ, y; pos_weight = 1) = (1 - y) * logŷ + (1 + (pos_weight - 1) * y) * (log(1 + exp(-abs(logŷ))) + max(-logŷ, 0))
adjacency_matrix_diag(g) = adjacency_matrix(g) + Matrix(I, size(g)...)
randn_like(target::A) where A<:AbstractArray{T} where T = randn(T, size(target))
mean(a::AbstractArray...) = sum(a) / length(a)


#
# Plotting helpers
#

function markersize(xy)
    (xmin, xmax), (ymin, ymax) = extrema(xy, dims = 2)
    0.03 * max(xmax - xmin, ymax - ymin)
end

markersize(xy::TrackedArray) = markersize(xy.data)


end
