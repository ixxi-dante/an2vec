module Utils


export mean, scale_center, softmaxcategoricallogprob, logitbinarycrossentropy, adjacency_matrix_diag, randn_like, markersize, loadparams!

using Statistics, Flux, LinearAlgebra, LightGraphs
import Statistics.mean
import Flux.loadparams!


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

"""Total probability of `y` for the categorical distributions defined by softmax(unormp)."""
function softmaxcategoricallogprob(unormp, y)
    shiftedunormp = unormp .- maximum(unormp, dims = 1)
    sum(y .* (shiftedunormp .- log.(sum(exp.(shiftedunormp), dims = 1))))
end

logitbinarycrossentropy(logŷ, y; pos_weight = 1.0f0) = (1.0f0 - y) * logŷ + (1.0f0 + (pos_weight - 1.0f0) * y) * (log(1.0f0 + exp(-abs(logŷ))) + max(-logŷ, 0.0f0))
adjacency_matrix_diag(g) = adjacency_matrix(g) + Matrix(I, size(g)...)
randn_like(target::A) where A<:AbstractArray{T} where T = randn(T, size(target))
mean(a::AbstractArray...) = sum(a) / length(a)

function loadparams!(ps::Tracker.Params, xs)
  for (p, x) in zip(ps, xs)
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x))")
    copyto!(Tracker.data(p), Tracker.data(x))
  end
end

onehotmaxbatch(a::AbstractArray) = Flux.onehotbatch(a, 1:maximum(a))


#
# Plotting helpers
#

function markersize(xy)
    (xmin, xmax), (ymin, ymax) = extrema(xy, dims = 2)
    0.03 * max(xmax - xmin, ymax - ymin)
end
markersize(xy::TrackedArray) = markersize(xy.data)


end
