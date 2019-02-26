module Utils


export mean, scale_center, softmaxcategoricallogprob, logitbinarycrossentropy, threadedlogitbinarycrossentropy, adjacency_matrix_diag, randn_like, markersize, loadparams!

using Statistics, Flux, LinearAlgebra, LightGraphs, SparseArrays
import Statistics.mean
import Flux.loadparams!


#
# Miscellaneous
#

function supertypes(T)
    current = T
    println(current)
    while current != Any
        current = supertype(current)
        println(current)
    end
end

rowinmatrix(r::AbstractVector, m::AbstractMatrix) = any(all(reshape(r, 1, :) .== m, dims = 2))


#
# Neural network helpers
#

function scale_center(x; dims = 1)
    x = x .- mean(x, dims = dims)
    norm = sqrt.(sum(x .^ 2, dims = dims))
    zeros = findall(norm .== 0)
    norm[zeros] = fill(one(eltype(norm)), size(zeros))
    x ./ norm
end

"""Total probability of `y` for the categorical distributions defined by softmax(unormp)."""
function softmaxcategoricallogprob(unormp, y)
    shiftedunormp = unormp .- maximum(unormp, dims = 1)
    sum(y .* (shiftedunormp .- log.(sum(exp.(shiftedunormp), dims = 1))))
end

logitbinarycrossentropy(logŷ::T, y::T; pos_weight = 1f0) where T = (1f0 - y) * logŷ + (1f0 + (pos_weight - 1f0) * y) * (log(1f0 + exp(-abs(logŷ))) + max(-logŷ, 0f0))


import Flux.Tracker: @grad, track, data, nobacksies

function threadedlogitbinarycrossentropy!(out::AbstractArray{T}, logŷ::AbstractArray{T}, y::AbstractArray{T}; pos_weight) where T
    @assert size(logŷ) == size(y)
    Threads.@threads for i in eachindex(logŷ)
        @inbounds out[i] = logitbinarycrossentropy(logŷ[i], y[i]; pos_weight = pos_weight)
    end
    return out
end
threadedlogitbinarycrossentropy(logŷ, y; pos_weight = 1f0) = threadedlogitbinarycrossentropy!(similar(logŷ), logŷ, y, pos_weight = pos_weight)
threadedlogitbinarycrossentropy(logŷ::TrackedArray, y; pos_weight = 1f0) = track(threadedlogitbinarycrossentropy, logŷ, y, pos_weight = pos_weight)

function ∇threadedlogitbinarycrossentropy_logits!(out::AbstractArray, Δ::AbstractArray, logŷ::AbstractArray, y::AbstractArray; pos_weight)
    Threads.@threads for i in eachindex(out)
        @inbounds out[i] = Δ[i] * (σ(logŷ[i]) * (y[i] * (pos_weight - 1) + 1) - y[i] * pos_weight)
    end
    return out
end

function ∇threadedlogitbinarycrossentropy_labels!(out::AbstractArray, Δ::AbstractArray, logŷ::AbstractArray, y::AbstractArray; pos_weight)
    Threads.@threads for i in eachindex(out)
        @inbounds out[i] = Δ[i] * (max(logŷ[i], 0) * (pos_weight - 1) - pos_weight * logŷ[i] + (pos_weight - 1) * log(1 + exp(-abs(logŷ[i]))))
    end
    return out
end

∇threadedlogitbinarycrossentropy_logits(Δ, logŷ, y; pos_weight) = ∇threadedlogitbinarycrossentropy_logits!(similar(Δ), Δ, logŷ, y; pos_weight = pos_weight)
∇threadedlogitbinarycrossentropy_labels(Δ, logŷ, y; pos_weight) = ∇threadedlogitbinarycrossentropy_labels!(similar(Δ), Δ, logŷ, y; pos_weight = pos_weight)

@grad function threadedlogitbinarycrossentropy(logŷ::AbstractArray{T}, y::AbstractArray{T}; pos_weight) where T
    threadedlogitbinarycrossentropy(data(logŷ), data(y); pos_weight = pos_weight),
        Δ -> nobacksies(:threadedlogitbinarycrossentropy,
            (∇threadedlogitbinarycrossentropy_logits(data(Δ), data(logŷ), data(y); pos_weight = pos_weight),
             nothing))
end


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
