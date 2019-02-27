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

#
# Losses
#

import Flux.Tracker: @grad, track, data, nobacksies


regularizer(params; l = 0.01f0) = l * sum(x -> sum(x.^2), params)


klnormal(μ, logσ) = (exp(2logσ) + μ^2 - 1 - 2logσ) / 2


"""Total probability of `y` for the categorical distributions defined by softmax(unormp)."""
function softmaxcategoricallogprob(unormp, y)
    shiftedunormp = unormp .- maximum(unormp, dims = 1)
    sum(y .* (shiftedunormp .- log.(sum(exp.(shiftedunormp), dims = 1))))
end


logitbinarycrossentropy(logŷ, y; pos_weight = 1) = (1 - y) * logŷ + (1 + (pos_weight - 1) * y) * (log(1 + exp(-abs(logŷ))) + max(-logŷ, 0))

function threadedlogitbinarycrossentropy!(out::AbstractArray, logŷ::AbstractArray, y::AbstractArray; kw...)
    @assert size(logŷ) == size(y)
    Threads.@threads for i in eachindex(out)
        @inbounds out[i] = logitbinarycrossentropy(logŷ[i], y[i]; kw...)
    end
    return out
end
threadedlogitbinarycrossentropy(logŷ, y; kw...) = threadedlogitbinarycrossentropy!(similar(logŷ), logŷ, y; kw...)
threadedlogitbinarycrossentropy(logŷ::TrackedArray, y; kw...) = track(threadedlogitbinarycrossentropy, logŷ, y; kw...)

function ∇threadedlogitbinarycrossentropy_logits!(out::AbstractArray, Δ::AbstractArray, logŷ::AbstractArray, y::AbstractArray; pos_weight)
    Threads.@threads for i in eachindex(out)
        @inbounds out[i] = Δ[i] * (σ(logŷ[i]) * (y[i] * (pos_weight - 1) + 1) - y[i] * pos_weight)
    end
    return out
end
∇threadedlogitbinarycrossentropy_logits(Δ, logŷ, y; kw...) = ∇threadedlogitbinarycrossentropy_logits!(similar(Δ), Δ, logŷ, y; kw...)

function ∇threadedlogitbinarycrossentropy_labels!(out::AbstractArray, Δ::AbstractArray, logŷ::AbstractArray, y::AbstractArray; pos_weight)
    Threads.@threads for i in eachindex(out)
        @inbounds out[i] = Δ[i] * (max(logŷ[i], 0) * (pos_weight - 1) - pos_weight * logŷ[i] + (pos_weight - 1) * log(1 + exp(-abs(logŷ[i]))))
    end
    return out
end
∇threadedlogitbinarycrossentropy_labels(Δ, logŷ, y; kw...) = ∇threadedlogitbinarycrossentropy_labels!(similar(Δ), Δ, logŷ, y; kw...)

@grad function threadedlogitbinarycrossentropy(logŷ::AbstractArray, y::AbstractArray; kw...)
    threadedlogitbinarycrossentropy(data(logŷ), data(y); kw...),
        Δ -> nobacksies(:threadedlogitbinarycrossentropy,
            (∇threadedlogitbinarycrossentropy_logits(data(Δ), data(logŷ), data(y); kw...),
             # Ignore the ∇threadedlogitbinarycrossentropy_labels gradient
             nothing))
end


normallogprobloss(μ, logσ, y) = log(2π) / 2 + logσ + (y - μ)^2 * exp(-2logσ) / 2

function threadednormallogprobloss!(out::AbstractArray, μ::AbstractArray, logσ::AbstractArray, y::AbstractArray)
    @assert size(μ) == size(logσ) == size(y)
    Threads.@threads for i in eachindex(out)
        @inbounds out[i] = normallogprobloss(μ[i], logσ[i], y[i])
    end
    return out
end
threadednormallogprobloss(μ, logσ, y) = threadednormallogprobloss!(similar(μ), μ, logσ, y)
threadednormallogprobloss(μ::TrackedArray, logσ::TrackedArray, y) = track(threadednormallogprobloss, μ, logσ, y)

function ∇threadednormallogprobloss_μ_logσ!(outμ::AbstractArray, outlogσ::AbstractArray, Δ::AbstractArray, μ::AbstractArray, logσ::AbstractArray, y::AbstractArray)
    Threads.@threads for i in eachindex(outμ)
        @inbounds begin
            gradμ = (μ[i] - y[i]) * exp(-2logσ[i])
            outμ[i] = Δ[i] * gradμ
            outlogσ[i] = Δ[i] * (1 - (μ[i] - y[i]) * gradμ)
        end
    end
    return outμ, outlogσ
end
∇threadednormallogprobloss_μ_logσ(Δ, μ, logσ, y) = ∇threadednormallogprobloss_μ_logσ!(similar(Δ), similar(Δ), Δ, μ, logσ, y)

@grad function threadednormallogprobloss(μ::AbstractArray, logσ::AbstractArray, y::AbstractArray)
    threadednormallogprobloss(data.([μ, logσ, y])...),
        Δ -> nobacksies(:threadednormallogprobloss,
            # The first two gradients (w.r.t. μ and logσ) are computed together
            (∇threadednormallogprobloss_μ_logσ(data.([Δ, μ, logσ, y])...)...,
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
