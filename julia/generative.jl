module Generative


include("utils.jl")
using .Utils

using LightGraphs
using StatsBase
using Random


"""Generate an SBM graph, returning it and its communities as a vector"""
function make_sbm(l, k, p_in, p_out; gseed = -1)
    g = LightGraphs.SimpleGraphs.stochastic_block_model(
        p_in * (k - 1), p_out * k,
        k .* ones(UInt, l), seed = gseed
    )
    communities = [c for c in 1:l for i in 1:k]
    g, communities
end

"""Generate features taken as the adjacency matrix of an SBM graph, shuffling a portion `1 - correlation` of them."""
function make_sbmfeatures(l, k, p_in, p_out, correlation; gseed = -1)
    g, communities = make_sbm(l, k, p_in, p_out, gseed = gseed)
    A = adjacency_matrix_diag(g)

    nnodes = length(communities)
    nshuffle = Int(round((1 - correlation) * nnodes))
    idx = sample(1:nnodes, nshuffle, replace = false)
    shuffledidx = shuffle(idx)

    communities[idx] = communities[shuffledidx]
    A[:, idx] = A[:, shuffledidx]

    A, communities
end

"""Generate colors (i.e. class indexes) for `communities`, shuffling `1 - correlation` of the nodes"""
function make_colors(communities, correlation)
    nnodes = length(communities)
    nshuffle = Int(round((1 - correlation) * nnodes))
    idx = sample(1:nnodes, nshuffle, replace = false)

    colors = copy(communities)
    colors[idx] = colors[shuffle(idx)]

    colors
end

"""Generate features of a given rank clustered at a given correlation level around their community centers"""
function make_clusteredhyperplane(el::Type, shape::Tuple, rank::Int, communities::AbstractArray{T, 1}, correlation::Float64) where T
    plane = randn(shape[1], rank) * randn(rank, shape[2])
    out = zeros(el, shape)
    for i = 1:shape[2]
        community = findall(communities .== communities[i])
        m = mean(plane[:, community], dims = 2)
        out[:, i] = (1 - correlation) .* plane[:, i] .+ correlation .* m
    end
    out
end
make_clusteredhyperplane(shape, rank, communities, correlation) = make_clusteredhyperplane(Float64, shape, rank, communities, correlation)


end
