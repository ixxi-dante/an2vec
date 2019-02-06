module Generative


using LightGraphs
using StatsBase
using Random


function make_sbm(l, k, p_in, p_out; gseed = -1)
    g = LightGraphs.SimpleGraphs.stochastic_block_model(
        p_in * (k - 1), p_out * k,
        k .* ones(UInt, l), seed = gseed
    )
    communities = [c for c in 1:l for i in 1:k]
    g, communities
end

function make_colors(communities, correlation)
    nnodes = length(communities)
    nshuffle = Int(round((1 - correlation) * nnodes))
    shuffleidx = sample(1:nnodes, nshuffle, replace = false)

    colors = copy(communities)
    colors[shuffleidx] = colors[shuffle(shuffleidx)]

    colors
end


end
