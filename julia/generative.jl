module Generative


include("utils.jl")
import .Utils: rowinmatrix
using LinearAlgebra
using SparseArrays
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

"""Generate colors (i.e. class indexes) for `communities`, shuffling `1 - correlation` of the nodes"""
function make_colors(communities, correlation)
    nnodes = length(communities)
    nshuffle = Int(round((1 - correlation) * nnodes))
    shuffleidx = sample(1:nnodes, nshuffle, replace = false)

    colors = copy(communities)
    colors[shuffleidx] = colors[shuffle(shuffleidx)]

    colors
end

"""Creat a test graph from `g` with `p` percent of deleted edges, returning the new graph, the list of removed edges, and a list of as many negative edges (edges that are absent in `g`)"""
function make_blurred_test_set(g::SimpleGraph, p)
    Atriu = sparse(UpperTriangular(adjacency_matrix(g)))
    A_size = size(Atriu, 1)
    is, js, _ = findnz(Atriu)
    edges_triu = [is js]
    nedges = size(edges_triu, 1)
    test_size = Int64(round(nedges * p))

    test_true_idx = sample(1:nedges, test_size, replace = false)
    test_true_edges = edges_triu[test_true_idx, :]

    gathered_false_edges = 0
    test_false_edges = similar(test_true_edges)
    while gathered_false_edges < test_size
        candidate = sample(1:A_size, 2, replace = false)
        if (rowinmatrix(candidate, edges_triu)
                || rowinmatrix(candidate[end:-1:1], edges_triu)
                || rowinmatrix(candidate, test_false_edges)
                || rowinmatrix(candidate[end:-1:1], test_false_edges))
            continue
        end
        test_false_edges[gathered_false_edges+1, :] = candidate
        gathered_false_edges += 1
    end

    gtrain = copy(g)
    for i in 1:test_size
        @assert rem_edge!(gtrain, test_true_edges[i, :]...)
    end

    gtrain, test_true_edges, test_false_edges
end



end
