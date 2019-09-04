module Dataset


include("utils.jl")
import .Utils: rowinmatrix
using LinearAlgebra
using SparseArrays
using LightGraphs
using StatsBase
using GraphIO


"""Create a test graph from `g` with `p` percent of deleted edges, returning the new graph, the list of removed edges, and a list of as many negative edges (edges that are absent in `g`)"""
function make_edges_test_set(g::SimpleGraph, p)
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


"""Create a test graph from `g` with `p` percent of deleted nodes, returning the new graph, the list of removed (test) nodes, and the list of remaining (train) nodes"""
function make_nodes_test_set(g::SimpleGraph, p)
    nnodes = nv(g)
    test_size = Int64(round(nnodes * p))
    test_nodes = sample(1:nnodes, test_size, replace = false)
    train_nodes = setdiff(1:nnodes, test_nodes)

    gtrain = copy(g)
    for i in sort(test_nodes, rev = true)
        @assert rem_vertex!(gtrain, i)
    end

    gtrain, sort(test_nodes), train_nodes
end

function make_nodes_test_set(edgelistpath::AbstractString, p)
    g = SimpleGraph(loadgraph(edgelistpath, "", EdgeListFormat()))
    gtrain, test_nodes, train_nodes = make_nodes_test_set(g, p)
    (path, io) = mktemp()
    savegraph(io, gtrain, "", EdgeListFormat())
    close(io)
    path, test_nodes, train_nodes
end


end
