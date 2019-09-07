module Dataset


include("utils.jl")
import .Utils: rowinmatrix
using LinearAlgebra
using SparseArrays
using LightGraphs
using MetaGraphs
using StatsBase


"""Create a test graph from `g` with `p` percent of deleted edges, returning the new graph, the list of removed edges, and a list of as many negative edges (edges that are absent in `g`)"""
function make_edges_test_set(g::SimpleGraph, p)
    Atriu = sparse(UpperTriangular(adjacency_matrix(g)))
    A_size = size(Atriu, 1)
    is, js, _ = findnz(Atriu)
    edges_triu = [is js]
    nedges = size(edges_triu, 1)
    test_size = Int64(round(nedges * p))

    test_true_idx = sample(1:nedges, test_size, replace = false)
    edges_test_true = edges_triu[test_true_idx, :]

    gathered_false_edges = 0
    edges_test_false = similar(edges_test_true)
    while gathered_false_edges < test_size
        candidate = sample(1:A_size, 2, replace = false)
        if (rowinmatrix(candidate, edges_triu)
                || rowinmatrix(candidate[end:-1:1], edges_triu)
                || rowinmatrix(candidate, edges_test_false)
                || rowinmatrix(candidate[end:-1:1], edges_test_false))
            continue
        end
        edges_test_false[gathered_false_edges+1, :] = candidate
        gathered_false_edges += 1
    end

    gtrain = copy(g)
    for i in 1:test_size
        @assert rem_edge!(gtrain, edges_test_true[i, :]...)
    end

    gtrain, edges_test_true, edges_test_false
end

function make_edges_test_set(mg::MetaGraph, p)
    gtrain, edges_test_true, edges_test_false = make_edges_test_set(SimpleGraph(mg), p)
    metragraph_with_props_from(gtrain, mg, collect(1:nv(mg))), edges_test_true, edges_test_false
end

function make_edges_test_set(edgelist::Array{Int64,2}, p)
    mg = mg_from_idedgelist(edgelist)
    mgtrain, edges_test_true, edges_test_false = make_edges_test_set(mg, p)

    # Convert edge arrays to ids
    medges_test_true = map(n -> node2id(mg, n), edges_test_true)
    medges_test_false = map(n -> node2id(mg, n), edges_test_false)

    mg_to_idedgelist(mgtrain), medges_test_true, medges_test_false
end

"""Create a test graph from `g` with `p` percent of deleted nodes, returning the new graph, the list of removed (test) nodes, and the list of remaining (train) nodes"""
function make_nodes_test_set(g::SimpleGraph, p)
    nnodes = nv(g)
    test_size = Int64(round(nnodes * p))
    nodes_test = sample(1:nnodes, test_size, replace = false)
    nodes_train = collect(1:nnodes)

    gtrain = copy(g)
    for i in sort(nodes_test, rev = true)
        @assert rem_vertex!(gtrain, i)
        nodes_train[i] = nodes_train[length(nodes_train)]
        nodes_train = nodes_train[1:end-1]
    end

    gtrain, sort(nodes_test), nodes_train
end

function make_nodes_test_set(mg::MetaGraph, p)
    gtrain, nodes_test, nodes_train = make_nodes_test_set(SimpleGraph(mg), p)
    metragraph_with_props_from(gtrain, mg, nodes_train), nodes_test, nodes_train
end

function make_nodes_test_set(edgelist::Array{Int64,2}, p)
    mg = mg_from_idedgelist(edgelist)
    mgtrain, nodes_test, nodes_train = make_nodes_test_set(mg, p)

    # Convert node arrays to ids
    mnodes_test = map(n -> node2id(mg, n), nodes_test)
    mnodes_train = map(n -> node2id(mg, n), nodes_train)

    mg_to_idedgelist(mgtrain), mnodes_test, mnodes_train
end

function metragraph_with_props_from(g::SimpleGraph, mg::MetaGraph, g2mg_nodes_map::Array{Int64})
    mgout = MetaGraph(g)
    # Copy over all node properties
    for i in 1:nv(mgout)
        set_props!(mgout, i, props(mg, g2mg_nodes_map[i]))
    end
    # Copy over all edge properties
    for e in edges(mgout)
        i, j = Tuple(e)
        set_props!(mgout, e, props(mg, Edge(g2mg_nodes_map[i], g2mg_nodes_map[j])))
    end
    mgout
end

node2id(mg, node) = props(mg, node)[:id]

function mg_from_idedgelist(edgelist::Array{Int64,2})
    # Check we have source->destination pairs
    @assert size(edgelist)[2] == 2
    # Get node ids
    nodeids = sort(collect(Set(edgelist)))
    nnodes = length(nodeids)
    idnodes = Dict(id => node for (node, id) in enumerate(nodeids))
    # Build the graph
    g = SimpleGraph(nnodes)
    for i in 1:size(edgelist)[1]
        source = idnodes[edgelist[i, 1]]
        target = idnodes[edgelist[i, 2]]
        add_edge!(g, Tuple([source, target]))
    end
    # Add attributes
    mg = MetaGraph(g)
    for i in 1:nv(g)
        set_prop!(mg, i, :id, nodeids[i])
    end
    mg
end

function mg_to_idedgelist(mg::MetaGraph)
    edgelist_nodes = Tuple.(edges(mg))
    nodes_in_edgelist = union(Set.(edgelist_nodes)...)
    nodes_lost = setdiff(1:nv(mg), nodes_in_edgelist)
    if length(nodes_lost) > 0
        println("Warning: nodes lost in conversion to edgelist:")
        println(nodes_lost)
    end
    map(e -> (node2id(mg, e[1]), node2id(mg, e[2])), edgelist_nodes)
end


end
