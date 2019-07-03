include("utils.jl")
include("dataset.jl")
include("vae.jl")
using .Utils
using .Dataset
using .VAE

using Flux
using LightGraphs
import BSON
using ArgParse
using Profile
import JLD
using NPZ
using ScikitLearn
using Random
using StatsBase
using LinearAlgebra
@sk_import metrics: (roc_auc_score, average_precision_score, f1_score)
@sk_import linear_model: LogisticRegression


# Parameters
const profile_losses_filename = "an2vec-losses.jlprof"

"""Parse CLI arguments."""
function parse_cliargs()
    parse_settings = ArgParseSettings()
    @add_arg_table parse_settings begin
        "--dataset"
            help = "path to a npz file containing adjacency and features of a dataset"
            arg_type = String
            required = true
        "--forced-correlation"
            help = "use manual structure-features decorrelation of the dataset; 0 corresponds to completely random, 1 to the unaltered dataset; defaults to 1"
            arg_type = Float64
            default = 1.0
        "--testprop"
            help = "percentage of edges/nodes to add (or add and remove, for edges) to training dataset for reconstruction/classification testing"
            arg_type = Float64
            # default = 0.15
            required = true
        "--testtype"
            help = "test type; if provided, must be either \"nodes\" for classification, or \"edges\" for link prediction"
            arg_type = String
        "--seed"
            help = "Random seed for test set generation"
            arg_type = Int
        "--label-distribution"
            help = "which distribution do the labels follow; must be one of " * join(keys(VAE.label_distributions), ", ")
            arg_type = String
            required = true
        "--diml1enc"
            help = "dimension of intermediary encoder layer"
            arg_type = Int
            # default = 10
            required = true
        "--diml1dec"
            help = "dimension of intermediary decoder layer"
            arg_type = Int
            # default = 10
            required = true
        "--dimxiadj"
            help = "embedding dimensions for adjacency"
            arg_type = Int
            # default = 2
            required = true
        "--dimxifeat"
            help = "embedding dimensions for features"
            arg_type = Int
            # default = 2
            required = true
        "--overlap"
            help = "overlap of adjacency and feature embeddings"
            arg_type = Int
            # default = 1
            required = true
        "--catdiag"
            help = """
                concatenate I_N to features and possibly labels too;
                if provided, must be either "input" for catdiag on encoder input and not on decoder output, or "both" for input and output"""
            arg_type = String
        "--bias"
            help = "activate/deactivate bias in the VAE"
            arg_type = Bool
            required = true
        "--sharedl1"
            help = "share/unshare encoder first layer across features and adjacency"
            arg_type = Bool
            required = true
        "--decadjdeep"
            help = "deep/shallow adjacency decoder"
            arg_type = Bool
            required = true
        "--nepochs"
            help = "number of epochs to train for"
            arg_type = Int
            default = 1000
        "--savehistory"
            help = "file to save the training history (as npz)"
            arg_type = String
            required = true
        "--saveweights"
            help = "file to save the final model weights and creation parameters (as Bson)"
            arg_type = String
        "--savedataset"
            help = "file to save the training dataset (as Bson)"
            arg_type = String
        "--profile"
            help = """
                profile n loss runs instead of training the model;
                overrides nepochs and save* options; results are saved to "$(profile_losses_filename)"."""
            arg_type = Int
    end

    parsed = parse_args(ARGS, parse_settings)

    @assert parsed["testtype"] in [nothing, "nodes", "edges"]
    @assert parsed["catdiag"] in [nothing, "input", "both"]
    parsed["initb"] = parsed["bias"] ? (s) -> zeros(Float32, s) : VAE.Layers.nobias
    parsed["label-distribution"] = VAE.label_distributions[parsed["label-distribution"]]
    parsed
end


"""Load the adjacency matrix and features."""
function dataset(args)
    data = npzread(args["dataset"])

    features = convert(Array{Float32}, transpose(data["features"]))
    classes = transpose(data["labels"])

    # Make sure we have a non-weighted graph
    @assert Set(data["adjdata"]) == Set([1])

    # Remove any diagonal elements in the matrix
    rows = data["adjrow"]
    cols = data["adjcol"]
    nondiagindices = findall(rows .!= cols)
    rows = rows[nondiagindices]
    cols = cols[nondiagindices]
    # Make sure indices start at 0
    @assert minimum(rows) == minimum(cols) == 0

    # Construct the graph
    edges = LightGraphs.SimpleEdge.(1 .+ rows, 1 .+ cols)
    g = SimpleGraphFromIterator(edges)

    # Check sizes for sanity
    @assert nv(g) == size(g, 1) == size(g, 2) == size(features, 2)

    # Randomize to the level requested
    nnodes = nv(g)
    correlation = args["forced-correlation"]
    nshuffle = Int(round((1 - correlation) * nnodes))
    idx = StatsBase.sample(1:nnodes, nshuffle, replace = false)
    shuffledidx = shuffle(idx)
    features[:, idx] = features[:, shuffledidx]
    classes[:, idx] = classes[:, shuffledidx]

    # Concatenate an identity matrix if asked to
    if args["catdiag"] == nothing
        labels = features
    elseif args["catdiag"] == "input"
        labels = features
        features = vcat(features, Array(Diagonal(ones(Float32, nnodes))))
    else
        @assert args["catdiag"] == "both"
        features = vcat(features, Array(Diagonal(ones(Float32, nnodes))))
        labels = features
    end

    @assert eltype(features) == Float32
    @assert eltype(labels) == Float32
    g, features, labels, classes
end


"""Define the function computing AUC and AP scores for link reconstruction"""
function make_edges_perf_scorer(;enc, sampleξ, dec, greal::SimpleGraph, test_true_edges, test_false_edges)
    # Convert test edge arrays to indices
    test_true_indices = CartesianIndex.(test_true_edges[:, 1], test_true_edges[:, 2])
    test_false_indices = CartesianIndex.(test_false_edges[:, 1], test_false_edges[:, 2])

    # Prepare ground truth values for test edges
    Areal = Array(adjacency_matrix(greal))
    real_true = Areal[test_true_indices]
    @assert real_true == ones(length(test_true_indices))
    real_false = Areal[test_false_indices]
    @assert real_false == zeros(length(test_false_indices))
    real_all = vcat(real_true, real_false)

    function perf(x)
        μ = enc(x)[1]
        Alogitpred = dec(μ)[1].data
        pred_true = Utils.threadedσ(Alogitpred[test_true_indices])
        pred_false = Utils.threadedσ(Alogitpred[test_false_indices])
        pred_all = vcat(pred_true, pred_false)

        roc_auc_score(real_all, pred_all), average_precision_score(real_all, pred_all)
    end

    perf
end


"""Define the function computing Macro and Micro F1 scores for node classification"""
function make_nodes_perf_scorer(;enc, greal, feature_size, label_size, args, trainparams, features, classes, test_nodes, train_nodes)
    # Create the model with full adjacency for testing
    println("Info: making nodes perf scoring model")
    testenc, _, _, testparamsenc, testparamsdec = VAE.make_vae(g = greal, feature_size = feature_size, label_size = label_size, args = args)
    testparams = Flux.Params()
    push!(testparams, testparamsenc..., testparamsdec...)

    # Create numerical encodings of the classes
    classesnum = dropdims(map(p -> p[1], findmax(classes, dims = 1)[2]), dims = 1)
    @assert size(classesnum) == (size(classes, 2),)

    function perf(x)
        # Get embeddings for train nodes
        μtrain = copy(transpose(enc(x)[1].data))

        # Train a logistic regression on μtrain/classestrain
        reg = LogisticRegression(multi_class = :ovr, solver = :liblinear)
        ScikitLearn.fit!(reg, μtrain, classesnum[train_nodes])

        # Get embeddings for test nodes
        loadparams!(testparams, trainparams)
        μtest = copy(transpose(testenc(features)[1][:, test_nodes].data))

        # Get LogisticRegression predictions for test nodes
        testpred = ScikitLearn.predict(reg, μtest)

        # Score predictions
        f1_score(classesnum[test_nodes], testpred, average = :macro), f1_score(classesnum[test_nodes], testpred, average = :micro)
    end
end


function main()
    args = parse_cliargs()
    savehistory, saveweights, savedataset, profilen = args["savehistory"], args["saveweights"], args["savedataset"], args["profile"]
    if profilen != nothing
        println("Profiling $profilen loss runs. Ignoring any \"save*\" arguments.")
    else
        saveweights == nothing && println("Warning: will not save model weights after training")
    end
    seed = args["seed"]
    if seed != nothing
        println("Setting random seed to $seed")
        Random.seed!(seed)
    end

    println("Loading the dataset")
    g, _features, labels, classes = dataset(args)
    feature_size = size(_features, 1)
    label_size = size(labels, 1)
    if args["testtype"] == nothing
        gtrain = g
        test_nodes = nothing
        train_nodes = 1:nv(g)
    elseif args["testtype"] == "edges"
        gtrain, test_true_edges, test_false_edges = Dataset.make_edges_test_set(g, args["testprop"])
        test_nodes = nothing
        train_nodes = 1:nv(g)
    else
        @assert args["testtype"] == "nodes"
        gtrain, test_nodes, train_nodes = Dataset.make_nodes_test_set(g, args["testprop"])
    end
    fnormalise = normaliser(_features[:, train_nodes])
    features_train = fnormalise(_features[:, train_nodes])
    features_all = fnormalise(_features)
    labels_train = labels[:, train_nodes]

    println("Making the model")
    enc, sampleξ, dec, paramsenc, paramsdec = VAE.make_vae(
        g = gtrain, feature_size = feature_size, label_size = label_size, args = args)
    paramsvae = Flux.Params()
    push!(paramsvae, paramsenc..., paramsdec...)
    losses, loss = VAE.make_losses(
        g = gtrain, labels = labels_train, args = args,
        enc = enc, sampleξ = sampleξ, dec = dec,
        paramsenc = paramsenc, paramsdec = paramsdec)
    perf_edges = if args["testtype"] == "edges"
        make_edges_perf_scorer(
            enc = enc, sampleξ = sampleξ, dec = dec,
            greal = g, test_true_edges = test_true_edges, test_false_edges = test_false_edges)
    else
        nothing
    end
    perf_nodes = if args["testtype"] == "nodes"
        make_nodes_perf_scorer(
            enc = enc,
            greal = g, feature_size = feature_size, label_size = label_size, args = args, trainparams = paramsvae,
            features = features_all, classes = classes,
            test_nodes = test_nodes, train_nodes = train_nodes)
    else
        nothing
    end

    if profilen != nothing
        println("Profiling loss runs...")
        Utils.repeat_fn(1, loss, features_train)  # Trigger compilation
        Profile.clear()
        Profile.init(n = 10000000)
        @profile Utils.repeat_fn(profilen, loss, features_train)
        li, lidict = Profile.retrieve()
        println("Saving profile results to \"$(profile_losses_filename)\"")
        JLD.@save profile_losses_filename li lidict

        return
    end

    if savedataset != nothing
        println("Saving training dataset to \"$savedataset\"")
        test_info = Dict()
        if args["testtype"] == "edges"
            test_info["test_true_edges"] = test_true_edges
            test_info["test_false_edges"] = test_false_edges
        end
        if args["testtype"] == "nodes"
            test_info["test_nodes"] = test_nodes
            test_info["train_nodes"] = train_nodes
        end
        BSON.@save savedataset gtrain test_info
    else
        println("Not saving training dataset")
    end

    if saveweights != nothing
        saveweights0 = saveweights * "-0"
        println("Saving initial model weights and creation parameters to \"$saveweights0\"")
        weights = Tracker.data.(paramsvae)
        BSON.@save saveweights0 weights args
    else
        println("Not saving initial model weights or creation parameters")
    end

    println("Training...")
    history = VAE.train_vae!(
        args = args, features = features_train,
        losses = losses, loss = loss, perf_edges = perf_edges, perf_nodes = perf_nodes,
        paramsvae = paramsvae)

    println("Final losses and performance metrics:")
    for (name, values) in history
        println("  $name = $(values[end])")
    end

    # Save results
    println("Saving training history to \"$savehistory\"")
    npzwrite(savehistory, Dict{String, Any}(history))
    if saveweights != nothing
        println("Saving final model weights and creation parameters to \"$saveweights\"")
        weights = Tracker.data.(paramsvae)
        BSON.@save saveweights weights args
    else
        println("Not saving model weights or creation parameters")
    end
end

main()
