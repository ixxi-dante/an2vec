include("utils.jl")
include("layers.jl")
include("generative.jl")
include("dataset.jl")
using .Utils
using .Layers
using .Generative
using .Dataset

using Flux
using LightGraphs
using ProgressMeter
using Statistics
using Distributions
using Random
import BSON
using ArgParse
using Profile
import JLD
using NPZ
using PyCall


# Parameters
const klscale = 1f-3
const regscale = 1f-3
const profile_losses_filename = "an2vec-losses.jlprof"
const supported_feature_distributions = [Bernoulli, Categorical, Normal]
const feature_distributions_dict = Dict(lowercase(repr(d)) => d for d in supported_feature_distributions)

"""Parse CLI arguments."""
function parse_cliargs()
    parse_settings = ArgParseSettings()
    @add_arg_table parse_settings begin
        "--dataset"
            help = "path to a npz file containing adjacency and features of a dataset"
            arg_type = String
            required = true
        "--blurring"
            help = "percentage of edges to add/remove to training dataset for reconstruction testing"
            arg_type = Float64
            # default = 0.15
            required = true
        "--feature-distribution"
            help = "which distribution do the features follow; must be one of " * join(keys(feature_distributions_dict), ", ")
            arg_type = String
            required = true
        "--diml1"
            help = "dimension of intermediary layers"
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
        "--nepochs"
            help = "number of epochs to train for"
            arg_type = Int
            default = 1000
        "--savehistory"
            help = "file to save the training history (as Bson)"
            arg_type = String
            required = true
        "--saveweights"
            help = "file to save the final model weights and creation parameters (as Bson)"
            arg_type = String
        "--profile"
            help = """
                profile n loss runs instead of training the model;
                overrides nepochs and save* options; results are saved to "$(profile_losses_filename)"."""
            arg_type = Int
    end

    parsed = parse_args(ARGS, parse_settings)
    parsed["feature-distribution"] = feature_distributions_dict[parsed["feature-distribution"]]
    parsed
end


"""Load the adjacency matrix and features."""
function dataset(args)
    adjfeatures = npzread(args["dataset"])

    features = transpose(adjfeatures["features"])

    # Make sure we have a non-weighted graph
    @assert Set(adjfeatures["adjdata"]) == Set([1])

    # Remove any diagonal elements in the matrix
    rows = adjfeatures["adjrow"]
    cols = adjfeatures["adjcol"]
    nondiagindices = findall(rows .!= cols)
    rows = rows[nondiagindices]
    cols = cols[nondiagindices]
    # Make sure indices start at 0
    @assert minimum(rows) == minimum(cols) == 0

    # Construct the graph
    edges = LightGraphs.SimpleEdge.(1 .+ rows, 1 .+ cols)
    g = SimpleGraphFromIterator(edges)

    # Check sizes for sanity
    @assert size(g, 1) == size(g, 2) == size(features, 2)
    g, convert(Array{Float32}, features), convert(Array{Float32}, scale_center(features))
end


"""Make the model."""
function make_vae(;g, feature_size, args)
    diml1, dimξadj, dimξfeat, overlap = args["diml1"], args["dimxiadj"], args["dimxifeat"], args["overlap"]

    # Encoder
    l1 = Layers.GC(g, feature_size, diml1, Flux.relu, initb = Layers.nobias)
    lμ = Layers.Apply(Layers.VOverlap(overlap),
        Layers.GC(g, diml1, dimξadj, initb = Layers.nobias),
        Layers.GC(g, diml1, dimξfeat, initb = Layers.nobias))
    llogσ = Layers.Apply(Layers.VOverlap(overlap),
        Layers.GC(g, diml1, dimξadj, initb = Layers.nobias),
        Layers.GC(g, diml1, dimξfeat, initb = Layers.nobias))
    enc(x) = (h = l1(x); (lμ(h), llogσ(h)))
    encparams = Flux.params(l1, lμ, llogσ)

    # Sampler
    sampleξ(μ, logσ) = μ .+ exp.(logσ) .* randn_like(μ)

    # Decoder
    decadj = Layers.Bilin()
    decfeat, decparams = if args["feature-distribution"] == Normal
        println("Info: using Gaussian feature decoder")
        decfeatl1 = Dense(dimξfeat, diml1, Flux.relu)
        decfeatlμ = Dense(diml1, feature_size)
        decfeatllogσ = Dense(diml1, feature_size)
        decfeat(ξ) = (h = decfeatl1(ξ); (decfeatlμ(h), decfeatllogσ(h)))
        decfeat, Flux.params(decadj, decfeatl1, decfeatlμ, decfeatllogσ)
    else
        println("Info: using non-Gaussian feature decoder")
        decfeat = Chain(
            Dense(dimξfeat, diml1, Flux.relu),
            Dense(diml1, feature_size),
        )
        decfeat, Flux.params(decadj, decfeat)
    end
    dec(ξ) = (decadj(ξ[1:dimξadj, :]), decfeat(ξ[end-dimξfeat+1:end, :]))

    enc, sampleξ, dec, encparams, decparams
end


"""Define the function compting AUC and AP scores for model predictions (adjacency only)"""
function make_perf_scorer(;enc, sampleξ, dec, greal::SimpleGraph, test_true_edges, test_false_edges)
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

    metrics = pyimport("sklearn.metrics")

    function perf(x)
        μ = enc(x)[1]
        Alogitpred = dec(μ)[1].data
        pred_true = Utils.threadedσ(Alogitpred[test_true_indices])
        pred_false = Utils.threadedσ(Alogitpred[test_false_indices])
        pred_all = vcat(pred_true, pred_false)

        metrics[:roc_auc_score](real_all, pred_all), metrics[:average_precision_score](real_all, pred_all)
    end

    perf
end


"""Define the model losses."""
function make_losses(;g, labels, feature_size, args, enc, sampleξ, dec, paramsenc, paramsdec)
    feature_distribution = args["feature-distribution"]
    dimξadj, dimξfeat, overlap = args["dimxiadj"], args["dimxifeat"], args["overlap"]
    Adiag = Array{Float32}(adjacency_matrix_diag(g))
    densityA = Float32(mean(adjacency_matrix(g)))
    densitylabels = Float32(mean(labels))

    # TODO check normalisation constants

    # Kullback-Leibler divergence
    Lkl(μ, logσ) = sum(Utils.threadedklnormal(μ, logσ))
    κkl = Float32(size(g, 1) * (dimξadj - overlap + dimξfeat))

    # Adjacency loss
    Ladj(logitApred) = (
        sum(threadedlogitbinarycrossentropy(logitApred, Adiag, pos_weight = (1f0 / densityA) - 1))
        / (2 * (1 - densityA))
    )
    κadj = Float32(size(g, 1)^2 * log(2))

    # Features loss
    Lfeat(logitFpred, ::Type{Bernoulli}) = (
        sum(threadedlogitbinarycrossentropy(logitFpred, labels, pos_weight = (1f0 / densitylabels) - 1))
        / (1 - densitylabels)
    )
    κfeat_bernoulli = Float32(prod(size(labels)) * log(2))
    κfeat(::Type{Bernoulli}) = κfeat_bernoulli

    Lfeat(unormFpred, ::Type{Categorical}) = - softmaxcategoricallogprob(unormFpred, labels)
    κfeat_categorical = Float32(size(g, 1) * log(feature_size))
    κfeat(::Type{Categorical}) = κfeat_categorical

    Lfeat(Fpreds, ::Type{Normal}) = ((μ, logσ) = Fpreds; sum(Utils.threadednormallogprobloss(μ, logσ, labels)))
    κfeat_normal = Float32(prod(size(labels)) * (log(2π) + mean(labels.^2)) / 2)
    κfeat(::Type{Normal}) = κfeat_normal

    # Total loss
    function losses(x)
        μ, logσ = enc(x)
        logitApred, unormfeatpred = dec(sampleξ(μ, logσ))
        Dict("kl" => klscale * Lkl(μ, logσ) / κkl,
            "adj" => Ladj(logitApred) / κadj,
            "feat" => Lfeat(unormfeatpred, feature_distribution) / κfeat(feature_distribution),
            "reg" => regscale * Utils.regularizer(paramsdec))
    end

    function loss(x)
        sum(values(losses(x)))
    end

    losses, loss
end


"""Profile runs of a function"""
function profile_fn(n::Int64, fn, args...)
    for i = 1:n
        fn(args...)
    end
end


"""Train a VAE."""
function train_vae!(;args, features, losses, loss, perf, paramsvae)
    nepochs = args["nepochs"]

    history = Dict(name => zeros(nepochs) for name in keys(losses(features)))
    history["total loss"] = zeros(nepochs)
    history["auc"] = zeros(nepochs)
    history["ap"] = zeros(nepochs)

    opt = ADAM(0.01)
    @showprogress for i = 1:nepochs
        Flux.train!(loss, paramsvae, [(features,)], opt)

        lossparts = losses(features)
        for (name, value) in lossparts
            history[name][i] = value.data
        end
        history["total loss"][i] = sum(values(lossparts)).data
        history["auc"][i], history["ap"][i] = perf(features)
    end

    history
end


function main()
    args = parse_cliargs()
    savehistory, saveweights, profilen = args["savehistory"], args["saveweights"], args["profile"]
    if profilen != nothing
        println("Profiling $profilen loss runs. Ignoring any \"save*\" arguments.")
    else
        saveweights == nothing && println("Warning: will not save model weights after training")
    end

    println("Loading the dataset")
    g, labels, features = dataset(args)
    gtrain, test_true_edges, test_false_edges = Dataset.make_blurred_test_set(g, args["blurring"])
    feature_size = size(features, 1)

    println("Making the model")
    enc, sampleξ, dec, paramsenc, paramsdec = make_vae(
        g = gtrain, feature_size = feature_size, args = args)
    losses, loss = make_losses(
        g = gtrain, labels = labels, feature_size = feature_size, args = args,
        enc = enc, sampleξ = sampleξ, dec = dec,
        paramsenc = paramsenc, paramsdec = paramsdec)
    perf = make_perf_scorer(
        enc = enc, sampleξ = sampleξ, dec = dec,
        greal = g, test_true_edges = test_true_edges, test_false_edges = test_false_edges)

    if profilen != nothing
        println("Profiling loss runs...")
        profile_fn(1, loss, features)  # Trigger compilation
        Profile.clear()
        Profile.init(n = 10000000)
        @profile profile_fn(profilen, loss, features)
        li, lidict = Profile.retrieve()
        println("Saving profile results to \"$(profile_losses_filename)\"")
        JLD.@save profile_losses_filename li lidict

        return
    end

    println("Training...")
    paramsvae = Flux.Params()
    push!(paramsvae, paramsenc..., paramsdec...)
    history = train_vae!(
        args = args, features = features,
        losses = losses, loss = loss, perf = perf,
        paramsvae = paramsvae)

    println("Final losses and performance metrics:")
    for (name, values) in history
        println("  $name = $(values[end])")
    end

    # Save results
    println("Saving training history to \"$savehistory\"")
    BSON.@save savehistory history
    if saveweights != nothing
        println("Saving final model weights and creation parameters to \"$saveweights\"")
        weights = Tracker.data.(paramsvae)
        BSON.@save saveweights weights args
    else
        println("Not saving model weights or creation parameters")
    end
end

main()
