include("utils.jl")
include("layers.jl")
include("generative.jl")
using .Utils
using .Layers
using .Generative

using Flux
using LightGraphs
using ProgressMeter
using Statistics
using Distributions
using Random
using BSON: @save
using ArgParse


# Parameters
const klscale = 1e-3
const regscale = 1e-3

"""Parse CLI arguments."""
function parse_cliargs()
    parse_settings = ArgParseSettings()
    @add_arg_table parse_settings begin
        "-l"
            help = "number of communities"
            arg_type = Int
            # default = 10
            required = true
        "-k"
            help = "size of each community"
            arg_type = Int
            # default = 10
            required = true
        "--p_in"
            help = "intra-community connection probability"
            arg_type = Float64
            default = 0.4
        "--p_out"
            help = "extra-community connection probability"
            arg_type = Float64
            default = 0.01
        "--correlation"
            help = "correlation between colors and SBM communities"
            arg_type = Float64
            required = true
        "--gseed"
            help = "seed for generation of the graph"
            arg_type = Int
            default = -1
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
        "--savedataset"
            help = "file to save the training dataset (as Bson)"
            arg_type = String
    end

    parsed = parse_args(ARGS, parse_settings)
    @assert 0 <= parsed["correlation"] <= 1
    parsed["diml1"] = Int64(round(sqrt(parsed["l"] * (parsed["dimxiadj"] + parsed["dimxifeat"]))))
    parsed
end


"""Define the graph and features."""
function dataset(args)
    l, k, p_in, p_out, gseed, correlation = args["l"], args["k"], args["p_in"], args["p_out"], args["gseed"], args["correlation"]

    g, communities = Generative.make_sbm(l, k, p_in, p_out, gseed = gseed)
    colors = Generative.make_colors(communities, correlation)
    colorsoh = Utils.onehotmaxbatch(colors)
    features = scale_center(colorsoh)

    g, colorsoh, features
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

    # Sampler
    sampleξ(μ, logσ) = μ .+ exp.(logσ) .* randn_like(μ)

    # Decoder
    decadj = Chain(
        Dense(dimξadj, diml1, Flux.relu, initb = Layers.nobias),
        Layers.Bilin(diml1)
    )
    decfeat = Chain(
        Dense(dimξfeat, diml1, Flux.relu, initb = Layers.nobias),
        Dense(diml1, feature_size, initb = Layers.nobias),
    )
    @views dec(ξ) = (decadj(ξ[1:dimξadj, :]), decfeat(ξ[end-dimξfeat+1:end, :]))

    enc, sampleξ, dec, Flux.params(l1, lμ, llogσ), Flux.params(decadj, decfeat)
end


"""Define the model losses."""
function make_losses(;g, labels, feature_size, args, enc, sampleξ, dec, paramsenc, paramsdec)
    dimξadj, dimξfeat, overlap = args["dimxiadj"], args["dimxifeat"], args["overlap"]
    Adiag = adjacency_matrix_diag(g)
    densityA = mean(adjacency_matrix(g));

    # Decoder regularizer
    decregularizer(l = 0.01) = l * sum(x -> sum(x.^2), paramsdec)

    # Kullback-Leibler divergence
    Lkl(μ, logσ) = 0.5 * sum(exp.(2 .* logσ) + μ.^2 .- 1 .- 2 .* logσ)
    κkl = size(g, 1) * (dimξadj - overlap + dimξfeat)

    # Adjacency loss
    Ladj(logitApred) = (
        0.5 * sum(logitbinarycrossentropy.(logitApred, Adiag, pos_weight = (1 / densityA) - 1))
        / (1 - densityA)
    )
    κadj = size(g, 1)^2 * log(2)

    # Features loss
    Lfeat(unormfeatpred) = - softmaxcategoricallogprob(unormfeatpred, labels)
    κfeat = size(g, 1) * log(feature_size)

    # Total loss
    function losses(x)
        μ, logσ = enc(x)
        logitApred, unormfeatpred = dec(sampleξ(μ, logσ))
        Dict("kl" => klscale * Lkl(μ, logσ) / κkl,
            "adj" => Ladj(logitApred) / κadj,
            "feat" => Lfeat(unormfeatpred) / κfeat,
            "reg" => regscale * decregularizer())
    end

    function loss(x)
        sum(values(losses(x)))
    end

    losses, loss
end


### Profile

#function profile_test(n)
#    for i = 1:n
#        vae(features)
#    end
#end
#
#profile_test(1)  # Trigger compilation
#using Profile
#@profile profile_test(10)
#li, lidict = Profile.retrieve()
#using JLD
#@save "gvae.jlprof" li lidict


"""Train a VAE."""
function train_vae!(;args, features, losses, loss, paramsvae)
    nepochs = args["nepochs"]

    history = Dict(name => zeros(nepochs) for name in keys(losses(features)))
    history["total"] = zeros(nepochs)

    opt = ADAM(0.01)
    @showprogress for i = 1:nepochs
        Flux.train!(loss, paramsvae, [(features,)], opt)

        lossparts = losses(features)
        for (name, value) in lossparts
            history[name][i] = value.data
        end
        history["total"][i] = sum(values(lossparts)).data
    end

    history
end


function main()
    args = parse_cliargs()
    savehistory, saveweights, savedataset = args["savehistory"], args["saveweights"], args["savedataset"]
    saveweights == nothing && println("Warning: will not save model weights after training")
    savedataset == nothing && println("Warning: will not save training dataset after training")

    println("Making the dataset")
    g, labels, features = dataset(args)
    feature_size = size(features, 1)

    println("Making the model")
    enc, sampleξ, dec, paramsenc, paramsdec = make_vae(
        g = g, feature_size = feature_size, args = args)
    losses, loss = make_losses(
        g = g, labels = labels, feature_size = feature_size, args = args,
        enc = enc, sampleξ = sampleξ, dec = dec,
        paramsenc = paramsenc, paramsdec = paramsdec)

    println("Training...")
    paramsvae = Flux.Params()
    push!(paramsvae, paramsenc..., paramsdec...)
    history = train_vae!(
        args = args, features = features,
        losses = losses, loss = loss,
        paramsvae = paramsvae)
    println("Final losses:")
    for (name, values) in history
        println("  $name = $(values[end])")
    end

    # Save results
    println("Saving training history to \"$savehistory\"")
    @save savehistory history
    if saveweights != nothing
        println("Saving final model weights and creation parameters to \"$saveweights\"")
        weights = Tracker.data.(paramsvae)
        @save saveweights weights args
    else
        println("Not saving model weights or creation parameters")
    end
    if savedataset != nothing
        println("Saving training dataset to \"$savedataset\"")
        @save savedataset g labels features
    else
        println("Not saving training dataset")
    end
end

main()
