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
import BSON
using ArgParse
using Profile
import JLD


# Parameters
const klscale = 1e-3
const regscale = 1e-3
const profile_losses_filename = "an2vec-losses.jlprof"

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
        "--profile"
            help = """
                profile n loss runs instead of training the model;
                overrides nepochs and save* options; results are saved to "$(profile_losses_filename)"."""
            arg_type = Int
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
    colorsoh = Array{Float32}(Utils.onehotmaxbatch(colors))
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
    dec(ξ) = (decadj(ξ[1:dimξadj, :]), decfeat(ξ[end-dimξfeat+1:end, :]))

    enc, sampleξ, dec, Flux.params(l1, lμ, llogσ), Flux.params(decadj, decfeat)
end


"""Define the model losses."""
function make_losses(;g, labels, feature_size, args, enc, sampleξ, dec, paramsenc, paramsdec)
    dimξadj, dimξfeat, overlap = args["dimxiadj"], args["dimxifeat"], args["overlap"]
    Adiag = Array{Float32}(adjacency_matrix_diag(g))
    densityA = Float32(mean(adjacency_matrix(g)))

    # Decoder regularizer
    decregularizer(l = 0.01f0) = l * sum(x -> sum(x.^2), paramsdec)

    # Kullback-Leibler divergence
    Lkl(μ, logσ) = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1f0 .- 2f0 .* logσ)
    κkl = Float32(size(g, 1) * (dimξadj - overlap + dimξfeat))

    # Adjacency loss
    Ladj(logitApred) = (
        0.5f0 * sum(logitbinarycrossentropy.(logitApred, Adiag, pos_weight = (1f0 / densityA) - 1f0))
        / (1f0 - densityA)
    )
    κadj = Float32(size(g, 1)^2 * log(2))

    # Features loss
    Lfeat(unormfeatpred) = - softmaxcategoricallogprob(unormfeatpred, labels)
    κfeat = Float32(size(g, 1) * log(feature_size))

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


"""Profile runs of a function"""
function profile_fn(n::Int64, fn, args...)
    for i = 1:n
        fn(args...)
    end
end


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
    savehistory, saveweights, savedataset, profilen = args["savehistory"], args["saveweights"], args["savedataset"], args["profile"]
    if profilen != nothing
        println("Profiling $profilen loss runs. Ignoring any \"save*\" arguments.")
    else
        saveweights == nothing && println("Warning: will not save model weights after training")
        savedataset == nothing && println("Warning: will not save the training dataset after training")
    end

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
        losses = losses, loss = loss,
        paramsvae = paramsvae)
    println("Final losses:")
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
    if savedataset != nothing
        println("Saving training dataset to \"$savedataset\"")
        BSON.@save savedataset g labels features
    else
        println("Not saving training dataset")
    end
end

main()
