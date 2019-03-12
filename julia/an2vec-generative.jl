include("utils.jl")
include("generative.jl")
include("vae.jl")
using .Utils
using .Generative
using .VAE

using Flux
import BSON
using ArgParse
using Profile
import JLD
using NPZ


# Parameters
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
    @assert 0 <= parsed["correlation"] <= 1
    parsed["diml1"] = Int64(round(sqrt(parsed["l"] * (parsed["dimxiadj"] + parsed["dimxifeat"]))))
    parsed["initb"] = if parsed["bias"]; zeros; else VAE.Layers.nobias; end
    parsed["feature-distribution"] = VAE.Categorical
    parsed
end


"""Define the graph and features."""
function dataset(args)
    l, k, p_in, p_out, gseed, correlation = args["l"], args["k"], args["p_in"], args["p_out"], args["gseed"], args["correlation"]

    g, communities = Generative.make_sbm(l, k, p_in, p_out, gseed = gseed)
    colors = Generative.make_colors(communities, correlation)
    colorsoh = Array{Float32}(Utils.onehotmaxbatch(colors))

    g, convert(Array{Float32}, colorsoh), convert(Array{Float32}, scale_center(colorsoh))
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
    enc, sampleξ, dec, paramsenc, paramsdec = VAE.make_vae(
        g = g, feature_size = feature_size, args = args)
    losses, loss = VAE.make_losses(
        g = g, labels = labels, feature_size = feature_size, args = args,
        enc = enc, sampleξ = sampleξ, dec = dec,
        paramsenc = paramsenc, paramsdec = paramsdec)

    if profilen != nothing
        println("Profiling loss runs...")
        Utils.repeat_fn(1, loss, features)  # Trigger compilation
        Profile.clear()
        Profile.init(n = 10000000)
        @profile Utils.repeat_fn(profilen, loss, features)
        li, lidict = Profile.retrieve()
        println("Saving profile results to \"$(profile_losses_filename)\"")
        JLD.@save profile_losses_filename li lidict

        return
    end

    println("Training...")
    paramsvae = Flux.Params()
    push!(paramsvae, paramsenc..., paramsdec...)
    history = VAE.train_vae!(
        args = args, features = features,
        losses = losses, loss = loss,
        paramsvae = paramsvae)
    println("Final losses:")
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
    if savedataset != nothing
        println("Saving training dataset to \"$savedataset\"")
        BSON.@save savedataset g labels features
    else
        println("Not saving training dataset")
    end
end

main()
