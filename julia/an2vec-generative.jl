include("utils.jl")
include("generative.jl")
include("vae.jl")
using .Utils
using .Generative
using .VAE

using LinearAlgebra
using Flux
import BSON
using ArgParse
using Profile
import JLD
using NPZ
using StatsBase


# Parameters
const profile_losses_filename = "an2vec-losses.jlprof"

"""Parse CLI arguments."""
function parse_cliargs()
    parse_settings = ArgParseSettings()
    @add_arg_table parse_settings begin
        "-l"
            help = "number of structural communities"
            arg_type = Int
            # default = 10
            required = true
        "-k"
            help = "size of each structural community"
            arg_type = Int
            # default = 10
            required = true
        "--p_in"
            help = "structural intra-community connection probability"
            arg_type = Float64
            default = 0.4
        "--p_out"
            help = "structural extra-community connection probability"
            arg_type = Float64
            default = 0.01
        "--correlation"
            help = "correlation between features and structural communities"
            arg_type = Float64
            required = true
        "--featuretype"
            help = "generative model for the features; one of \"colors\", \"sbm\", \"lowrank\""
            arg_type = String
            required = true
        "--fsbm_l"
            help = "number of sbmfeatures communities (ignore when using colors)"
            arg_type = Int
        "--fsbm_k"
            help = "size of each sbmfeatures community (ignore when using colors)"
            arg_type = Int
        "--fsbm_p_in"
            help = "sbmfeatures intra-community connection probability (ignore when using colors)"
            arg_type = Float64
        "--fsbm_p_out"
            help = "sbmfeatures extra-community connection probability (ignore when using colors)"
            arg_type = Float64
        "--fsbm_seed"
            help = "sbmfeatures graph seed"
            arg_type = Int
            default = -1
        "--flowrank_dim"
            help = "Dimension for low rank clustered hyperplane features"
            arg_type = Int
        "--flowrank_rank"
            help = "Rank for low rank clustered hyperplane features"
            arg_type = Int
        "--flowrank_noisescale"
            help = "Scale of noise around each community centroid"
            arg_type = Float64
        "--flowrank_catdiag"
            help = "Whether or not to concatenate a diagonal matrix to the features"
            arg_type = Bool
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
    parsed["initb"] = parsed["bias"] ? (s) -> zeros(Float32, s) : VAE.Layers.nobias
    parsed["label-distribution"], dimfeat, dimlabels = if parsed["featuretype" ] == "colors"
        (VAE.Categorical, parsed["l"], parsed["l"])
    elseif parsed["featuretype" ] == "sbm"
        @assert all((k) -> parsed[k] != nothing, ["fsbm_l", "fsbm_k", "fsbm_p_in", "fsbm_p_out"])
        @assert parsed["l"] * parsed["k"] == parsed["fsbm_l"] * parsed["fsbm_k"]
        (VAE.Bernoulli, parsed["l"] * parsed["k"], parsed["l"] * parsed["k"])
    else
        @assert parsed["featuretype"] == "lowrank"
        @assert parsed["flowrank_rank"] != nothing
        @assert parsed["flowrank_dim"] != nothing
        @assert parsed["flowrank_noisescale"] != nothing
        @assert parsed["flowrank_catdiag"] != nothing
        (VAE.Normal, parsed["flowrank_dim"] + (parsed["flowrank_catdiag"] ? parsed["l"] * parsed["k"] : 0), parsed["flowrank_dim"] )
    end
    parsed["diml1enc"] = Int64(round(geomean([dimfeat, parsed["dimxiadj"] + parsed["dimxifeat"]])))
    parsed["diml1dec"] = Int64(round(geomean([dimlabels, parsed["dimxiadj"] + parsed["dimxifeat"]])))
    parsed
end


"""Define the graph and features."""
function dataset(args)
    l, k, p_in, p_out, gseed, correlation = args["l"], args["k"], args["p_in"], args["p_out"], args["gseed"], args["correlation"]

    g, communities = Generative.make_sbm(l, k, p_in, p_out, gseed = gseed)

    features, labels = if args["featuretype"] == "colors"
        colors = Generative.make_colors(communities, correlation)
        features = Array{Float32}(Utils.onehotmaxbatch(colors))
        features, features
    elseif args["featuretype"] == "sbm"
        fsbm_l, fsbm_k, fsbm_p_in, fsbm_p_out, fsbm_seed = args["fsbm_l"], args["fsbm_k"], args["fsbm_p_in"], args["fsbm_p_out"], args["fsbm_seed"]
        sbmfeatures, _ = Generative.make_sbmfeatures(fsbm_l, fsbm_k, fsbm_p_in, fsbm_p_out, correlation; gseed = fsbm_seed)
        features = Array{Float32}(sbmfeatures)
        features, features
    else
        @assert args["featuretype"] == "lowrank"
        flowrank_dim, flowrank_rank, flowrank_noisescale = args["flowrank_dim"], args["flowrank_rank"], args["flowrank_noisescale"]
        features = Generative.make_clusteredhyperplane(Float32, (flowrank_dim, l * k), flowrank_rank, flowrank_noisescale, communities, correlation)
        @assert rank(features) == flowrank_rank
        labels = features
        if args["flowrank_catdiag"]
            features = vcat(features, Diagonal(ones(Float32, l * k)))
        end
        features, labels
    end

    @assert eltype(features) == Float32
    @assert eltype(labels) == Float32
    g, features, labels
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
    g, _features, labels = dataset(args)
    feature_size = size(_features, 1)
    label_size = size(labels, 1)
    features = normaliser(_features)(_features)

    println("Making the model")
    enc, sampleξ, dec, paramsenc, paramsdec = VAE.make_vae(
        g = g, feature_size = feature_size, label_size = label_size, args = args)
    losses, loss = VAE.make_losses(
        g = g, labels = labels, label_size = label_size, args = args,
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
        BSON.@save savedataset g labels
    else
        println("Not saving training dataset")
    end
end

main()
