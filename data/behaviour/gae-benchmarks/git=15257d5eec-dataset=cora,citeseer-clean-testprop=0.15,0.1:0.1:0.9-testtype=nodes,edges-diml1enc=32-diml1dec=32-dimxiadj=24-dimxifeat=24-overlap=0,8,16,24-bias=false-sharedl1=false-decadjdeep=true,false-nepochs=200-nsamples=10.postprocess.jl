using DataFrames, NPZ, GZip, CSV, ProgressMeter


folder = "git=15257d5eec-dataset=cora,citeseer-clean-testprop=0.15,0.1:0.1:0.9-testtype=nodes,edges-diml1enc=32-diml1dec=32-dimxiadj=24-dimxifeat=24-overlap=0,8,16,24-bias=false-sharedl1=false-decadjdeep=true,false-nepochs=200-nsamples=10"
pattern = Regex(folder * "/dataset=([^-]+(?:-clean)?)-testtype=([^-]+)-testprop=([^-]+)-decadjdeep=([^-]+)-dimxi=([^-]+)-sample=([^-]+).npz")
basedir = "data/behaviour/gae-benchmarks/"
refnfiles = 3200

for (root, dirs, files) in walkdir(basedir * folder)
    println("root = $root")

    npzfiles = filter((f) -> f[end-2:end] == "npz", files)
    nfiles = length(npzfiles)
    println("$nfiles files")
    if nfiles != refnfiles
        continue
    end

    outpath = "$root.csv.gz"
    if stat(outpath).size > 0
        println("$outpath already exists")
        continue
    end

    df = Array{Dict}(undef, nfiles)
    nepochs = nothing
    @showprogress for (i, file) in enumerate(npzfiles)
        path = joinpath(root, file)
        m = match(pattern, path)
        if m === nothing
            println("No match: $path")
            continue
        end

        history = npzread(path)
        r = Dict()
        for (k, v) in history
            if nepochs === nothing
                nepochs = length(v)
            else
                @assert nepochs == length(v)
            end
            r[k] = v
        end

        dataset, testtype, testprop, decadjdeep, dimξ, sample = m[1], m[2], parse(Float32, m[3]), parse(Bool, m[4]), parse(Int, m[5]), parse(Int, m[6])
        r["epoch"] = 1:nepochs
        r["dataset"] = repeat([dataset], nepochs)
        r["testtype"] = repeat([testtype], nepochs)
        r["testprop"] = repeat([testprop], nepochs)
        r["decadjdeep"] = repeat([decadjdeep], nepochs)
        r["dimξ"] = repeat([dimξ], nepochs)
        r["sample"] = repeat([sample], nepochs)

        df[i] = r
    end

    println("equalizing dicts")
    cols = union(Set.(keys.(df))...)
    for r in df
        for col in cols
            if !haskey(r, col)
                r[col] = repeat([0f0], nepochs)
            end
        end
    end

    println("saving to $outpath")
    df = vcat(DataFrame.(df)...)
    GZip.open(outpath, "w") do io
        CSV.write(io, df)
    end

end
