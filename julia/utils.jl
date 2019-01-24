module Utils

export supertypes, finaleltype, scale_center, colorbar

using Makie, Statistics, Flux.Tracker

function supertypes(T)
    current = T
    println(current)
    while current != Any
        current = supertype(current)
        println(current)
    end
end

#finaleltype(::Type{<:AbstractArray{T}}) where {T} = T
#finaleltype(x::Type{<:TrackedArray{T}}) where {T<:Real} = T

#function scale_center(x; dims = 1)
#    x = x .- mean(x, dims = dims)
#    norm = sqrt.(sum(x .^ 2, dims = dims))
#    x ./ norm
#end

#function colorbar(x)
#    a = range(minimum(x), stop = maximum(x), length = 100) |> collect
#    heatmap(reshape(a, 1, :))
#end

end
