module Layers


include("utils.jl")
using Flux, LightGraphs, LinearAlgebra, Memoize, .Utils


#
# Helpers
#

"""Helper to have no bias in Dense and GC layers."""
nobias(out::Integer) = fill(nothing, out)
Flux.param(n::AbstractArray{Nothing}) = fill(0, size(n))


#
# Helper layers
#

struct VOverlap{I<:Integer,F}
    overlap::I
    reducer::F
end
Flux.@treelike VOverlap
VOverlap(overlap) = VOverlap(overlap, mean)

function Base.show(io::IO, o::VOverlap)
    print(io, "VOverlap(", o.overlap)
    o.reducer == mean || print(io, ", ", o.reducer)
    print(io, ")")
end

@views function (o::VOverlap)(x1, x2)
    vcat(
        x1[1:end-o.overlap, :],
        o.reducer(x1[end-o.overlap+1:end, :], x2[1:o.overlap, :]),
        x2[1+o.overlap:end, :]
    )
end


struct Apply{V,T<:NTuple}
    f::V
    args::T
    Apply(f, args...) = new{typeof(f), typeof(args)}(f, args)
end
Flux.@treelike Apply

children(a::Apply) = (a.f, a.args...)
mapchildren(f, a::Apply) = Apply(f(a.f), f.(a.args)...)

function Base.show(io::IO, a::Apply)
    print(io, "Apply(", a.f, ", ")
    join(io, a.args, ", ")
    print(io, ")")
end

function (a::Apply)(x)
    f, args = a.f, a.args
    f(map(l -> l(x), args)...)
end


#
# Graph-convolutional layer
#

@memoize function adjacency_matrix_diag_norm(g::SimpleGraph)
    Adiag = adjacency_matrix_diag(g)
    Adiag_sumin_inv_sqrt = 1 ./ sqrt.(dropdims(sum(Adiag, dims = 1), dims = 1))
    Adiag_sumout_inv_sqrt = 1 ./ sqrt.(dropdims(sum(Adiag, dims = 2), dims = 2))
    diagm(0 => Adiag_sumout_inv_sqrt) * Adiag * diagm(0 => Adiag_sumin_inv_sqrt)
end

struct GC{S<:AbstractArray,T,U,F}
    Anorm::S
    W::T
    b::U
    σ::F
    function GC(g::SimpleGraph, W::T, b::U, σ::F) where {T,U,F}
        Anorm = adjacency_matrix_diag_norm(g)
        new{typeof(Anorm),T,U,F}(Anorm, W, b, σ)
    end
end
GC(g, W, b) = GC(g, W, b, identity)
Flux.@treelike GC

function GC(g::SimpleGraph, in::Integer, out::Integer, σ = identity;
        initW = Flux.glorot_uniform, initb = zeros)
    return GC(g, param(initW(out, in)), param(initb(out)), σ)
end

function Base.show(io::IO, l::GC)
    print(io, "GC(g ~ ", size(l.Anorm, 1), ", W ~ ", (size(l.W, 2), size(l.W, 1)), ", b ~ ")
    isa(l.b, TrackedArray) ? print(io, size(l.b, 1)) : print(io, "nothing")
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end

function (a::GC)(x::AbstractArray)
    W, Anorm, b, σ = a.W, a.Anorm, a.b, a.σ
    σ.(W * x * Anorm .+ b)
end


#
# Bilinear layer
#

struct Bilin{T,F}
    W::T
    σ::F
end
Bilin() = Bilin(identity)
Flux.@treelike Bilin

function Bilin(in::Integer = nothing, σ = identity; initW = Flux.glorot_uniform)
    W = in == nothing ? 1 : param(initW(in, in))
    return Bilin(W, σ)
end

function (a::Bilin)(x::AbstractArray)
    W, σ = a.W, a.σ
    σ.(transpose(x) * W * x)
end


end
