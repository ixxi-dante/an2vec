using Flux, Flux.Data.MNIST, Statistics, CuArrays, Random, Distributions
using Flux: throttle, params
using Juno: @progress

MaybeTrackedArray{A} = Union{A, TrackedArray{T,N,A} where {T,N}} where A

# Have a numerically stable logpdf for `p` close to 1 or 0.
# TODO: question: why does .+ instead of + between the two log() hang?
logpdfbernoulli(p::MaybeTrackedArray, y::MaybeTrackedArray) = y .* log.(p .+ eps()) + (1 .- y) .* log.(1 .- p .+ eps())

# Load data, binarise it, and partition into mini-batches of M.
X = (float.(hcat(vec.(MNIST.images())...)) .> 0.5) |> gpu
N, M = size(X, 2), 100
data = [X[:,i] for i in Iterators.partition(1:N, M)]


################################# Define Model #################################

# Latent dimensionality, # hidden units.
Dz, Dh = 2, 10

# Components of recognition model / "encoder" MLP.
A = Dense(28^2, Dh, tanh) |> gpu
μ = Dense(Dh, Dz) |> gpu
logσ = Dense(Dh, Dz) |> gpu
g(X) = (h = A(X); (μ(h), logσ(h)))
# TODO: update to this line once CURAND improvements have hit tagged CuArrays + Flux (https://discourse.julialang.org/t/gpu-randn-way-slower-than-rand/18236/4)
#randn_like(target::MaybeTrackedArray{A}) where {A<:CuArray{T,N}} where {T,N} = (r = gpu(Array{T}(undef, size(target))); randn!(r); r)
randn_like(target::A) where A<:AbstractArray{T} where T = randn(T, size(target))
randn_like(target::MaybeTrackedArray{<:CuArray{T,N}}) where {T,N} = gpu(randn(T, size(target)))
z(μ, logσ) = μ .+ exp.(logσ) .* randn_like(μ)

# Generative model / "decoder" MLP.
f = Chain(Dense(Dz, Dh, tanh), Dense(Dh, 28^2, σ)) |> gpu


####################### Define ways of doing things with the model. #######################

# KL-divergence between approximation posterior and N(0, 1) prior.
kl_q_p(μ, logσ) = 0.5 * sum(exp.(2 .* logσ) + μ.^2 .- 1 .+ logσ.^2)

# logp(x|z) - conditional probability of data given latents.
logp_x_z(x, z) = sum(logpdfbernoulli(f(z), x))

# Monte Carlo estimator of mean ELBO using M samples.
L̄(X) = ((μ̂, logσ̂) = g(X); (logp_x_z(X, z(μ̂, logσ̂)) - kl_q_p(μ̂, logσ̂)) / M)

loss(X) = -L̄(X) + 0.01 * sum(x->sum(x.^2), params(f))

# Sample from the learned model.
modelsample() = rand.(Bernoulli.(cpu(f(gpu(z(zeros(Float32, Dz), zeros(Float32, Dz)))))))


################################# Learn Parameters ##############################

evalcb = throttle(() -> @show(-L̄(X[:, rand(1:N, M)])), 30)
opt = ADAM(params(A, μ, logσ, f))
@progress for i = 1:20
  @info "Epoch $i"
  Flux.train!(loss, zip(data), opt, cb=evalcb)
end


################################# Sample Output ##############################

using Images, FileIO

img(x) = Gray.(reshape(x, 28, 28))

cd(@__DIR__)
samples = hcat(img.([modelsample() for i = 1:10])...)
save(FileIO.File{FileIO.DataFormat{:PNG}}("sample.png"), samples)
