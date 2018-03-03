#
# Importance Sampler
#

export AbstractImportanceSampler, ImportanceSampler

abstract type AbstractImportanceSampler end

function update!(is::AbstractImportanceSampler;
        X::Union{Void, AbstractMatrix{<:Real}}=nothing,  # samples
        F::Union{Void, AbstractMatrix{<:Real}}=nothing,  # function values
        W::Union{Void, AbstractVector{<:Real}}=nothing,  # weights
        N::Int=0, nbatches=0, batchsize::Int=0,
        kwds...)
    isa(X, Void) || size(X, 1) == length(ImportanceSampler.q) ||
        error("X must share the same dimension as q")
    isa(F, Void) || size(F, 1) == length(ImportanceSampler.μ) ||
        error("f must share the same dimension as μ")
    sum([N, nbatches, batchsize] .== 0) == 2 ||
        error("exactly two of N, nbatches, and batchsize must be provided")
    all([N, nbatches, batchsize] .≥ 0) ||
        error("N, nbatches, and batchsize must non-negative")

    isa(W, Void) || (W = W')
    all_equal( size.(filter(A -> !isa(A, Void), [X, F, W]), 2) ) ||
        error("X, F, and W must all have the same number of observations")

    N == 0 && (N = batchsize*nbatches)
    nbatches == 0 && (nbatches = N/batchsize)
    batchsize == 0 && (batchsize = N/nbatches)

    return _update!(is, X, F, W, N, nbatches, batchsize; kwds...)
end

mean(is::AbstractImportanceSampler) = mean(is.μ)
cov(is::AbstractImportanceSampler) = cov(is.μ)
var(is::AbstractImportanceSampler) = var(is.μ)

proposal(is::AbstractImportanceSampler) = is.q

diagnostics(is::AbstractImportanceSampler) = is.d

weightfun(is::AbstractImportanceSampler) = is.w

gen_default_w(p, q<:Distribution) =
    default_w(x::AbstractVector{<:Real}) = exp(logpdf(p, x) - logpdf(q, x))


_calcF(is::ImportanceSampler, X::Matrix{<:Real}) =
    _calcF!(Matrix{Float64}(length(is.μ), size(X, 2)), is, X)

@inline function _calcF!(F::Matrix{Float64}, is::ImportanceSampler, X::Matrix{<:Real})
    n = size(X, 2)

    for i in 1:n
        @inbounds is.f!(view(F, :, i), view(X, :, i))
    end

    return F
end

_calcW(is::ImportanceSampler, X::Matrix{<:Real}) =
    _calcW!(Vector{Float64}(size(X, 2)), is, X)

@inline function _calcW!(W::Vector{Float64}, is::ImportanceSampler, X::Matrix{<:Real})
    n = size(X, 2)

    for i in 1:n
        @inbounds W[i] = is.w(view(X, :, i))
    end

    return W
end

##########################################################################################
#                   Default IS
##########################################################################################
struct ImportanceSampler <: AbstractImportanceSampler
    q::Distribution
    μ::MeanVariance
    d::Diagnostics
    w
    f!

    function ImportanceSampler(f!, lengthf::Int, q<:Distribution ; p=nothing, w=nothing)
        xor(isa(p, Void), isa(w, Void)) ||
            error("Only one of p or w must be provided")
        isa(p, Void) || isa(p, Distribution) ||
            error("p must <: Distributions.Distribution")
        is(w, Void) && (w = default_w(p, q))

        lengthf ≥ 1 ||
            error("f must have a dimension greater than 0")

        return new(q, MeanVariance(lengthf), Diagnostics(), w, f!)
    end
end

function _update!(is::ImportanceSampler,
        X::AbstractMatrix{<:Real},
        F::Union{Void, AbstractMatrix{<:Real}},
        W::Union{Void, AbstractMatrix{<:Real}},
        N::Int, nbatches::Int, batchsize::Int;
        updateμ::Bool=true, _...)

    updateμ || return

    isa(F, Void) && (F = _calcF(is, X))
    isa(W, Void) && (W = _calcW(is, X))

    F .*= W'
    update!(is.μ, F)
    update!(is.d, W)
end
