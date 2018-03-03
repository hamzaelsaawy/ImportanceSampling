#
# Importance Sampler
#

export AbstractImportanceSampler, ImportanceSampler,
    proposal, diagnostics, weightfun #coeffs

abstract type AbstractImportanceSampler end

function update!(is::AbstractImportanceSampler;
        X::Union{Void, AbstractVecOrMat{<:Real}}=nothing,  # samples
        F::Union{Void, AbstractVecOrMat{<:Real}}=nothing,  # function values
        W::Union{Void, AbstractVecOrMat{<:Real}}=nothing,  # weights
        niters::Int=0, nbatches::Int=0, batchsize::Int=0,
        kwds...)

    isXvoid = isa(X, Void)
    isFvoid = isa(F, Void)
    isWvoid = isa(W, Void)

    X = _reshape_vector(X)
    F = _reshape_vector(F)
    W = _reshape_vector(W)

    # dimensions checking
    isXvoid || size(X, 1) == length(is.q) ||
        error("X must share the same dimension as q")

    isWvoid || size(W, 1) == 1 ||
        error("W must be singleton observations only")

    isFvoid || size(F, 1) == length(is.μ) ||
        error("f must share the same dimension as μ")

    all_equal( size.(filter(A -> !isa(A, Void), [X, F, W]), 2) ) ||
        error("X, F, and W must all have the same number of observations")

    if isXvoid
        xor(isFvoid, isWvoid)
            error("both F or W must be provided if X is not")

        # no data passed, so pretty up batching and iters
        if isFvoid && isWvoid
            all([niters, nbatches, batchsize] .≥ 0) ||
                error("niters, nbatches, and batchsize must non-negative")

            sum([niters, nbatches, batchsize] .== 0) == 1 ||
                error("exactly two of niters, nbatches, and batchsize must be provided")

            niters == 0 && (niters = Int(batchsize*nbatches))
            nbatches == 0 && ((niters, nbatches) = round_div(niters, batchsize))
            batchsize == 0 && ((niters, batchsize) = round_div(niters, nbatches))
        end
    end


    return _update!(is, X, F, W, niters, nbatches, batchsize; kwds...)
end

mean(is::AbstractImportanceSampler) = mean(is.μ)
cov(is::AbstractImportanceSampler) = cov(is.μ)
var(is::AbstractImportanceSampler) = var(is.μ)

proposal(is::AbstractImportanceSampler) = is.q

diagnostics(is::AbstractImportanceSampler) = is.d

weightfun(is::AbstractImportanceSampler) = is.w

rand(is::AbstractImportanceSampler) = rand(is.q)
rand(is::AbstractImportanceSampler, n::Int...) = rand(is.q, n...)
rand!(X::AbstractArray, is::AbstractImportanceSampler) = rand(X, is.q)

default_w(p, q::UnivariateDistribution) =
    (x::AbstractVector{<:Real}) -> begin
        x = first(x)
        exp(logpdf(p, x) - logpdf(q, x))
    end

default_w(p, q::Distribution) =
    (x::AbstractVector{<:Real}) -> exp(logpdf(p, x) - logpdf(q, x))

calcF(is::AbstractImportanceSampler, X::Matrix{<:Real}) =
    calcF!(Matrix{Float64}(length(is.μ), size(X, 2)), is, X)

@inline function calcF!(F::Matrix{Float64}, is::AbstractImportanceSampler, X::Matrix{<:Real})
    n = size(X, 2)

    for i in 1:n
        @inbounds is.f!(view(F, :, i), view(X, :, i))
    end

    return F
end

calcW(is::AbstractImportanceSampler, X::Matrix{<:Real}) =
    calcW!(Vector{Float64}(size(X, 2)), is, X)

@inline function calcW!(W::Vector{Float64}, is::AbstractImportanceSampler, X::Matrix{<:Real})
    n = size(X, 2)

    for i in 1:n
        @inbounds W[i] = is.w(view(X, :, i))
    end

    return W
end

##########################################################################################
#                   Default IS
##########################################################################################
mutable struct ImportanceSampler <: AbstractImportanceSampler
    q::Distribution
    μ::MeanVariance
    d::Diagnostic
    w
    f!

    function ImportanceSampler(f!, lengthf::Int, q::Distribution ; p=nothing, w=nothing)
        xor(isa(p, Void), isa(w, Void)) ||
            error("Only one of p or w must be provided")
        isa(p, Void) || isa(p, Distribution) ||
            error("p must <: Distributions.Distribution")
        isa(w, Void) && (w = default_w(p, q))

        lengthf ≥ 1 ||
            error("f must have a dimension greater than 0")

        return new(q, MeanVariance(lengthf), Diagnostic(), w, f!)
    end
end

function _update!(is::ImportanceSampler,
        X::AbstractMatrix{<:Real},
        F::Union{Void, AbstractMatrix{<:Real}},
        W::Union{Void, AbstractMatrix{<:Real}},
        niters::Int, nbatches::Int, batchsize::Int;
        updateμ::Bool=true, _...)

    updateμ || return

    isa(F, Void) && (F = calcF(is, X))
    isa(W, Void) && (W = calcW(is, X))

    F .*= W'
    update!(is.μ, F)
    update!(is.d, W)
end
