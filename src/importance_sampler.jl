#
# Importance Sampler
#

export AbstractImportanceSampler, ImportanceSampler,
    proposal, diagnostics, weightfun,
    CvImportanceSampler, updateμ!, updateβ!, coeffs

abstract type AbstractImportanceSampler end

function update!(is::AbstractImportanceSampler;
        X::Union{Void, AbstractVecOrMat{<:Real}}=nothing,  # samples
        F::Union{Void, AbstractVecOrMat{<:Real}}=nothing,  # function values
        W::Union{Void, AbstractVector{<:Real}}=nothing,  # weights
        niters::Int=0, nbatches::Int=0, batchsize::Int=0,
        kwds...)

    isXvoid = isa(X, Void)
    isFvoid = isa(F, Void)
    isWvoid = isa(W, Void)

    # make sure X, F are matrices (1 × something)
    # W cant handle matrices, X and F need it
    X = reshape_vector(X)
    F = reshape_vector(F)

    # dimensions checking
    isXvoid || size(X, 1) == length(is.q) ||
        error("X must share the same dimension as q")

    isFvoid || size(F, 1) == length(is.μ) ||
        error("f must share the same dimension as μ")

    all_equal( datasize.(filter(A -> !isa(A, Void), [X, F, W])) ) ||
        error("X, F, and W must all have the same number of observations")

    # decide if we need to sample X
    sample = false
    if isXvoid
        xor(isFvoid, isWvoid) &&
            error("both F and W must be provided if X is not")

        # no data passed, so pretty up batching and iters
        if isFvoid && isWvoid
            all([niters, nbatches, batchsize] .≥ 0) ||
                error("niters, nbatches, and batchsize must non-negative")

            sum([niters, nbatches, batchsize] .== 0) ≤ 1 ||
                error("atleast two of niters, nbatches, and batchsize must be provided")

            niters == 0 && (niters = Int(batchsize*nbatches))
            nbatches == 0 && ((niters, nbatches) = round_div(niters, batchsize))
            batchsize == 0 && ((niters, batchsize) = round_div(niters, nbatches))

            # if user provides all 3
            (niters == nbatches * batchsize) ||
                error("niters == nbatches * batchsize")

            sample = true
        end
    end

    return _update!(Val{sample}(), is, X, F, W, niters, nbatches, batchsize; kwds...)
end

mean(is::AbstractImportanceSampler) = mean(is.μ)
# divide by N to get estimator (co)var, not sample
cov(is::AbstractImportanceSampler) = cov(is.μ) / is.μ.N
var(is::AbstractImportanceSampler) = var(is.μ) / is.μ.N

for fun in Symbol.(["mean_eff_sample_size", "ne", "eff_sample_size", "neμ",
    "var_eff_sample_size", "neσ", "skew_eff_sample_size", "neγ"])
    eval( quote
        $(fun)(is::AbstractImportanceSampler) = $(fun)(diagnostics(is))
    end )
end

proposal(is::AbstractImportanceSampler) = is.q

diagnostics(is::AbstractImportanceSampler) = is.d

weightfun(is::AbstractImportanceSampler) = is.w

rand(is::AbstractImportanceSampler) = rand(is.q)
rand(is::AbstractImportanceSampler, n::Int...) = rand(is.q, n...)
rand!(X::AbstractArray, is::AbstractImportanceSampler) = rand(X, is.q)

default_w(p, q::UnivariateDistribution) =
    (x::AbstractVector{<:Real}) -> begin
        x = x[1]
        exp(logpdf(p, x) - logpdf(q, x))
    end

default_w(p, q::Distribution) =
    (x::AbstractVector{<:Real}) -> exp(logpdf(p, x) - logpdf(q, x))

calcW(is::AbstractImportanceSampler, X::Matrix{<:Real}) =
    calcW!(Vector{Float64}(size(X, 2)), is, X)

@inline function calcW!(W::AbstractVector{Float64}, is::AbstractImportanceSampler, X::Matrix{<:Real})
    n = size(X, 2)

    for i in 1:n
        @inbounds W[i] = is.w(view(X, :, i))
    end

    return W
end

calcQ(is::AbstractImportanceSampler, X::Matrix{<:Real}) = safe_pdf(is.q, X)

calcQ!(Q::AbstractVector{Float64}, is::AbstractImportanceSampler, X::Matrix{<:Real}) =
    safe_pdf!(Q, is.q, X)

calcF(is::AbstractImportanceSampler, X::Matrix{<:Real}) =
    calcF!(Matrix{Float64}(length(is.μ), size(X, 2)), is, X)

@inline function calcF!(F::AbstractMatrix{Float64}, is::AbstractImportanceSampler, X::Matrix{<:Real})
    n = size(X, 2)

    for i in 1:n
        @inbounds is.f!(view(F, :, i), view(X, :, i))
    end

    return F
end

calcG(is::AbstractImportanceSampler, X::Matrix{<:Real}) =
    calcG!(Matrix{Float64}(size(is.β, 2), size(X, 2)), is, X)

@inline function calcG!(G::AbstractMatrix{Float64}, is::AbstractImportanceSampler, X::Matrix{<:Real})
    n = size(X, 2)

    lens = length.(is.θs)
    cumlens = cumsum(vcat(1, lens))
    rs = [cumlens[i] : cumlens[i+1]-1 for i in 1:length(is.θs)]

    for i in 1:n
        for (j, g!) in enumerate(is.g!s)
            @inbounds g!(view(G, rs[j], i), view(X, :, i))
        end
    end

    return G
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
        isa(w, Void) && (w = default_w(p, q))

        lengthf ≥ 1 ||
            error("f must have a dimension greater than 0")

        return new(q, MeanVariance(lengthf), Diagnostic(), w, f!)
    end
end

function core_update(is::ImportanceSampler,
    F::AbstractMatrix{<:Real},
    W::AbstractVector{<:Real})

    F .= sign.(F) .* exp.(log.(abs.(F)) .+ log.(W'))

    update!(is.μ, F)
    update!(is.d, W)

    return is
end

function _update!(::Val{false},
        is::ImportanceSampler,
        X::Union{Void, AbstractMatrix{<:Real}},
        F::Union{Void, AbstractMatrix{<:Real}},
        W::Union{Void, AbstractVector{<:Real}},
        niters::Int, nbatches::Int, batchsize::Int;
        updateμ::Bool=true, _...)

    updateμ || return

    isa(F, Void) && (F = calcF(is, X))
    isa(W, Void) && (W = calcW(is, X))

    return core_update(is, F, W)
end

function _update!(::Val{true},
        is::ImportanceSampler,
        X::Union{Void, AbstractMatrix{<:Real}},
        F::Union{Void, AbstractMatrix{<:Real}},
        W::Union{Void, AbstractVector{<:Real}},
        niters::Int, nbatches::Int, batchsize::Int;
        updateμ::Bool=true, _...)

    updateμ || return

    X = Matrix{Float64}(length(is.q), batchsize)
    F = Matrix{Float64}(length(is.μ), batchsize)
    W = Vector{Float64}(batchsize)

    for _ in 1:nbatches
        rand!(is.q, X)
        calcF!(F, is, X)
        calcW!(W, is, X)

        core_update(is, F, W)
    end

    return is
end

##########################################################################################
#                   Control Variate IS
##########################################################################################
mutable struct CvImportanceSampler <: AbstractImportanceSampler
    q::Distribution
    μ::MeanVariance
    d::Diagnostic
    β::ControlVariate
    w
    f!
    g!s
    θs::Vector{Vector{Float64}}

    function CvImportanceSampler(f!, lengthf::Int,
            q::Distribution;
            g!s::Union{AbstractVector{<:Tuple{Any, Vector{Float64}}}, Void}=nothing, # [(g!, θ)...]
            p=nothing, w=nothing,
            use_q::Bool=false # use the q (or components of) for control variates
            )

        isa(g!s, Void) && !use_q &&
            error("either g!s must be provided, or q is used")

        xor(isa(p, Void), isa(w, Void)) ||
            error("Only one of p or w must be provided")
        isa(w, Void) && (w = default_w(p, q))

        lengthf ≥ 1 ||
            error("f must have a dimension greater than 0")

        μ = MeanVariance(lengthf)
        d = Diagnostic()

        if use_q
            _qs = (isa(q, MixtureDistribution)) ? q.components : [q]
            _g!s = [(r, x) -> safe_pdf!(r, q, x) for q in _qs]
            _θs = fill([1.0], length(_qs))
        end

        if !isa(g!s, Void)
            if use_q
                _g!s = append!(_g!s, first.(g!s))
                _θs = append!(_θs, last.(g!s))
            else
                _g!s = first.(g!s)
                _θs = last.(g!s)
            end
        end

        β = ControlVariate(lengthf, sum(length, _θs))

        return new(q, μ, d, β, w, f!, _g!s, _θs)
    end
end

# hacky method, but for edge case when length(f) == 1
mean(is::CvImportanceSampler) = mean(is.μ) + make_scalar(is.β.β' * vcat(is.θs...))

coeffs(is::CvImportanceSampler) = coeffs(is.β)

updateμ!(is::CvImportanceSampler; kwds...) = update!(is; kwds..., updateμ=true, updateβ=false)
updateβ!(is::CvImportanceSampler; kwds...) = update!(is; kwds..., updateμ=false, updateβ=true)

# the regression is f*w ≈ βᵀh/q, and μ ≈ βᵀθ + ∑ f*w - βᵀh/q,
# so compute f*w and h/q before doing anything else
# use exp and logs b/c W can be ≈ 0, or very, very large if q is poorly chosen
function core_update(is::CvImportanceSampler,
        F::AbstractMatrix{<:Real},
        G::AbstractMatrix{<:Real},
        W::AbstractVector{<:Real},
        Q::AbstractVector{<:Real},
        updateμ::Bool, updateβ::Bool)

    # f*w and h/q
    F .= sign.(F) .* exp.(log.(abs.(F)) .+ log.(W'))
    G .= sign.(G) .* exp.(log.(abs.(G)) .- log.(Q'))

    # β is g/q regressed onto f*w
    updateβ && update!(is.β, F, G)
    updateμ || return is

    β = is.β.β
    F .-= β' * G

    update!(is.μ, F)
    update!(is.d, W)

    return is
end

# use provided values
function _update!(::Val{false},
        is::CvImportanceSampler,
        X::Union{Void, AbstractMatrix{<:Real}},
        F::Union{Void, AbstractMatrix{<:Real}},
        W::Union{Void, AbstractVector{<:Real}},
        niters::Int, nbatches::Int, batchsize::Int;
        G::Union{Void, AbstractVecOrMat{<:Real}}=nothing,
        Q::Union{Void, AbstractVector{<:Real}}=nothing,
        updateμ::Bool=true, updateβ::Bool=false, _...)

    updateμ || updateβ || return

    # more dims checking for F & Q
    G = reshape_vector(G)

    isa(G, Void) || size(G, 1) == size(is.β, 2) ||
        error("h must share the same dimension as β")

    all_equal( datasize.(filter(A -> !isa(A, Void), [X, F, G, W, Q])) ) ||
        error("X, F, G, W, and Q must all have the same number of observations")

    # not sampling, so assume that either we have F&W, or X and some others
    isa(X, Void) && any(is(G, Void), isa(Q, Void)) &&
        error("F, G, W, and Q must be provided if X is not")

    isa(F, Void) && (F = calcF(is, X))
    isa(G, Void) && (G = calcG(is, X))
    isa(W, Void) && (W = calcW(is, X))
    isa(Q, Void) && (Q = calcQ(is, X))

    return core_update(is, F, G, W, Q, updateμ, updateβ)
end

# sample
function _update!(::Val{true},
        is::CvImportanceSampler,
        X::Union{Void, AbstractMatrix{<:Real}},
        F::Union{Void, AbstractMatrix{<:Real}},
        W::Union{Void, AbstractVector{<:Real}},
        niters::Int, nbatches::Int, batchsize::Int;
        G::Union{Void, AbstractVecOrMat{<:Real}}=nothing,
        Q::Union{Void, AbstractVector{<:Real}}=nothing,
        updateμ::Bool=true, updateβ::Bool=true, _...)

    updateμ || updateβ || return is

    X = Matrix{Float64}(length(is.q), batchsize)
    F = Matrix{Float64}(length(is.μ), batchsize)
    G = Matrix{Float64}(size(is.β, 2), batchsize)
    W = Vector{Float64}(batchsize)
    Q = Vector{Float64}(batchsize)

    for _ in 1:nbatches
        rand!(is.q, X)
        calcF!(F, is, X)
        calcG!(G, is, X)
        calcW!(W, is, X)
        calcQ!(Q, is, X)

        core_update(is, F, G, W, Q, updateμ, updateβ)
    end

    return is
end
