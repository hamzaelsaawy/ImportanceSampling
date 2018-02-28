#
# mixture distribution for importance sampling
#

export MixtureDistribution

struct MixtureDistribution{F<:VariateForm, S<:ValueSupport} <: Distribution{F, S}
    components::Vector{D} where D<:Distribution{F, S}
    prior::ProbabilityWeights
    log_prior::Vector{Float64}

    function MixtureDistribution{F, S}(
            components::AbstractArray{DD}, prior::AbstractArray{R}) where
            {F<:VariateForm, S<:ValueSupport, DD<:Distribution{F, S}, R<:Real}
        (length(prior) == length(components)) ||
                error("must be the same number of components and prior")

        all(α -> α ≥ 0, prior) || error("prior must be ≥ 0")
        _prior = ProbabilityWeights(vec(collect(prior))/sum(prior), 1.0)

        _components = vec(components)
        all_equal(size, _components) || error("components must have the same size")
        all_equal(length, _components) || error("components must have the same length")
        all_equal(eltype, _components) || error("components must have the same element type")

        # gives an error for support(Poisson)
        #(F == Univariate) && ( all_equal(support, _components) ||
        #       error("Distributions must share the same support") )

        return new(_components, _prior, log.(_prior))
    end
end

MixtureDistribution(components::AbstractArray{DD}, prior::AbstractArray{R}) where
        {F<:VariateForm, S<:ValueSupport, DD<:Distribution{F, S}, R<:Real} =
        MixtureDistribution{F, S}(components, prior)

MixtureDistribution(components::AbstractArray{DD}) where
        {F<:VariateForm, S<:ValueSupport, DD<:Distribution{F, S}} =
    MixtureDistribution(components, ones(length(components)))

for fun in Symbol.(["length", "size", "eltype"])
    eval( quote
        Base.$(fun)(d::MixtureDistribution) = $fun(first(d.components))
    end )
end

probs(d::MixtureDistribution) = d.prior.values
ncomponents(d::MixtureDistribution) = length(d.components)
components(d::MixtureDistribution) = d.components

#
# rand
#
rand(d::MixtureDistribution{Univariate}) = rand(d.components[sample(d.prior)])

_rand!(d::MixtureDistribution{Multivariate}, x::AbstractVector{T}) where T<:Real =
        _rand!(d.components[sample(d.prior)], x)

_rand!(d::MixtureDistribution{Matrixvariate}, x::AbstractMatrix{T}) where T<:Real =
        _rand!(d.components[sample(d.prior)], x)

#
# (log) pdf
#
# univariate
logpdf(d::MixtureDistribution{Univariate}, x::Real) =
        logsumexp(d.log_prior + logpdf.(d.components, x))
pdf(d::MixtureDistribution{Univariate}, x::Real) = exp(logpdf(d, x))

# multivariate
@inline function _logpdf(d::MixtureDistribution{Multivariate}, x::AbstractVector{T}) where T<:Real
    t = Vector{Float64}(ncomponents(d))

    for j in 1:ncomponents(d)
        @inbounds t[j] = _logpdf(d.components[j], x)
    end

    return logsumexp(d.log_prior .+ t)
end

@inline function _logpdf!(r::AbstractArray, d::MixtureDistribution{Multivariate},
        x::AbstractMatrix{T}) where T<:Real
    t = Matrix{Float64}(size(x, 2), ncomponents(d))

    for j in 1:ncomponents(d)
        _logpdf!(view(t, :, j), d.components[j], x)
    end

    t .+= d.log_prior'

    for i in 1:size(x, 2)
        @inbounds r[i] = logsumexp(t[i, :])
    end

    return r
end

# matrixvariate
@inline function _logpdf(d::MixtureDistribution{Matrixvariate}, x::AbstractMatrix)
    t = Vector{Float64}(ncomponents(d))

    for j in 1:ncomponents(d)
        @inbounds t[j] = _logpdf(d.components[j], x)
    end

    return logsumexp(d.log_prior .+ t)
end

#
# show
#
function show(io::IO, d::MixtureDistribution)
    println(io, string(typeof(d)))
    println(io, "prior: ", d.prior)
    println(io, "components:")
    println.(io, d.components)
end
