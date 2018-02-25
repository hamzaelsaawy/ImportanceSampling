#
# A Mixture Distribution for Q sampling
#

export MixtureDistribution

struct MixtureDistribution{F<:VariateForm, S<:ValueSupport} <: Distribution{F, S}
    αs::ProbabilityWeights
    qs::Vector{D} where D<:Distribution{F, S}

    function MixtureDistribution{F, S}(
            qs::AbstractArray{DD}, αs::AbstractArray{R}) where
            {F<:VariateForm, S<:ValueSupport, DD<:Distribution{F, S}, R<:Real}
        (length(αs) == length(qs)) || error("αs and qs must be the same length")

        all(α -> α ≥ 0, αs) || error("αs must be ≥ 0")
        _αs = ProbabilityWeights(vec(collect(αs))/sum(αs), 1.0)

        _qs = vec(qs)
        all_equal(size, _qs) || error("Distributions must share the same size")
        all_equal(length, _qs) || error("Distributions must share the same length")
        all_equal(eltype, _qs) || error("Distributions must share the same element type")

        # gives an error for support(Poisson)
        #(F == Univariate) && ( all_equal(support, _qs) ||
        #       error("Distributions must share the same support") )

        return new(_αs, _qs)
    end
end

MixtureDistribution(qs::AbstractArray{DD}, αs::AbstractArray{R}) where
        {F<:VariateForm, S<:ValueSupport, DD<:Distribution{F, S}, R<:Real} =
        MixtureDistribution{F, S}(qs, αs)

MixtureDistribution(qs::AbstractArray{DD}) where
        {F<:VariateForm, S<:ValueSupport, DD<:Distribution{F, S}} =
    MixtureDistribution(qs, ones(length(qs)))

for fun in Symbol.(["length", "size", "eltype"])
    eval( quote
        Base.$(fun)(q::MixtureDistribution) = $fun(first(q.qs))
    end )
end

Base.Random.rand(q::MixtureDistribution{Univariate}) = rand(q.qs[sample(q.αs)])
Distributions._rand!(q::MixtureDistribution{Multivariate}, x::AbstractVector) =
        Distributions._rand!(q.qs[sample(q.αs)], x)
