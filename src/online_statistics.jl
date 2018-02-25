#
# Compute Statistics Online
#

abstract type OnlineSampler end

mutable struct MeanVariance{T} <: OnlineSampler where T <: Real
    N::Int
    μ::T
end
