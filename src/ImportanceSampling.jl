#
# Importance Sampling main module
#

module ImportanceSampling

using Distributions

import Base: show, size, ndims, mean, cov, var, std
import Base.Random: rand

import StatsBase: ProbabilityWeights, pweights, sample
import StatsFuns: logsumexp
import Distributions: probs, ncomponents, components, _rand!,
        pdf, logpdf, _logpdf, _logpdf!

include("auxiliary.jl")
include("mixture_distribution.jl")
include("online_statistics.jl")

end
