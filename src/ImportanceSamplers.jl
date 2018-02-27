module ImportanceSamplers

using Distributions

import Base: show, size, mean, cov, var, std
import Base.Random: rand

import StatsBase: ProbabilityWeights, pweights, sample
import StatsFuns: logsumexp
import Distributions: probs, ncomponents, components, _rand!,
        pdf, logpdf, _logpdf, _logpdf!

include("auxiliary.jl")
include("online_statistics.jl")
include("mixture_distribution.jl")

end
