module ImportanceSamplers

import StatsBase: ProbabilityWeights, pweights, sample
using Distributions

include("auxiliary.jl")
include("online_statistics.jl")
include("mixture_distribution.jl")

end
