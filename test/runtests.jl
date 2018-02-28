#
# main file to run tests
#

using ImportanceSampling
using Base.Test

approx_eqaul(x1, x2=x1; tol=1e-13) =
    all(map((a,b) -> abs(a-b) ≤ tol, x1, x2))

include("test_online_statistics.jl")
