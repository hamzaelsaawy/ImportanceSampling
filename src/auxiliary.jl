#
# helper functions
#

all_equal(f, xs) = all(x -> f(x) == f(first(xs)), xs[2:end])
all_equal(xs) = all_equal(identity, xs)

outer(v, w=v) = A_mul_Bt(v, w)
outer!(A, v, w=v) = A_mul_Bt!(A, v, w)

_reshape_vector(A::Vector) = A'
_reshape_vector(A) = A

_safe_rand(q::UnivariateDistribution, ns::Int...) = rand(q, 1, ns...)
_safe_rand(q::Distribution, ns::Int...) = rand(q, ns...)

function round_div(n::Int, a::Int)
    b = ceil(Int, n/a)
    n = b*a
    return n, b
end
