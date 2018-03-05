#
# helper functions
#

all_equal(f, xs) = all(x -> f(x) == f(first(xs)), xs[2:end])
all_equal(xs) = all_equal(identity, xs)

outer(v, w=v) = A_mul_Bt(v, w)
outer!(A, v, w=v) = A_mul_Bt!(A, v, w)

reshape_vector(A::Vector) = reshape(A, 1, length(A))
reshape_vector(A) = A

safe_rand(q::Distribution, ns::Int...) = reshape_vector(rand(q, ns...))

function round_div(n::Int, a::Int)
    b = ceil(Int, n/a)
    n = b*a
    return n, b
end
