#
# helper functions
#

all_equal(f, xs) = all(x -> f(x) == f(first(xs)), xs[2:end])
all_equal(xs) = all_equal(identity, xs)

outer(v, w=v) = A_mul_Bt(v, w)
outer!(A, v, w=v) = A_mul_Bt!(A, v, w)

reshape_vector(A::Vector) = reshape(A, 1, length(A))
reshape_vector(A) = A

datasize(X::AbstractArray) = size(X, ndims(X))

safe_rand(q::Distribution, ns::Int...) = reshape_vector(rand(q, ns...))

#
# "safer" lol
#
safe_pdf(q::UnivariateDistribution, X::AbstractArray) = pdf.(q, X)
safe_pdf(q::Distribution, X::AbstractArray) = pdf(q, X)

safe_pdf!(r::AbstractVector, q::UnivariateDistribution, X::AbstractArray) = (r .= vec(pdf.(q, X)))
safe_pdf!(r::AbstractArray, q::UnivariateDistribution, X::AbstractArray) = (r .= pdf.(q, X))
safe_pdf!(r::AbstractArray, q::Distribution, X::AbstractArray) = pdf!(r, q, X)

function round_div(n::Int, a::Int)
    b = ceil(Int, n/a)
    n = b*a
    return n, b
end
