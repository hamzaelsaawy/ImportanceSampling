#
# Helper functions
#

all_equal(f, xs) = all(x -> f(x) == f(first(xs)), xs[2:end])
all_equal(xs) = all_equal(identity, xs)

outer(v::AbstractVector{<:Real}, w::AbstractVector{<:Real}=v) = v * w'
outer(v::Real, w::Real=v) = v*w # i know...
"""
    outer!(A::AbstractVector{<:Real},
            v::AbstractVector{<:Real},
            w::AbstractVector{<:Real}=v)

Set `A` to `A = v * w'`
"""
function outer!(A::AbstractMatrix{<:Real},
        v::AbstractVector{<:Real}, w::AbstractVector{<:Real}=v)
    M = length(v)
    N = length(w)
    size(A) == (M, N) ||
            throw(DimensionMismatch("A must have size `(length(v), length(w))`"))

    for j in 1:N
        for i in 1:M
            @inbounds A[i, j] = v[i] * w[j]
        end
    end

    return A
end
