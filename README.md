# Online Importance Sampling
**Hamza El-Saawy**
**Stanford Stats 362 Final Project**

This package supports online (batched)
[Importance Sampling](https://en.wikipedia.org/wiki/Importance_sampling) (IS)
with or without [control variates](https://en.wikipedia.org/wiki/Control_variates)

The package provides a `MixtureDistribution{F, S} <: Distribution{F, S}`
for mixture importance sampling.

*Note*: Even if `f!` (or `g!`) is a scalar function, it must write its output to a vector.
Moreover, `w` should only return a scalar.
Also, even if `q <: UnivariateDistribution`, `x` will be a `Vector` of length 1.
The exception is with respect to `p`, it should follow the convention of `Distributions.jl`,
which, unfortunetely, wants `logpdf(p::UnivariateDistrbution, x::Real)` but
`logpdf(p::MultivariateDistrbution, x::Vector)`.
So, for the univariate case, `w` accepts a vector of length 1, but `logpdf(p, x)`
takes a scalar.

*Note*: when passing in external data (`update!(is, X=X, F=F, W=W,...)`), `F` and `G`
will be modified in place (`F .*= W'` and `G ./= Q)`)

## IS

### Basic IS
```julia
ImportanceSampler(f!, lengthf::Int, q::Distribution ; p=nothing, w=nothing)
```

`f!(r::AbstractVector, x::AbstractVector)` is a function (or anything callable) that
modifies `r`, its first argument. Note that `x` will always have the same length
as `q`, the sampling distribution. `r` will always have the same length as `lengthf`, the
output dimension of `f`.

Either `p` or `w` should be provided. `p` should have
`logpdf(p, x::AbstractVector)` defined, for `x = rand(q)`.
`w(x)` should compute `p(x)/q(x)`, the ratio of their pdfs.

### IS with Control Variates:
```julia
CvImportanceSampler(f!, lengthf::Int, q::Distribution;
        g!s::Union{AbstractVector{<:Tuple{Any, Vector{Float64}}}, Void}=nothing,
        p=nothing, w=nothing,
        use_q::Bool=false)
```

Here, `g!s` is a vector of tuples, `(g!, θ)`, where
`g!(r::AbstractVector, x::AbstractVector)` takes a vector always of size `length(q)`
and writes the result in `r`, which always has the size `length(θ)`. `θ` is the integral
of `g` over the support of `q`. `use_q` uses `q`, or its components if
`q <: MixtureDistribution`, as control variates as well, with a `θ` of `[1.0]`.

## Running IS
The general syntax to run an `ImportanceSampler` is:
```julia
update!(is; X [, F, W])
update!(is; F, W)
update!(is; niters, nbatches, batchsize)
```
All arguments except for `is::ImportanceSampler` use keywords. `X` is optional if
`F` and `W` are provided.

If the date is not provided, `update` will generate random `nbatches` batches of data
sized `batchsize`, for a total number of iterations: `niters == nbatches * batchsize`.

Only two of `niters, nbatches, batchsize`, all `::Int`, should be provided.

For a `CvImportanceSampler`, it is:
```julia
update!(is; X [, F, G, W, Q])
update!(is; F, G, W, Q)
update!(is; niters, nbatches, batchsize)
```
Here `G` is the value of all functions in `g!s`, concatenated together vertically.
`Q` is `q(x)` at each point, which is used to generate `p(x) = w(x)*q(x)`.
Similar to above, if `X` is omitted, `F, G, W,` and `Q` must be provided.

There are also the keywords `updateμ::Bool=true` and `updateβ::Bool=false` that update the estimates
of the mean and regression coefficients, respectively.
The functions `updateμ!(...)` and `updateβ!(...)` with similar syntax to above can also be used
as well to update only one of the two. (Updating both together introduces a bias.)
