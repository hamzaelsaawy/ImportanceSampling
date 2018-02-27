#
# Compute Statistics Online
#

export OnlineStatistic, update!, update_batch,
    MeanVariance,
    Diagnostic,
    mean_eff_sample_size, ne, eff_sample_size, neμ,
    var_eff_sample_size, neσ, skew_eff_sample_size, neγ

abstract type OnlineStatistic{S<:Tuple} end

size(os::OnlineStatistic{S}) where S<:Tuple = tuple(S.parameters...)
ndims(os::OnlineStatistic{S}) where S<:Tuple = length(S.parameters)

function update!(os::OnlineStatistic, xs::Union{<:Real, AbstractVector{<:Real}}...)
    ndims(os) == length(xs) ||
        throw(DimensionMismatch("Number of samples not equal to number of dimensions"))

    xs = [xs...]
    are_scalars = ndims.(xs) == 0
    xs[are_scalars] = map(x -> [x], xs[are_scalars]

    all(size(os) .== length.(xs)) ||
        throw(DimensionMismatch("Sample sizes inconsistent with dimension lengths."))

    _update!(os, xs...)
end

function update_batch!(os::OnlineStatistic, xs::AbstractVecOrMat{R}...) where R<:Real
    ndims(os) == length(xs) ||
        throw(DimensionMismatch("Number of samples not equal to number of dimensions"))

    xs = [xs...]
    are_vectors = ndims.(xs) == 1
    xs[are_vectors] = map(x -> reshape(x, 1, length(x)), xs[are_vectors])

    all(size(os) .== size.(xs, 1)) ||
        throw(DimensionMismatch("Sample sizes inconsistent with dimension lengths."))

    all_equal(size.(xs, 2)) ||
        throw(DimensionMismatch("Samples should have the same number of observations."))

    _update_batch!(os, xs...)
end

# default implementation
@inline function _update_batch!(os::OnlineStatistic, xs::AbstractMatrix{R}...) where R<:Real
    for i = 1:size(x, 2)
        _update!(os, map(x -> view(x, :, i), xs)...)
    end

    return os
end

##########################################################################################
#                   Mean & Variance
# see "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances" Chang & Golub
##########################################################################################
mutable struct MeanVariance{D} <: OnlineStatistic{Tuple{D}}
    N::Int
    m::Vector{Float64}
    S::Matrix{Float64}

    function MeanVariance{D}() where D
        (isa(D, Int) && D > 0) || error("D must be an integer greater than 0")

        return new{D}(0, zeros(D), zeros(D, D))
    end
end

MeanVariance(D::Int) = MeanVariance{D}()

_mean(mv::MeanVariance{D}) where D = (mv.N ≥ 1) ? mv.m/mv.N : fill(NaN, D)
_cov(mv::MeanVariance{D}) where D = (mv.N ≥ 2) ? mv.S / (mv.N-1) : fill(NaN, D, D)
_var(mv::MeanVariance) = diag(cov(mv))

#
# scalar
#
mean(mv::MeanVariance{1}) = first(_mean(mv))
cov(mv::MeanVariance{1}) = first(_cov(mv))
var(mv::MeanVariance{1}) = first(_var(mv))
std(mv::MeanVariance{1}) = sqrt(var(mv))

#
# vector
#
mean(mv::MeanVariance) = _mean(mv)
cov(mv::MeanVariance) = _mean(mv)
var(mv::MeanVariance) = _var(mv)

@inline function _update!(mv::MeanVariance, x::AbstractVector{R}) where R<:Real
    mv.N += 1
    N = mv.N
    mv.m += x

    # for N == 1, x == μnew, so the update is [0]
    if N > 1
        mv.S += outer(N*x - mv.m) / (N*(N-1))
    end

    return mv
end

# batch update
function _update_batch!(mv::MeanVariance, x::AbstractMatrix{R}) where R<:Real
    N = mv.N
    M = size(x, 2)

    sumx = vec(sum(x, 2))
    μx = sumx / M

    # if M has a variance, else this is [0]
    if M > 1
        centeredx = x .- μx
        Sx = A_mul_Bt(centeredx, centeredx)
        mv.S += Sx
    end

    # becomes infinite b/c of N == 0
    if N > 0
        mv.S += M/(N*(M+N)) * outer(N*μx - mv.m)
    end

    mv.m += sumx
    mv.N += M

    return mv
end

##########################################################################################
#                   Diagnostics
# See Stanford Stats 362: Monte Carlo with Art Owen
##########################################################################################
mutable struct Diagnostic <: OnlineStatistic{Tuple{1}}
    N::Int
    sumw::Float64
    sumw2::Float64
    sumw3::Float64
    sumw4::Float64

    function Diagnostic()
        return new(0, 0, 0, 0, 0)
    end
end

_update!(d::Diagnostic, w::AbstractVector{R}) where R<:Real = _update!(d, first(w))
@inline function _update!(d::Diagnostic, w::Real)
    d.N += 1
    d.sumw += w
    d.sumw2 += w^2
    d.sumw3 += w^3
    d.sumw4 += w^4

    return d
end

mean_eff_sample_size(d::Diagnostic) = d.sumw^2/d.sumw2
ne(d::Diagnostic) = mean_eff_sample_size(d)
eff_sample_size(d::Diagnostic) = mean_eff_sample_size(d)
neμ(d::Diagnostic) = mean_eff_sample_size(d)

var_eff_sample_size(d::Diagnostic) = d.sumw2^2/d.sumw4
neσ(d::Diagnostic) = var_eff_sample_size(d)

skew_eff_sample_size(d::Diagnostic) = d.sumw2^3/d.sumw3^2
neγ(d::Diagnostic) = skew_eff_sample_size(d)

##########################################################################################
#                   Control Variates
##########################################################################################
mutable struct ControlVariate{D1, D2} <: OnlineStatistic{Tuple{D1, D2}}
    N::Int
    β::Vector{Float64}
    fmv::MeanVariance{D2}
    gmv::MeanVariance{D1}
    C::Matrix{Float64}

    function ControlVariate{D1, D2}() where {D1, D2}
        (isa(D1, Int) && D1 > 0) || error("D1 must be an integer greater than 0")
        (isa(D2, Int) && D2 > 0) || error("D2 must be an integer greater than 0")
        return new(0, zeros(D))
    end
end

ControlVariate(D1::Int, D2::Int) = ControlVariate{D1, D2}()

function _update!(cv::ControlVariate, g::Union{<:Real, AbstractVector{<:Real}},
        f::Union{<:Real, AbstractVector{<:Real}}, _...)
    length(x) == D ||
        throw(DimensionMismatch("Sample size inconsistent with statistic length."))
    mv.N += 1
    N = mv.N
    mv.m += x

    # for N == 1, x == μnew, so the update is [0]
    if N > 1
        mv.S += outer(N*x - mv.m) / (N*(N-1))
    end

    return mv
end

end
