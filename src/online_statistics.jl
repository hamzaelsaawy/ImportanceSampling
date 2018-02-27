#
# Compute Statistics Online
#

export OnlineStatistic, update!,
    MeanVariance, ControlVariate,
    Diagnostic,
    mean_eff_sample_size, ne, eff_sample_size, neμ,
    var_eff_sample_size, neσ, skew_eff_sample_size, neγ

abstract type OnlineStatistic{S<:Tuple} end

size(os::OnlineStatistic{S}) where S<:Tuple = tuple(S.parameters...)
ndims(os::OnlineStatistic{S}) where S<:Tuple = length(S.parameters)

# single value update
function update!(os::OnlineStatistic, xs::AbstractVector{R}...) where R<:Real
    ndims(os) == length(xs) ||
        throw(DimensionMismatch("Number of samples not equal to number of dimensions"))

    all(size(os) .== length.(xs)) ||
        throw(DimensionMismatch("Sample sizes inconsistent with dimension lengths."))

    _update!(os, xs...)
end

# batch update
function update!(os::OnlineStatistic, xs::AbstractMatrix{R}...) where R<:Real
    ndims(os) == length(xs) ||
        throw(DimensionMismatch("Number of samples not equal to number of dimensions"))

    all(size(os) .== size.(xs, 1)) ||
        throw(DimensionMismatch("Sample sizes inconsistent with dimension lengths."))

    all_equal(size.(xs, 2)) ||
        throw(DimensionMismatch("Samples should have the same number of observations."))

    _update!(os, xs...)
end

# default implementation
@inline function _update!(os::OnlineStatistic, xs::AbstractMatrix{R}...) where R<:Real
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
_var(mv::MeanVariance) = diag(_cov(mv))

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
cov(mv::MeanVariance) = _cov(mv)
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
function _update!(mv::MeanVariance, x::AbstractMatrix{R}) where R<:Real
    N = mv.N
    M = size(x, 2)

    sumx = vec(sum(x, 2))
    μx = sumx / M

    # if M has a variance, else this is [0]
    if M > 1
        centeredx = x .- μx
        mv.S += outer(centeredx, centeredx)
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

update!(d::Diagnostic, w::Real) = _update!(d, w)
function update!(d::Diagnostic, ws::Vector{R}) where R<:Real
    for i = 1:length(ws)
        _update!(d, ws[i])
    end

    return d
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
mutable struct ControlVariate{Df, Dg} <: OnlineStatistic{Tuple{Df, Dg}}
    N::Int
    β::Matrix{Float64}
    mf::Vector{Float64}
    mg::Vector{Float64}
    Sg::Matrix{Float64}
    Cgf::Matrix{Float64}

    function ControlVariate{Df, Dg}() where {Df, Dg}
        (isa(Df, Int) && Df > 0) || error("Df must be an integer greater than 0")
        (isa(Dg, Int) && Dg > 0) || error("Dg must be an integer greater than 0")
        return new(0, zeros(Dg, Df), zeros(Df), zeros(Dg), zeros(Dg, Dg), zeros(Dg, Df))
    end
end

ControlVariate(Df::Int, Dg::Int) = ControlVariate{Df, Dg}()

function _update!(cv::ControlVariate, f::AbstractVector{R1},
        g::AbstractVector{R2}) where {R1<:Real, R2<:Real}

    cv.N += 1
    N = cv.N
    cv.mf += f
    cv.mg += g

    if N > 1
        cv.Sg += outer(N*g - cv.mg) / (N*(N-1))
        cv.Sg += outer(N*g - cv.mg, N*f - cv.mf) / (N*(N-1))
    end

    cv.β = cv.Sg \ cv.Cgf

    return cv
end

# batch update
function _update!(cv::ControlVariate, fs::AbstractMatrix{R1},
        gs::AbstractMatrix{R2}) where {R1<:Real, R2<:Real}
    N = cv.N
    M = size(fs, 2)

    sumf = vec(sum(fs, 2))
    sumg = vec(sum(gs, 2))
    μf = sumf / M
    μg = sumg / M

    # if M has a variance, else this is [0]
    if M > 1
        centeredf = fs.- μf
        centeredg = gs.- μg
        cv.Sg += outer(centeredg, centeredg)
        cv.Cgf += outer(centeredg, centeredf)
    end

    # becomes infinite b/c of N == 0
    if N > 0
        cv.S += M/(N*(M+N)) * outer(N*μg - cv.mg, N*μf - cv.mf)
    end

    cv.mf += sumf
    cv.mg += sumg
    cv.N += M
    cv.β = cv.Sg \ cv.Cgf

    return mv
end
