#
# Compute Statistics Online
#

export OnlineStatistic, update!,
    MeanVariance,
    Diagnostic,
    mean_eff_sample_size, ne, eff_sample_size, neμ,
    var_eff_sample_size, neσ, skew_eff_sample_size, neγ

abstract type OnlineStatistic{D} end

#
# scalar
#
update!(os::OnlineStatistic{1}, x::Real) = _update!(os, x)
update!(os::OnlineStatistic{1}, x::AbstractVector{R}) where R<:Real =
        _update_batch!(os, reshape(x, 1, length(x)))

#
# vector
#
function update!(os::OnlineStatistic{D}, x::AbstractVector{R}) where {D, R<:Real}
    length(x) == D ||
        throw(DimensionMismatch("Sample size inconsistent with statistic length."))

    _update!(os, x)
end

# for batch updates
function update!(os::OnlineStatistic{D}, x::AbstractMatrix{R}) where {D, R<:Real}
    size(x, 1) == D ||
        throw(DimensionMismatch("Sample size inconsistent with statistic length."))

    _update_batch!(os, x)
end

# default implementation
function _update_batch!(os::OnlineStatistic, x::AbstractMatrix{R}) where R<:Real
    for i = 1:size(x, 2)
        _update!(os, view(x, :, i))
    end

    return os
end

##########################################################################################
#                   Mean & Variance
# see "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances" Chang & Golub
##########################################################################################
mutable struct MeanVariance{D} <: OnlineStatistic{D}
    N::Int
    m::Vector{Float64}
    S::Matrix{Float64}

    function MeanVariance{D}() where D
        (isa(D, Int) && D > 0) || error("D must be an integer greater than 0")

        return new{D}(0, zeros(D), zeros(D, D))
    end
end

MeanVariance(D::Int) = MeanVariance{D}()

size(mv::MeanVariance{D}) where D = D

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

function _update!(mv::MeanVariance, x::Union{<:Real, AbstractVector{<:Real}})
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
mutable struct Diagnostic <: OnlineStatistic{1}
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
function _update!(d::Diagnostic, w::Real)
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
mutable struct ControlVariate{D} <: OnlineStatistic{D}
    N::Int
    β::Vector{Float64}

    function ControlVariate{D}() where D
        (isa(D, Int) && D > 0) || error("D must be an integer greater than 0")
        return new(0, zeros(D))
    end
end

ControlVariate(D::Int) = ControlVariate{D}()
