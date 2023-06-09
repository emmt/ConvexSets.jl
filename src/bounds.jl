# Implement lower/upper bounds on the variables and boxed sets.

const Lower = :lower
const Upper = :upper
const BAD_BOUND_TYPE = ArgumentError(
    "invalid bound type expecting `$Lower` or `$Upper`")

const NOT_SAME_INDICES = DimensionMismatch("arrays have different indices")

abstract type AbstractBound{W,T,N} <: AbstractArray{T,N} end

const LowerBound{T,N} = AbstractBound{Lower,T,N}
const UpperBound{T,N} = AbstractBound{Upper,T,N}

struct FreeBound{W,T,N,I<:ArrayAxes{N}} <: AbstractBound{W,T,N}
    indices::I
    function FreeBound{W,T}(inds::I) where {W,T,N,I<:ArrayAxes{N}}
        ((W === Lower) | (W === Upper)) || throw(BAD_BOUND_TYPE)
        return new{W,T,N,I}(inds)
    end
end

const FreeLowerBound{T,N} = FreeBound{Lower,T,N}
const FreeUpperBound{T,N} = FreeBound{Upper,T,N}

struct UniformBound{W,T,N,I<:ArrayAxes{N}} <: AbstractBound{W,T,N}
    value::T
    indices::I
    function UniformBound{W,T}(val, inds::I) where {W,T,N,I<:ArrayAxes{N}}
        ((W === Lower) | (W === Upper)) || throw(BAD_BOUND_TYPE)
        return new{W,T,N,I}(val, inds)
    end
end

const UniformLowerBound{T,N} = UniformBound{Lower,T,N}
const UniformUpperBound{T,N} = UniformBound{Upper,T,N}

struct ElementWiseBound{W,T,N,L,A<:AbstractArray{T,N}} <: AbstractBound{W,T,N}
    values::A
    function ElementWiseBound{W,T}(vals::A) where {W,T,N,A<:AbstractArray{T,N}}
        ((W === Lower) | (W === Upper)) || throw(BAD_BOUND_TYPE)
        L = IndexStyle(A) === IndexLinear()
        return new{W,T,N,L,A}(vals)
    end
end

const ElementWiseLowerBound{T,N} = ElementWiseBound{Lower,T,N}
const ElementWiseUpperBound{T,N} = ElementWiseBound{Upper,T,N}

"""
    LowerBound{T,N}(nothing, dims)
    LowerBound{T,N}(val, dims)
    LowerBound{T,N}(vals)

yield an abstract array representing a lower bound. Parameters `T` and `N` are
the element type and the number of dimensions; if not specified, they are
determined from the arguments. In the first case, a free (unlimited) bound for
variables of dimensions `dims` is built. In the second case, a uniform bound
with value `val` and for variables of dimensions `dims` is built. In the third
case, an element-wise bound specified by array `vals` is built.

See also [`ConvexSets.UpperBound`](@ref) and [`ConvexSets.BoxedSet`](@ref).

""" LowerBound

"""
    UpperBound{T,N}(nothing, dims)
    UpperBound{T,N}(val, dims)
    UpperBound{T,N}(vals)

yield an abstract array representing an upper bound. Parameters `T` and `N` are
the element type and the number of dimensions; if not specified, they are
determined from the arguments. In the first case, a free (unlimited) bound for
variables of dimensions `dims` is built. In the second case, a uniform bound
with value `val` and for variables of dimensions `dims` is built. In the third
case, an element-wise bound specified by array `vals` is built.

See also [`ConvexSets.LowerBound`](@ref) and [`ConvexSets.BoxedSet`](@ref).

""" UpperBound

for (func, W) in ((:LowerBound, :Lower),
                  (:UpperBound, :Upper))
    @eval begin
        # Build free bound.
        $func{T,N}(::Nothing, inds::Vararg{LengthOrUnitRange,N}) where {T,N} =
            $func{T}(nothing, inds)
        $func{T,N}(::Nothing, inds::NTuple{N,LengthOrUnitRange}) where {T,N} =
            $func{T}(nothing, inds)
        $func{T}(::Nothing, inds::LengthOrUnitRange...) where {T} =
            $func{T}(nothing, inds)
        $func{T}(::Nothing, inds::NTuple{N,LengthOrUnitRange}) where {T,N} =
            $func{T}(nothing, map(to_indices, inds))
        $func{T}(::Nothing, inds::ArrayAxes{N}) where {T,N} =
            FreeBound{$W,T}(inds)

        # Build elementwise bound.
        $func{T,N}(val::Number, inds::Vararg{LengthOrUnitRange,N}) where {T,N} =
            $func{T}(val, inds)
        $func{T,N}(val::Number, inds::NTuple{N,LengthOrUnitRange}) where {T,N} =
            $func{T}(val, inds)
        $func{T}(val::Number, inds::LengthOrUnitRange...) where {T} =
            $func(as(T, val), inds)
        $func{T}(val::Number, inds::NTuple{N,LengthOrUnitRange}) where {T,N} =
            $func(as(T, val), inds)
        $func(val::Number, inds::LengthOrUnitRange...) =
            $func(val, inds)
        $func(val::Number, inds::Tuple{Vararg{LengthOrUnitRange}}) =
            $func(val, map(to_indices, inds))
        $func(val::T, inds::I) where {T<:Number,N,I<:ArrayAxes{N}} =
            UniformBound{$W,T}(val, inds)

        # Build elementwise bound.
        $func{T,N}(vals::AbstractArray{T,N}) where {T,N} = $func(vals)
        $func{T,N}(vals::AbstractArray{<:Any,N}) where {T,N} = $func{T}(vals)
        $func{T}(vals::AbstractArray{T}) where {T} = $func(vals)
        $func{T}(vals::AbstractArray) where {T} =
            $func(copyto!(similar(vals, T), vals))
        $func(vals::AbstractArray{T}) where {T} = ElementWiseBound{$W,T}(vals)
    end
end

# Implement abstract array API for free and uniform bounds. Each sub-type
# implements a specific version of `IndexStyle`, `getindex`, and `setindex!`.
Base.axes(A::AbstractBound) = A.indices
Base.size(A::AbstractBound) = map(length, axes(A))
Base.length(A::AbstractBound) = prod(size(A))

Base.IndexStyle(::Type{<:Union{FreeBound,UniformBound}}) = IndexLinear()
@inline Base.getindex(A::FreeLowerBound{T}) where {T} = typemin(T)
@inline Base.getindex(A::FreeUpperBound{T}) where {T} = typemax(T)
@inline Base.getindex(A::UniformBound) = A.value
@inline function Base.getindex(A::Union{FreeBound,UniformBound}, i::Int)
    @boundscheck checkbounds(A, i)
    return @inbounds getindex(A)
end

Base.parent(A::ElementWiseBound) = A.values
Base.axes(A::ElementWiseBound) = axes(parent(A))
Base.size(A::ElementWiseBound) = size(parent(A))
Base.length(A::ElementWiseBound) = length(parent(A))

const FastElementWiseBound{W,T,N} = ElementWiseBound{W,T,N,true}
Base.IndexStyle(::Type{<:FastElementWiseBound}) = IndexLinear()
@inline function Base.getindex(A::FastElementWiseBound, i::Int)
    @boundscheck checkbounds(A, i)
    return @inbounds getindex(parent(A), i)
end
#=
@inline function Base.setindex!(A::FastUniformBound, x, i::Int)
    @boundscheck checkbounds(A, i)
    @inbounds setindex!(parent(A), x, i)
    return A
end
=#
Base.IndexStyle(::Type{<:ElementWiseBound}) = IndexCartesian()
@inline function Base.getindex(A::FastElementWiseBound{N},
                               I::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(A, I...)
    return @inbounds getindex(parent(A), I...)
end
#=
@inline function Base.setindex!(A::FastElementWiseBound{T,N}, x,
                                I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    @inbounds setindex!(parent(A), x, I...)
    return A
end
=#
"""
    AbstractBoxedSet{T,N}

is the super-type of boxed set types for `N`-dimensional variables with values
of type `T`. Any boxed set instance, say `Ω`, shall have the 2 following
properties:

    Ω.lower # is the lower bound
    Ω.upper # is the upper bound

"""
abstract type AbstractBoxedSet{T,N} <: ConvexSet{T,N} end

"""
    BoxedSet{T,N}(lo, hi)

yields an object specifying lower bound `lo` and upper bound `hi` for
`N`-dimensional variables of type `T`. If not specified, parameters `T` and `N`
are determined from the arguments.

Method `isempty(Ω)` can be called to figure out whether the boxed set `Ω` is
empty. Expression `x ∈ Ω` yields whether variables `x` belong to the boxed set
`Ω`.

See also [`ConvexSets.LowerBound`](@ref) and [`ConvexSets.UpperBound`](@ref).

"""
struct BoxedSet{T,N,L<:LowerBound{T,N},U<:UpperBound{T,N}} <: ConvexSet{T,N}
    lower::L
    upper::U
    function BoxedSet(lower::L, upper::U) where {T,N,
                                                 L<:LowerBound{T,N},
                                                 U<:UpperBound{T,N}}
        axes(lower) == axes(upper) || throw(DimensionMismatch(
            "lower and upper bounds must have the same indices"))
        return new{T,N,L,U}(lower, upper)
    end
end

is_bounded_below(A::FreeLowerBound) = false
is_bounded_below(A::UniformLowerBound{T}) where {T} = A.value > typemin(T)
is_bounded_below(A::LowerBound) = true
is_bounded_below(B::BoxedSet) = is_bounded_below(B.lower)

is_bounded_above(A::FreeUpperBound) = false
is_bounded_above(A::UniformUpperBound{T}) where {T} = A.value < typemax(T)
is_bounded_above(A::UpperBound) = true
is_bounded_above(B::BoxedSet) = is_bounded_above(B.upper)

is_uniform(A::AbstractBound) = is_uniform(typeof(A))
is_uniform(::Type{<:FreeBound}) = true
is_uniform(::Type{<:UniformBound}) = true
is_uniform(::Type{<:AbstractBound}) = false

function Base.isempty(Ω::BoxedSet)
    lo, hi = Ω.lower, Ω.upper
    if is_uniform(lo) && is_uniform(hi)
        return lo[] > hi[]
    else
        flag = false
        @inbounds @simd for i in eachindex(lo, hi)
            flag |= lo[i] > hi[i]
        end
        return flag
    end
end

function Base.in(x::AbstractArray{T,N}, Ω::BoxedSet{T,N}) where {T,N}
    inds = axes(x)
    lo, hi = Ω.lower, Ω.upper
    axes(lo) == inds || throw(NOT_SAME_INDICES)
    axes(hi) == inds || throw(NOT_SAME_INDICES)
    flag = true
    bounded_below = is_bounded_below(lo)
    bounded_above = is_bounded_above(hi)
    if bounded_below && bounded_above
        @inbounds @simd for i in eachindex(x, lo, hi)
            flag &= lo[i] ≤ x[i] ≤ hi[i]
        end
    elseif bounded_below
        @inbounds @simd for i in eachindex(x, lo)
            flag &= lo[i] ≤ x[i]
        end
    elseif bounded_above
        @inbounds @simd for i in eachindex(x, hi)
            flag &= x[i] ≤ hi[i]
        end
    elseif dst !== x
        copyto!(dst, x)
    end
    return flag
end

# Default implementations.
function project_variables(x::AbstractArray{T,N},
                           Ω::AbstractBoxedSet{T,N}) where {T,N}
    return min.(max.(x, Ω.lower), Ω.upper)
end

function project_variables!(dst::AbstractArray{T,N}, x::AbstractArray{T,N},
                            Ω::AbstractBoxedSet{T,N}) where {T,N}
    dst .= min.(max.(x, Ω.lower), Ω.upper)
    return dst
end

# Implementation for "standard" arrays.
function project_variables!(dst::AbstractArray{T,N},
                            x::AbstractArray{T,N},
                            Ω::BoxedSet{T,N}) where {T,N}
    inds = axes(x)
    axes(dst) == inds || throw(NOT_SAME_INDICES)
    lo, hi = Ω.lower, Ω.upper
    axes(lo) == inds || throw(NOT_SAME_INDICES)
    axes(hi) == inds || throw(NOT_SAME_INDICES)
    bounded_below = is_bounded_below(lo)
    bounded_above = is_bounded_above(hi)
    if bounded_below && bounded_above
        @inbounds @simd for i in eachindex(dst, x, lo, hi)
            dst[i] = fastclamp(x[i], lo[i], hi[i])
        end
    elseif bounded_below
        @inbounds @simd for i in eachindex(dst, x, lo)
            dst[i] = fastmax(x[i], lo[i])
        end
    elseif bounded_above
        @inbounds @simd for i in eachindex(dst, x, hi)
            dst[i] = fastmin(x[i], hi[i])
        end
    elseif dst !== x
        copyto!(dst, x)
    end
    return dst
end

@inline fastmin(x::T, y::T) where {T} = x > y ? y : x
@inline fastmax(x::T, y::T) where {T} = x < y ? y : x
@inline fastclamp(x::T, lo::T, hi::T) where {T} =
    fastmin(fastmax(x, lo), hi)

function project_direction!(dst::AbstractArray{T,N},
                            x::AbstractArray{T,N},
                            pm::PlusMinus,
                            d::AbstractArray{T,N},
                            Ω::BoxedSet{T,N}) where {T,N}
    inds = axes(x)
    axes(dst) == inds || throw(NOT_SAME_INDICES)
    axes(d) == inds || throw(NOT_SAME_INDICES)
    lo, hi = Ω.lower, Ω.upper
    bounded_below = is_bounded_below(lo)
    bounded_above = is_bounded_above(hi)
    if bounded_below && bounded_above
        @inbounds @simd for i in eachindex(dst, x, d, lo, hi)
            unblocked = ifelse(isnegative(pm, d[i]), x[i] > lo[i], x[i] < hi[i])
            dst[i] = ifelse(unblocked, d[i], zero(T))
        end
    elseif bounded_below
        @inbounds @simd for i in eachindex(dst, x, d, lo)
            unblocked = (x[i] > lo[i]) | ispositive(pm, d[i])
            dst[i] = ifelse(unblocked, d[i], zero(T))
        end
    elseif bounded_above
        @inbounds @simd for i in eachindex(dst, x, d, g, hi)
            unblocked = (x[i] < hi[i]) | isnegative(pm, d[i])
            dst[i] = ifelse(unblocked, d[i], zero(T))
        end
    elseif dst !== d
        copyto!(dst, d)
    end
    return dst
end

@inline ispositive(x::T) where {T} = x > zero(T)
@inline ispositive(::Plus,  x::T) where {T} = ispositive(x)
@inline ispositive(::Minus, x::T) where {T} = isnegative(x)

@inline isnegative(x::T) where {T} = x < zero(T)
@inline isnegative(::Plus,  x::T) where {T} = isnegative(x)
@inline isnegative(::Minus, x::T) where {T} = ispositive(x)

# Default implementations.
function unblocked_variables!(dst::AbstractArray{<:Any,N},
                              x::AbstractArray{T,N},
                              pm::PlusMinus,
                              d::AbstractArray{T,N},
                              Ω::AbstractBoxedSet{T,N}) where {T,N}
    if pm isa Plus
        dst .=
            ((x .> Ω.lower) .| (d .≥ zero(eltype(d)))) .&
            ((x .< Ω.upper) .| (d .≤ zero(eltype(d))))
    else
        dst .=
            ((x .> Ω.lower) .| (d .≤ zero(eltype(d)))) .&
            ((x .< Ω.upper) .| (d .≥ zero(eltype(d))))
    end
    return dst
end

# Implementation for "standard" arrays.
function unblocked_variables!(dst::AbstractArray{<:Union{T,Bool},N},
                              x::AbstractArray{T,N},
                              pm::PlusMinus,
                              d::AbstractArray{T,N},
                              Ω::BoxedSet{T,N}) where {T,N}
    inds = axes(x)
    axes(dst) == inds || throw(NOT_SAME_INDICES)
    axes(d) == inds || throw(NOT_SAME_INDICES)
    lo, hi = Ω.lower, Ω.upper
    axes(lo) == inds || throw(NOT_SAME_INDICES)
    axes(hi) == inds || throw(NOT_SAME_INDICES)
    bounded_below = is_bounded_below(lo)
    bounded_above = is_bounded_above(hi)
    if bounded_below && bounded_above
        @inbounds @simd for i in eachindex(dst, x, d, lo, hi)
            dst[i] =
                ((x[i] > lo[i]) | !isnegative(pm, d[i])) &
                ((x[i] < hi[i]) | !ispositive(pm, d[i]))
        end
    elseif bounded_below
        @inbounds @simd for i in eachindex(dst, x, d, lo)
            dst[i] = (x[i] > lo[i]) | !isnegative(pm, d[i])
        end
    elseif bounded_above
        @inbounds @simd for i in eachindex(dst, x, d, hi)
            dst[i] = (x[i] < hi[i]) | !ispositive(pm, d[i])
        end
    else
        fill!(dst, oneunit(eltype(dst)))
    end
    return dst
end

function line_search_limits(x::AbstractArray{T,N},
                            pm::PlusMinus,
                            d::AbstractArray{T,N},
                            Ω::BoxedSet{T,N}) where {T,N}
    # Check arguments.
    inds = axes(x)
    axes(d) == inds || throw(NOT_SAME_INDICES)
    lo, hi = Ω.lower, Ω.upper
    axes(lo) == inds || throw(NOT_SAME_INDICES)
    axes(hi) == inds || throw(NOT_SAME_INDICES)
    # Initialize `αmin` and `αmax`. We rely on the behavior of comparions when
    # one of the operands is a NaN to update `αmax` as:
    #
    #     αmax = αmax > α ? αmax : α
    #
    # which yields `α` if `αmax` is NaN as the test is false in this case.
    αmin = typemax(T)
    αmax = as(T, NaN)
    bounded_below = is_bounded_below(lo)
    bounded_above = is_bounded_above(hi)
    if bounded_below && bounded_above
        @inbounds @simd for i in eachindex(x, d, lo, hi)
            if isnegative(pm, d[i])
                α = step_to_bound(x[i], pm, d[i], lo[i])
                αmin = αmin < α ? αmin : α
                αmax = αmax > α ? αmax : α
            elseif ispositive(pm, d[i])
                α = step_to_bound(x[i], pm, d[i], hi[i])
                αmin = αmin < α ? αmin : α
                αmax = αmax > α ? αmax : α
            end
        end
    elseif bounded_below
        @inbounds @simd for i in eachindex(x, d, lo)
            if isnegative(pm, d[i])
                α = step_to_bound(x[i], pm, d[i], lo[i])
                αmin = αmin < α ? αmin : α
                αmax = αmax > α ? αmax : α
            end
        end
    elseif bounded_above
        @inbounds @simd for i in eachindex(x, d, hi)
            if ispositive(pm, d[i])
                α = step_to_bound(x[i], pm, d[i], hi[i])
                αmin = αmin < α ? αmin : α
                αmax = αmax > α ? αmax : α
            end
        end
    end
    if isnan(αmax)
        αmax = typemax(T)
    end
    return αmin, αmax
end

@inline step_to_bound(x::T, ::Plus,  d::T, b::T) where {T} = (b - x)/d
@inline step_to_bound(x::T, ::Minus, d::T, b::T) where {T} = (x - b)/d
