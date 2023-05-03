# Common constants, types, and methods for convex sets.

const Plus = typeof(+)
const Minus = typeof(-)
const PlusMinus = Union{Plus,Minus}

const ArrayAxis = AbstractUnitRange{Int}
const ArrayAxes{N} = NTuple{N,ArrayAxis}
const LengthOrUnitRange = Union{Integer,AbstractUnitRange{<:Integer}}

# Convert any object of type LengthOrUnitRange into an instance of
# AbstractUnitRange{Int}.
to_indices(dim::Integer) = Base.OneTo{Int}(dim)
to_indices(rng::AbstractUnitRange{Int}) = rng
to_indices(rng::Base.OneTo{<:Integer}) = Base.OneTo{Int}(length(rng))
to_indices(rng::AbstractUnitRange{<:Integer}) =
    as(Int, first(rng)):as(Int, last(rng))

"""
    ConvexSet{T,N}

is the abstract type of convex sets for `N`-dimensional variables whose values
have type `T`.

"""
abstract type ConvexSet{T,N} end

"""
    ConvexSets.project_variables(x, Ω)

yields the orthogonal projection of the variables `x` on the convex set `Ω`.

"""
project_variables(x::AbstractArray{T,N}, Ω::ConvexSet{T,N}) where {T,N} =
    project_variables!(similar(x), x, Ω)

"""
    ConvexSets.project_variables!(x, Ω) -> x

overwrites the entries of `x` with the orthogonal projection of the variables
`x` on the convex set `Ω`. This is a shortcut for `project_variables!(x,x,Ω)`
where:

    ConvexSets.project_variables!(dst, x, Ω) -> dst

overwrites the entries of `dst` with the orthogonal projection of the variables
`x` on the convex set `Ω`. This latter method is intended to be extended for
specific concrete types of variables `x` and/or convex sets `Ω`.

"""
project_variables!(x::AbstractArray{T,N}, Ω::ConvexSet{T,N}) where {T,N} =
    project_variables!(x, x, Ω)

"""
    ConvexSets.project_direction(x0, ±, d, Ω) -> p

yields the feasible search direction `p` such that the following properties
hold:

    ∀ α ≥ 0, project_variables(x0 ± α⋅p, Ω) = project_variables(x0 ± α⋅d, Ω)
    ∀ α ∈ [0,ϵ], project_variables(x0 ± α⋅p, Ω) = x0 ± α⋅p

for some `ϵ > 0`.

See [`ConvexSets.project_direction!`](@ref) for an in-place version.

!!! warning
    It is assumed that the variables `x0` are feasible, i.e. that `x0 ∈ Ω`.

"""
function project_direction(x0::AbstractArray{T,N},
                           pm::PlusMinus, d::AbstractArray{T,N},
                           Ω::ConvexSet{T,N}) where {T,N}
    return project_direction!(similar(d), x0, pm, d, Ω)
end

"""
    ConvexSets.project_direction!(dst, x0, ±, d, Ω) -> dst

overwrites `dst` by the feasible search direction and returns it. See
[`ConvexSets.project_direction`](@ref) for more details.

"""
function project_direction!(dst::AbstractArray{T,N}, x0::AbstractArray{T,N},
                            pm::PlusMinus, d::AbstractArray{T,N},
                            Ω::ConvexSet{T,N}) where {T,N}
    dst === d && error("projected direction and direction must be different objects")
    unblocked_variables!(dst, x0, pm, d, Ω)
    dst .*= d
    return dst
end

"""
    ConvexSets.unblocked_variables(S = Bool, x0, ±, d, Ω) -> u

yields an array similar to `x0` but with elements of type `S` whose entries are
`zero(S)` or `one(S)` depending whether variables are blocked or not by the
constraints imposed by `Ω` when moving from `x0` in the direction `±d`.

This method requires that the in-place version
[`ConvexSets.unblocked_variables!`](@ref) be extended for the types of the
arguments.

!!! warning
    It is assumed that the variables `x0` are feasible, i.e. that `x0 ∈ Ω`.

"""
function unblocked_variables(x0::AbstractArray{T,N},
                             pm::PlusMinus, d::AbstractArray{T,N},
                             Ω::ConvexSet{T,N}) where {T,N}
    return unblocked_variables(Bool, x0, pm, d, Ω)
end

function unblocked_variables(::Type{S}, x0::AbstractArray{T,N},
                             pm::PlusMinus, d::AbstractArray{T,N},
                             Ω::ConvexSet{T,N}) where {S,T,N}
    return unblocked_variables!(similar(x0, S), x0, pm, d, Ω)
end

"""
    ConvexSets.unblocked_variables!(dst, x0, ±, d, Ω) -> dst

overwrites the entries of `dst` with `zero(eltype(dst))` or `one(eltype(dst))`
depending whether variables are blocked or not by the constraints imposed by
`Ω` when moving from `x0` in the direction `±d`.

!!! warning
    It is assumed that the variables `x0` are feasible, i.e. that `x0 ∈ Ω`.

See [`ConvexSets.unblocked_variables`](@ref).

""" unblocked_variables!

"""
    ConvexSets.line_search_limits(x0, ±, d, Ω) -> αmin, αmax

yields the limits `αmin` and `αmax` for the step length `α` in a line-search
where iterates `x` are given by:

    x = project_variables(x0 ± α⋅d, Ω)

where `project_variables(x, Ω)` yields the orthogonal projection of the
variables `x` on the convex set `Ω`.

Output value `αmin` is the largest nonnegative step length such that
`0 ≤ α ≤ amin` implies:

    project_variables(x0 ± α*d, Ω) = x0 ± α*d

Output value `αmax` is the least nonnegative step length such that `α ≥ amax`
implies:

    project_variables(x0 ± α⋅d, Ω) = project_variables(x0 ± αmax*d, Ω)

In other words, the variables remain in `Ω` if `0 ≤ α ≤ αmin` and the projected
variables are all the same for any `α` such that `α ≥ αmax ≥ 0`.

!!! warning
    It is assumed that the variables `x0` are feasible, i.e. that `x0 ∈ Ω`; this is
    not verified for efficiency reasons.

See also: [`ConvexSets.project_variables`](@ref),
[`ConvexSets.unblocked_variables`](@ref), and
[`ConvexSets.line_search_max_step`](@ref).

"""
line_search_limits

"""
    ConvexSets.line_search_max_step(x0, ±, d, Ω) -> αmax

yields the least nonnegative step length such that `α ≥ amax` implies:

    project_variables(x0 ± α⋅d, Ω) = project_variables(x0 ± αmax*d, Ω)

In other words, the projected variables are all the same for any `α` such that
`α ≥ αmax ≥ 0`.

!!! warning
    It is assumed that the variables `x0` are feasible, i.e. that `x0 ∈ Ω`; this is
    not verified for efficiency reasons.

See also: [`ConvexSets.project_variables`](@ref),
[`ConvexSets.unblocked_variables`](@ref), and
[`ConvexSets.line_search_limits`](@ref).

"""
function line_search_max_step(x0::AbstractArray{T,N},
                              pm::PlusMinus,
                              d::AbstractArray{T,N},
                              Ω::ConvexSet{T,N}) where {T,N}
    αmin, αmax = line_search_limits(x0, pm, d, Ω)
    return αmax
end
