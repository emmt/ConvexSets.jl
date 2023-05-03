module TestingConvexSets

using Test

using ConvexSets
using ConvexSets:
    AbstractBoxedSet,
    FreeBound, FreeLowerBound, FreeUpperBound,
    UniformBound, UniformLowerBound, UniformUpperBound,
    ElementWiseBound, ElementWiseLowerBound, ElementWiseUpperBound,
    is_bounded_below, is_bounded_above

# Simple structure to check reefrence implementations.
struct SimpleBox{T,N,
                 L<:AbstractArray{T,N},
                 U<:AbstractArray{T,N}} <: AbstractBoxedSet{T,N}
    lower::L
    upper::U
end



@testset "Convex sets" begin
    dims = (4,3,2)
    N = length(dims)
    T = Float32
    @testset "Free bounds" begin
        b = @inferred LowerBound{T}(nothing, dims...)
        @test b isa AbstractArray{T,N}
        @test b isa FreeLowerBound{T,N}
        @test size(b) == dims
        @test all(b .=== typemin(T))
        @test !is_bounded_below(b)
        @test_throws MethodError UpperBound{T,N+1}(nothing, dims)
        b = @inferred UpperBound{T,N}(nothing, dims...)
        @test b isa AbstractArray{T,N}
        @test b isa FreeUpperBound{T,N}
        @test size(b) == dims
        @test all(b .=== typemax(T))
        @test !is_bounded_above(b)
    end
    @testset "Uniform bounds" begin
        # Uniform lower bound equivalent to free bound.
        b = @inferred LowerBound(-T(Inf), dims...)
        @test b === @inferred LowerBound{T}(-Inf, dims...)
        @test b === @inferred LowerBound{T,N}(-Inf, dims...)
        @test b isa AbstractArray{T,N}
        @test b isa UniformLowerBound{T,N}
        @test size(b) == dims
        @test all(b .=== typemin(T))
        @test !is_bounded_below(b)
        # Uniform upper bound equivalent to free bound.
        @test_throws MethodError UpperBound{T,N+1}(+Inf, dims)
        b = @inferred UpperBound(T(Inf), dims...)
        @test b === @inferred UpperBound{T}(Inf, dims...)
        @test b === @inferred UpperBound{T,N}(Inf, dims...)
        @test b isa AbstractArray{T,N}
        @test b isa UniformUpperBound{T,N}
        @test size(b) == dims
        @test all(b .=== typemax(T))
        @test !is_bounded_above(b)
        # Constraining uniform bounds.
        b = @inferred LowerBound(zero(T), dims...)
        @test all(b .=== zero(T))
        @test is_bounded_below(b)
        b = @inferred UpperBound(one(T), dims...)
        @test all(b .=== one(T))
        @test is_bounded_above(b)
    end
    xmin = zeros(T, dims)
    xmax = reshape(zero(T):length(xmin)-1, dims)
    @testset "Element-wise bounds" begin
        # Lower bound.
        @test_throws MethodError UpperBound{T,N+1}(xmin)
        b = @inferred LowerBound(xmin)
        @test b === @inferred LowerBound{T}(xmin)
        @test b === @inferred LowerBound{T,N}(xmin)
        @test b isa AbstractArray{T,N}
        @test b isa ElementWiseLowerBound{T,N}
        @test size(b) == dims
        @test all(b .=== xmin)
        @test is_bounded_below(b)
        # Upper bound.
        @test_throws MethodError UpperBound{T,N+1}(xmax)
        b = @inferred UpperBound(xmax)
        @test b === @inferred UpperBound{T}(xmax)
        @test b === @inferred UpperBound{T,N}(xmax)
        @test b isa AbstractArray{T,N}
        @test b isa ElementWiseUpperBound{T,N}
        @test size(b) == dims
        @test all(b .=== xmax)
        @test is_bounded_above(b)
    end
    @testset "Boxed sets" begin
        lo = @inferred LowerBound(xmin)
        @test lo == xmin
        hi = @inferred UpperBound(xmax)
        @test hi == xmax
        Ω = @inferred BoxedSet(lo, hi)
        Ωs = SimpleBox(xmin, xmax)
        @test is_bounded_below(Ω) == is_bounded_below(lo)
        @test is_bounded_above(Ω) == is_bounded_above(hi)
        x = similar(xmin, T)
        flag = false
        for i in eachindex(x, lo, hi)
            x[i] = flag ? lo[i] - 1 : hi[i] + 1
            flag = !flag
        end
        xp = @inferred project_variables(x, Ω)
        @test xp == min.(max.(x, xmin), xmax)
        @test xp == min.(max.(x, lo), hi)
        @test xp == @inferred project_variables(x, Ωs)
        x === @inferred project_variables!(x, Ω)
        @test all(x .== xp)
        d = Array{T}(undef, dims)
        for i in firstindex(d):lastindex(d)
            d[i] = isodd(i) ? -2i : 3i
        end
        w = @inferred unblocked_variables(x, +, d, Ω)
        @test eltype(w) === Bool
        @test size(w) == dims
        @test w == (((x .> lo) .| (d .≥ zero(T))) .& ((x .< hi) .| (d .≤ zero(T))))
        p = @inferred project_direction(x, +, d, Ω)
        @test p == (w .* d)
        w = @inferred unblocked_variables(x, -, d, Ω)
        @test eltype(w) === Bool
        @test size(w) == dims
        @test w == (((x .> lo) .| (d .≤ zero(T))) .& ((x .< hi) .| (d .≥ zero(T))))
        p = @inferred project_direction(x, -, d, Ω)
        @test p == (w .* d)
        i1 = (d .< zero(eltype(d)))
        i2 = (d .> zero(eltype(d)))
        s = vcat((xmin[i1] - x[i1])./d[i1], (xmax[i2] - x[i2])./d[i2])
        αmin, αmax = @inferred line_search_limits(x, +, d, Ω)
        @test extrema(s) == (αmin, αmax)
        @test αmax == @inferred line_search_max_step(x, +, d, Ω)
        s = vcat((x[i2] - xmin[i2])./d[i2], (x[i1] - xmax[i1])./d[i1])
        αmin, αmax = @inferred line_search_limits(x, -, d, Ω)
        @test extrema(s) == (αmin, αmax)
        @test αmax == @inferred line_search_max_step(x, -, d, Ω)
    end
end

end # module
