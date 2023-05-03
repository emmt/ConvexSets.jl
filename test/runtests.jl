module TestingConvexSets

using Test

using ConvexSets
using ConvexSets:
    FreeBound, FreeLowerBound, FreeUpperBound,
    UniformBound, UniformLowerBound, UniformUpperBound,
    ElementWiseBound, ElementWiseLowerBound, ElementWiseUpperBound,
    is_bounded_below, is_bounded_above

@testset "Box constraints" begin
    dims = (2,3,4)
    N = length(dims)
    T = Float32
    # Free bounds.
    b = @inferred LowerBound{T}(nothing, dims...)
    @test b isa AbstractArray{T,N}
    @test b isa FreeLowerBound{T,N}
    @test size(b) == dims
    @test all(b .== typemin(T))
    @test !is_bounded_below(b)
    @test_throws MethodError UpperBound{T,N+1}(nothing, dims)
    b = @inferred UpperBound{T,N}(nothing, dims...)
    @test b isa AbstractArray{T,N}
    @test b isa FreeUpperBound{T,N}
    @test size(b) == dims
    @test all(b .== typemax(T))
    @test !is_bounded_above(b)

    # Uniform bounds.
    b = @inferred LowerBound(-T(Inf), dims...)
    c = @inferred LowerBound{T}(-Inf, dims...)
    d = @inferred LowerBound{T,N}(-Inf, dims...)
    @test c === b
    @test d === b
    @test b isa AbstractArray{T,N}
    @test b isa UniformLowerBound{T,N}
    @test size(b) == dims
    @test all(b .=== typemin(T))
    @test !is_bounded_below(b)
    b = @inferred LowerBound(zero(T), dims...)
    @test all(b .=== zero(T))
    @test is_bounded_below(b)
    @test_throws MethodError UpperBound{T,N+1}(nothing, dims)
    b = @inferred UpperBound(T(Inf), dims...)
    c = @inferred UpperBound{T}(Inf, dims...)
    d = @inferred UpperBound{T,N}(Inf, dims...)
    @test c === b
    @test d === b
    @test b isa AbstractArray{T,N}
    @test b isa UniformUpperBound{T,N}
    @test size(b) == dims
    @test all(b .=== typemax(T))
    @test !is_bounded_above(b)
    b = @inferred UpperBound(one(T), dims...)
    @test all(b .=== one(T))
    @test is_bounded_above(b)
end

end # module
