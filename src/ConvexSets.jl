module ConvexSets

export
    AbstractBound,
    BoxedSet,
    ConvexSet,
    LowerBound,
    UpperBound,
    project_variables,
    project_variables!,
    project_direction!,
    unblocked_variable!,
    line_search_limits

using AsType

include("common.jl")
include("bounds.jl")

end  # module
