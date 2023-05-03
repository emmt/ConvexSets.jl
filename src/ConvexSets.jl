module ConvexSets

export
    AbstractBound,
    BoxedSet,
    ConvexSet,
    LowerBound,
    UpperBound,
    project_variables,
    project_variables!,
    project_direction,
    project_direction!,
    unblocked_variables,
    unblocked_variables!,
    line_search_limits,
    line_search_max_step

using AsType

include("common.jl")
include("bounds.jl")

end  # module
