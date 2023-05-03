# Convex sets for large scale optimization problems in Julia

[![Build Status](https://github.com/emmt/ConvexSets.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/ConvexSets.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/ConvexSets.jl?svg=true)](https://ci.appveyor.com/project/emmt/ConvexSets-jl) [![Coverage](https://codecov.io/gh/emmt/ConvexSets.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/ConvexSets.jl)

This package provides types and methods to deal with convex sets used to
represent strict constraints in large scale optimization problems in Julia.

## Boxed sets

Boxed sets are simple convex sets implementing separable lower and/or upper
bounds for the variables. A boxed set is built as follows:

``` julia
box = BoxedSet(lo, hi)
```

where `lo` and `hi` are objects respectively representing the lower and the
upper bounds on the variables.

The `ConvexSets` package provides a number of different bound types:

* **Free bounds** (that is unlimited bounds) are built by:

  ``` julia
  lo = LowerBound{T}(nothing, dims)
  hi = UpperBound{T}(nothing, dims)
  ```

  with `T` the element type of the variables and `dims` their dimensions.

* **Uniform bounds** are built by:

  ``` julia
  lo = LowerBound(val, dims)
  hi = UpperBound(val, dims)
  ```

  with `val` the value of the bound and `dims` the dimensions of the variables.
  Type parameter `T` may be used to specify the element type of the variables:

  ``` julia
  lo = LowerBound{T}(val, dims)
  hi = UpperBound{T}(val, dims)
  ```

  will automatically convert the value `val` to type `T`. Free bounds may also
  be built by:

  ``` julia
  lo = LowerBound(typemin(T), dims)
  hi = UpperBound(typemax(T), dims)
  ```

  but may be slightly less efficient.

* **Element-wise bounds** are  built by:

  ``` julia
  lo = LowerBound(vals)
  hi = UpperBound(vals)
  ```

  with `vals` the array of bound values which must have the same indices as the
  variables. Note that if all values of `vals` are the same, it is more
  efficient to build a uniform bound object which requires almost no storage
  and which is faster to use.

Of course, the lower and upper bounds in a boxed set may be of different types.
Other means to specify the bounds can be defined by sub-typing the abstract
type `ConvexSets.AbstractBound` and by extending the abstract array API for
these types.


## Methods

To make variables `x` feasible, their orthogonal projection on a convex set `Ω`
is obtained by:

``` julia
project_variables(x, Ω)
```

The in-place versions:

``` julia
project_variables!(x, Ω) -> x
project_variables!(dst, x, Ω) -> dst
```

overwrite their first argument with the projected variables and return them. If
new convex sets are defined by sub-typing the abstract type `ConvexSet{T,N}`
with `T` the element type of the variables and `N` their number of dimensions,
the latter above method shall be specialized for the types of `dst`, `x`, and
`Ω`.

Many large scale numerical optimization methods update the variables as
follows:

```
x = x0 ± α⋅d
```

with `x0` the variables at the start of the line-search, `d` the search
direction, and `α ≥ 0` the step length which has to be optimized typically by
solving:

```
α ≈ arg min f(x0 ± α⋅d)   s.t.   α ≥ 0
```

where `f(x)` is the objective function. Here, whether `±` is a `+` or a `-` is
not a free parameter, it depends on how the search direction is computed and on
the objective. If (as assumed above) the objective is to minimize `f(x)`, then
`±d` shall be a descent direction; otherwise, `±d` shall be an ascent direction
to maximize the objective function.

If strict constraints are implemented on the variables by a convex set `Ω`, the
above statement becomes:

```
α ≈ arg min f(project_variables(x0 ± α⋅d, Ω))   s.t.   α ≥ 0
```

and a few methods are useful in this constrained case.

A first such method is:

``` julia
p = project_direction(x, ±, d, Ω)
```

yields the **feasible direction** `p` such that the following properties hold:

``` julia
∀ α ≥ 0, project_variables(x0 ± α⋅p, Ω) = project_variables(x0 ± α⋅d, Ω)
∀ α ∈ [0,ϵ], project_variables(x0 ± α⋅p, Ω) = x0 ± α⋅p
```

for some `ϵ > 0`. In other words, searching along the direction `p` is
equivalent to searching along `d` (starting at `x0`) but the initial part of
the line-search along `p` is inside the feasible set `Ω`.

Another useful method is:

``` julia
u = unblocked_variables(S, x0, ±, d, Ω)
```

which yields an array `u` of same size as `x0` and `d`, whose elements are of
type `S`, and such that:

``` julia
u[i] = one(S)   if project_variables(x0 ± α⋅d, Ω)[i] != x0[i]
       zero(S)  else
```

for any non-negligible value of `α` and assuming that the variables `x0` are
feasible, i.e. such that `x0 ∈ Ω`.

Note that the feasible direction `p` could also have been defined by:

``` julia
p = unblocked_variables(x0 ± α⋅d, Ω) .* d
```

Finally, in a line-search, to avoid projecting variables when this is
unnecessary or searching with too long steps, the following step limits are of
interest:

* `αmin` is the largest nonnegative step length such that `0 ≤ α ≤ αmin`
  implies:

  ``` julia
  project_variables(x0 ± α*d, Ω) = x0 ± α*d
  ```

* `αmax` is the least nonnegative step length such that `α ≥ αmax` implies:

  ``` julia
  project_variables(x0 ± α⋅d, Ω) = project_variables(x0 ± αmax*d, Ω)
  ```

In other words, the variables remain in `Ω` if `0 ≤ α ≤ αmin` and the projected
variables are all the same for any `α` such that `α ≥ αmax`.

These limits are given by:

``` julia
αmin, αmax = line_search_limits(x0, ±, d, Ω)
αmax = line_search_max_step(x0, ±, d, Ω)
```
