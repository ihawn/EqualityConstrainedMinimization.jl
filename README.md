# EqualityConstrainedMinimization

This is a non-linear minimization solver that works with all convex problems
with linear equality constraints. It is intended to solve problems in the form
"minimize f(x) such that Ax = b" and "minimize f(x)" for the unconstrained case.
I am taking the algorithms directly from "Convex Optimization" by Stephen Boyd
and Lieven Vandenberghe. This package is intended for learning purposes and while
it may not be the fastest solver, it is certainly one of the easiest to understand.

See "UsageExamples.jl" for some usage examples.
See the two ipynb notebooks for even more examples and some mathematical insight into how these algorithms work.
