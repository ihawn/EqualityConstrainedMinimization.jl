module EqualityConstrainedMinimization
export Solve_ECM
using Calculus
using LinearAlgebra
include("algorithms.jl")


#######################################################################
#######################################################################
#=

This is a non-linear minimization solver that works with all convex problems
with linear equality constraints. It is intended to solve problems in the form
"minimize f(x) such that Ax = b" and "minimize f(x)" for the unconstrained case.
I am taking the algorithms used directly from "Convex Optimization" by Stephen Boyd
and Lieven Vandenberghe. This package is intended for learning purposes and while
it may not be the fastest solver, it is certainly one of the easiest to understand.

Algorithms used for unconstrained problem:
    -Gradient descent
    -Newton's method
    -Unless force_gradient = true, the solver will use gradient descent method until
     stopping criterion η is met. Then, it will switch to Newton's Method until ϵ is met
     Note that η > ϵ.

Algorithms used for equality constrained problem:
    -Feasible Start Newton Method
    -Infeasible Start Newton Method

# Arguments
    - 'f': The convex objective. Must be convex or the solution may converge to a local solution.
    - 'x': The starting point. May be feasible or infeasible though starting feasible may converge faster.
    - 'A': The coefficient matrix on the Ax = b constraint. This system needs to be underdetermined. Not including this will imply an unconstrained problem.
    - 'b': The constant vector on the Ax = b constraint. Not including this will imply feasible start.
    - 'α': A parameter used in the backtracking line search. Note that α ∈ (0, 0.5)
    - 'β': Another parameter used in the backtracking line search. β ∈ (0, 1)
    - 'η': A stopping criterion used to determine when to switch from gradient descent to Newton's method.
    - 'ϵ': Overall stopping criterion.
    - 'maxIt': Max iteration stopping criterion. Meant as a failsafe.
    - 'verbose': Setting this to true will show data from each iteration.
    - 'force_gradient': Setting this to true will force only the gradient descent method for unconstrained problems. Do this when ϵ is relatively large.


Main solver function:
Solve_ECM(
        f,
        x::Array{Float64,1};
        A::Array{Float64,2} = zeros(2, 2),
        b::Array{Float64,1} = zeros(2),
        α = 0.3,
        β = 0.8,
        η = 1e-4,
        ϵ = 1e-8,
        maxIt = 1e3,
        verbose = true,
        force_gradient = false
    )

See 'UsageExamples.jl' for some examples.

=#
#######################################################################
#######################################################################





#####################################
#####################################
#
# Main Solver
#
#####################################
#####################################
function Solve_ECM(
        f,
        x::Array{Float64,1};
        A::Array{Float64,2} = zeros(2, 2),
        b::Array{Float64,1} = zeros(2),
        α = 0.3,
        β = 0.8,
        η = 1e-4,
        ϵ = 1e-8,
        maxIt = 1e3,
        verbose = true,
        force_gradient = false
    )

    n = length(x)
    m = size(A,1)

    uncon = true
    feasible = true

    if η >= ϵ && n >= m

        if A != zeros(2, 2)
            if b != zeros(2)
                feasible = false
            end
            uncon = false
        end

        sol = Tuple{Array{Float64,1},Float64}

        #Determine which method to use
        if uncon

            print("\n\nUnconstrained objective detected")

            if force_gradient
                print("\nSolving using gradient descent\n")
                sol = Unconstrained_Grad_Descent(f, x, n, α, β, ϵ, maxIt, verbose)
            else
                print("\nSolving using gradient descent and Newton's Method\n")
                sol = Solve_UMP(f, x, n, α, β, η, ϵ, maxIt, verbose)
            end
        else
            if feasible
                print("\n\nVector b not given. Therefore, x is assumed feasible")
                print("\nSolving using Feasible Start Newton Method")
                sol = Newton_FS(f, x, A, n, m, α, β, ϵ, maxIt, verbose)
            else
                print("\n\nVector b given. Therefore, x is assumed infeasible")
                print("\nSolving using Infeasible Start Newton Method")
                sol = Newton_IFS(f, x, A, b, n, m, α, β, ϵ, maxIt, verbose)
            end
        end

        print("\n\n------------------------")
        print("\nSolution found at point:")
        print("\n\nx = ", sol[1],"\n")
        print("\nObjective Value = ", sol[2], "\n\n")
    else
        print("\nParameters not correct. Either η < ϵ or Ax = b is an overdetermined system")
    end

    return sol[2]
end

end
