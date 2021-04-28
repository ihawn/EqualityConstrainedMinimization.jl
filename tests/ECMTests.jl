using Test
using ECMSolver
using LinearAlgebra


#Test objective for constrained problem (see Boyd and Vandenberghe problem 10.15)
function Test_Objective_Con(_x)
    _n = length(_x)
    sum = 0

    for i = 1:_n
        sum += _x[i] * log(_x[i])
    end

    return sum
end

l = 3
w = 2
mtx = [1 2 3; 4 5 6]
x = ones(l) #ECM[2] #set equal to ECM[2] to test feasible start
vect = [7; 8]

f(_x) = Test_Objective_Con(_x)

#Solves the problem "minimize f(x) subject to Ax = b." Since b is given, we assume infeasible start
IF_Sol = ECMSolver.Solve_ECM(f, x, A = mtx, b = vect, verbose = false)

#Solves the problem "minimize f(x) subject to Ax = b." Since b not given, we assume feasible start
x = ECM[2]
F_Sol = ECMSolver.Solve_ECM(f, x, A = mtx, verbose = false)

#Solves the problem "minimize f(x)."
#Setting force_gradient = true will force the gradient method
#Do this if ϵ is relatively large.
#Otherwise, the solver will first use Gradient Descent and then switch to Newton's Method
U_Sol = ECMSolver.Solve_ECM(f, x, verbose = false)
