#using .ECMSolver

include("EqualityConstrainedMinimization.jl")
using LinearAlgebra

#Generates ECM problem with feasible starting point
function Generate_ECM(_n, _p, A_bound)
    _x = rand(_n)
    _A = rand(_p, _n) * A_bound

    it = 0
    #Ensure A has full rank
    while rank(_A) != max(_n, _p) && it < 100 #Cap generation at 100 tries. The odds of needing this many are very low
        _A = rand(_p, _n) * A_bound
        it += 1
    end

    _b = _A * _x

    return _A, _x, _b
end

#Test objective for constrained problem (see Boyd and Vandenberghe problem 10.15)
function Test_Objective_Con(_x)
    _n = length(_x)
    sum = 0

    for i = 1:_n
        sum += _x[i] * log(_x[i])
    end

    return sum
end

l = 5
w = 3 #Constraint coefficient mtx will be an n x m matrix
ECM = Generate_ECM(l, w, 1)
mtx = ECM[1] #the A in Ax = b
x = ones(l)*0.5 #ECM[2] #set equal to ECM[2] to test feasible start
vect = ECM[3] #the b in Ax = b

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
