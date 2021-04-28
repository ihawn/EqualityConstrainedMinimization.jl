using .EqualityConstrainedMinimization
using Test
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

l = 3
w = 2
mtx = [0.52 0.96 0.59; 0.42 0.37 0.25]
feasX = [0.2; 0.3; 0.4]
x = ones(l)
vect = mtx*feasX


n = 100
p = 30
ECM = Generate_ECM(n, p, 1)

f(_x) = Test_Objective_Con(_x)


@testset "ECMSolver.jl" begin
    @test Solve_ECM(f, x, verbose = false) ≈ -1.1036383232 rtol = 1e-8
    @test Solve_ECM(f, x, A = mtx, b = vect, verbose = false) ≈ -1.0587361852 rtol = 1e-8
    @test Solve_ECM(f, feasX, A = mtx, verbose = false) ≈ -1.0587361852 rtol = 1e-8
    @test Solve_ECM(f, feasX, A = mtx, b = vect, verbose = false) ≈ -1.0587361852 rtol = 1e-8

    @test Solve_ECM(f, ECM[2], A = ECM[1], verbose = false) ≈ Solve_ECM(f, ones(n), A = ECM[1], b = ECM[3], verbose = false) rtol = 1e-8
end
