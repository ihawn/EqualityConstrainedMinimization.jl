


#######################################################################
#=
Section of code dealing with unconstrained minimization problems.
Some functions are reused in the equality constrained section
=#
#######################################################################



#################
#Gradient Descent
#################

#Use Calculus.jl library to compute the gradient numerically
function Compute_Gradient(_f, _x)
    return Calculus.gradient(g -> _f(g),_x)
end

#Use Calculus.jl library to compute the hessian numerically
function Compute_Hessian(_f, _x)
    return Calculus.hessian(h -> _f(h),_x)
end

#Choose t using backgracking line search (p. 464, alg. 9.2)
function Back_Line_Search(_x, _f, _fx, _∇f, _Δx, _α, _β, _n)

    _t = 1

    while _f((_x + _t*_Δx)) > _fx + _α*_t*transpose(_∇f)*_Δx
        _t *= _β
    end

    return _t
end

#Main solution method for Unconstrained Gradient Descent Method (p. 466, alg 9.3)
function Unconstrained_Grad_Descent(_f, _x, _n, _α, _β, _ϵ, maxIt, verbose)
    it = 0;
    t = 1
    ∇f = Compute_Gradient(_f, _x)
    normGrad = norm(∇f)

    val = _f(_x)

    #Stopping conditions: norm(∇f) <= η or iterations exceeds maximum iteration tolerance
    while normGrad > _ϵ && it <= maxIt
        ∇f = Compute_Gradient(_f, _x) #Evaluate gradient at current point
        normGrad = norm(∇f) #Compute euclidean norm using LinearAlgebra library


        #Perform backgracking line search to pick t
        bls = Back_Line_Search(_x, _f, val, ∇f, -∇f, _α, _β, _n)
        t = bls[1]

        #update step
        _x -= t*∇f

        val = _f(_x)


        if(verbose)
            print("\n\nIteration: ", it)
            print("\nGradient Norm = ", normGrad)
        #    print("\nDescent Direction = ", -∇fVal)
            print("\nt = ", t)
        #    print("\nx = ",_x)
            print("\nf(x) = ", val)
        end

        it += 1
    end

    if it > maxIt
        print("\n\nMax Iterations Reached!")
    end

    return _x, val
end



#################
#Newton Method
#################

#Computes Newton step (p. 487, alg. 9.5)
function Newton_Step(_∇f, _∇2f, _x, _n)
    return _∇2f\-_∇f
end

#Computes Newton decrement (p. 487, alg. 9.5)
function Newton_Decrement(_∇2f, _Δxnt)
    return transpose(_Δxnt)*_∇2f*_Δxnt
end

#Main solver method for Unconstrained Newton Method (p. 487, alg. 9.5)
function Unconstrained_Newton(_f, _x, _n, _α, _β, _ϵ, maxIt, verbose)

    λ2 = 1 #prime λ
    it = 0
    val = _f(_x)

    while λ2 > _ϵ && it <= maxIt

        #Objective, Gradient, and Hessian
        val = _f(_x)
        ∇f = Compute_Gradient(_f, _x)
        ∇2f = Compute_Hessian(_f, _x)


        #Newton step
        Δxnt = Newton_Step(∇f, ∇2f, _x, _n)

        #Newton decrement
        λ2 = Newton_Decrement(∇2f, Δxnt)

        #Choose t with backtracking line search
        t = Back_Line_Search(_x, _f, val, ∇f, Δxnt, _α, _β, _n)


        #Print data
        if verbose
            print("\n\nIteration: ", it)
        #    print("\nNewton step = ", Δxnt)
            print("\nλ^2 = ", λ2)
            print("\nt = ", t)
        #    print("\nx = ", _x)
            print("\nf(x) = ", val)
        end

        #Update x
        _x += t*Δxnt


        it+=1
    end

    if it > maxIt
        print("\n\nMax Iterations Reached!")
    end

    return _x, val
end




#=
This function combines the consistent convergent rate of gradient descent
with the fast local convergence of Newton's Method by using gradient
descent at first, and then switching to Newton's method when the solution
gets within η > ϵ. Then, Newton's method gets the solution to within ϵ
=#
function Solve_UMP(_f, _x, _n, _α, _β, _η, _ϵ, maxIt, verbose)
    sol = Unconstrained_Grad_Descent(_f, _x, _n, _α, _β, _η, maxIt, verbose)

    print("\n\nCurrent solution is ", sol[2] ,"\nSwitching to Newton's Method\n")
    #use output point of gradient descent solution as starting point for Newton's Method
    return Unconstrained_Newton(_f, sol[1], _n, _α, _β, _ϵ, maxIt, verbose)
end




#######################################################################
#=
Section of code dealing with equality constrained minimization problems.
=#
#######################################################################



#################
#Feasible Start NM
#################

#Computes Newton Step (p. 526 (10.11))
function Compute_Newton_Step(_f, _∇f, _∇2f, _A, _At, _x, _n, _m)
    return ([_∇2f _At; _A zeros(_m,_m)]  \ [-_∇f; zeros(_m)])[1:_n]
end

#Compute Newton Decrement (p. 527 (10.12))
function Compute_Newton_Decrement(_Δxn, _∇2f)
    return abs(transpose(_Δxn) * _∇2f * _Δxn)
end

#Main solver function for Feasible Start Newton Method (p. 528, alg. 10.1)
function Newton_FS(_f, _x, _A, _n, _m, _α, _β, _ϵ, maxIt, verbose)

    obj = Float64
    λ2 = 10 #prime λ with a big number to ensure first iteration is run
    it = 0
    _At = transpose(_A)

    while λ2 / 2 > _ϵ && it <= maxIt

        obj = _f(_x)
        ∇f = Compute_Gradient(_f, _x)
        ∇2f = Compute_Hessian(_f, _x)
        Δxn = Compute_Newton_Step(_f, ∇f, ∇2f, _A, _At, _x, _n, _m)

        λ2 = Compute_Newton_Decrement(Δxn, ∇2f) #returns λ^2

        t = Back_Line_Search(_x, _f, obj, ∇f, Δxn, _α, _β, _n)

        #Update x
        _x += t*Δxn

        if verbose
            print("\n\nIteration ", it)
            print("\nλ^2 =         ",λ2)
            print("\nObjective =   ", obj)
        end

        it+=1
    end

    return _x, obj
end



#################
#Infeasible Start NM
#################

# This function computes the dual. (p. 531)
function Compute_Dual(_∇f, _At)
    return _At \ -_∇f
end

#This function computes the residual see page 533 (10.21)
function Compute_Residual(_∇f, _At, _v, _vr)
    return [_∇f + _At*_v; _vr]
end

#This function computes the residual just like Compute_Residual but in a way that we can use it for backgracking line search.
function R(_x, _v, _At, _A, _b, _f)
    return [Compute_Gradient(_f, _x) + _At*_v; _A*_x - _b]
end

#This function computes the Newton and Dual Step. See page 533 (10.22)
function Compute_Steps(_∇f, _∇2f, _x, _A, _At, _vr, _p, _v, _res)
    return [_∇2f _At; _A zeros(_p,_p)] \ -_res
end

#Backgracking line search for infeasible start method. (p. 534, Alg. 10.2)
function BLS_IF(_x, _v, _A, _At, _b, _vr, _f, _∇f, _Δxnt, _Δvnt, _α, _β, _nrm)
    _t = 1

    while norm(R(_x + _t*_Δxnt, _v + _t*_Δvnt, _At, _A, _b, _f)) > (1 - _α*_t)* _nrm
        _t *= _β
    end

    return _t
end

#Main solver function
function Newton_IFS(_f, _x, _A, _b, _n, _m, _α, _β, _ϵ, maxIt, verbose)

    #Initialize necessary values for first iteration
    At = transpose(_A)
    vr = _A*_x - _b
    ∇f = Compute_Gradient(_f, _x)
    _v = Compute_Dual(∇f, At)
    it = 0
    nrm = 100


    while  (norm(vr) > _ϵ || nrm > _ϵ) && it < maxIt

        it += 1

        ∇f = Compute_Gradient(_f, _x)
        ∇2f = Compute_Hessian(_f, _x)
        _v = Compute_Dual(∇f, At)

        vr = _A*_x - _b
        res = Compute_Residual(∇f, At, _v, vr)

        Δ = Compute_Steps(∇f, ∇2f, _x, _A, At, vr, _m, _v, res)
        Δxnt = Δ[1:_n]
        Δvnt = Δ[_n+1:_n+_m]

        nrm = norm(res)
        t = BLS_IF(_x, _v, _A, At, _b, vr, _f, ∇f, Δxnt, Δvnt, _α, _β, nrm)

        _x += t*Δxnt
        _v += t*Δvnt

        if verbose
            print("\n\nIteration: ", it)
            print("\nt = ", t)
            print("\nnorm(r) = ", nrm)
    #        print("\nx = ", x)
            print("\nObjective = ", _f(_x))
    #        print("\nΔx = ", Δxnt)
        end

    end

    return _x, _f(_x)
end
