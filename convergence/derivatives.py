from sympy import *

'''
The output needs to be modified in order to use numpy functions such as np.exp,
np.pi and np.sqrt.
'''
x, U, t, dt, k, sigsq0, M = symbols('x U T dt k sigsq0 M')
sigsq = sigsq0 + 2 * k*t
a_coeff =  M / sqrt(2 * pi * sigsq)
b_coeff = 2*sigsq
C = a_coeff*exp( - (x - U*t)**2/b_coeff )

Cx = simplify(diff(C,x))
Cxx = simplify(diff(Cx,x))
Cxxx = simplify(diff(Cxx,x))
Cxxxx = simplify(diff(Cxxx,x))

# for RK schemes (Heun and Dirk)
Ct =  simplify(diff(C,t))
Ctt =  simplify(diff(Ct,t))
Cttt = simplify(diff(Ctt,t))

# We take the Taylor expansion from time t = 0
# so we need subsitute and simplify Cxxx, Cxxxx and Cttt accordingly
Cxxx = simplify(Cxxx.subs(t,0))
Cxxxx = simplify(Cxxxx.subs(t,0))
Cttt = simplify(Cttt.subs(t,0))

# for the trapezoidal rule
f = (t-dt/2)*Ctt;
errT = simplify(integrate(f,(t,0,dt)) )
#errS1 = simplify(integrate(Cxxx,(t,0,dt)))
#errS2 = simplify(integrate(Cxxxx,(t,0,dt)))
#
#errT = simplify(errT.subs(t,0))