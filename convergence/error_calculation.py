#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 22:37:35 2017

@author: john
"""
import numpy as np

def num_error(x,sigsq0,M,k,U,dx,dt,scheme, adv_order):
    
    # derivatives for a 2nd order upwind scheme
    Cxxx = np.sqrt(2)*M*x*(3*sigsq0 - x**2)*np.exp(-x**2/(2*sigsq0))/(2*np.sqrt(np.pi)*sigsq0**(7/2))
    
    Cxxxx = np.sqrt(2)*M*(3*sigsq0**2 - 6*sigsq0*x**2 + x**4)*np.exp(-x**2/(2*sigsq0))/(2*np.sqrt(np.pi)*sigsq0**(9/2))

    Cttt = np.sqrt(2)*M*(-3*U**3*sigsq0**4*x + U**3*sigsq0**3*x**3 + 9*U**2*k*sigsq0**4 - 18*U**2*k*sigsq0**3*x**2 + 3*U**2*k*sigsq0**2*x**4 + 45*U*k**2*sigsq0**3*x - 30*U*k**2*sigsq0**2*x**3 + 3*U*k**2*sigsq0*x**5 - 15*k**3*sigsq0**3 + 45*k**3*sigsq0**2*x**2 - 15*k**3*sigsq0*x**4 + k**3*x**6)*np.exp(-x**2/(2*sigsq0))/(2*np.sqrt(np.pi)*sigsq0**(13/2))
    
    
    if adv_order == 2:
        space_error = dt*(U/3*Cxxx+k/12*Cxxxx)*dx**2
    elif adv_order == 3:
        space_error = dt*(-U/12*Cxxxx*dx+k/12*Cxxxx)*dx**2        
    
    if scheme == 'Heun':
        time_error = -1/6*Cttt*dt**3
        
    if scheme == 'DIRK2':
        
        alpha = 1 - (2**.5)/2   
        time_error = (3*alpha**2 - 2*alpha**3 - 1/6)*Cttt*dt**3
        
    if scheme == 'tRule':
        time_error = 1/12*Cttt*dt**3 
    
    total_error = time_error + space_error
    
    return total_error


def analytical_sol(sigsq0,M,U,k,x,t,l):
    
    sigsq = sigsq0 + 2 * k *t
    a_coeff =  M / np.sqrt(2 * np.pi * sigsq)
    b_coeff = 2*sigsq
    
    Xad = x - U*t
    Xp = Xad + l
    Xp_mod = np.mod(Xp,2*l)
    Xtra = Xp_mod - l
    
    an_sol = a_coeff*np.exp(- Xtra**2/b_coeff )
    
    
    
    return an_sol