import numpy as np
import pylab

def domain(xL,xR,value,use):
    xL = float(xL)
    xR = float(xR)
    if use == 'Nx':
        return np.linspace(xL,xR,value),  (xR - xL)/float(value-1)
    elif use == 'dx':
        return np.arange(xL,xR+1,value),int((xR - xL)/value+1) 
        # you need to do xE+1 to include the end value

def plot_domain(X,Y):
    pylab.plot(X,Y, "-o", label="initial solution")

def gaussian(x,C00,a):
# x must be numpy array list
    return C00*np.exp(-x**2/a**2);


def diffusion(Nx,k,dx,bc='periodic'):
    diagm = np.eye(Nx)
    diagl = np.eye(Nx,Nx,-1) 
    diagu = np.eye(Nx,Nx,1)     
    M = -2*diagm + diagl + diagu
    
    if bc == 'periodic':
        M[0,-2] = 1.0
        M[-1,1] = 1.0

    return k/dx**2*M

def advection(Nx,U,dx,order,bc='periodic'):
    diagm = np.eye(Nx)
    diagl = np.eye(Nx,Nx,-1)  
    
    if order == 1:       
        M = diagm - diagl
        
        if bc == 'periodic':
            M[0,-2] = -1.0    
        
        M = -U/dx*M
        
    if order == 2:
        diagll = np.eye(Nx,Nx,-2)
        M =  3*diagm - 4*diagl + diagll
                      
        if bc == 'periodic':
            M[0,-2] = -4
            M[0,-3] = 1             
            M[1,-2] = 1
             
        M = -U/(2.0*dx)*M
               
    if order == 3:
        diagu = np.eye(Nx,Nx,1) 
        diagll = np.eye(Nx,Nx,-2)        
        M = 2*diagu + 3*diagm - 6*diagl + diagll
  
        if bc == 'periodic':
            M[0,-2] = -6
            M[0,-3] = 1             
            M[1,-2] = 1 
            M[-1,1] = 2; 
 
        M = -U/(6*dx)*M;                    
              
    return M