import numpy as np
import matplotlib.pyplot as plt

'''
Assuming we are solving a time ODE of the form dy/dt = f(y), where y = y(x,t)
If the system is linear, it may be discretized as Y(t0+dt) = M*Y(t<=t0)
   
dt: time step
tend: final time
Y0: value of Y at t initial (initial conditions)
'''

def fly_plot(X,Y,tpause = 0.001):

    conc = plt.plot(X,Y,"-o",color='k') 
    plt.pause(tpause)
    conc[0].remove()
    
def rdiv(A,B):
    
    return np.transpose(np.linalg.solve(np.transpose(A),np.transpose(B)))     
              
#### Explicit schemes    
def fEuler(dt,tend,M,Y0,X,flyplot=False):
    
    Y = Y0
    
    for i in range(0,int(tend/dt+.5)):
        
              Y = Y + dt*np.dot(M,Y)  
              
              if flyplot:
                  fly_plot(X,Y)
                  
    return Y



def Heun(dt,tend,M,Y0,X,flyplot=False):
    
    Y = Y0
    
    for i in range(0,int(tend/dt+.5)):        
        
              Y = Y + dt/2*(np.dot(M,Y) + np.dot(M,Y + dt*np.dot(M,Y)))  
               
              if flyplot:
                  fly_plot(X,Y)
                 
    return Y   


def DIRK2(dt,tend,M,Y0,X,flyplot=False):
    
    Y = Y0  
    
    lam = 1 - (2**.5)/2
    beta = 1 - lam
    
    I = np.eye(M.shape[0])
    M0 = I - lam*dt*M 
    M1 = I + dt*beta*rdiv(M0,M)
    M = np.linalg.solve(M0,M1)
    for i in range(0,int(tend/dt+.5)): 
        
        Y = np.dot(M,Y)
    
        if flyplot:
            fly_plot(X,Y)
            
    return Y    

def tRule(dt,tend,M,Y0,X,flyplot=False):
    
    Y = Y0

    M1 = np.eye(M.shape[0]) - dt/2*M
    M2 = np.eye(M.shape[0]) + dt/2*M
    M = np.linalg.solve(M1,M2);

    for i in range(0,int(tend/dt+.5)): 
        
        Y = np.dot(M,Y)   
        
        if flyplot:
            fly_plot(X,Y)
            
    return Y    