import os
file_dir= os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)

import numpy as np
from space_discretization import *
from time_integration import *
from matplotlib import pyplot as plt
from error_calculation import *

print('Finite-Difference Advection Diffusion Solver')

# choose parameters for the initial profile
k = 0.0e-3 # dispersion coefficient [m^2/s]
l = 500.0
xW = -l # x-coordinate west boundary [m]
xE = l # x-coordinate east boundary [m]
C00 = .5e0 # maximum of IC [kg/m3]
a = 100.0 # spreading parameter for IC [m]
sigsq0 = a**2/2
M = C00*(2*np.pi*sigsq0)**.5
U = 1

# PERFORM TEST
nx = 500 # number of segments
Nx = nx + 1 # number of points
adv_order = 3
X,dx = domain(xW,xE,Nx,'Nx')
Mdiff = diffusion(Nx,k,dx)
Madv = advection(Nx,U,dx,adv_order,'periodic')
Mad = Mdiff + Madv

C0 = gaussian(X,C00,a)
dt = 1

fst = 30
fsst = 30
fslb = 30
fsg = 40
lw = 2
ms = 30
mew = 10
wim = 1920
###############################################3
him = wim
my_dpi = 96

######################
def rdiv(A,B):
    
    return np.transpose(np.linalg.solve(np.transpose(A),np.transpose(B)))     
# choose a range of time steps
q = 10
T0 = 2**q
p = list(range(0,15))
DT = T0/np.power(2*np.ones(len(p)),p)

tend = T0
# advected coordinates
Can = analytical_sol(sigsq0,M,U,k,X,tend,l)
check = np.linalg.norm(Can-C0)

EH = np.zeros(len(DT))
ED = np.zeros(len(DT))
ET = np.zeros(len(DT))

EnumH = np.zeros(len(DT)) 
EnumD = np.zeros(len(DT)) 
EnumT = np.zeros(len(DT)) 

for i in range(len(DT)):
    print(i)
    dt = DT[i]
    
    enumH = 0  
    enumD = 0  
    enumT = 0  
    
###############################################################################    
    CH = C0.copy()
    for m in range(0,int(tend/dt+.5)):        
        

            CH = CH + dt/2*(np.dot(Mad,CH) + np.dot(Mad,CH + dt*np.dot(Mad,CH)))  
            enumH += num_error(X,sigsq0,M,k,U,dx,dt,'Heun',adv_order)            
            
###############################################################################    
    lam = 1 - (2**.5)/2
    beta = 1 - lam
    I = np.eye(Mad.shape[0])
    M0 = I - lam*dt*Mad 
    M1 = I + dt*beta*rdiv(M0,Mad)
    Mdirk2 = np.linalg.solve(M0,M1)
          
    CD = C0.copy()
    for m in range(0,int(tend/dt+.5)):            

        CD = np.dot(Mdirk2,CD)
        enumD += num_error(X,sigsq0,M,k,U,dx,dt,'DIRK2',adv_order)              
###############################################################################    
    M1 = np.eye(Mad.shape[0]) - dt/2*Mad
    M2 = np.eye(Mad.shape[0]) + dt/2*Mad
    Mtrule = np.linalg.solve(M1,M2);

    CT = C0.copy()            
    for m in range(0,int(tend/dt+.5)):            

        CT = np.dot(Mtrule,CT)
        enumT += num_error(X,sigsq0,M,k,U,dx,dt,'tRule',adv_order)              
###############################################################################  
        
    EH[i] = np.linalg.norm(Can-CH)*dx**.5
    EnumH[i] = np.linalg.norm(enumH)        
    
    ED[i] = np.linalg.norm(Can-CD)*dx**.5
    EnumD[i] = np.linalg.norm(enumD)
    
    ET[i] = np.linalg.norm(Can-CT)*dx**.5
    EnumT[i] = np.linalg.norm(enumT)




#fig1 = plt.figure()

#plt.loglog(DT[sp0:],EH[sp0:],':or')
#plt.loglog(DT[sp0:],EnumH[sp0:],'-xr')

#plt.loglog(DT[sp0:],ED[sp0:],':ob')
#plt.loglog(DT[sp0:],EnumD[sp0:],'-xb')
#
#plt.loglog(DT[sp0:],ET[sp0:],':og')
#plt.loglog(DT[sp0:],EnumT[sp0:],'-xg')
#


##########################


def relerr(x,xnum):
    
    return np.abs(x - xnum)
#
relH = relerr(EH,EnumH)
relD = relerr(ED,EnumD)
relT = relerr(ET,EnumT)

#plt.loglog(DT[sp0:],relH[sp0:],':or')

fst = 30
fsst = 30
fslb = 30
fsg = 40
lw = 2
ms = 30
mew = 10
wim = 1920
him = wim
my_dpi = 96

sp0 = 6
##########################
def conv_rate(x,DT,expected_cr,end_pt = len(DT)):
    
    lx0 = np.log(np.abs(x[0]))
    lxEnd = np.log(np.abs(x[-1]))    
    lt0 = np.log(np.abs(DT[0]))
    ltEnd = np.log(np.abs(DT[-1]))    

    lEx = np.log(np.abs(x[end_pt-1])) - expected_cr*np.log(np.abs(DT[end_pt-1])) 
    
    cr = (lxEnd - lx0 )/(ltEnd - lt0)   
    conv_line = np.exp(lEx + expected_cr*np.log(DT))
    
    return cr, conv_line

cr, cl = conv_rate(relT,DT,2,sp0)
sp1 = sp0
ep= sp1 + 6
sp2 = sp0 + 3
plt.close("all")
fig2 = plt.figure(figsize=(wim/my_dpi, him/my_dpi), dpi=my_dpi)
#########################
l1, = plt.loglog(DT[sp0:],relD[sp0:],':xb',markersize=ms,markeredgewidth = mew, linewidth = lw,label='DIRK2')
l2, = plt.loglog(DT[sp0:],relT[sp0:],':xg',markersize=ms,markeredgewidth = mew, linewidth = lw,label= 'tapezoidal')
l3, = plt.loglog(DT[sp1-1:ep],cl[sp1-1:ep]*2,'-.k',label='2nd order')
l4, = plt.loglog(DT[sp2:],relH[sp2:],':xr',markersize=ms,markeredgewidth = mew, linewidth = lw,label= 'Heun')

plt.legend(handles=[l1,l2,l3,l4],fontsize = fsg)
plt.tick_params(axis='both', which='major', labelsize=40)
plt.title("relative error of the analytic error approximation",fontsize=fsst)  


fig2.savefig('gte_DIRK_TR.png', bbox_inches='tight',dpi=400)