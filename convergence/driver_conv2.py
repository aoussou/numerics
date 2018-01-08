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
nx = 1000 # number of segments
Nx = nx + 1 # number of points
adv_order = 2
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
# choose a range of time steps
q = 8
T0 = 2**q
p = list(range(3,13))
DT = T0/np.power(2*np.ones(len(p)),p)

tend = T0
# advected coordinates
Can = analytical_sol(sigsq0,M,U,k,X,tend,l)
check = np.linalg.norm(Can-C0)

EexactHeun = np.zeros(len(DT))
for i in range(len(DT)):
    
    Cend = Heun(DT[i],tend,Mad,C0,X) 

    EexactHeun[i] = np.linalg.norm(Can-Cend)*dx**.5

EexactDIRK2 = np.zeros(len(DT))
for i in range(len(DT)):
    
    Cend = DIRK2(DT[i],tend,Mad,C0,X) 

    EexactDIRK2[i] = np.linalg.norm(Can-Cend)*dx**.5
    
EexacttRule = np.zeros(len(DT))
for i in range(len(DT)):
    
    Cend = tRule(DT[i],tend,Mad,C0,X) 

    EexacttRule[i] = np.linalg.norm(Can-Cend)*dx**.5


def conv_rate(x,DT,expected_cr,end_pt = len(DT)):
    
    lx0 = np.log(np.abs(x[0]))
    lxEnd = np.log(np.abs(x[-1]))    
    lt0 = np.log(np.abs(DT[0]))
    ltEnd = np.log(np.abs(DT[-1]))    

    lEx = np.log(np.abs(x[end_pt-1])) - expected_cr*np.log(np.abs(DT[end_pt-1])) 
    
    cr = (lxEnd - lx0 )/(ltEnd - lt0)   
    conv_line = np.exp(lEx + expected_cr*np.log(DT))
    
    return cr, conv_line

cr, cl = conv_rate(EexacttRule,DT,2,3)
plt.close("all")

him = wim
my_dpi = 96
sp0 = 6
sp1 = 0
ep=-4


fig2 = plt.figure(figsize=(wim/my_dpi, him/my_dpi), dpi=my_dpi)

l2, = plt.loglog(DT[sp1:],EexactDIRK2[sp1:],':+',markersize=ms,markeredgewidth = mew, linewidth = lw,label='DIRK2')
l3, = plt.loglog(DT[sp1:],EexacttRule[sp1:],':x',markersize=ms,markeredgewidth = mew, linewidth = lw,label='trap rule')
l4, = plt.loglog(DT[0:ep],cl[0:ep]/1.5,'-.k',label='2nd order')
l1, = plt.loglog(DT[sp0:],EexactHeun[sp0:],':o',markersize=ms,linewidth = lw,label='Heun', markeredgewidth=mew,markeredgecolor='r', markerfacecolor='None')
plt.title("time converge \n 2nd order upwind space discretization",fontsize=fsst)    

plt.legend(handles=[l1,l2,l3,l4],fontsize = fsg)
plt.tick_params(axis='both', which='major', labelsize=40)

fig2.savefig('conv2.png', bbox_inches='tight',dpi=400)