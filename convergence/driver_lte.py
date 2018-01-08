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
k = 1.0e-3 # dispersion coefficient [m^2/s]
l = 500.0
xW = -l # x-coordinate west boundary [m]
xE = l # x-coordinate east boundary [m]
C00 = .5e0 # maximum of IC [kg/m3]
a = 100.0 # spreading parameter for IC [m]
sigsq0 = a**2/2
M = C00*(2*np.pi*sigsq0)**.5
U = 1

# PERFORM TEST
nx = 4000 # number of segments
Nx = nx + 1 # number of points
adv_order = 2
X,dx = domain(xW,xE,Nx,'Nx')
Mdiff = diffusion(Nx,k,dx)
Madv = advection(Nx,U,dx,adv_order,'periodic')
Mad = Mdiff + Madv

C0 = gaussian(X,C00,a)
dt = 1


# choose a range of time steps
T0 = 1
p = list(range(-12,5))
DT = T0/np.power(2*np.ones(len(p)),p)

EexactHeun = np.zeros(len(DT))
EnumHeun =  np.zeros(len(DT))
for i in range(len(DT)):
    
    tend = DT[i]    
    Cend = tRule(DT[i],tend,Mad,C0,X) 
    Can = Can = analytical_sol(sigsq0,M,U,k,X,DT[i],l)
    EexactHeun[i] = np.linalg.norm(Can-Cend)
    enum = num_error(X,sigsq0,M,k,U,dx,DT[i],'tRule')
    EnumHeun[i] = np.linalg.norm(enum)
    
    
#EexactDIRK2 = np.zeros(len(DT))
#EnumDIRK2 =  np.zeros(len(DT))
#for i in range(len(DT)):
#    
#    tend = DT[i]    
#    Cend = DIRK2(DT[i],tend,Mad,C0,X) 
#    Can = analytical_sol(sigsq0,M,U,k,X,DT[i])
#    EexactHeun[i] = np.linalg.norm(Can-Cend)*dx**.5
#    enum = num_error(X,sigsq0,M,k,U,dx,DT[i],'DIRK2')
#    EnumDIRK2[i] = np.linalg.norm(enum)    

def conv_rate(x,DT,expected_cr):
    
    lx0 = np.log(np.abs(x[0]))
    lxEnd = np.log(np.abs(x[-1]))    
    lt0 = np.log(np.abs(DT[0]))
    ltEnd = np.log(np.abs(DT[-1]))    

    lEx = lxEnd - expected_cr*ltEnd
    
    cr = (lxEnd - lx0 )/(ltEnd - lt0)   
    conv_line = np.exp(lEx + expected_cr*np.log(DT))
    
    return cr, conv_line

cr, cl = conv_rate(EexactHeun,DT,1)
print(cr)

diffE = np.abs(EexactHeun-EnumHeun)
relE = diffE/EexactHeun

fst = 30
fsst = 30
fslb = 30
fsg = 40
lw = 2
ms = 20
mew = 5
wim = 1920
###############################################3
him = wim
my_dpi = 96

fig1 = plt.figure(figsize=(wim/my_dpi, him/my_dpi), dpi=my_dpi)
l1, = plt.loglog(DT,EexactHeun,'-*b',markersize=ms,markeredgewidth = mew, linewidth = lw,label='$e_{exact} = y_{numerical} - y_{exact}$')
l2, = plt.loglog(DT,EnumHeun,'--or',markersize=ms,linewidth = lw,label='$e_{analytical}$', markeredgewidth=mew,markeredgecolor='r', markerfacecolor='None')

l3, = plt.loglog(DT,diffE,'+:g',markersize=ms,markeredgewidth = mew/2,linewidth = lw,label='$=|e_{num} - e_{exact}|$')
l4, = plt.loglog(DT,relE,':xb',markersize=ms,markeredgewidth = mew/2,linewidth = lw,label='$= \\frac{|e_{num} - e_{analytical}|}{|e_{numerical}|}$')
plt.title("Heun's method total local truncation error \n PDE: advection-diffusion $t_{end} = dt$",fontsize=fsst)    

plt.xlabel('dt',fontsize=fslb)

#plt.legend(handles=[l1,l2,l3,l4],fontsize = fsg)
#plt.legend(handles=[l1,l2,l3,l4],fontsize = fsg,loc='center left', bbox_to_anchor=(0.0, .75))
plt.legend(handles=[l1,l2,l3,l4],fontsize = fsg,loc='upper center')

plt.tick_params(axis='both', which='major', labelsize=40)
#fig1.savefig('lte_AD_Heun.png', bbox_inches='tight',dpi=400)
