from matplotlib import pyplot as plt
import numpy as np
import copy

t0 = 0 # initial time
y0 = np.exp(t0) # initial value

# choose a range of time steps
T0  = 10
p = list(range(0,14))
DT = T0/np.power(2*np.ones(len(p)),p)


E = []; Enum = []



for dt in DT:
    
    Ts = dt # simulation time set to one time step
    nT = int(Ts/dt)  # number of time steps    
    T = t0 + Ts # final time
 
    # initialize y  and the final numerical error
    y = copy.copy(y0)
    y_exact = np.exp(T)
    e_num = 0
    
    for i in range(nT):
        
        e_num +=  -dt**3/6*y
        y_tilde = y + dt*y
        y = y + dt/2*(y + y_tilde)

    e_num_exact = y - y_exact 
    E.append(e_num_exact)
    Enum.append(e_num)
    
###################
# PLOTTING AND POSTPROCESSING #
###################    
    
fst = 30
fsst = 30
fslb = 30
fsg = 40
lw = 2
ms = 30
mew = 10

diffE = np.abs(np.asarray(E)-np.asarray(Enum))
relDiffE = diffE/np.abs(np.asarray(E))

wim = 1920
him = .9*wim
my_dpi = 96
fig = plt.figure(figsize=(wim/my_dpi, him/my_dpi), dpi=my_dpi)
#fig, ax = plt.subplots()

#plt.rcParams["figure.figsize"] = [wim,him]
plt.title("Heun's method local truncation error \n ODE: $y=y'$, $t_{end} = dt$, $t_0 = %0g$" %t0,fontsize=fsst)

l1, = plt.loglog(DT,np.abs(E),'-*b',markersize=ms,markeredgewidth = mew, linewidth = lw,label='$e_{num} = y_{numerical} - y_{exact}$')
l2, = plt.loglog(DT,np.abs(Enum),'--or',markersize=ms,linewidth = lw,label='$e_{analytical}$', markeredgewidth=mew,markeredgecolor='r', markerfacecolor='None')
l3, = plt.loglog(DT,relDiffE,':xb',markersize=ms,markeredgewidth = mew/2,linewidth = lw,label='$\\frac{|e_{num} - e_{analytical}|}{|e_{numerical}|}$')
l4, = plt.loglog(DT,diffE,'+:g',markersize=ms,markeredgewidth = mew/2,linewidth = lw,label='$|e_{num} - e_{exact}|$')

plt.xlabel('dt',fontsize=fslb)
plt.tick_params(axis='both', which='major', labelsize=40)

plt.legend(handles=[l1,l2,l3,l4],fontsize = fsg)
fig.savefig('lte_ode_Heun.png', bbox_inches='tight',dpi=400)
