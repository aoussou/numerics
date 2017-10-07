import datetime
import calendar
import pylab


from space_discretization import *
from time_integration import *
print("Simulation started on", datetime.date.today(), 
      '(',calendar.day_name[datetime.datetime.today().weekday()],')'
      "at",datetime.datetime.now().time() )

print('Finite-Difference Advection Diffusion Solver')

tend = 1000.0;
dt = 1
k = 0.0e0 # dispersion coefficient [m^2/s]
l = 500.0
xW = -l # x-coordinate west boundary [m]
xE = l # x-coordinate east boundary [m]

C00 = .5e0 # maximum of IC [kg/m3]
a = 100.0 # spreading parameter for IC [m]
U = 1
''' if you want to impose the spatial step dx '''
#dx = 100
#X,Nx = domain(xW,xE,dx,'dx')


''' if you want to impose the number of points '''
nx = 100 # number of segments
Nx = nx + 1 # number of points
X,dx = domain(xW,xE,Nx,'Nx')
C0 = gaussian(X,C00,a)

adv_order = 3

plot_domain(X,C0)
pylab.legend()
pylab.xlabel('x')
pylab.ylabel('C')
testfilename='initial_solution.png'

Mdiff = diffusion(Nx,k,dx)
Madv = advection(Nx,U,dx,adv_order,'periodic')

Mad = Mdiff + Madv


flyplot = True # set to true if you want to plot on the fly
Cend = DIRK3(dt,tend,Mad,C0,X,flyplot) 
#other time integration schemes can be used, look in time_integration.py


pylab.plot(X,Cend, "-o", color="r")
pylab.legend()
pylab.xlabel('x')
pylab.ylabel('C')
testfilename='solution.png'

print("Simulation ended on", datetime.date.today(), 
      '(',calendar.day_name[datetime.datetime.today().weekday()],')'
      "at",datetime.datetime.now().time() )