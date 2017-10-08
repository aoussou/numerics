An example of how to solve the advection-diffusion equation using finite difference.

You can choose between a number of time integration solvers, namely the Forward Euler, Heun and 4th order Runge-Kutta explicit schemes, and the trapezoidal rule, DIRK2 and DIRK3 implicit schemes.

There is still a lot of room for improvement, in particular using sparse matrices to speed up the code.

<hr>

In the FD advection-diffusion error derivations document, the function for the spatial and temporal errors of various schemes are derived. It is also shown that 2nd order convergence cannot be observed with a second order time scheme if the spatial discretization is not at least of 3rd order.
