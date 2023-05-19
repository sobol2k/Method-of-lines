import numpy as np
from scipy.integrate import solve_ivp

"""

Implements the non adaptive method of lines for 

u_t(x) = a u'(x)

where a \in R assuming dirichlet boundary condition at the left and right side of the domain. We approximate u'(x) with second order FD

u'(x) ~ u(x_i-1) - u(x_i+1) / 2h where h is the spacing between the nodes.

"""

' Problem parameters'
a = 1
dirichlet_boundary_value = 1

' Create spatial domain '
x_start = 0
x_end = 5
number_of_nodes = 1000
x = np.linspace(x_start,x_end,number_of_nodes)
h = x[1]-x[0] #TODO

' Create spatial discretization '
ones_offdiag = np.ones(number_of_nodes-3)
upper = np.diagflat(-ones_offdiag, k=-1)
lower = np.diagflat(ones_offdiag,  k=1)
system_matrix = upper+lower
system_matrix *= a/2*h

' Dirichlet boundaries '
dirichlet_boundary_vector = np.zeros(number_of_nodes-2)
dirichlet_boundary_vector[0]  = dirichlet_boundary_value * a/2*h
dirichlet_boundary_vector[-1] = dirichlet_boundary_value * a/2*h

' Create time domain'
t_start = 0
t_end = 100
t_span = [t_start,t_end]

' Initial values'
y0 = np.ones(number_of_nodes-2)

' create complete right hand side '
def fun(t,y,system_matrix,dbv):
    return system_matrix @ y + dbv

' Solve ODE system '
res = solve_ivp(fun, t_span, y0, method='RK45', args = (system_matrix,dirichlet_boundary_vector))

' Plot solution '
