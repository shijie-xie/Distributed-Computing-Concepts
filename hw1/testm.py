# Scatter.py

import numpy as np
from mpi4py import MPI



#init u,m,mu (local)
alpha = 4
u = np.ones((5,5))
u[1,1]=5
u[2,2]=7
v = u[1:3,1:3]

print(v)
print (np.shape(v))
#print the result



