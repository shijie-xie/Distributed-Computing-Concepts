# Scatter.py

import numpy as np
from mpi4py import MPI



#init u,m,mu (local)
alpha = 4
u = np.ones((5,5,5))
v = u[1:3,3,1:3]
print (np.shape(v))
#print the result



