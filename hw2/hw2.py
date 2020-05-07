############# Project 2 #############

import numpy as np
import scipy
import matplotlib
import time

# can't use mpi yet
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


from devito import configuration
configuration['log-level'] = 'WARNING'

path='project_data/project_data/'
geo = np.load(path+'common.npy',allow_pickle=True).item()

nshots = 51  # Number of shots to create gradient from
edge_num = 4  # Edges used,1-4 for top/left+right/top+left+right/all
nreceivers = 51*edge_num  # Number of receiver locations per shot ,as 51,102,153,204 
fwi_iterations = 5  # Number of outer FWI iterations

label_left = [0,51,102,51,0]
label_right=[0,102,204,204,204]


start_time=time.time()


from examples.seismic import demo_model, plot_velocity, plot_perturbation

# Define true and initial model
shape = (101, 51)  # Number of grid point (nx, nz)
spacing = (10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0.)  # Need origin to define relative source and receiver locations(on the Top Left conner)

v=np.empty(shape, dtype=np.float32)
v0=np.empty(shape, dtype=np.float32)
v0[:,:15]=2.
v0[:,15:40]=2.
v0[:,40:]=2.

# v=v0, just used in test example, don't use it
v[:,:15]=2
v[:,15:40]=1.8
v[:,40:]=3.4

model = demo_model('constant-isotropic', vp=v, origin=origin, shape=shape, spacing=spacing, nbl=20)  # the real model,do not use
model0 = demo_model('constant-isotropic', vp=v0, origin=origin, shape=shape, spacing=spacing, nbl=20)  # the initial model

#plot_velocity(model)
#plot_velocity(model0)

# Define acquisition geometry: source
from examples.seismic import AcquisitionGeometry,TimeAxis

t0 = 0.      # start time = 0s
tn = 600.    # 600ms = 0.6s
dt = 0.3     # 0.3ms
f0 = 0.015   # f of Ricker Source
time_range = TimeAxis(start=t0, stop=tn, step=dt)


label0=0    # the first source
src_coordinates = np.empty((1, 2))
src_coordinates.data[0,0] = geo['src_coordinates'][label0,0]
src_coordinates.data[0,1] = 500-geo['src_coordinates'][label0,1]
rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 0] = geo['rec_coordinates'][label_left[edge_num]:label_right[edge_num],0] # can be replaced as [:,51:102]...
rec_coordinates[:, 1] = 500-geo['rec_coordinates'][label_left[edge_num]:label_right[edge_num],1] # for the origin point is at top left


# Geometry

geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')
geometry.resample(dt)

#print(geometry.time_axis.num)
# all source locations
source_locations = np.empty((nshots, 2), dtype=np.float32)
source_locations[:, 0] = geo['src_coordinates'][:,0]
source_locations[:, 1] = 500 - geo['src_coordinates'][:,1] # for the origin point is at top left


from examples.seismic.acoustic import AcousticWaveSolver

solver = AcousticWaveSolver(model, geometry, space_order=4)

from devito import Function, TimeFunction
from examples.seismic import Receiver



def fwi_gradient(vp_in):

    # Create symbols to hold the gradient and residual
    grad = Function(name="grad", grid=model.grid)
    residual = Receiver(name='rec', grid=model0.grid,
                        time_range=geometry.time_axis, 
                        coordinates=geometry.rec_positions)
    objective = 0.
    myshots_left=nshots*rank//size
    myshots_right=nshots*(rank+1)//size
    for i in range(myshots_left,myshots_right):
        # Update source location
        geometry.src_positions[0, :] = source_locations[i, :]
        
        # true data from file
        rec= np.load(path+'project2_data_shot_'+str(i)+'.npy',allow_pickle=True)
        true_d = Receiver(name='rec', grid=model.grid, npoint=nreceivers, time_range=geometry.time_axis,coordinates=geometry.rec_positions,data=rec[:,label_left[edge_num]:label_right[edge_num]])
        #true_d, _, _ = solver.forward(vp=model.vp)
        #print(true_d.data[:,3])
        #print(true_d.data.shape)
        # Compute smooth data and full forward wavefield u0
        smooth_d, u0, _ = solver.forward(vp=vp_in, save=True)
        #print(smooth_d.data.shape)
        # Compute gradient from data residual and update objective function 
        residual.data[:] = smooth_d.data[:] - true_d.data[:]
        
        objective += .5*np.linalg.norm(residual.data.flatten())**2
        solver.gradient(rec=residual, u=u0, vp=vp_in, grad=grad)
        objective = comm.reduce(objective, root=0,op=MPI.SUM)
        grad = comm.reduce(grad, root=0,op=MPI.SUM)
    if (rank == 0):
      return objective, -grad.data

#ff, update = fwi_gradient(model0.vp)


from examples.seismic import plot_image

# Plot the FWI gradient
#plot_image(update, vmin=-1e5, vmax=1e5, cmap="jet")

# Plot the difference between the true and initial model.
# This is not known in practice as only the initial model is provided.
#plot_image(model0.vp.data - model.vp.data, vmin=-1e-1, vmax=1e-1, cmap="jet")

# Show what the update does to the model
#alpha = .5 / np.abs(update).max() #.5
#plot_image(model0.vp.data - alpha*update, vmin=1., vmax=5., cmap="jet")


def apply_box_constraint(vp):
    # Maximum possible 'realistic' velocity is 1.5 km/sec
    # Minimum possible 'realistic' velocity is 4.0 km/sec
    return np.clip(vp, 1.5, 4.0)


# Run FWI with gradient descent
history = np.zeros((fwi_iterations, 1))
gradold = np.zeros(model0.vp.data.shape,dtype=np.float32)
vpold = np.zeros(model0.vp.data.shape,dtype=np.float32)  #used for Barzilai-Borwein step
for i in range(0, fwi_iterations):
    # Compute the functional value and gradient for the current
    # model estimate

    phi, direction = fwi_gradient(model0.vp)
    

    # Store the history of the functional values
    history[i] = phi
    
    # Artificial Step length for gradient descent
    # In practice this would be replaced by a Linesearch (Wolfe, ...)
    # that would guarantee functional decrease Phi(m-alpha g) <= epsilon Phi(m)
    # where epsilon is a minimum decrease constant
    #alpha = 0.05 / np.abs(direction).max()
    bbsteps = model0.vp.data - vpold
    bbstepy = direction - gradold
    gradold = np.copy(direction)
    vpold = np.copy(model0.vp.data)
    if i>0 :
   #     print(np.abs(vpold).max())
   #     print(np.abs(gradold).max())
   #     print(np.abs(bbsteps).max())
    #    print(np.abs(bbstepy).max())
    # take Barzilai-Borwein step instead
        alpha = np.clip(np.dot(bbsteps.flatten(),bbstepy.flatten())/np.dot(bbstepy.flatten(),bbstepy.flatten()),0,.7/np.abs(direction).max())
    else :
        alpha = .5 / np.abs(direction).max()
    print('alpha=%f'%alpha)

    # Update the model estimate and enforce minimum/maximum values
    model0.vp = apply_box_constraint(model0.vp.data - alpha * direction)
    
    # Log the progress made
    print('Objective value is %f at iteration %d' % (phi, i+1))


finish_time=time.time()
print('time = %f s'%(finish_time-start_time))
# Plot inverted velocity model
plot_velocity(model0)
np.savetxt('model0',model0.vp.data)
for i in range(10):
    print('%d th vp is %f'%(i,np.mean(model0.vp.data[20:121,23+5*i])))
# Plot objective function decrease
import matplotlib.pyplot as plt
plt.figure()
plt.plot(history)
plt.xlabel('Iteration number')
plt.ylabel('Misift value Phi')
plt.title('Convergence')
plt.show()














