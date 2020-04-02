# Scatter.py

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#set the constant
lamda = 10.0
alpha = 1000.0
# change the parameter

# optimization of u
# solve ( 1 - \alpha \Delta) u = ( nimg - \nabla \cdot (\mu + \alpha m))
def op_u():
    global comm, size,rank, alpha, mu,m,u,count,localimg
    global localnx,ny,nz
    #X = (\mu + \alpha m)
    X = mu + alpha * m
    #Y = localnimg - \nabla \cdot (\mu + \alpha m)
    Y =   np.zeros((localnx+2,ny+2,nz+2))#init Y
    # MPI of Y
    Xfor = X[0,localnx,:,:]#send the last X to n+1 process
    Xbac = X[0,1,:,:]
    #NOTE both side need MPI
    for_send_buf =np.copy(Xfor)
    bac_send_buf =np.copy(Xbac)
    #print(for_send_buf.shape)
    for_recv =np.empty([ny+2,nz+2], dtype=np.float64)
    bac_recv =np.empty([ny+2,nz+2], dtype=np.float64)
    

    # only n to n+1 is in need
    if rank==0 :
        for_send_req =comm.Isend(for_send_buf, dest = rank+1, tag = rank+350)
        for_send_req.Wait()
        bac_recv_req =comm.Irecv(bac_recv, source = rank+1 , tag = rank+451)
        bac_recv_req.Wait()
        X[0,localnx+1,:,:] = bac_recv
  
    elif rank == size-1:
        for_recv_req = comm.Irecv(for_recv, source = rank-1 , tag = rank+349)
        for_recv_req.Wait()
        bac_send_req =comm.Isend(bac_send_buf, dest = rank-1, tag = rank+450)
        bac_send_req.Wait()
        X[0,0,:,:] = for_recv


    else :
        #pass in both direction
        for_send_req =comm.Isend(for_send_buf,dest = rank+1, tag = rank+350)
        bac_send_req =comm.Isend(bac_send_buf,dest = rank-1, tag = rank+450)
        #recv
        bac_recv_req =comm.Irecv(bac_recv, source = rank+1 , tag = rank+451)
        for_recv_req =comm.Irecv(for_recv, source = rank-1 , tag = rank+349)

        for_send_req.Wait()
        for_send_req.Wait()
        bac_recv_req.Wait()
        for_recv_req.Wait()
        X[0,localnx+1,:,:] = bac_recv
        X[0,0,:,:] = for_recv
    # local part of Y
    #use matrix instead
    #for i in range(1,localnx+1):
    #    for j in range(1,ny+1):
    #        for k in range(1,nz+1):
    #            #NOTE: localimg & Y has diffrent indices
    #            Y[i][j][k] = localimg[i-1][j-1][k-1]  -  ( X[i][j][k][0]-X[i-1][j][k][0] )-( X[i][j][k][1]-X[i][j-1][k][1] ) - ( X[i][j][k][2]-X[i-1][j][k-1][2] )
    Y[1:localnx+1 , 1:ny+1 ,1:nz+1] = localimg -( X[0,2:localnx+2,1:ny+1,1:nz+1] - X[0,0:localnx,1:ny+1,1:nz+1])/2 -( X[1,1:localnx+1,2:ny+2,1:nz+1] - X[1,1:localnx+1,0:ny,1:nz+1])/2  -( X[2,1:localnx+1,1:ny+1,2:nz+2] - X[2,1:localnx+1,1:ny+1,0:nz])/2             
    #print(np.shape(Y[1:localnx+1 , 1:ny+1 ,1:nz+1]))
    #print("in iteration ",count,"rank",rank, "finish assemble the matrix" )



  
    # solve ( 1 - \alpha \Delta) u = Y
    # central diff
    for t in range(0,15):#TODO 15 is a parameter here to op
        # solve (1 + 1.5 * \alpha ) u = Y + 0.25*\alpha ( \sum u^{\pm 1})
        #u[1:localnx,1:ny,1:nz] =1.0/(1+1.5*alpha)*( Y[1:localnx,1:ny,1:nz]+ 0.25*alpha*(u[0:localnx-1,1:ny,1:nz]+u[2:localnx+1,1:ny,1:nz]+u[1:localnx,0:ny-1,1:nz]+u[1:localnx,2:ny+1,1:nz]+u[1:localnx,1:ny,0:nz-1]+u[1:localnx,1:ny,2:nz+1])
        # solve (1 + 6 * \alpha ) u = Y + \alpha ( \sum u^{\pm 1})
        #u[localnx,:,:] = u[localnx-1,:,:]
        u[1:localnx+1,1:ny+1,1:nz+1] =(1.0/(1+6*alpha))*( Y[1:localnx+1,1:ny+1,1:nz+1]+ alpha*(u[0:localnx,1:ny+1,1:nz+1]+u[2:localnx+2,1:ny+1,1:nz+1]+u[1:localnx+1,0:ny,1:nz+1]+u[1:localnx+1,2:ny+2,1:nz+1]+u[1:localnx+1,1:ny+1,0:nz]+u[1:localnx+1,1:ny+1,2:nz+2]))
        #MPI part
        for_send_buf =np.copy(u[localnx,:,:])
        bac_send_buf =np.copy(u[1,:,:])
        for_recv =np.empty([ny+2,nz+2], dtype=np.float64)
        bac_recv =np.empty([ny+2,nz+2], dtype=np.float64)
        #print(count, " ",t,rank)
        if rank==0 :
            #pass forward
            for_send_req =comm.Isend(for_send_buf, dest = rank+1, tag = rank+350)
            for_send_req.Wait()
            bac_recv_req =comm.Irecv(bac_recv, source = rank+1 , tag = rank+451)
            bac_recv_req.Wait()
            u[localnx+1,:,:] = bac_recv
            
        elif rank == size-1:
            for_recv_req = comm.Irecv(for_recv, source = rank-1 , tag = rank+349)
            for_recv_req.Wait()

            #pass backward

            bac_send_req =comm.Isend(bac_send_buf, dest = rank-1, tag = rank+450)
            #recv 
            bac_send_req.Wait()
            u[0,:,:] = for_recv


        else :
            #pass in both direction
            for_send_req =comm.Isend(for_send_buf,dest = rank+1, tag = rank+350)
            bac_send_req =comm.Isend(bac_send_buf,dest = rank-1, tag = rank+450)
            #recv
            bac_recv_req =comm.Irecv(bac_recv, source = rank+1 , tag = rank+451)
            for_recv_req =comm.Irecv(for_recv, source = rank-1 , tag = rank+349)

            for_send_req.Wait()
            for_send_req.Wait()
            bac_recv_req.Wait()
            for_recv_req.Wait()
            u[localnx+1,:,:] = bac_recv
            u[0,:,:] = for_recv
        #print("finish MPI in u op",count, " ",t,rank)


    return

def cal_grad_u():
    global u,grad_u
    grad_u[0,1:localnx+1,1:ny+1,1:nz+1] =(- u[0:localnx,1:ny+1,1:nz+1] + u[2:localnx+2,1:ny+1,1:nz+1])/2
    grad_u[1,1:localnx+1,1:ny+1,1:nz+1] =(- u[1:localnx+1,0:ny,1:nz+1] + u[1:localnx+1,2:ny+2,1:nz+1])/2
    grad_u[2,1:localnx+1,1:ny+1,1:nz+1] =(- u[1:localnx+1,1:ny+1,0:nz] + u[1:localnx+1,1:ny+1,2:nz+2])/2
#NOTE the following ones is quite slow
#    for i in range(1,localnx+1):
#        for j in range(1,ny+1):
#            for k in range(1,nz+1):
#                grad_u[i][j][k][0] = u[i][j][k] - u[i-1][j][k]
#                grad_u[i][j][k][1] = u[i][j][k] - u[i][j-1][k]
#                grad_u[i][j][k][2] = u[i][j][k] - u[i][j][k-1]

    return

    


nimg = None
nx, ny , nz =200,200,200
if rank == 0:
    
    start = time.time()
    # init the pic in rank 0
    
    # Generate image
    img = 100.0*np.ones((nx,ny,nz)) 
    img[75:150,75:150,75:150] = 150.0
    #img[15:30,15:30,15:30] = 150.0

    # Adding Gaussian noise
    nmean, nsigma = 0.0, 12.0
    nimg = np.random.normal(nmean,nsigma,(nx,ny,nz)) + img
    #print("shape of nimg",np.shape(nimg))
    
# init local img
localnx = nx//size
#print(localnx)
#TODO the program work only when localnx is an integer
localimg =  np.empty([localnx,ny,nz], dtype=np.float64)
comm.Scatter(nimg   , localimg, root=0)
#print("shape of local nimg in rank ",rank, np.shape(localimg))
#init u,m,mu (local)
#store data in 1:localnx, 1:ny,1:nz. 
#store MPI data in 0,1:ny,1:nz and localnx+1,1:ny,1:nz
if rank== 0 :
    del nimg
    del img
    
u = 100.0* np.ones((localnx+2,ny+2,nz+2)) # rank(zero) = empty
m = 100.0* np.ones((3,localnx+2,ny+2,nz+2)) # first indice 0,1,2 means dx ,dy ,dz
mu= 100.0* np.zeros((3,localnx+2,ny+2,nz+2))
grad_u =  100.0* np.zeros((3,localnx+2,ny+2,nz+2)) # calculate grad_u use cal_grad_u()   

def op_m():# use the shrink 
    #NO MPI
    #Anisotropic
    global m , grad_u ,mu, alpha,lamda
    X = grad_u +( 1.0/alpha) * mu
    Y= lamda/alpha* np.ones((3,localnx+2,ny+2,nz+2))
    #print(np.max(np.abs(X) - Y))
    m = np.multiply(np.sign(X) , np.maximum(np.abs(X) - Y ,0 ))
    #m =  np.maximum(np.abs(X) - Y , np.zeros((3,localnx+2,ny+2,nz+2))) 
    #print(np.max(m))
    return

def op_mu():# NO NEED for grad_u
    #NO MPI
    global m,grad_u,mu,alpha #grad_u has been update in cal_u()
    mu = mu + alpha*(m - grad_u)
    return


count = 0
while count<3:#TODO stability 
    op_u()
    cal_grad_u()
    op_m()
    op_mu()

    count = count+1
#    if rank == 0:
#        plt.subplot(4,4,count+1)
#        plt.imshow(u[80,:,:], cmap=plt.cm.gray)
#        plt.subplot(4,4,count+1+4)
#        plt.imshow(grad_u[1,80,:,:], cmap=plt.cm.gray)
#        plt.subplot(4,4,count+1+8)
#        plt.imshow(m[1,80,:,:], cmap=plt.cm.gray)
#        plt.subplot(4,4,count+1+12)
#        plt.imshow(mu[1,80,:,:], cmap=plt.cm.gray)

#        print("    end of ietration " ,count)

if rank ==0:
    #plt.figure()
    end = time.time()
    print("Execution Time: ", end - start)
#print the result



