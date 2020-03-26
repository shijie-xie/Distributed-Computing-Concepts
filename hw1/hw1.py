# Scatter.py

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#set the constant
lamda = 10.0
alpha = 10.0


# optimization of u
# solve ( 1 - \alpha \Delta) u = ( nimg - \nabla \cdot (\mu + \alpha m))
def op_u():
    global comm, size,rank, alpha, mu,m,u,count,localimg
    global localnx,ny,nz
    #X = (\mu + \alpha m)
    X = mu + alpha * m
    #Y = localnimg - \nabla \cdot (\mu + \alpha m)
    Y =  np.zeros((localnx+2,ny+2,nz+2))
    # MPI of Y
    Xf = X[localnx+1,:,:,0]
    for_send_buf =np.copy(Xf)
    #print(for_send_buf.shape)
    for_recv =np.empty([ny+2,nz+2], dtype=np.float64)
    

    #TODO only for is in need
    if rank==0 :
        #pass forward
        for_send_req =comm.Isend(for_send_buf, dest = rank+1, tag = rank+50)
        for_send_req.Wait()
    elif rank == size-1:
        #recv 
        for_recv_req = comm.Irecv(for_recv, source = rank-1 , tag = rank+49)
        for_recv_req.Wait()
        X[0,:,:,0] = for_recv
        #save recv to local part


    else :
        #pass in both direction
        for_send_req =comm.Isend(for_send_buf,dest = rank+1, tag = rank+50)
        #recv
        for_recv_req =comm.Irecv(for_recv, source = rank-1 , tag = rank+49)

        for_send_req.Wait()       
        for_recv_req.Wait()
        X[0,:,:,0] = for_recv
    # local part of Y
    for i in range(1,localnx+1):
        for j in range(1,ny+1):
            for k in range(1,nz+1):
                #NOTE: localimg & Y has diffrent indices
                Y[i][j][k] = localimg[i-1][j-1][k-1]  -  ( X[i][j][k][0]-X[i-1][j][k][0] )-( X[i][j][k][1]-X[i][j-1][k][1] ) - ( X[i][j][k][2]-X[i-1][j][k-1][2] )
               

    #print("in iteration ",count,"rank",rank, "finish assemble the matrix" )





  
    # solve ( 1 - \alpha \Delta) u = Y
    # solve ( 1 + 6 \alpha) u[i][j][k] = Y[i][j][k] - \alpha*( u[i \pm 1][j \pm 1][k \pm 1] )
    for t in range(0,20):#TODO 100 is a parameter here to op
        for i in range(1,localnx+1 ):
            for j in range(1,ny+1):
                for k in range(1,nz+1):
                    u[i][j][k] =( Y[i][j][k] + alpha/4.0*(u[i][j][k+1]+u[i][j][k-1]+u[i][j+1][k]+u[i][j-1][k]+u[i-1][j][k]+u[i+1][j][k]))/(1+6.0/4.0*alpha)

        #MPI part
        for_send_buf =np.copy(u[localnx+1,:,:])
        bac_send_buf =np.copy(u[0,:,:])
        for_recv =np.empty([1,ny+2,nz+2], dtype=np.float64)
        bac_recv =np.empty([1,ny+2,nz+2], dtype=np.float64)
        #print(count, " ",t,rank)
        if rank==0 :
            #pass forward
            for_send_req =comm.Isend(for_send_buf, dest = rank+1, tag = rank+350)
            #recv 
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
    for i in range(1,localnx+1):
        for j in range(1,ny+1):
            for k in range(1,nz+1):
                grad_u[i][j][k][0] = u[i][j][k] - u[i-1][j][k]
                grad_u[i][j][k][1] = u[i][j][k] - u[i][j-1][k]
                grad_u[i][j][k][2] = u[i][j][k] - u[i][j][k-1]

    return

    


nimg = None
nx, ny , nz =40,40,40
if rank == 0:
    # init the pic in rank 0
    
    # Generate image
    img = 100.0*np.ones((nx,ny,nz)) 
    #img[75:150,75:150,75:150] = 150.0
    img[15:30,15:30,15:30] = 150.0

    # Adding Gaussian noise
    nmean, nsigma = 0.0, 12.0
    nimg = np.random.normal(nmean,nsigma,(nx,ny,nz)) + img
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(nimg[18,:,:], cmap=plt.cm.gray)


 
# init local img
localnx = nx//size
#TODO the program work only when localnx is an integer
localimg =  np.empty([localnx,ny,nz], dtype=np.float64)
comm.Scatter(nimg   , localimg, root=0)

#init u,m,mu (local)
u = 100.0* np.ones((localnx+2,ny+2,nz+2)) # rank(zero) = empty
m = 100.0* np.ones((localnx+2,ny+2,nz+2,3)) # first indice 0,1,2 means dx ,dy ,dz
mu= 100.0* np.ones((localnx+2,ny+2,nz+2,3))
grad_u =  100.0* np.zeros((localnx+2,ny+2,nz+2,3)) # calculate grad_u use cal_grad_u()   

def op_m():# use the shrink 
    #NO MPI
    #Anisotropic
    global m , grad_u ,mu, alpha,lamda
    X = grad_u + 1.0/alpha * mu
    Y= lamda/alpha* np.ones((localnx+2,ny+2,nz+2,3))
    m = np.multiply(np.sign(X) , np.maximum(np.abs(X) - Y , np.zeros((localnx+2,ny+2,nz+2,3)) ))
    return

def op_mu():# NO NEED for grad_u
    #NO MPI
    global m,grad_u,mu,alpha #grad_u has been update in cal_u()
    mu = mu + alpha*(m - grad_u)
    return


count = 0
while count< 3:#TODO set the right condition
    op_u()
    print("finish op of u, in rank",rank)
    cal_grad_u()
    print("finish calculate grad u, in rank",rank)
    op_m()
    print("finish op of m , in rank",rank)
    op_mu()
    print("finish op of \mu, in rank",rank)
    count = count+1
    print("    end of ietration " ,count)

if rank ==0:
    #plt.figure()
    plt.subplot(1,4,4)
    plt.imshow(u[18,:,:], cmap=plt.cm.gray)
    plt.show()


#print the result



