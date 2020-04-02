# hw1
李昊臻，夏君毅，解士杰

## 数值结果

* MPI边界两侧的结果，子图2,3
* y横切面
* z横切面，靠近方块边界-5个格点

## ![Figure_1](/Users/xieshijie/Desktop/Distributed-Computing-Concepts.git/hw1/Figure_1.png)u subproblem

$$
(I - \alpha \Delta)u = f - \nabla \cdot (\mu + \alpha m)
$$

* 使用Jacobi迭代法求解，Jabobi迭代法的并行不需要随时更新数据。可以直接矩阵的形式进行迭代。

```python
u[1:localnx+1,1:ny+1,1:nz+1] =(1.0/(1+6*alpha))*( Y[1:localnx+1,1:ny+1,1:nz+1]+ alpha*(u[0:localnx,1:ny+1,1:nz+1]+u[2:localnx+2,1:ny+1,1:nz+1]+u[1:localnx+1,0:ny,1:nz+1]+u[1:localnx+1,2:ny+2,1:nz+1]+u[1:localnx+1,1:ny+1,0:nz]+u[1:localnx+1,1:ny+1,2:nz+2]))
```



* 使用Isend和Irecv进行MPI通信，不需要考虑锁死。
* 矩阵运算比for循环快

```python

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
if rank==0 :
    for_send_req =comm.Isend(for_send_buf, dest = rank+1, tag = rank+350)
    
    bac_recv_req =comm.Irecv(bac_recv, source = rank+1 , tag = rank+451)
    bac_recv_req.Wait()
    for_send_req.Wait()
    X[0,localnx+1,:,:] = bac_recv
  
elif rank == size-1:
    for_recv_req = comm.Irecv(for_recv, source = rank-1 , tag = rank+349)
    
    bac_send_req =comm.Isend(bac_send_buf, dest = rank-1, tag = rank+450)
    bac_send_req.Wait()
    for_recv_req.Wait()
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
```
## 计算$\nabla u$

因为$m,\mu$的计算都需要$\nabla u$所以我们直接算好$\nabla u$这样之后的两个subproblem就不需要MPI通讯了.

* 我们这里也没有MPI,是因为我们在Jacobi迭代完成后也完成了MPI通讯
* 写成矩阵计算速度更快

```python
def cal_grad_u():
    global u,grad_u
    grad_u[0,1:localnx+1,1:ny+1,1:nz+1] =(- u[0:localnx,1:ny+1,1:nz+1] + u[2:localnx+2,1:ny+1,1:nz+1])/2
    grad_u[1,1:localnx+1,1:ny+1,1:nz+1] =(- u[1:localnx+1,0:ny,1:nz+1] + u[1:localnx+1,2:ny+2,1:nz+1])/2
    grad_u[2,1:localnx+1,1:ny+1,1:nz+1] =(- u[1:localnx+1,1:ny+1,0:nz] + u[1:localnx+1,1:ny+1,2:nz+2])/2
#NOTE the following one is quite slow
#    for i in range(1,localnx+1):
#        for j in range(1,ny+1):
#            for k in range(1,nz+1):
#                grad_u[i][j][k][0] = u[i][j][k] - u[i-1][j][k]
#                grad_u[i][j][k][1] = u[i][j][k] - u[i][j-1][k]
#                grad_u[i][j][k][2] = u[i][j][k] - u[i][j][k-1]

```



## m subproblem

$$
m_{x_i} = \text{shrink} ( \nabla_{x_i}u+\frac{1}{\alpha} \mu_{x_i} , \frac{\lambda}{\alpha})\\
\text{shrink}(x,\gamma) = \frac{x}{\vert x\vert}\max \{ x - \gamma , 0\}
$$



```python
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

```

### 关于$\alpha$和$\lambda$的取值讨论

* $\lambda$选用10，和hw0相同
* $\alpha$的选取基于如下的考虑：希望边界处的$\text{shirnk}(x, \gamma )$取$\frac{x}{\vert x\vert}(x - \gamma)$,其他地方的取0。这样可以降噪。
  * 而且$\alpha$是和差分算子中的h关联的，即$\nabla u = \frac{u^{i+1}- u^{i-1}}{h}$  
  * 测试了一下shrink中x的值，边界处大约在0.01的量级。取$\alpha$=1000

## $\mu$ subproblem

```python
def op_mu():# NO NEED for grad_u
    #NO MPI
    global m,grad_u,mu,alpha #grad_u has been update in cal_u()
    mu = mu + alpha*(m - grad_u)
    return

```



## 运行时间和线程数的关系

在笔记本上运行时时间大约为6.778627157211304

在天河2号上进行计算，使用$400^3$的规模，使用2，4，8，10个线程。得到的结果比较奇怪。

下一步还要再看。

![9531585801426_.pic](/Users/xieshijie/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/0e5e4e81b22452c948b7a5fcd291d2d2/Message/MessageTemp/e35ba67ea281252ff69b107da4786394/Image/9531585801426_.pic.jpg)

## 内存使用

内存占用较大，

![image-20200402131124480](/Users/xieshijie/Library/Application Support/typora-user-images/image-20200402131124480.png)