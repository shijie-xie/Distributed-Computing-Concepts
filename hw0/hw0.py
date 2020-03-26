import numpy as np
import matplotlib.pyplot as plt
nx, ny = 200, 200
# Generate image
img = 100.0*np.ones((nx,ny))
img[75:150,75:150] = 150.0
# Adding Gaussian noise
nmean, nsigma = 0.0, 12.0
nimg = np.random.normal(nmean,nsigma,(nx,ny)) + img#nimg = noisy img
plt.figure()
plt.subplot(1,4,1)
plt.imshow(nimg, cmap=plt.cm.gray)


#parameter
eps1=0.001
lam1=1e3
step1=1


#----first one--------

grad=np.zeros((nx,ny))
auxm = grad #store sqrt( (u[i][j] - u[i+1][j])**2 + (u[i][j] - u[i][j+1])**2 + eps1)

sum1=0 #L2 part + TV part(\epsilon)
sumold=2e10
sumnew=1e10
sum12=0
step=0
timg=100.0*np.ones((nx+1,ny+1))
for i in range(nx):
    for j in range(ny):
        timg[i][j]=nimg[i][j]
temp=0

#########################    L1 grad norm  ############################## 
while (step<0):
#while ((sumold/sumnew-1)>0.001):
    sum1=0
    sum2=0
    sumh=0
    step+=1
    resimg=timg[0:nx,0:ny]-nimg
    for i in resimg:
        for j in i:
            sum1+=j*j
    for i in range(200):
        for j in range(200):
           auxm[i][j] = np.sqrt( (timg[i][j] - timg[i+1][j])**2 + (timg[i][j] - timg[i][j+1])**2 + eps1)
           sum1+=lam1*auxm[i][j]
    for i in range(200):
        for j in range(200):
           grad[i][j] = 2*( timg[i][j]-nimg[i][j]) + lam1/auxm[i][j]*(2*timg[i][j] - timg[i+1][j]- timg[i][j+1])
           if i > 1:
               grad[i][j] -= lam1/auxm[i-1][j]*(timg[i-1][j]-timg[i][j]) 
           if j > 1:
               grad[i][j] -= lam1/auxm[i][j-1]*(timg[i][j-1]-timg[i][j])
    grad= grad/np.max(grad)            
##################################################   
    for i in range(200):
        for j in range(200):
            timg[i][j]-=step1*grad[i][j]*(1- step*0.004)
    sumold = sumnew
    sumnew = sum1
    print("step=",step,"sum=",sumnew,'\n')
 

#print(grad)
plt.subplot(1,4,2)
plt.imshow(timg, cmap=plt.cm.gray)

#-------------------third one------



grad=np.zeros((nx,ny))
auxm = grad #store sqrt( (u[i][j] - u[i+1][j])**2 + (u[i][j] - u[i][j+1])**2 + eps1)
sum1=0 #L2 part + TV part(\epsilon)
sumold=2e10
sumnew=1e10
sum12=0
step=0
timg=100.0*np.ones((nx+1,ny+1))
for i in range(nx):
    for j in range(ny):
        timg[i][j]=nimg[i][j]
temp=0

#########################    L1 grad norm  ############################## 
while (step<0):
#while ((sumold/sumnew-1)>0.001):
    sum1=0
    sum2=0
    sumh=0
    step+=1
    resimg=timg[0:nx,0:ny]-nimg
    for i in resimg:
        for j in i:
            sum1+=j*j
    for i in range(200):
        for j in range(200):
           auxm[i][j] =  (timg[i][j] - timg[i+1][j])**2 + (timg[i][j] - timg[i][j+1])**2 
           sum1+=lam1*auxm[i][j]
    for i in range(200):
        for j in range(200):
           grad[i][j] = 2*( timg[i][j]-nimg[i][j]) + lam1*2*(2*timg[i][j] - timg[i+1][j]- timg[i][j+1])
           if i > 1:
               grad[i][j] -= lam1*2*(timg[i-1][j]-timg[i][j])
           if j > 1:
               grad[i][j] -= lam1*2*(timg[i][j-1]-timg[i][j])
    grad= grad/np.max(grad)            
##################################################   
    for i in range(200):
        for j in range(200):
            timg[i][j]-=step1*grad[i][j]*(1- step*0.004)
    sumold = sumnew
    sumnew = sum1
    print("step=",step,"sum=",sumnew,'\n')
 

#print(grad)
plt.subplot(1,4,4)
plt.imshow(timg, cmap=plt.cm.gray)







##############################

for i in range(nx):
    for j in range(ny):
        timg=nimg/(1+lam1)
plt.subplot(1,4,3)
plt.imshow(timg, cmap=plt.cm.gray)

plt.show()

