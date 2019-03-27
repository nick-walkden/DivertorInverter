from inverter import read_geom_matrix_pol,generate_psf_matrix,invert_psf_SART
from read_movie import read_movie
from background_subtractor import *
import matplotlib.pyplot as plt
import numpy as np
from equilibrium import equilibrium as eq 

#Movie reading parameters
moviefile = 'Movies/29724/C001H001S0001/C001H001S0001-04.mraw'
Nframes = 40
starttime = 0.35
transforms = ['transpose','reverse_x']
gfile = 'gfiles/29724_0.35.g'
#Movie enhancement paramters
NBGsub = 20


#Inversion routine parameters
betaL = 0.0001
w = 1.1
Niter = 200000
tol = 1e-5



#Read in the movie frames and subtract background
frames = read_movie(moviefile,Nframes=Nframes,starttime=starttime,transforms=transforms)
frames = run_bgsub_min(frames[:],NBGsub)  


#Load in the geometry matrix and generate the PSF matrix
gmatfile = '29724_GeoMat.npz'
gmat_in,r_in,z_in,nx,ny,npsi_in,npol_in = read_geom_matrix_pol(gmatfile)

psf_in = generate_psf_matrix(gmat_in)


#Load in the equilibrium for plotting purposes
E=eq(gfile=gfile)

i = 0
#Loop over each frame, perform inversion and save image and data
for frame in frames[NBGsub:]:

    plt.subplot(131)
    plt.imshow(frame,cmap='gray')
    
    plt.subplot(132)
    
    #Perform fieldline convolution
    frame_sub_conv_in = (gmat_in.toarray()*frame.flatten()[:,np.newaxis]).sum(axis=0)

    #Perform inversion
    inv_in = invert_psf_SART(psf_in,frame_sub_conv_in,np.ones(npsi_in),np.ones(npol_in),w=w,betaL=betaL,Niter=Niter,tol=tol)
  
    plt.contour(E.R,E.Z,E.psiN[:],300,colors='c',linewidths=0.5,alpha=0.5)
    plt.contour(E.R,E.Z,E.psiN[:],[1.0],colors='c',linewidths=2.0,alpha=0.75)
    plt.contourf(r_in.reshape(npsi_in,npol_in),z_in.reshape(npsi_in,npol_in),inv_in.reshape(npsi_in,npol_in),np.linspace(0,2000,400),cmap='afmhot')
    
    #plt.contour(E.R,E.Z,E.psiN[:],100,colors='w',linewidths=0.3,alpha=0.5)
    plt.xlim(r_in.min(),r_in.max())
    plt.ylim(z_in.min(),z_in.max())
    
    plt.gca().set_aspect('equal')
    
    plt.subplot(133)
    
    plt.imshow(gmat_in.dot(inv_in).reshape(nx,ny),cmap='gray')
    plt.savefig('images/29724_inner_'+str(i)+'.png',bbox_inches='tight')
    plt.clf()
    #plt.show()
    np.savez('data/29724_inner_'+str(i),inv_in=inv_in,r_in=r_in,z_in=z_in,npsi_in=npsi_in,npol_in=npol_in,frame=frame,moviefile=moviefile,Nframes=Nframes,starttime=starttime,NBGsub=NBGsub,betaL=betaL,w=w,Niter=Niter,tol=tol,gmatfile=gmatfile)
    i += 1
    
