#!/usr/bin/env python
from __future__ import print_function
"""Classes for inverting camera data for filament detection

Example usage:
TODO: Add example usage
"""
import logging
import os
import time
import sys
#from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


    
def read_geom_matrix_pol(fn):
    """Read geometry matrix file for poloidal plans"""
    from scipy.sparse import csc_matrix
    dat = np.load(fn)
    # Make the geometry matrix
    
    exluded = ['values','rows','columns','mat_shape','r_values','tor_values','nx','ny']
    
    shape = dat['mat_shape']
    
    #shape[0] = (dat['nx']+1)*(dat['ny'])
    
    matrix = csc_matrix((dat['values'],(dat['rows'],dat['columns'])) ,shape=shape, dtype=np.float)  
    return matrix,dat['r_values'],dat['z_values'],dat['nx'],dat['ny'],dat['npsi'],dat['npol']

def plot_geom_matrix(mat,nx,ny):
    
    import matplotlib.pyplot as plt
    
    im = mat.dot(np.ones(mat.shape[1])).reshape(nx,ny)
    
    plt.imshow(im)
    plt.show()
    
def twoD_Gaussian(x,y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Simple function to create a 2D Gaussian function
    
    Arguments:
        (x,y) -- 2D meshgrid arrays specifying the two coordinates
        
    """
    
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.T.ravel()

def reproject_image(geo_mat,emissivity,calibfactor=60.0):
    """
    Simple wrapper to reproject an emissivity
    
    Keyword:
        calibfactor -- float -- a calibration factor used to normalize the image intensity to that of a typical fastcam image
    """
    
    return geo_mat.dot(emissivity.flatten())*calibfactor
       
def reproject_gaussian(geo_mat,r,tor,A,r0,t0,dr,dt,ang):
    """
    Create a reprojected image of a single Gaussian emission profile
    """
    rr,tt = np.meshgrid(r,tor)
    
    return reproject_image(geo_mat,twoD_Gaussian((rr,tt),A,r0,t0,dr,dt,ang,0.0))
    
def reproject_multi_gaussian(geo_mat,r,tor,A,r0,t0,dr,dt,ang):
    """
    Create a reprojected image of a set of multiple Gaussians
    
    All arguments here must be iterables of the same length
    """
    
    rr,tt = np.meshgrid(r,tor)
    
    dens = np.zeros(len(r)*len(tor))
    
    for i in np.arange(len(A)):
        
        dens += twoD_Gaussian((rr,tt),A[i],r0[i],t0[i],dr[i],dt[i],ang[i],0.0)
        
    return reproject_image(geo_mat,dens),dens
    
def reproject_random_gaussians(geo_mat,Nfils,r,tor,R_dist=None,tor_dist=None,dR_dist=None,dtor_dist=None,amp_dist=None,suppress_output=True):
    """
    Create a reprojection of a random distribution of filaments
    
    All distribution functions should be callable objects that take only the number of samples as a required input
    
    Keywords:
        R_dist -- distribution function -- Distribution of radial positions
        tor_dist -- distribution function -- Distribution of toroidal positions
        dR_dist -- distribution function -- Distribution of radial widths
        dtor_dist -- distribution function -- Distribution of toroidal widths
        amp_dist -- distribution function -- Distribution of amplitudes
        suppress_output -- bool -- If true only return the image, if False also return all filament properties
    
    """
    if R_dist is None:
        def R_dist(nfils):
            return np.random.uniform(size=nfils)*0.15 + 1.33
    
    if tor_dist is None:
        def tor_dist(nfils):
            return np.random.uniform(size=nfils)*(np.max(tor) - np.min(tor)) + np.min(tor)
        
    
    if dR_dist is None:
        def dR_dist(nfils): 
            return np.random.uniform(size=nfils)*0.025 + 0.005
    
    if dtor_dist is None:
        dtor_dist = dR_dist
    
    if amp_dist is None:
        def amp_dist(nfils):
            return np.ones(nfils)
            
    amps = amp_dist(Nfils)
    Rs = R_dist(Nfils)
    tors = tor_dist(Nfils)
    dRs = dR_dist(Nfils)
    dtors = dtor_dist(Nfils)
       
    img,emissivity = reproject_multi_gaussian(geo_mat,r,tor,amps,Rs,tors,dRs,dtors,np.zeros(Nfils))
    
    if suppress_output:
        return img
    else:
        return img,emissivity,amps,Rs,tors,dRs,dtors
    

def generate_psf_matrix(geom_matrix):
    """
    Generate the psf matrix from a given geometry matrix 
    """
    #res = np.zeros((geom_matrix.shape[1],geom_matrix.shape[1]))
    res = (geom_matrix.T).dot(geom_matrix)
    res = res.toarray()

    return res 

    
def invert_psf_SART(A,b,r_values,tor_values,w=1.0,Niter=100,tol=1e-4,visualise=False,verbose=True,no_neg=True,betaL=1e-8,npsi=None,npol=None):
    """
    Invert using the SART algorithm with Laplacian regularization (see eg https://link.springer.com/article/10.1186/BF03352821)

    """

    #Set initial guess 
    x0 = np.zeros(b.shape) + np.exp(-1)
    #x0 = b
    #nr,nt = len(r_values),len(tor_values)
    

    #Append matrices to regularize
    lpl = gen_lpl_matrix(A,r_values,tor_values)#,npsi=npsi,npol=npol)    
        
    A = np.concatenate([A,betaL*lpl],axis=0)
    b = np.concatenate([b,np.zeros(len(b))])
    
    A_Si = np.sum(np.abs(A),axis=1) 
    A_Sj = np.sum(np.abs(A),axis=0)
    At = np.ascontiguousarray(A.T)
    w_over_A_Sj = w / A_Sj
    one_over_A_Si = 1.0 / A_Si
    if visualise:
        import matplotlib.pyplot as plt
        im = plt.imshow(x0.reshape((nr,nt))/np.max(x0),interpolation='none')
        plt.ion()
        plt.show()
     
    for i in np.arange(Niter):        
        #newx = x0 +  (w/A_Sj)*(A*((1.0/A_Si)*(b - np.dot(A,x0)))[:,np.newaxis]).sum(axis=0)
        newx = x0 + w_over_A_Sj * np.dot(At, one_over_A_Si * (b - np.dot(A, x0)))
        if no_neg: newx[newx < 0] = 0.0 
        
        Lc = np.max(np.abs(newx - x0))/np.max(np.abs(newx))
        L2 = np.sqrt(np.mean((newx - x0)**2.0)/np.mean(newx**2.0))
        
        if verbose:
            print(i)
            print('Lc norm = '+str(np.max(np.abs(newx - x0))/np.max(np.abs(newx))))
            print('L2 norm = '+str(np.sqrt(np.mean((newx - x0)**2.0)/np.mean(newx**2.0))))
        
        x0 = newx
        if np.max([Lc,L2]) < tol:
            break
        
        
        
        
        if visualise:
            im.set_data(x0.reshape((nr,nt))/np.max(x0))
            plt.gcf().canvas.draw()
        
    if visualise: plt.ioff()    
    return x0

 
def gen_lpl_matrix(A,r_values,tor_values): 
    """Generate the Laplacian smoothing matrix to use in regularization"""
    
    #NOTE: This function could probably be made more efficient, however it is not the bottleneck
    # for the inversion routines at the moment. If this changes this should be re-addressed
       
    nr,nt = len(r_values),len(tor_values)
    
    #Construct laplacian smoothing matrix
    lpl = np.zeros(A.shape)
    
    for i in np.arange(A.shape[0]):
        
        #Temporary 2D array to store stencil in to ensure flattening is done properly
        a2 = np.zeros((nr,nt))
        jj = i % nt
        ii = int(i/nt)
         
        
        #The normal stencil for points inside the domain boundary is 
        # -1 -1 -1
        # -1  8 -1
        # -1 -1 -1
        
        #For points in a boundary we have 
        
        # ---------
        # -1  5 -1
        # -1 -1 -1
        
        #And for points at a corner
        
        # |--------
        # |  3 -1
        # | -1 -1
        
        if ii == 0:
            if jj == 0:
                a2[ii,jj] = 3
                a2[ii+1,jj] = -1
                a2[ii,jj+1] = -1
                a2[ii+1,jj+1] = -1
         
            elif jj == nt - 1:
                a2[ii,jj] = 3
                a2[ii+1,jj] = -1
                a2[ii,jj-1] = -1
                a2[ii+1,jj-1] = -1
                
            else:
                a2[ii,jj] = 5
                a2[ii+1,jj] = -1
                a2[ii,jj+1] = -1
                a2[ii+1,jj+1] = -1
                a2[ii,jj-1] = -1
                a2[ii+1,jj-1] = -1
                
        elif ii == nr - 1:
            if jj == 0:
                a2[ii,jj] = 3
                a2[ii-1,jj] = -1
                a2[ii,jj+1] = -1
                a2[ii-1,jj+1] = -1
            
            elif jj == nt - 1:
                a2[ii,jj] = 3
                a2[ii-1,jj] = -1
                a2[ii,jj-1] = -1
                a2[ii-1,jj-1] = -1
                    
            else:     
                a2[ii,jj] = 5
                a2[ii-1,jj] = -1
                a2[ii,jj+1] = -1
                a2[ii-1,jj+1] = -1
                a2[ii,jj-1] = -1
                a2[ii-1,jj-1] = -1
        else:
            if jj == 0:
                a2[ii,jj] = 5
                a2[ii-1,jj] = -1
                a2[ii,jj+1] = -1
                a2[ii-1,jj+1] = -1
                a2[ii+1,jj] = -1
                a2[ii+1,jj+1] = -1
            
            elif jj == nt - 1:
                a2[ii,jj] = 5
                a2[ii-1,jj] = -1
                a2[ii,jj-1] = -1
                a2[ii-1,jj-1] = -1
                a2[ii+1,jj] = -1
                a2[ii+1,jj-1] = -1
                    
            else:     
                a2[ii,jj] = 8
                a2[ii-1,jj] = -1
                a2[ii+1,jj] = -1
                a2[ii-1,jj+1] = -1
                a2[ii+1,jj+1] = -1
                a2[ii-1,jj-1] = -1
                a2[ii+1,jj-1] = -1
                a2[ii,jj+1] = -1
                a2[ii,jj-1] = -1
        
        
        lpl[:,i] = a2.flatten()
           
    return lpl
        
def generate_poloidal_RZ_points(gfile,bounds,npsi=100,npol=200,direction='inboard',ds=1e-2):
    
    from scipy.interpolate import interp1d
    try:
        from cyfieldlineTracer import RK4Tracer
    except:
        try:
            from pyFieldlineTracer.fieldlineTracer import RK4Tracer
            print("No cyFieldlineTracer module found, reverting to python version")
        except:
            Error_text = "Cannot find cyFieldlineTracer or pyFieldlineTracer, unable to compute fieldline_array. Exiting. \n"
            Error_text += "For cyFieldlineTracer please visit https://git.ccfe.ac.uk/SOL_Transport/cyFieldlineTracer \n"
            Error_text += "Alternatively, for the slower pyFieldlineTracer please visit https://git.ccfe.ac.uk/SOL_Transport/pyFieldlineTracer."
            raise ImportError(Error_text)
            
    tracer = RK4Tracer(gfile=gfile)
    
    #bounds should be a 2 element list of coordinates defining the bounding line eg. [(r0,z0),(r1,z1)]
    
    #First sample npsi points along the bounding line
    
    r0,z0 = bounds[0]
    r1,z1 = bounds[1]
    
    #Get coefficients of the line
    m = (z1 - z0)/(r1 - r0)
    c = z0 - m*r0
    
    Rinit = np.linspace(r0,r1,npsi)
    Zinit = np.linspace(z0,z1,npsi)
    
    if direction is not 'inboard':
        ds *= -1

    coordR = np.zeros((npsi,npol))
    coordZ = np.zeros((npsi,npol))
    #psi = np.linspace(psimin,psimax,npsi)
    import matplotlib.pyplot as plt
    
    for i in np.arange(npsi):
        print(i)
        #Should be organised so that the desired target (inboard or outboard) is at the 0th index
        fline = tracer.trace(Rinit[i],Zinit[i],0.0,5e5,ds,verbose=False)
        #fline = fline.filter(['Z','R'],[[fline.Z[0],Zinit[i]],[fline.R[0],Rinit[i]]])
        fline = fline.filter(['R'],[[np.min([fline.R[0],Rinit[i]]),np.max([fline.R[0],Rinit[i]])]])
        if np.abs(fline.Z[-1] - Zinit[i]) > ds:
            fline = fline.filter(['Z'],[[fline.Z[0],Zinit[i]]])
        #plt.plot(fline.R,fline.Z,lw=0.5)
        
        dL = np.zeros(fline.R.shape)
        dL[1:] = ((fline.R[1:] - fline.R[0:-1])**2.0 + (fline.Z[1:] - fline.Z[0:-1])**2.0)**0.5
        L = np.cumsum(dL)
        
        Rfunc = interp1d(L,fline.R)
        Zfunc = interp1d(L,fline.Z)
        
        newL = np.linspace(L.min(),L.max(),npol)
        
        coordR[i] = Rfunc(newL)
        coordZ[i] = Zfunc(newL)
             
    return np.array([coordR,coordZ]) 
    
def generate_geom_matrix_pol_plane(fn,gfile, calib,bounds,npsi,npol,fl_tor_lim,phi_offset=0.0,raylengths=None,ds=5e-4,ROI=[(0,None),(0,None)],neutrals_dist=None,direction='inboard',xlim=None,ylim=None):
    
    import time
    
    try:
        from cyfieldlineTracer import RK4Tracer
    except:
        try:
            from pyFieldlineTracer.fieldlineTracer import RK4Tracer
            print("No cyFieldlineTracer module found, reverting to python version")
        except:
            Error_text = "Cannot find cyFieldlineTracer or pyFieldlineTracer, unable to compute fieldline_array. Exiting. \n"
            Error_text += "For cyFieldlineTracer please visit https://git.ccfe.ac.uk/SOL_Transport/cyFieldlineTracer \n"
            Error_text += "Alternatively, for the slower pyFieldlineTracer please visit https://git.ccfe.ac.uk/SOL_Transport/pyFieldlineTracer."
            raise ImportError(Error_text)
    
    print("---Generating poloidal geometry matrix---")
    
    tracer = RK4Tracer(gfile=gfile)
    coords = generate_poloidal_RZ_points(gfile,bounds,npsi,npol, direction)
    
    mat_rows = []#np.empty(0,dtype=np.uint16)
    mat_cols = []#np.empty(0,dtype=np.uint32)
    mat_vals = []#np.empty(0,dtype=np.float16)
    
    shape = calib.get_image().shape
    
    nx = shape[0]
    ny = shape[1]
    Ppos = calib.get_pupilpos()
    
    xmin = ROI[0][0]
    xmax = ROI[0][1]
    ymin = ROI[1][0]
    ymax = ROI[1][1]
    
    Rs = coords[0].flatten()
    Zs = coords[1].flatten()
    t0 = time.time()  
    if direction is 'outboard':
        ds = -ds
    for i in np.arange(Rs.shape[0]):
        #print(Rs[i],Zs[i])
        tin = time.time()
        
        fline = tracer.trace(Rs[i],Zs[i],phi_offset,5e5,ds,verbose=False,tor_lim=fl_tor_lim)
        fline = fline.filter('phi',[phi_offset-fl_tor_lim,phi_offset+fl_tor_lim])
        phi0 = phi_offset 
        
        X = fline.R*np.cos(fline.phi)#+ phi0)
        Y = fline.R*np.sin(fline.phi)#+ phi0)
        Z = fline.Z
        
        r2 = (X - Ppos[0])*(X - Ppos[0]) + (Y - Ppos[1])*(Y - Ppos[1]) + (Z - Ppos[2])*(Z - Ppos[2])
        
        if neutrals_dist is not None:
            scale_factors = neutrals_dist(fline.R,fline.Z)
        else:
            scale_factors = np.ones(fline.R.shape)
        lines = calib.project_points(np.array([X.flatten(),Y.flatten(),Z.flatten()]).T)[0]
        #print(np.sum(lines))
        del X,Y,Z
        lines_x = lines[:,1].astype(np.float)
        lines_y = lines[:,0].astype(np.float)
        del lines
        
        lines_x[lines_x < 0] = np.nan
        lines_x[lines_x > nx-1] = np.nan
        lines_x[lines_y < 0] = np.nan
        lines_x[lines_y > ny-1] = np.nan
        
        nan_filter = np.logical_and(np.isfinite(lines_x),np.isfinite(lines_y)) 
        lines_x = lines_x[nan_filter].astype(np.int)
        lines_y = lines_y[nan_filter].astype(np.int)
        scale_factors = scale_factors[nan_filter]
        #scale_factor = scale_factors[nan_filter]
        img = np.zeros((nx,ny))
        
        r2 = r2[nan_filter]
        if raylengths is not None:
                
            r2_filter = np.less(np.sqrt(r2),raylengths[[lines_x,lines_y]])
                
            r2 = r2[r2_filter]
            lines_x = lines_x[r2_filter]
            lines_y = lines_y[r2_filter]
            scale_factors = scale_factors[r2_filter]
            
        np.add.at(img,[lines_x,lines_y],scale_factors*np.abs(ds)/r2)
        del lines_x,lines_y
        
        
        
        
        
        img = img[xmin:,ymin:]
        
        if xmax is not None:
            img = img[:xmax]
        if ymax is not None:
            img = img[:,:ymax]
        
        #import matplotlib.pyplot as plt
        #
        #plt.imshow(img)
        #plt.show()
        shape_im = img.shape
        
        nx_im = shape_im[0]
        ny_im = shape_im[1]
        
        #img = img[:121,65:]
          
        img = img.flatten()
        
        
        
        #Figure out which values need to be saved
        indx = np.where(img > 0)
        #Set only non-zero values of the sparse matrix
        nix = indx[0].shape[0]
        #mat_cols = np.append(mat_cols, np.ones(nix,dtype=np.uint32)*(np.uint32(j)+np.uint32(i*ntor)))
        #mat_rows = np.append(mat_rows,indx[0])
        #mat_vals = np.append(mat_vals, img[indx].astype(np.float16))
            
        mat_cols.append(np.ones(nix,dtype=np.uint32)*(np.uint32(i)))
        mat_rows.append(indx[0])
        mat_vals.append(img[indx].astype(np.float16))
        
        print('\r',end='')
        sys.stdout.flush()
        tout = time.time()
        print('Status:\t {:03.3f} %  \t Estimated TOA {}'.format(100.0*float(i+1)/float(Rs.shape[0]),time.ctime(t0 + ((tout - t0)*(float(Rs.shape[0])/float(i+1))))),end='')
           
 
    indx = None
    img = None
        
    t1 = time.time()
    print("\nTime taken in image generation: "+str(t1-t0))
    dt2 = t1-t0
        
    #print("Trace/generation time: "+str(dt1/dt2))
    #print("Estimated TOA: "+str(time.ctime(time.time() + (nr-i-1)*(dt1+dt2))))
          
    mat_shape = (nx_im*ny_im,npsi*npol)
    
    mat_cols = np.concatenate(mat_cols)
    mat_rows = np.concatenate(mat_rows)
    mat_vals = np.concatenate(mat_vals)
    
    # Write matrix to file fn
    np.savez_compressed(fn,rows=mat_rows,
                        columns=mat_cols,values=mat_vals,
                        mat_shape=mat_shape,nx=nx_im,ny=ny_im,npsi=npsi,npol=npol,r_values=Rs,z_values=Zs,gfile=gfile,calibration=calib,
                        fl_tor_lim=fl_tor_lim,phi_offset=phi_offset,bounds=bounds,ROI=ROI)

    del mat_rows
    del mat_cols
    del mat_vals
    
    return read_geom_matrix_pol(fn)
        
        
        