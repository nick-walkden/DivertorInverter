from inverter import generate_poloidal_RZ_points,generate_geom_matrix_pol_plane,read_geom_matrix_pol
from equilibrium import equilibrium as eq
import matplotlib.pyplot as plt
import numpy as np
import calcam

#Specify the limiting surface for generatio of the poloidal grid
line_inner = [(0.515,-1.20),(0.58,-1.39)]

#Specify number of points in grid
npsi = 15
npol = 20

#Limit of field-line extent around the machine
fl_limit = np.pi

#Equilibrium gfile to use
gfile = 'gfiles/29724_0.35.g'

#Load the calcam calibration file
cal = calcam.Calibration('/Users/nwalkden/Documents/Calcam 2/Calibrations/29724.ccc')

#Use these files to generate the geometry matrix and save to disk
mat,r,z,nx,ny,npsi,npol=generate_geom_matrix_pol_plane('29724_inner_GeoMat.npz',gfile,cal,line_inner,npsi,npol,fl_limit)