import numpy as np
import os
from radmc3dPy import *
import matplotlib.pylab as plt

analyze.writeDefaultParfile('ppdisk') 										#create the parameter file

setup.problemSetupDust('ppdisk', mdisk='1e-5*ms', gap_rin='[10.0*au, 30.0*au, 60.*au]', 
	gap_rout='[15.*au, 45.*au, 90.*au]', gap_drfact='[1e-10, 1e-10, 1e-10]', nz='0') 						# setup the model

data_den = analyze.readData(ddens=True)										# read the input density distribution

#Plot opacities
opac = analyze.readOpac(ext=['silicate_ice_thin_ossenkopf'])									# read the silicate opacity
plt.loglog(opac.wav[0], opac.kabs[0], opac.ksca[0])
plt.xlabel(r'$\lambda$ [$\mu$m]')
plt.ylabel(r'$\kappa_{\rm abs}$ [cm$^2$/g]')
plt.show()


##Plot density##
c = plt.contourf(data_den.grid.x/natconst.au, np.pi/2.-data_den.grid.y, np.log10(data_den.rhodust[:,:,0,0].T), 30)
plt.xlabel('r [AU]')
plt.ylabel(r'$\pi/2-\theta$')
plt.xscale('log')
cb = plt.colorbar(c)														# add colorbar
cb.set_label(r'$\log_{10}{\rho}$', rotation=270., labelpad=25)

#Calculate the optical depth at an specific wavelength
data_den.getTau(wav=0.5)
c_tau = plt.contour(data_den.grid.x/natconst.au, np.pi/2.-data_den.grid.y, data_den.taux[:,:,0].T, [1.0],  colors='w', linestyles='solid')
plt.clabel(c_tau, inline=1, fontsize=10)
plt.show()


#Run MC
import os
os.system('radmc3d mctherm')

#Read and plot Temperature
data_tem = analyze.readData(dtemp=True)
c_tem = plt.contourf(data_tem.grid.x/natconst.au, np.pi/2.-data_tem.grid.y, data_tem.dusttemp[:,:,0,0].T, 30, cmap=plt.cm.jet)
plt.xlabel('r [AU]')
plt.ylabel(r'$\pi/2-\theta$')
plt.xscale('log')
cb = plt.colorbar(c_tem)
cb.set_label('T [K]', rotation=270., labelpad=25)

c = plt.contour(data_tem.grid.x/natconst.au, np.pi/2.-data_tem.grid.y, data_tem.dusttemp[:,:,0,0].T, 10,  colors='k', linestyles='solid')
plt.clabel(c, inline=1, fontsize=10)
plt.show()

#exit()
#Calculate SED and plot
#os.system('radmc3d sed')
os.system('radmc3d spectrum incl 60 phi 0 lambdarange 1 50 nlam 50 setthreads 8')
sed = analyze.readSpectrum('spectrum.out')
analyze.plotSpectrum(sed,xlg=True, ylg=True,nufnu=True,micron=True,dpc=140.)
plt.show()

"""
#Create images using RADMC-3D
image.makeImage(npix=300., wav=1300, incl=60., phi=0., sizeau=300.)
im = image.readImage()

image.plotImage(im, au=True, log=True, cmap=plt.cm.gist_heat)
plt.show()

image.plotImage(im, arcsec=True, dpc=140., log=True, bunit='snu', cmap=plt.cm.gist_heat) #image shown in angular scale
plt.show()


cim = im.imConv(fwhm=[0.035, 0.035], pa=0., dpc=140.) 															   #convoluted image
image.plotImage(cim, arcsec=True, dpc=140., log=True, bunit='snu', cmap=plt.cm.gist_heat)
plt.show()


image.plotImage(cim, arcsec=True, dpc=140., log=True, bunit='snu', cmask_rad=0.17, cmap=plt.cm.gist_heat) # corographaphic mask
plt.show()

dat = [1,10,50]
image.radmc3dImage.getVisibility(im,bl=dat,pa=43,dpc=240)
"""


exit()
