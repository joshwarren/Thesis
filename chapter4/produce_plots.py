## ==================================================================
## 		Plot the outputted dynamics maps
## ==================================================================
## warrenj 20150330 Process to plot the results of pPXF and GANDALF 
## routines.
## warrenj 20150727 Changing to a python script
## warrenj 20150917 Altered to plot and save all 8 plots.
## warrenj 20151216 Added section to plot residuals.
## warrenj 20160111 Add section to plot histgrams of the fields.
## warrenj 20160405 Added keyword CO to overlay CO maps from ALMA if avaible.
## This supersedes plot_results_CO.py
## warrenj 20160726 Now plots in a more object orientated way and creates a
## grid of plots too. This supersedes plot_results2.py

## *************************** KEYWORDS ************************* ##
# galaxy 		Name of the galaxy being plotted: used to find 
#				correct files and to print onto the plot.
# discard	0	Interger giving the number of rows and columns 
#				to be removed from the plot to remove edge 
#				effects.
# wav_range 	null	Imposed wavelength range on top of the automated 
#				limits.	
# vLimit 	2 	Integer giving the number of lowest and highest 
#				results in the plot to be discarded. Defualt 
#				ignores 2 highest and 2 lowest bins.
# norm		"lwv"	Normalisation methods for velocity fields:
#				lwv: luminosity weighted mean of the whole 
#				field is set to 0.
#				lum: velocity of the brightest spaxel is set 
#				to 0.
#				sig: Noralised to the mean velocity of 5 bins with the
#				highest LOSVD.
# plots 	False   Boolean to show plots as routine runs.
# nointerp 	False 	Boolean to use interpolation between bins in 
#				plots or not.
# residual 	False	Method to measure the residuals:
#			mean: use the mean of the residuals in each 
#				bin.
#			median: use the median of the residuals in 
#				each bin.
#			max: use the maximum of the residuals in 
#				each bin.
#			False: do not calculate and produce plot of 
#				residuals.
# CO	   False	Boolean to show ALMA CO plots overlaied (if they exist)
# D 		None Option to pass in the Data object instead of loading it.
## ************************************************************** ##

import numpy as np # for array handling
import glob # for searching for files
from astropy.io import fits # reads fits files (is from astropy)
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt # used for plotting
from plot_velfield_nointerp import plot_velfield_nointerp # for plotting with no interpolations. 
from plot_histogram import plot_histogram
import os
from sauron_colormap2 import sauron2 as sauron
import cPickle as pickle
from plot_results import set_lims, add_
from checkcomp import checkcomp
cc = checkcomp()


#-----------------------------------------------------------------------------
def plot(galaxy, norm="fit_disk", overplot={}, D=None, instrument='vimos'):	
	opt='kin'

	vin_dir = '%s/Data/%s/analysis' % (cc.base_dir, instrument)
	out_dir = '%s/Documents/thesis/chapter4/%s' % (cc.home_dir, 
		instrument)

	data_file =  "%s/galaxies.txt" % (vin_dir)
	# different data types need to be read separetly
	


	if instrument=='vimos':
		file_headings = np.loadtxt(data_file, dtype=str)[0]
		col = np.where(file_headings=='SN_%s' % (opt))[0][0]

		z_gals, SN_target_gals= np.loadtxt(data_file, unpack=True, 
			skiprows=1, usecols=(1,6))
		galaxy_gals = np.loadtxt(data_file, skiprows=1, usecols=(0,),
			dtype=str)
		i_gal = np.where(galaxy_gals==galaxy)[0][0]
		z = z_gals[i_gal]
		SN_target=SN_target_gals[i_gal]

	elif instrument=='muse':
		file_headings = np.loadtxt(data_file, dtype=str)[0]
		col = np.where(file_headings=='SN_%s' % (opt))[0][0]
	
		x_cent_gals, y_cent_gals, SN_target_gals = np.loadtxt(data_file, 
			unpack=True, skiprows=1, usecols=(1,2,col), 
			dtype='int,int,float')
		galaxy_gals = np.loadtxt(data_file, skiprows=1, usecols=(0,),
			dtype=str)
		i_gal = np.where(galaxy_gals==galaxy)[0][0]
		SN_target=SN_target_gals[i_gal]-10
		center = (x_cent_gals[i_gal], y_cent_gals[i_gal])

		data_file =  "%s/Data/vimos/analysis/galaxies.txt" % (cc.base_dir)
		z_gals = np.loadtxt(data_file, unpack=True, skiprows=1, 
			usecols=(1))
		galaxy_gals = np.loadtxt(data_file, skiprows=1, usecols=(0,),
			dtype=str)
		i_gal = np.where(galaxy_gals==galaxy)[0][0]
		z = z_gals[i_gal]

	cubeFile = fits.open(get_dataCubeDirectory(galaxy))
	header = cubeFile[0].header
	cubeFile.close()
# ------------== Reading pickle file and create plot  ===----------

	# Load pickle file from pickler.py
	if D is None:
		pickleFile = open("%s/%s/%s/pickled/dataObj.pkl" % (vin_dir, 
			galaxy, opt), 'rb')
		D = pickle.load(pickleFile)
		pickleFile.close()
	
	if D.norm_method != norm:
		D.norm_method = norm
		D.find_restFrame()

	plots = [
		'flux',
		"components['stellar'].plot['vel']",
		"components['stellar'].plot['sigma']",
		"components['stellar'].plot['vel'].uncert",
		"components['stellar'].plot['sigma'].uncert",
	]

	saveTo = [
		"%s/%s_stellar_img.png" % (out_dir, galaxy),
		"%s/%s_stellar_vel.png" % (out_dir, galaxy),
		"%s/%s_stellar_sigma.png" % (out_dir, galaxy),
		"%s/%s_stellar_vel_uncert.png" % (out_dir, galaxy),
		"%s/%s_stellar_sigma_uncert.png" % (out_dir, galaxy),
	]

	attr, vmin, vmax = np.loadtxt('%s/Data/%s/analysis/lims.txt' % (
		cc.base_dir, instrument), dtype=str, usecols=(0,1,2))
	min_v, max_v = min_v.astype(float), max_v.astype(float)

	for i, p in enumerate(plots):
		i_att = np.where(attr == p)[0][0]

		if 'flux' in p:
			ax = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, 
				D.yBar, getattr(D, p), header, vmin=vmin[i_att], 
				vmax=vmax[i_att], cmap='gist_yarg', 
				flux_unbinned=D.unbinned_flux, center=center)
		else:
			ax = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, 
				D.yBar, getattr(D, p), header, vmin=vmin[i_att], 
				vmax=vmax[i_att], cmap=sauron, 
				flux_unbinned=D.unbinned_flux, center=center, 
				signal_noise=D.SNRatio, signal_noise_target=SN_target)
		if overplot:
			for o, color in overplot.iteritems():
				add_(o, color, ax, galaxy)

		fig = plt.gcf()
		fig.savefig(saveTo[i])






	opt = 'pop'
	if instrument=='vimos':
		file_headings = np.loadtxt(data_file, dtype=str)[0]
		col = np.where(file_headings=='SN_%s' % (opt))[0][0]

		z_gals, SN_target_gals= np.loadtxt(data_file, unpack=True, 
			skiprows=1, usecols=(1,6))
		galaxy_gals = np.loadtxt(data_file, skiprows=1, usecols=(0,),
			dtype=str)
		i_gal = np.where(galaxy_gals==galaxy)[0][0]
		z = z_gals[i_gal]
		SN_target=SN_target_gals[i_gal]

	elif instrument=='muse':
		file_headings = np.loadtxt(data_file, dtype=str)[0]
		col = np.where(file_headings=='SN_%s' % (opt))[0][0]
	
		x_cent_gals, y_cent_gals, SN_target_gals = np.loadtxt(data_file, 
			unpack=True, skiprows=1, usecols=(1,2,col), 
			dtype='int,int,float')
		galaxy_gals = np.loadtxt(data_file, skiprows=1, usecols=(0,),
			dtype=str)
		i_gal = np.where(galaxy_gals==galaxy)[0][0]
		SN_target=SN_target_gals[i_gal]-10
		center = (x_cent_gals[i_gal], y_cent_gals[i_gal])

		data_file =  "%s/Data/vimos/analysis/galaxies.txt" % (cc.base_dir)
		z_gals = np.loadtxt(data_file, unpack=True, skiprows=1, 
			usecols=(1))
		galaxy_gals = np.loadtxt(data_file, skiprows=1, usecols=(0,),
			dtype=str)
		i_gal = np.where(galaxy_gals==galaxy)[0][0]
		z = z_gals[i_gal]

	if D is None:
		pickleFile = open("%s/%s/%s/pickled/dataObj.pkl" % (vin_dir, 
			galaxy, opt), 'rb')
		D = pickle.load(pickleFile)
		pickleFile.close()
	
	if D.norm_method != norm:
		D.norm_method = norm
		D.find_restFrame()

	plots = [
		"absorption_line('G4300')",
		"absorption_line('Fe4383')"
		"absorption_line('Ca4455')"
		"absorption_line('Fe4531')"
		"absorption_line('H_beta')"
		"absorption_line('Fe5015')"
		"absorption_line('Mg_b')"
		"absorption_line('G4300', uncert=True)[1]",
		"absorption_line('Fe4383', uncert=True)[1]"
		"absorption_line('Ca4455', uncert=True)[1]"
		"absorption_line('Fe4531', uncert=True)[1]"
		"absorption_line('H_beta', uncert=True)[1]"
		"absorption_line('Fe5015', uncert=True)[1]"
		"absorption_line('Mg_b', uncert=True)[1]"
	]



	saveTo = [
		"%s/%s_G4300.png" % (out_dir, galaxy),
		"%s/%s_Fe4384.png" % (out_dir, galaxy),
		"%s/%s_Ca445.png" % (out_dir, galaxy),
		"%s/%s_Fe4531.png" % (out_dir, galaxy),
		"%s/%s_H_beta.png" % (out_dir, galaxy),
		"%s/%s_Fe5015.png" % (out_dir, galaxy),
		"%s/%s_Mg_b.png" % (out_dir, galaxy),
		"%s/%s_G4300_uncert.png" % (out_dir, galaxy),
		"%s/%s_Fe4384_uncert.png" % (out_dir, galaxy),
		"%s/%s_Ca445_uncert.png" % (out_dir, galaxy),
		"%s/%s_Fe4531_uncert.png" % (out_dir, galaxy),
		"%s/%s_H_beta_uncert.png" % (out_dir, galaxy),
		"%s/%s_Fe5015_uncert.png" % (out_dir, galaxy),
		"%s/%s_Mg_b_uncert.png" % (out_dir, galaxy)
	]


	for i, p in enumerate(plots):
		i_att = np.where(attr == p)[0][0]
		ax = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, 
			D.yBar, getattr(D, p), header, vmin=vmin[i_att], 
			vmax=vmax[i_att], cmap='gnuplot2', 
			flux_unbinned=D.unbinned_flux, center=center, 
			signal_noise=D.SNRatio, signal_noise_target=SN_target)
		if overplot:
			for o, color in overplot.iteritems():
				add_(o, color, ax, galaxy)

		fig = plt.gcf()
		fig.savefig(saveTo[i])

	

	


##############################################################################

# Use of plot_results.py

if __name__ == '__main__':

	galaxies = ['eso443-g024', 'ic1459', 'ic1531', 'ic4296', 'ngc0612', 
		'ngc1399', 'ngc3100', 'ngc3557', 'ngc7075', 'pks0718-34']
	# galaxies = [galaxies[8]]

	for galaxy in galaxies:
		print galaxy

		plot_results(galaxy, instrument='vimos', 
			overplot={'CO':'c', 'radio':'r'})