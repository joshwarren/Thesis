import cPickle as pickle
import matplotlib.pyplot as plt 
import numpy as np 
from plot_velfield_nointerp import plot_velfield_nointerp
from plot_results import add_
from errors2 import get_dataCubeDirectory
from prefig import Prefig
from checkcomp import checkcomp
cc = checkcomp()
from astropy.io import fits 
from sauron_colormap import sauron#2 as sauron
from plot_results import set_lims


from Bin import myArray

galaxy = 'ngc1399'
opt = 'kin'
out_dir = '%s/Documents/thesis/chapter2' % (cc.home_dir)


if True:
	Prefig(size=(8,8))
	fig, ax = plt.subplots()#, sharex=True, sharey=True)



	print galaxy

	vin_dir = '%s/Data/vimos/analysis' % (cc.base_dir)
	data_file =  "%s/galaxies.txt" % (vin_dir)
	file_headings = np.loadtxt(data_file, dtype=str)[0]
	col = np.where(file_headings=='SN_%s' % (opt))[0][0]
	x_gals, y_gals, SN_target_gals = np.loadtxt(data_file, 
		unpack=True, skiprows=1, usecols=(4,5,col,), dtype='int,int,float')
	galaxy_gals = np.loadtxt(data_file, skiprows=1, usecols=(0,),dtype=str)
	i_gal = np.where(galaxy_gals==galaxy)[0][0]
	SN_target=SN_target_gals[i_gal]
	center = (x_gals[i_gal], y_gals[i_gal])


	vin_dir += '/%s/%s' % (galaxy, opt) 

	pickle_file = '%s/pickled' % (vin_dir)
	pickleFile = open("%s/dataObj.pkl" % (pickle_file), 'rb')
	D = pickle.load(pickleFile)
	pickleFile.close()


	f = fits.open(get_dataCubeDirectory(galaxy))
	header = f[0].header
	f.close()


	# Velocity
	vmin_vel, vmax_vel = set_lims(D.components['stellar'].plot['vel'],
		symmetric=True, n_std=5)

	ax = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
		D.xBar, D.yBar, D.components['stellar'].plot['vel'], header, 
		vmin=vmin_vel, vmax=vmax_vel, cmap=sauron, signal_noise=D.SNRatio, 
		colorbar=True, label=r'Mean velocity (km s$^{-1}$)', 
		signal_noise_target=SN_target, ax=ax, galaxy='VIMOS', center=center,
		galaxy_labelcolor='w')


	ax.ax_dis.tick_params(top=True, bottom=True, left=True, 
		right=True, direction='in', which='major', length=15,
		width=2, labelsize='large', color='w')
	ax.ax_dis.tick_params(top=True, bottom=True, left=True, 
		right=True, direction='in', which='minor', length=7,
		width=2, color='w')
	ax.ax_dis.xaxis.label.set_size(22)
	ax.ax_dis.yaxis.label.set_size(22)

	fig.savefig('%s/VIMOS_NGC1399_vel.png' % (out_dir), bbox_inches='tight',
		dpi=120)
	plt.close()






if False:
	Prefig(size=(20,10))
	fig, ax = plt.subplots(1,2)
	from Bin import Data

	tessellation_File = "/mnt/x/Data/vimosindi/analysis_sav_2016-02-09"\
		+"/%s/voronoi_2d_binning_output.txt" % (galaxy)
	tessellation_File2 = "/mnt/x/Data/vimosindi/analysis_sav_2016-02-09"\
		+"/%s/voronoi_2d_binning_output2.txt" % (galaxy)
	D = Data(np.loadtxt(tessellation_File, unpack=True, skiprows = 1, 
		usecols=(0,1,2)), sauron=True)
	D.xBar, D.yBar = np.loadtxt(tessellation_File2, unpack=True, skiprows = 1)

	vel_file = "/mnt/x/Data/vimosindi/analysis_sav_2016-02-09"\
		+"/%s/results/4200-/no_MC/gal_vel.dat" % (galaxy)
	vel = np.loadtxt(vel_file, unpack=True, usecols=(0,))
	print min(vel), max(vel)
	D.components_no_mask['stellar'].setkin('vel', vel)

	f = fits.open('/mnt/x/Data/vimosindi/reduced/ngc1399/cube/VIMOS'
		+'.2012-07-27T08-49-07.769_crcl_oextr1_fluxcal_vmcmb_cor_cexp_cube.fits')
	header=f[1].header

	vmin_vel, vmax_vel = set_lims(D.components['stellar'].plot['vel'],
		symmetric=True, n_std=5)

	ax[1] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
		D.xBar, D.yBar, D.components['stellar'].plot['vel'], header, 
		vmin=vmin_vel, vmax=vmax_vel, cmap=sauron, 
		colorbar=True, label=r'Mean velocity (km s$^{-1}$)', 
		ax=ax[1])


	x, y = np.meshgrid(range(40), range(40))
	x, y = x.flatten(), y.flatten()
	bin_num = np.arange(40**2)

	image = np.nansum(f[1].data, axis=0).flatten()

	ax[0] = plot_velfield_nointerp(x, y, bin_num, 
		x, y, image, header, 
		cmap='hot', 
		# colorbar=True, label=r'Mean velocity (km s$^{-1}$)', 
		ax=ax[0])

	for a in ax:
		a.ax_dis.tick_params(top=True, bottom=True, left=True, 
			right=True, direction='in', which='major', length=15,
			width=2, labelsize='large', color='w')
		a.ax_dis.tick_params(top=True, bottom=True, left=True, 
			right=True, direction='in', which='minor', length=7,
			width=2, color='w')
		a.ax_dis.xaxis.label.set_size(22)
		a.ax_dis.yaxis.label.set_size(22)


	# _new added to file name so that photoshoped version is not accidentally 
	# overwritten
	fig.savefig('%s/P3D_NGC1399_new.png' % (out_dir), bbox_inches='tight',
		dpi=120)


