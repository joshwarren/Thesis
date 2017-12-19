from checkcomp import checkcomp
cc = checkcomp()
if 'home' not in cc.device:
	import matplotlib # 20160202 JP to stop lack-of X-windows error
	matplotlib.use('Agg') # 20160202 JP to stop lack-of X-windows error
import cPickle as pickle
import matplotlib.pyplot as plt 
import numpy as np 
from plot_velfield_nointerp import plot_velfield_nointerp
from plot_results import add_
from errors2_muse import get_dataCubeDirectory
from prefig import Prefig
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

	vin_dir = '%s/Data/muse/analysis' % (cc.base_dir)
	data_file =  "%s/galaxies.txt" % (vin_dir)
	file_headings = np.loadtxt(data_file, dtype=str)[0]
	col = np.where(file_headings=='SN_%s' % (opt))[0][0]
	SN_target_gals = np.loadtxt(data_file, 
		unpack=True, skiprows=1, usecols=(col,))
	galaxy_gals = np.loadtxt(data_file, skiprows=1, usecols=(0,),dtype=str)
	i_gal = np.where(galaxy_gals==galaxy)[0][0]
	SN_target=SN_target_gals[i_gal]


	vin_dir += '/%s/%s' % (galaxy, opt) 

	pickle_file = '%s/pickled' % (vin_dir)
	pickleFile = open("%s/dataObj.pkl" % (pickle_file), 'rb')
	D = pickle.load(pickleFile)
	pickleFile.close()


	f = fits.open(get_dataCubeDirectory(galaxy))
	header = f[1].header
	f.close()


	# Velocity
	vmin_vel, vmax_vel = set_lims(D.components['stellar'].plot['vel'],
		symmetric=True, n_std=5)

	ax = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
		D.xBar, D.yBar, D.components['stellar'].plot['vel']-20, header, 
		vmin=vmin_vel+30, vmax=vmax_vel-30, cmap=sauron, signal_noise=D.SNRatio, 
		colorbar=True, label=r'Mean velocity (km s$^{-1}$)', 
		signal_noise_target=SN_target, ax=ax, galaxy='MUSE', 
		galaxy_labelcolor='w')






	ax.ax_dis.tick_params(top=True, bottom=True, left=True, 
		right=True, direction='in', which='major', length=15,
		width=2, labelsize='large', color='w')
	ax.ax_dis.tick_params(top=True, bottom=True, left=True, 
		right=True, direction='in', which='minor', length=7,
		width=2, color='w')
	ax.ax_dis.xaxis.label.set_size(22)
	ax.ax_dis.yaxis.label.set_size(22)

	fig.savefig('%s/MUSE_NGC1399_vel.png' % (out_dir), bbox_inches='tight',
		dpi=240)



if True:
	Prefig(size=(10,10))
	fig, ax = plt.subplots()#, sharex=True, sharey=True)
	galaxy = 'ic1459'

	print galaxy

	vin_dir = '%s/Data/muse/analysis' % (cc.base_dir)
	data_file =  "%s/galaxies.txt" % (vin_dir)
	file_headings = np.loadtxt(data_file, dtype=str)[0]
	col = np.where(file_headings=='SN_%s' % (opt))[0][0]
	SN_target_gals = np.loadtxt(data_file, 
		unpack=True, skiprows=1, usecols=(col,))
	galaxy_gals = np.loadtxt(data_file, skiprows=1, usecols=(0,),dtype=str)
	i_gal = np.where(galaxy_gals==galaxy)[0][0]
	SN_target=SN_target_gals[i_gal]


	vin_dir += '/%s/%s' % (galaxy, opt) 

	pickle_file = '%s/pickled' % (vin_dir)
	pickleFile = open("%s/dataObj.pkl" % (pickle_file), 'rb')
	D = pickle.load(pickleFile)
	pickleFile.close()


	f = fits.open(get_dataCubeDirectory(galaxy))
	header = f[1].header
	f.close()

	ax = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
		D.xBar, D.yBar, D.SNRatio, header, 
		cmap=sauron, colorbar=True, label=r'S/N', 
		ax=ax)


	ax.ax_dis.tick_params(top=True, bottom=True, left=True, 
		right=True, direction='in', which='major', length=15,
		width=2, labelsize='large', color='w')
	ax.ax_dis.tick_params(top=True, bottom=True, left=True, 
		right=True, direction='in', which='minor', length=7,
		width=2, color='w')
	ax.ax_dis.xaxis.label.set_size(22)
	ax.ax_dis.yaxis.label.set_size(22)

	fig.savefig('%s/egSNR.png' % (out_dir), bbox_inches='tight', dpi=240)
	plt.close()