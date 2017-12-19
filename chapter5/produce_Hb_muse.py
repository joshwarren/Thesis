from checkcomp import checkcomp
cc = checkcomp()
if 'home' not in cc.device:
	import matplotlib # 20160202 JP to stop lack-of X-windows error
	matplotlib.use('Agg') # 20160202 JP to stop lack-of X-windows error
import cPickle as pickle
import matplotlib.pyplot as plt 
import numpy as np 
from plot_velfield_nointerp import plot_velfield_nointerp
from plot_results_muse import add_
from errors2_muse import get_dataCubeDirectory
from prefig import Prefig
from astropy.io import fits 
from sauron_colormap import sauron#2 as sauron
from plot_results import set_lims


def plot(galaxies):
	opt = 'pop'
	overplot={'CO':'c', 'radio':'r'}
	Prefig(size=np.array((4, 1))*8)
	fig, axs = plt.subplots(1, 4)#, sharex=True, sharey=True)
	out_dir = '%s/Documents/thesis/chapter5/muse' % (cc.home_dir)


	# from Bin import myArray
	# class Ds(object):
	# 	def __init__(self):
	# 		self.x=np.array([0,0,0,1,1,1,2,2,40])
	# 		self.y=np.array([0,1,2,0,1,2,0,1,40])
	# 		self.bin_num = np.array([0,0,1,0,1,1,2,2,3])
	# 		self.xBar = np.array([0.5,1.5,2,40])
	# 		self.yBar = np.array([0.5,1.5,1,40])
	# 		self.SNRatio = np.array([0,1,1,2])
	# 		self.unbinned_flux = np.zeros((40,40))
	# 		self.number_of_bins = 4
	# 		self.components = {'stellar':comp(),'[OIII]5007d':comp()}
	# 		self.e_line = {f:v for f,v in self.components.iteritems()
	# 			if f!='stellar'}
	# 		self.flux = np.array([0,1,1,2])

	# class comp(object):
	# 	def __init__(self):
	# 		self.plot = {'vel':myArray([0,1,1,2], [0,1,1,2]), 
	# 			'sigma': myArray([0,1,1,2],[0,1,1,2])}
	# 		self.flux = np.array([0.5,1.5,1,40])
	# 		self.amp_noise = np.array([0.5,1.5,1,40])
	
	# D=Ds()


	for i in range(len(galaxies)):
		galaxy = galaxies[i]
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
		
		# attr, vmin, vmax = np.loadtxt('%s/lims.txt' % (vin_dir), dtype=str, 
		# 	usecols=(0,1,2), skiprows=1, unpack=True)
		# vmin, vmax = vmin.astype(float), vmax.astype(float)


		vin_dir += '/%s/%s' % (galaxy, opt) 

		pickle_file = '%s/pickled' % (vin_dir)
		pickleFile = open("%s/dataObj.pkl" % (pickle_file), 'rb')
		D = pickle.load(pickleFile)
		pickleFile.close()

		f = fits.open(get_dataCubeDirectory(galaxy))
		header = f[1].header
		f.close()

		axs[i] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
			D.xBar, D.yBar, D.components['[OIII]5007d'].flux, header,  
			# vmin=vmin[attr==plots[0]], vmax=vmax[attr==plots[0]], 
			cmap=sauron, flux_unbinned=D.unbinned_flux, galaxy_labelcolor='w',
			signal_noise=D.e_line['[OIII]5007d'].amp_noise, 
			signal_noise_target=4, galaxy=galaxy, ax=axs[i])
		if overplot:
			for o, color in overplot.iteritems():
				scale = 'log' if o == 'radio' else 'lin'
				add_(o, color, axs[i], galaxy, nolegend=True, scale=scale)

	for a in axs:
		if hasattr(a, 'ax_dis'):
			a.ax_dis.tick_params(top=True, bottom=True, left=True, 
				right=True, direction='in', which='major', length=20,
				width=3, labelsize='large')
			a.ax_dis.tick_params(top=True, bottom=True, left=True, 
				right=True, direction='in', which='minor', length=10,
				width=3)
			a.ax_dis.xaxis.label.set_size(22)
			a.ax_dis.yaxis.label.set_size(22)

	for a in axs:
		if hasattr(a, 'ax_dis'):
			a.ax_dis.tick_params(top=True, bottom=True, left=True, right=True, 
				direction='in', which='both')

	for a in axs[1:]:
		if hasattr(a, 'ax_dis'): 
			a.ax_dis.set_yticklabels([])
			a.ax_dis.set_ylabel('')

	# Add colorbar
	ax_loc = axs[3].get_position()
	cax = fig.add_axes([ax_loc.x1+0.03, ax_loc.y0, 0.02, ax_loc.height])
	cbar = plt.colorbar(axs[0].cs, cax=cax)
	cbar.ax.set_yticklabels([])


	fig.savefig('%s/Hb.png' % (out_dir), bbox_inches='tight', dpi=240)



if __name__=='__main__':
	plot(['ic1459', 'ic4296', 'ngc1316', 'ngc1399', ])