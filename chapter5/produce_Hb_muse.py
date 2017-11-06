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


def plot(galaxies):
	opt = 'pop'
	overplot={'CO':'c', 'radio':'r'}
	Prefig(size=np.array((4, 1))*10)
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
	# 		self.components = {'stellar':comp(),'Hbeta':comp()}
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
		header = f[0].header
		f.close()

		axs.flatten()[i] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
			D.xBar, D.yBar, D.components['Hbeta'].flux, header,  
			# vmin=vmin[attr==plots[0]], vmax=vmax[attr==plots[0]], 
			cmap=sauron, flux_unbinned=D.unbinned_flux, 
			signal_noise=D.e_line['Hbeta'].amp_noise, signal_noise_target=4,
			galaxy=galaxy, ax=axs.flatten()[i])
		if overplot:
			for o, color in overplot.iteritems():
				add_(o, color, axs.flatten()[i], galaxy, nolegend=True)

	for a in axs.flatten():
		if hasattr(a, 'ax_dis'):
			a.ax_dis.tick_params(top=True, bottom=True, left=True, right=True, 
				direction='in', which='both')

	for a in axs.flatten()[[1,2,3,5,6,7,10]]:
		if hasattr(a, 'ax_dis'): 
			a.ax_dis.set_yticklabels([])
			a.ax_dis.set_ylabel('')

	for a in axs.flatten()[[0,1,2,3,5,6]]:
		if hasattr(a, 'ax_dis'): 
			a.ax_dis.set_xticklabels([])
			a.ax_dis.set_xlabel('')



	fig.savefig('%s/Hb.png' % (out_dir), bbox_inches='tight')



if __name__=='__main__':
	plot(['ic1459', 'ic4296', 'ngc1316', 'ngc1399', ])