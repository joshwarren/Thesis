from checkcomp import checkcomp
cc = checkcomp()
if 'home' not in cc.device:
	import matplotlib # 20160202 JP to stop lack-of X-windows error
	matplotlib.use('Agg') # 20160202 JP to stop lack-of X-windows error
import cPickle as pickle
import matplotlib.pyplot as plt 
import numpy as np 
from plot_velfield_nointerp import plot_velfield_nointerp
from prefig import Prefig
from astropy.io import fits 
from sauron_colormap import sauron#2 as sauron
from plot_results import set_lims


from Bin import myArray

def plot(galaxies, str_galaxies, file_name, instrument):
	if instrument == 'vimos':
		from plot_results import add_
		from errors2 import get_dataCubeDirectory
	elif instrument == 'muse':
		from plot_results_muse import add_
		from errors2_muse import get_dataCubeDirectory



	opt = 'pop'
	overplot={'CO':'c', 'radio':'g'}
	Prefig(size=np.array((4, len(galaxies)))*7)
	fig, axs = plt.subplots(len(galaxies), 4)#, sharex=True, sharey=True)
	out_dir = '%s/Documents/thesis/chapter5/%s' % (cc.home_dir, instrument)

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
	# 		self.components = {'stellar':comp(), '[OIII]5007d':comp()}
	# 		self.flux = np.array([0,1,1,2])

	# class comp(object):
	# 	def __init__(self):
	# 		self.plot = {'vel':myArray([0,1,1,2], [0,1,1,2]), 
	# 			'sigma': myArray([0,1,1,2],[0,1,1,2])}

	# D=Ds()



	for i, galaxy in enumerate(galaxies):
	# for i in range(3):
		print galaxy

		vin_dir = '%s/Data/%s/analysis' % (cc.base_dir, instrument)
		data_file =  "%s/galaxies.txt" % (vin_dir)
		file_headings = np.loadtxt(data_file, dtype=str)[0]
		col = np.where(file_headings=='SN_%s' % (opt))[0][0]
		SN_target_gals = np.loadtxt(data_file, 
			unpack=True, skiprows=1, usecols=(col,))
		galaxy_gals = np.loadtxt(data_file, skiprows=1, usecols=(0,),dtype=str)
		i_gal = np.where(galaxy_gals==galaxy)[0][0]
		SN_target=SN_target_gals[i_gal]

		attr, vmin, vmax = np.loadtxt('%s/lims.txt' % (vin_dir), dtype=str, 
			usecols=(0,1,2), skiprows=1, unpack=True)
		vmin, vmax = vmin.astype(float), vmax.astype(float)

		vin_dir += '/%s/%s' % (galaxy, opt) 

		pickle_file = '%s/pickled' % (vin_dir)
		pickleFile = open("%s/dataObj.pkl" % (pickle_file), 'rb')
		D = pickle.load(pickleFile)
		pickleFile.close()


		f = fits.open(get_dataCubeDirectory(galaxy))
		if instrument == 'vimos':
			header = f[0].header
		elif instrument == 'muse':
			header = f[1].header
		f.close()

		plots = [
			"components['[OIII]5007d'].plot['vel']",
			"components['[OIII]5007d'].plot['sigma']",
			"components['[OIII]5007d'].plot['vel'].uncert",
			"components['[OIII]5007d'].plot['sigma'].uncert"
			]

		# Velocity
		if galaxy == 'ngc3100':
			vmin_vel, vmax_vel = -100, 100 #set_lims(
				# D.components['[OIII]5007d'].plot['vel'], symmetric=True, 
				# n_std=5)
			print 'NGC 3100 velocity scale:', vmin_vel,'km/s to ', \
				vmax_vel, 'km/s'
		else:
			vmin_vel=vmin[attr==plots[0]][0]
			vmax_vel=vmax[attr==plots[0]][0]

		axs[i,0] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
			D.xBar, D.yBar, D.components['[OIII]5007d'].plot['vel'], header, 
			vmin=vmin_vel, vmax=vmax_vel, cmap=sauron, 
			flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
			signal_noise_target=SN_target, ax=axs[i,0])
		if overplot:
			for o, color in overplot.iteritems():
				add_(o, color, axs[i,0], galaxy, nolegend=True)


		axs[i,1] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, 
			D.yBar, D.components['[OIII]5007d'].plot['vel'].uncert, header, 
			vmin=vmin[attr==plots[2]][0], vmax=vmax[attr==plots[2]][0], 
			cmap=sauron, flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
			signal_noise_target=SN_target, ax=axs[i,1])

		# Velocty dispersion
		axs[i,2] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, 
			D.yBar, D.components['[OIII]5007d'].plot['sigma'], header, 
			vmin=vmin[attr==plots[1]][0], vmax=vmax[attr==plots[1]][0], 
			cmap=sauron, flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
			signal_noise_target=SN_target, ax=axs[i,2])
		if overplot:
			for o, color in overplot.iteritems():
				add_(o, color, axs[i,2], galaxy, nolegend=True)


		axs[i,3] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar,
			D.yBar, D.components['[OIII]5007d'].plot['sigma'].uncert, header, 
			vmin=vmin[attr==plots[3]][0], vmax=vmax[attr==plots[3]][0], 
			cmap=sauron, flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
			signal_noise_target=SN_target, ax=axs[i,3])


	for a in axs.flatten():
		if hasattr(a, 'ax_dis'):
			a.ax_dis.tick_params(top=True, bottom=True, left=True, 
				right=True, direction='in', which='major', length=20,
				width=3, labelsize='large')
			a.ax_dis.tick_params(top=True, bottom=True, left=True, 
				right=True, direction='in', which='minor', length=10,
				width=3)
			a.ax_dis.xaxis.label.set_size(22)
			a.ax_dis.yaxis.label.set_size(22)


	for a in axs[:,1:].flatten():
		if hasattr(a, 'ax_dis'): 
			a.ax_dis.set_yticklabels([])
			a.ax_dis.set_ylabel('')

	# for a in axs[range(0, len(galaxies), 2), 1:].flatten():
	# 	if hasattr(a, 'ax_dis'): 
	# 		a.ax_dis.set_yticklabels([])
	# 		a.ax_dis.set_ylabel('')
	for a in axs[:-1,:].flatten():
		if hasattr(a, 'ax_dis'): 
			a.ax_dis.set_xticklabels([])
			a.ax_dis.set_xlabel('')

	# Create gap between galaxies
	for i in range(1, len(galaxies)):
		for a in axs[i, :].flatten():
			ax_loc = a.get_position()
			ax_loc.y0 -= i*0.01
			ax_loc.y1 -= i*0.01

			a.set_position(ax_loc)
			if hasattr(a, 'ax_dis'):
				a.ax_dis.set_position(ax_loc)

	loc = np.mean([axs[0,0].get_position().x0, axs[0,1].get_position().x1])
	fig.text(loc, 0.92, r'Velocity', va='top', ha='center', size='xx-large')
	loc = np.mean([axs[0,2].get_position().x0, axs[0,3].get_position().x1])
	fig.text(loc, 0.92, r'Velocity Dispersion', va='top', ha='center', 
		size='xx-large')

	for i, g in enumerate(str_galaxies):
		loc = np.mean([axs[i,0].get_position().y0, axs[i,0].get_position().y1])
		fig.text(0.07, loc, g, va='center', ha='right', 
			rotation='vertical', size='xx-large')
	
	# Add colorbar
	ax_loc = axs[0,3].get_position()
	cax = fig.add_axes([ax_loc.x1+0.03, ax_loc.y0, 0.02, ax_loc.height])
	cbar = plt.colorbar(axs[1,1].cs, cax=cax)
	cbar.ax.set_yticklabels([])

	# plt.show()
	fig.savefig('%s/%s.png' % (out_dir, file_name), bbox_inches='tight',
		dpi=60)



if __name__=='__main__':
	if 'home' in cc.device:
		plot(['ic1459', 'ngc0612', 'ngc3100'], 
			['IC 1459', 'NGC 612', 'NGC 3100'], 'kin', 'vimos')
	elif cc.device == 'uni':
		plot(['ic1459', 'ngc1316'], ['IC 1459', 'NGC 1316'], 'kin', 'muse')