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

def plot(galaxies, str_galaxies, file_name):
	opt = 'kin'
	overplot={'CO':'c', 'radio':'r'}
	Prefig(size=np.array((3, len(galaxies)*2))*10)
	fig, axs = plt.subplots(len(galaxies)*2, 3)#, sharex=True, sharey=True)
	out_dir = '%s/Documents/thesis/chapter4/vimos' % (cc.home_dir)

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
	# 		self.components = {'stellar':comp()}
	# 		self.flux = np.array([0,1,1,2])

	# class comp(object):
	# 	def __init__(self):
	# 		self.plot = {'vel':myArray([0,1,1,2], [0,1,1,2]), 
	# 			'sigma': myArray([0,1,1,2],[0,1,1,2])}

	# D=Ds()



	for i, galaxy in enumerate(galaxies):
	# for i in range(3):
		print galaxy

		vin_dir = '%s/Data/vimos/analysis' % (cc.base_dir)
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
		header = f[0].header
		f.close()

		plots = [
			'flux',
			"components['stellar'].plot['vel']",
			"components['stellar'].plot['sigma']",
			"components['stellar'].plot['vel'].uncert",
			"components['stellar'].plot['sigma'].uncert"
			]

		# Flux
		axs[2*i,0] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
			D.xBar, D.yBar, D.flux, header,  
			# vmin=vmin[attr==plots[0]], vmax=vmax[attr==plots[0]], 
			cmap='gist_yarg', flux_unbinned=D.unbinned_flux, 
			# signal_noise=D.SNRatio, signal_noise_target=SN_target, 
			ax=axs[2*i,0])
		if overplot:
			for o, color in overplot.iteritems():
				add_(o, color, axs[2*i,0], galaxy, nolegend=True)
		
		axs[2*i+1,0].remove()

		# Velocity
		if galaxy == 'ngc0612':
			vmin_vel, vmax_vel = set_lims(D.components['stellar'].plot['vel'],
				symmetric=True, n_std=5)
			print 'NGC 612 velocity scale:', vmin_vel,'km/s to ', vmax_vel, 'km/s'
		else:
			vmin_vel=vmin[attr==plots[1]]
			vmax_vel=vmax[attr==plots[1]], 
		axs[2*i,1] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
			D.xBar, D.yBar, D.components['stellar'].plot['vel'], header, 
			vmin=vmin_vel, vmax=vmax_vel, cmap=sauron, 
			flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
			signal_noise_target=SN_target, ax=axs[2*i,1])
		if overplot:
			for o, color in overplot.iteritems():
				add_(o, color, axs[2*i,1], galaxy, nolegend=True)


		axs[2*i+1,1] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, 
			D.yBar, D.components['stellar'].plot['vel'].uncert, header, 
			vmin=vmin[attr==plots[3]], vmax=vmax[attr==plots[3]], 
			cmap=sauron, flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
			signal_noise_target=SN_target, ax=axs[2*i+1,1])

		# Velocty dispersion
		axs[2*i,2] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, 
			D.yBar, D.components['stellar'].plot['sigma'], header, 
			vmin=vmin[attr==plots[2]], vmax=vmax[attr==plots[2]], 
			cmap=sauron, flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
			signal_noise_target=SN_target, ax=axs[2*i,2])
		if overplot:
			for o, color in overplot.iteritems():
				add_(o, color, axs[2*i,2], galaxy, nolegend=True)


		axs[2*i+1,2] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar,
			D.yBar, D.components['stellar'].plot['sigma'].uncert, header, 
			vmin=vmin[attr==plots[4]], vmax=vmax[attr==plots[4]], 
			cmap=sauron, flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
			signal_noise_target=SN_target, ax=axs[2*i+1,2])


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


	for a in axs[:,2].flatten():
		if hasattr(a, 'ax_dis'): 
			a.ax_dis.set_yticklabels([])
			a.ax_dis.set_ylabel('')

	for a in axs[range(0, len(galaxies)*2, 2), 1].flatten():
		if hasattr(a, 'ax_dis'): 
			a.ax_dis.set_yticklabels([])
			a.ax_dis.set_ylabel('')
	for a in axs[:-1,1:].flatten():
		if hasattr(a, 'ax_dis'): 
			a.ax_dis.set_xticklabels([])
			a.ax_dis.set_xlabel('')

	# Create gap between galaxies
	for i in range(2, len(galaxies)*2, 2):
		for a in axs[i:i+2, :].flatten():
			ax_loc = a.get_position()
			ax_loc.y0 -= i*0.01
			ax_loc.y1 -= i*0.01

			a.set_position(ax_loc)
			if hasattr(a, 'ax_dis'):
				a.ax_dis.set_position(ax_loc)
	for i in range(0, len(galaxies)*2, 2):
		for a in axs[i+1, :].flatten():
			ax_loc = a.get_position()
			ax_loc.y0 += 0.01
			ax_loc.y1 += 0.01

			a.set_position(ax_loc)
			if hasattr(a, 'ax_dis'):
				a.ax_dis.set_position(ax_loc)


	fig.text(0.24, 0.9, r'Flux', va='top', ha='center', size='xx-large')
	fig.text(0.51, 0.9, r'Velocty', va='top', ha='center', size='xx-large')
	fig.text(0.8, 0.9, r'Velocty Dispersion', va='top', ha='center', 
		size='xx-large')

	if len(galaxies) == 1:
		fig.text(0.07, 0.5, str_galaxies[0], va='center', ha='right', 
			rotation='vertical', size='xx-large')

	if len(galaxies) == 2:
		fig.text(0.07, 0.7, str_galaxies[0], va='center', ha='right', 
			rotation='vertical', size='xx-large')
		fig.text(0.07, 0.29, str_galaxies[1], va='center', ha='right', 
			rotation='vertical', size='xx-large')

	if len(galaxies) == 3:
		fig.text(0.07, 0.755, str_galaxies[0], va='center', ha='right', 
			rotation='vertical', size='xx-large')
		fig.text(0.07, 0.48, str_galaxies[1], va='center', ha='right',
			rotation='vertical', size='xx-large')
		fig.text(0.07, 0.19, str_galaxies[2], va='center', ha='right',
			rotation='vertical', size='xx-large')
	
	# Add colorbar
	ax_loc = axs[0,2].get_position()
	cax = fig.add_axes([ax_loc.x1+0.03, ax_loc.y0, 0.02, ax_loc.height])
	cbar = plt.colorbar(axs[1,1].cs, cax=cax)
	cbar.ax.set_yticklabels([])

	# plt.show()
	fig.savefig('%s/%s.png' % (out_dir, file_name), bbox_inches='tight')



if __name__=='__main__':
	plot(['eso443-g024', 'ic1459'], 
		['ESO 443-G24', 'IC 1459'], 'kin1')

	plot(['ic1531', 'ic4296', 'ngc1399'], 
		[' IC 1531', 'IC 4296', 'NGC 1399'], 'kin2')

	plot(['ngc3100', 'ngc3557', 'ngc7075'], 
		['NGC 3100', 'NGC 3557', 'NGC 7075'], 'kin3')

	plot(['pks0718-34', 'ngc0612'], 
		['PKS 718-34', 'NGC 612'], 'kin4')
