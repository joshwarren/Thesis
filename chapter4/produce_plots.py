import matplotlib.pyplot as plt 
import numpy as np 
from plot_velfield_nointerp import plot_velfield_nointerp
from prefig import Prefig
from checkcomp import checkcomp
cc = checkcomp()
from astropy.io import fits 
from sauron_colormap import sauron#2 as sauron
from plot_results import set_lims
from Bin import myArray
from matplotlib import ticker

# Fake Data class for fast testing plotting routines
class Ds(object):
	def __init__(self):
		self.x=np.array([0,0,0,1,1,1,2,2,40])
		self.y=np.array([0,1,2,0,1,2,0,1,40])
		self.bin_num = np.array([0,0,1,0,1,1,2,2,3])
		self.xBar = np.array([0.5,1.5,2,40])
		self.yBar = np.array([0.5,1.5,1,40])
		self.SNRatio = np.array([0,1,1,2])
		self.unbinned_flux = np.zeros((40,40))
		self.number_of_bins = 4
		self.components = {'stellar':comp(), '[OIII]5007d':comp(),
			'[NII]6583d':comp(), '[SII]6716':comp(), '[OI]6300d':comp(), 
			'Hbeta':comp(), 'Halpha':comp(), '[NI]d':comp()}
		self.e_line = {k:v for k,v in self.components.iteritems() 
			if k!='stellar'}
		self.e_components = self.e_line.keys()
		self.flux = np.array([0,1,1,2])

	def absorption_line(self, p, uncert=False):
		if uncert:
			return [0,1,1,2], [0,1,1,2]
		else:
			return [0,1,1,2]
# Fake components class
class comp(object):
	def __init__(self):
		self.plot = {'vel':myArray([0,1,1,2], [0,1,1,2]), 
			'sigma': myArray([0,1,1,2],[0,1,1,2])}
		self.flux = myArray([0,1,1,2], [0,1,1,2])
		self.equiv_width = myArray([0,1,1,2], [0,1,1,2])
	@property
	def age(self):
		return myArray([0,1,1,2], [0,1,1,2])
	@property
	def metalicity(self):
		return myArray([0,1,1,2], [0,1,1,2])
	@property
	def alpha(self):
		return myArray([0,1,1,2], [0,1,1,2])
	

def plot(galaxies, str_galaxies, file_name, debug=False, instrument='vimos'):
	opt = 'kin'
	overplot={'CO':'w', 'radio':'g'}
	Prefig(size=np.array((3, len(galaxies)*2))*7)
	fig, axs = plt.subplots(len(galaxies)*2, 3)#, sharex=True, sharey=True)
	out_dir = '%s/Documents/thesis/chapter4/%s' % (cc.home_dir, instrument)

	for i, galaxy in enumerate(galaxies):
	# for i in range(3):
		print galaxy

		vin_dir = '%s/Data/%s/analysis' % (cc.base_dir, instrument)
		data_file =  "%s/galaxies.txt" % (vin_dir)
		file_headings = np.loadtxt(data_file, dtype=str)[0]
		col = np.where(file_headings=='SN_%s' % (opt))[0][0]
		SN_target_gals = np.loadtxt(data_file, 
			unpack=True, skiprows=1, usecols=(col,))
		galaxy_gals = np.loadtxt(data_file, skiprows=1, usecols=(0,),
			dtype=str)
		i_gal = np.where(galaxy_gals==galaxy)[0][0]
		SN_target=SN_target_gals[i_gal]

		attr, vmin, vmax = np.loadtxt('%s/lims.txt' % (vin_dir), dtype=str, 
			usecols=(0,1,2), skiprows=1, unpack=True)
		vmin, vmax = vmin.astype(float), vmax.astype(float)

		if galaxy == 'ngc1316':
			vmin[attr==plots[3]] = 12
			vmax[attr==plots[3]] = 28

			vmin[attr==plots[4]] = 12
			vmax[attr==plots[4]] = 28

		if debug:
			D = Ds()
		else:
			from Bin2 import Data
			D = Data(galaxy, instrument=instrument, opt=opt)

		if instrument == 'vimos':
			from plot_results import add_
			from errors2 import get_dataCubeDirectory
			f = fits.open(get_dataCubeDirectory(galaxy))
			header = f[0].header
		elif instrument == 'muse':
			from plot_results_muse import add_
			from errors2_muse import get_dataCubeDirectory
			f = fits.open(get_dataCubeDirectory(galaxy))
			header = f[1].header
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
				scale = 'log' if o == 'radio' else 'lin'
				add_(o, color, axs[2*i,0], galaxy, nolegend=True, 
					scale=scale)

		axs[2*i+1,0].remove()

		# Velocity
		if instrument=='vimos' and galaxy == 'ngc0612':
			vmin_vel, vmax_vel = set_lims(D.components['stellar'].plot['vel'],
				symmetric=True, n_std=5)
			print 'NGC 612 velocity scale:', vmin_vel,'km/s to ', \
				vmax_vel, 'km/s'
		else:
			vmin_vel=vmin[attr==plots[1]][0]
			vmax_vel=vmax[attr==plots[1]][0]

		if instrument=='muse' and galaxy=='ngc1399':
			axs[2*i,1] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
				D.xBar, D.yBar, D.components['stellar'].plot['vel']-20, 
				header, vmin=vmin_vel, vmax=vmax_vel, cmap=sauron, 
				flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
				signal_noise_target=SN_target, ax=axs[2*i,1])
		else:
			axs[2*i,1] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
				D.xBar, D.yBar, D.components['stellar'].plot['vel'], header, 
				vmin=vmin_vel, vmax=vmax_vel, cmap=sauron, 
				flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
				signal_noise_target=SN_target, ax=axs[2*i,1])
		for o, color in overplot.iteritems():
				scale = 'log' if o == 'radio' else 'lin'
				add_(o, color, axs[2*i,1], galaxy, nolegend=True, 
					scale=scale)


		axs[2*i+1,1] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, 
			D.yBar, D.components['stellar'].plot['vel'].uncert, header, 
			vmin=vmin[attr==plots[3]][0], vmax=vmax[attr==plots[3]][0], 
			cmap=sauron, flux_unbinned=D.unbinned_flux, 
			signal_noise=D.SNRatio, signal_noise_target=SN_target, 
			ax=axs[2*i+1,1])

		# Velocty dispersion
		axs[2*i,2] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, 
			D.yBar, D.components['stellar'].plot['sigma'], header, 
			vmin=vmin[attr==plots[2]][0], vmax=vmax[attr==plots[2]][0], 
			cmap=sauron, flux_unbinned=D.unbinned_flux, 
			signal_noise=D.SNRatio, signal_noise_target=SN_target, 
			ax=axs[2*i,2])
		if overplot:
			for o, color in overplot.iteritems():
				scale = 'log' if o == 'radio' else 'lin'
				add_(o, color, axs[2*i,2], galaxy, nolegend=True, 
					scale=scale)

		axs[2*i+1,2] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar,
			D.yBar, D.components['stellar'].plot['sigma'].uncert, header, 
			vmin=vmin[attr==plots[4]][0], vmax=vmax[attr==plots[4]][0], 
			cmap=sauron, flux_unbinned=D.unbinned_flux, 
			signal_noise=D.SNRatio, signal_noise_target=SN_target, 
			ax=axs[2*i+1,2])


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
	fig.text(0.51, 0.9, r'Velocity', va='top', ha='center', size='xx-large')
	fig.text(0.8, 0.9, r'Velocity Dispersion', va='top', ha='center', 
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
	# if False:
	# 	ax_loc = axs[0,2].get_position()
	# 	cax = fig.add_axes([ax_loc.x1+0.03, ax_loc.y0, 0.02, ax_loc.height])
	# 	cbar = plt.colorbar(axs[1,1].cs, cax=cax)
	# 	cbar.ax.set_yticklabels([])
	# else:

	# This ticker locator is behaving very strangely! Symmetric seems 
	# to have to opposite effect intended... and min_n_ticks seems to 
	# need to doubled from what I expect...
	ticks_sym = ticker.MaxNLocator(nbins=4, symmetric=True, 
		min_n_ticks=6)
	ticks_pos = ticker.MaxNLocator(nbins=4, min_n_ticks=3)
	ax_loc = axs[0,2].get_position()

	# Left
	cax = fig.add_axes([ax_loc.x1+0.06, ax_loc.y0, 0.02, ax_loc.height])
	cbar = plt.colorbar(axs[0,1].cs, cax=cax, ticks=ticks_pos)
	fig.text(ax_loc.x1+0.02, (ax_loc.y0+ax_loc.y1)/2, 
		r'Velocity (km s$^{-1}$)', rotation=90, va='center', ha='center')

	# Right
	fig.text(ax_loc.x1+0.13, (ax_loc.y0+ax_loc.y1)/2, 
		r'Velocity Dispersion (km s$^{-1}$)', rotation=270, va='center', 
		ha='center')
	cax2 = cax.twinx()
	cax2.set_ylim(axs[0,2].cs.get_clim())
	cax2.yaxis.set_major_locator(ticks_sym)
	

	# Colorbar for Uncertainy
	ax_loc = axs[1,2].get_position()
	ticks_pos = ticker.MaxNLocator(nbins=4)

	# Left
	cax = fig.add_axes([ax_loc.x1+0.06, ax_loc.y0, 0.02, ax_loc.height])
	cbar = plt.colorbar(axs[1,1].cs, cax=cax, ticks=ticks_pos)
	fig.text(ax_loc.x1+0.02, (ax_loc.y0+ax_loc.y1)/2, 
		r'Velocity Uncertainy (km s$^{-1}$)', rotation=90, va='center', 
		ha='center')

	# Right
	fig.text(ax_loc.x1+0.13, (ax_loc.y0+ax_loc.y1)/2, 
		r'Velocity Dispersion Uncertainy(km s$^{-1}$)', rotation=270, 
		va='center', ha='center')
	cax2 = cax.twinx()
	cax2.set_ylim(axs[1,2].cs.get_clim())
	cax2.yaxis.set_major_locator(ticks_pos)

	# Add extra colorbar for NGC 0612
	if 'ngc0612' in galaxies:
		loc = np.where('ngc0612' in galaxies)[0][0]
		ax_loc = axs[loc, 2].get_position()
		ticks_sym = ticker.MaxNLocator(nbins=4, min_n_ticks=3)

		cax = fig.add_axes([ax_loc.x1+0.06, ax_loc.y0, 0.02, ax_loc.height])
		cbar = plt.colorbar(axs[loc,1].cs, cax=cax, ticks=ticks_sym)
		fig.text(ax_loc.x1+0.13, (ax_loc.y0+ax_loc.y1)/2, 
			r'Velocity for NGC 0612 (km s$^{-1}$)', rotation=270, 
			va='center', ha='center')


	if debug:
		fig.savefig('%s/%s.png' % (out_dir, 'test'), bbox_inches='tight',
			dpi=200)
	else:
		fig.savefig('%s/%s.png' % (out_dir, file_name), bbox_inches='tight',
			dpi=200)
	plt.close('all')


if __name__=='__main__':
	plot(['ic1459','ic4296','ngc1316'], ['IC 1459', 'IC 4296', 'NGC 1316'],
		'kin1', instrument='muse')

	plot(['ngc1399'], ['NGC 1399'], 'kin2', instrument='muse')

	plot(['eso443-g024', 'ic1459'], ['ESO 443-G24', 'IC 1459'], 'kin1', 
		instrument='vimos')

	plot(['ic1531', 'ic4296', 'ngc1399'], ['IC 1531', 'IC 4296', 'NGC 1399'],
		'kin2', instrument='vimos')

	plot(['ngc3100', 'ngc3557', 'ngc7075'], 
		['NGC 3100', 'NGC 3557', 'NGC 7075'], 'kin3', instrument='vimos')

	plot(['pks0718-34', 'ngc0612'], ['PKS 718-34', 'NGC 612'], 'kin4', 
		instrument='vimos')

