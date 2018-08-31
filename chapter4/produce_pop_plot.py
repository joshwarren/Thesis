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
from matplotlib import ticker

def plot(galaxies, str_galaxies, file_name, instrument='vimos', 
	debug=False):
	opt = 'pop'
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

		if debug:
			from produce_plots import Ds 
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

		# Age
		axs[2*i,0] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
			D.xBar, D.yBar, D.components['stellar'].age, header,  
			vmin=0, vmax=15, cmap='inferno', flux_unbinned=D.unbinned_flux, 
			signal_noise=D.SNRatio, signal_noise_target=SN_target, 
			ax=axs[2*i,0])
		if overplot:
			for o, color in overplot.iteritems():
				scale = 'log' if o == 'radio' else 'lin'
				add_(o, color, axs[2*i,0], galaxy, nolegend=True, scale=scale)
		

		axs[2*i+1,0] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
			D.xBar, D.yBar, D.components['stellar'].age.uncert, header, 
			vmin=0, vmax=2, cmap='inferno', flux_unbinned=D.unbinned_flux, 
			signal_noise=D.SNRatio, signal_noise_target=SN_target, 
			ax=axs[2*i+1,0])

		# Metalicity
		axs[2*i,1] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
			D.xBar, D.yBar, D.components['stellar'].metalicity, header, 
			vmin=-2.25, vmax=0.67, cmap='inferno', 
			flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
			signal_noise_target=SN_target, ax=axs[2*i,1])
		if overplot:
			for o, color in overplot.iteritems():
				scale = 'log' if o == 'radio' else 'lin'
				add_(o, color, axs[2*i,1], galaxy, nolegend=True, scale=scale)


		axs[2*i+1,1] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, 
			D.yBar, D.components['stellar'].metalicity.uncert, header, 
			vmin=0, vmax=0.4, cmap='inferno', flux_unbinned=D.unbinned_flux, 
			signal_noise=D.SNRatio, signal_noise_target=SN_target, 
			ax=axs[2*i+1,1])

		# Alpha
		axs[2*i,2] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, 
			D.yBar, D.components['stellar'].alpha, header, 
			vmin=-0.3, vmax=0.5, cmap='inferno', 
			flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
			signal_noise_target=SN_target, ax=axs[2*i,2])
		if overplot:
			for o, color in overplot.iteritems():
				scale = 'log' if o == 'radio' else 'lin'
				add_(o, color, axs[2*i,2], galaxy, nolegend=True, scale=scale)


		axs[2*i+1,2] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar,
			D.yBar, D.components['stellar'].alpha.uncert, header, 
			vmin=0, vmax=0.25, cmap='inferno', 
			flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
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

	for a in axs[:,1:].flatten():
		if hasattr(a, 'ax_dis'): 
			a.ax_dis.set_yticklabels([])
			a.ax_dis.set_ylabel('')
	for a in axs[:-1,:].flatten():
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

	ax_loc = axs[0,0].get_position()
	fig.text(np.mean([axs[0,0].get_position().x0,axs[0,0].get_position().x1]), 
		ax_loc.y1+0.04, r'Age', va='center', ha='center', size='xx-large')
	fig.text(np.mean([axs[0,1].get_position().x0,axs[0,1].get_position().x1]), 
		ax_loc.y1+0.04, r'Metalicity', va='center', ha='center', size='xx-large')
	fig.text(np.mean([axs[0,2].get_position().x0,axs[0,2].get_position().x1]), 
		ax_loc.y1+0.04, r'Alpha-enhancement', va='center', ha='center', 
		size='xx-large')

	for i in range(len(str_galaxies)):
		y_loc = np.mean([axs[i*2,0].get_position().y0, 
			axs[i*2+1,0].get_position().y1])
		fig.text(0.07, y_loc, str_galaxies[i], va='center', ha='right', 
			rotation='vertical', size='xx-large')

	# Add colorbar
	# ax_loc = axs[0,2].get_position()
	# cax = fig.add_axes([ax_loc.x1+0.03, ax_loc.y0, 0.02, ax_loc.height])
	# cbar = plt.colorbar(axs[0,0].cs, cax=cax)
	# cbar.ax.set_yticklabels([])

	# ticks = ticker.MaxNLocator(nbins=4, symmetric=False)
	# ticks = ticker.AutoLocator()
	ticks = ticker.MaxNLocator(nbins=4)#, symmetric=False)
	ax_loc = axs[0,2].get_position()

	# Left
	cax = fig.add_axes([ax_loc.x1+0.06, ax_loc.y0, 0.02, ax_loc.height])
	cbar = plt.colorbar(axs[0,0].cs, cax=cax, ticks=ticks)
	fig.text(ax_loc.x1+0.03, (ax_loc.y0+ax_loc.y1)/2, 
		'Age (Gyr)', rotation=90, va='center', ha='center')

	# Right
	fig.text(ax_loc.x1+0.13, (ax_loc.y0+ax_loc.y1)/2, 
		'Metalicity (dex)', rotation=270, va='center', 
		ha='center')
	ticks = ticker.MaxNLocator(nbins=4)
	cax2 = cax.twinx()
	cax2.set_ylim(axs[0,1].cs.get_clim())
	cax2.yaxis.set_major_locator(ticks)

	# Far Right
	fig.text(ax_loc.x1+0.2, (ax_loc.y0+ax_loc.y1)/2, 
		'Alpha (dex)', rotation=270, va='center', 
		ha='center')
	ticks = ticker.MaxNLocator(nbins=4)

	cax3 = cax.twinx()
	cax3.spines['right'].set_position(('axes', 5))
	cax3.set_frame_on(True)
	cax3.patch.set_visible(False)

	cax3.set_ylim(axs[0,2].cs.get_clim())
	cax3.yaxis.set_major_locator(ticks)

	# Uncertainties
	ticks = ticker.MaxNLocator(nbins=3)
	ax_loc = axs[1,2].get_position()

	# Left
	cax = fig.add_axes([ax_loc.x1+0.06, ax_loc.y0, 0.02, ax_loc.height])
	cbar = plt.colorbar(axs[1,0].cs, cax=cax, ticks=ticks)
	fig.text(ax_loc.x1+0.03, (ax_loc.y0+ax_loc.y1)/2, 
		'Age Uncertainty (Gyr)', rotation=90, va='center', ha='center')

	# Right
	fig.text(ax_loc.x1+0.13, (ax_loc.y0+ax_loc.y1)/2, 
		'Metalicity Uncertainty (dex)', rotation=270, va='center', 
		ha='center')
	# ticks = ticker.MaxNLocator(nbins=4)
	cax2 = cax.twinx()
	cax2.set_ylim(axs[1,1].cs.get_clim())
	cax2.yaxis.set_major_locator(ticks)

	# Far Right
	fig.text(ax_loc.x1+0.2, (ax_loc.y0+ax_loc.y1)/2, 
		'Alpha Uncertainty (dex)', rotation=270, va='center', 
		ha='center')

	ticks = ticker.MaxNLocator(nbins=4)

	cax3 = cax.twinx()
	cax3.spines['right'].set_position(('axes', 5))
	cax3.set_frame_on(True)
	cax3.patch.set_visible(False)

	cax3.set_ylim(axs[1,2].cs.get_clim())
	cax3.yaxis.set_major_locator(ticks)




	if debug:
		fig.savefig('%s/%s.png' % (out_dir, 'test'), bbox_inches='tight',
			dpi=200)
	else:
		fig.savefig('%s/%s.png' % (out_dir, file_name), bbox_inches='tight',
			dpi=200)
	plt.close('all')

if __name__=='__main__':
	plot(['ic1459', 'ic4296', 'ngc1316'], 
		['IC 1459', 'IC 4296', 'NGC 1316'], 'pop1', instrument='muse')

	plot(['ngc1399'], ['NGC 1399'], 'pop2', instrument='muse')

	plot(['eso443-g024', 'ic1459'], 
		['ESO 443-G24', 'IC 1459'], 'pop1', instrument='vimos')

	plot(['ic1531', 'ic4296', 'ngc0612'], 
		['IC 1531', 'IC 4296', 'NGC 612'], 'pop2', instrument='vimos')

	plot(['ngc1399', 'ngc3100', 'ngc3557'], 
		['NGC 1399', 'NGC 3100', 'NGC 3557'], 'pop3', instrument='vimos')

	plot(['ngc7075', 'pks0718-34'], 
		['NGC 7075', 'PKS 0718-34'], 'pop4', instrument='vimos')
