from checkcomp import checkcomp
cc = checkcomp()
if 'home' not in cc.device:
	import matplotlib # 20160202 JP to stop lack-of X-windows error
	matplotlib.use('Agg') # 20160202 JP to stop lack-of X-windows error
# import cPickle as pickle
import matplotlib.pyplot as plt 
import numpy as np 
from plot_velfield_nointerp import plot_velfield_nointerp
from prefig import Prefig
from astropy.io import fits 
from sauron_colormap import sauron#2 as sauron
from plot_results import set_lims
from tools import myerrorbar
from Bin import myArray
# from BPT import lbd, lnd, lsd, add_grids
from BPT import add_grids, NII_Ha_to_NI_Hb, log_NII_Ha_to_NI_Hb, EqW_Ha_to_EqW_Hb, \
	log_EqW_Ha_to_EqW_Hb
from Bin2 import Data
from scipy import ndimage # for gaussian blur
from matplotlib import ticker


def plot(galaxies, str_galaxies, file_name, instrument, debug=False):
	if instrument == 'vimos':
		from plot_results import add_
		from errors2 import get_dataCubeDirectory
	elif instrument == 'muse':
		from plot_results_muse import add_
		from errors2_muse import get_dataCubeDirectory

	opt = 'pop'
	overplot={'CO':'w', 'radio':'g'}
	Prefig(size=np.array((4, len(galaxies)))*7)
	fig, axs = plt.subplots(len(galaxies), 4)#, sharex=True, sharey=True)
	out_dir = '%s/Documents/thesis/chapter5/%s' % (cc.home_dir, instrument)

	for i, galaxy in enumerate(galaxies):
		if debug:
			from produce_plots import Ds
			D = Ds()
		else:
			D = Data(galaxy, instrument=instrument, opt=opt)
			opt = D.opt

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
				scale = 'log' if o == 'radio' else 'lin'
				add_(o, color, axs[i, 0], galaxy, nolegend=True, 
					scale=scale)


		axs[i,1] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, 
			D.yBar, D.components['[OIII]5007d'].plot['vel'].uncert, header, 
			vmin=vmin[attr==plots[2]][0], vmax=vmax[attr==plots[2]][0], 
			cmap=sauron, flux_unbinned=D.unbinned_flux, 
			signal_noise=D.SNRatio, signal_noise_target=SN_target, 
			ax=axs[i,1])

		# Velocty dispersion
		axs[i,2] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, 
			D.yBar, D.components['[OIII]5007d'].plot['sigma'], header, 
			vmin=vmin[attr==plots[1]][0], vmax=vmax[attr==plots[1]][0], 
			cmap=sauron, flux_unbinned=D.unbinned_flux, 
			signal_noise=D.SNRatio, signal_noise_target=SN_target, 
			ax=axs[i,2])
		if overplot:
			for o, color in overplot.iteritems():
				scale = 'log' if o == 'radio' else 'lin'
				add_(o, color, axs[i, 2], galaxy, nolegend=True, 
					scale=scale)

		axs[i,3] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar,
			D.yBar, D.components['[OIII]5007d'].plot['sigma'].uncert, header, 
			vmin=vmin[attr==plots[3]][0], vmax=vmax[attr==plots[3]][0], 
			cmap=sauron, flux_unbinned=D.unbinned_flux, 
			signal_noise=D.SNRatio, signal_noise_target=SN_target, 
			ax=axs[i,3])


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
	fig.text(loc, 0.92, r'Mean Velocity', va='top', ha='center', size='xx-large')
	loc = np.mean([axs[0,2].get_position().x0, axs[0,3].get_position().x1])
	fig.text(loc, 0.92, r'Velocity Dispersion', va='top', ha='center', 
		size='xx-large')

	for i, g in enumerate(str_galaxies):
		loc = np.mean([axs[i,0].get_position().y0, 
			axs[i,0].get_position().y1])
		fig.text(0.07, loc, g, va='center', ha='right', 
			rotation='vertical', size='xx-large')
	
	# Add colorbar
	ticks_sym = ticker.MaxNLocator(nbins=4, symmetric=True, 
		min_n_ticks=6)
	ticks_pos = ticker.MaxNLocator(nbins=4, min_n_ticks=3)
	ax_loc = axs[0,3].get_position()

	# Left
	cax = fig.add_axes([ax_loc.x1+0.06, ax_loc.y0, 0.02, ax_loc.height])
	cbar = plt.colorbar(axs[0,0].cs, cax=cax, ticks=ticks_pos)
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
	ax_loc = axs[1,3].get_position()
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
	cax2.set_ylim(axs[1,3].cs.get_clim())
	cax2.yaxis.set_major_locator(ticks_pos)


	# Add extra colorbar for NGC 3100
	if 'ngc3100' in galaxies:
		loc = np.where(np.array(galaxies)=='ngc3100')[0][0]
		ax_loc = axs[loc, 3].get_position()
		ticks_sym = ticker.MaxNLocator(nbins=4, min_n_ticks=3)

		cax = fig.add_axes([ax_loc.x1+0.06, ax_loc.y0, 0.02, ax_loc.height])
		cbar = plt.colorbar(axs[loc,1].cs, cax=cax, ticks=ticks_sym)
		fig.text(ax_loc.x1+0.13, (ax_loc.y0+ax_loc.y1)/2, 
			r'Velocity for NGC 3100 (km s$^{-1}$)', rotation=270, 
			va='center', ha='center')


	if debug:
		fig.savefig('%s/%s.png' % (out_dir, 'test'), bbox_inches='tight',
			dpi=240)
	else:
		fig.savefig('%s/%s.png' % (out_dir, file_name), bbox_inches='tight',
			dpi=240)

	plt.close('all')


def ngc3100_NI_Hb():
	galaxy = 'ngc3100'
	opt='pop'
	instrument='vimos'
	from plot_results import add_
	from errors2 import get_dataCubeDirectory

	Prefig(size=(8,8))
	fig, ax = plt.subplots()

	f = fits.open(get_dataCubeDirectory(galaxy))
	header = f[0].header
	f.close()

	D = Data(galaxy, instrument=instrument, opt=opt)

	ax = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, D.yBar, 
		D.components['[NI]d'].flux/D.components['Hbeta'].flux, header, 
		nodots=True, flux_unbinned=D.unbinned_flux, colorbar=True, ax=ax,
		label=r'$\mathrm{\frac{[NI]\lambda\lambda5197,5200}{H\beta}}$', 
		label_size=1.4)

	for o, color in {'radio':'g','CO':'w'}.iteritems():
		scale = 'log' if o == 'radio' else 'lin'
		add_(o, color, ax, galaxy, nolegend=True, scale=scale)
	ax.ax_dis.tick_params(top=True, bottom=True, left=True, right=True, 
		direction='in', which='major', length=20, width=3, labelsize='large')
	ax.ax_dis.tick_params(top=True, bottom=True, left=True, right=True, 
		direction='in', which='minor', length=10, width=3)

	fig.savefig('%s/Documents/thesis/chapter5/vimos/ngc3100_NI_Hb.png' % (
		cc.home_dir), dpi=300, bbox_inches='tight')


def ngc1316_inflow():
	galaxy = 'ngc1316'
	opt = 'pop'
	instrument = 'muse'
	from plot_results_muse import add_
	from errors2_muse import get_dataCubeDirectory
	import disk_fit_functions_binned as dfn

	Prefig(size=np.array((3.6, 1))*5.5)
	fig, ax = plt.subplots(1,3)

	f = fits.open(get_dataCubeDirectory(galaxy))
	header = f[1].header
	f.close()

	D = Data('ngc1316', instrument=instrument, opt=opt)

	vel = D.components['[OIII]5007d'].plot['vel']
	m = ~np.isnan(vel)

	d, params = dfn.disk_fit_exp(D.xBar[m], D.yBar[m], np.array(vel).copy()[m], 
		vel.uncert.copy()[m], sigclip=5.0, leeway=0., verbose=False, grid_length=150)
	d *= -1 # Seems to be slight quirk of the system.
	params[5:] *= -1
	disc = vel.copy() # reinsert nans
	disc[m] = d

	vmin, vmax = set_lims(D.components['[OIII]5007d'].plot['vel'], 
		symmetric=True, positive=False)
	ax[0] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, D.yBar, 
		vel, header, vmin=vmin, max=vmax, nodots=True, flux_unbinned=D.unbinned_flux,
		colorbar=False, ax=ax[0])
	ax[1] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, D.yBar, 
		disc, header, vmin=vmin, vmax=vmax, nodots=True, 
		flux_unbinned=D.unbinned_flux, colorbar=False, ax=ax[1])
	ax[2] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, D.yBar, 
		vel - disc, header, vmin=vmin, vmax=vmax, nodots=True, 
		flux_unbinned=D.unbinned_flux, colorbar=False, ax=ax[2])

	for a in ax:
		for o, color in {'radio':'g','CO':'w'}.iteritems():
			scale = 'log' if o == 'radio' else 'lin'
			add_(o, color, a, galaxy, nolegend=True, scale=scale)
		if hasattr(a, 'ax_dis'):
			a.ax_dis.tick_params(top=True, bottom=True, left=True, 
				right=True, direction='in', which='major', length=20,
				width=3, labelsize='large')
			a.ax_dis.tick_params(top=True, bottom=True, left=True, 
				right=True, direction='in', which='minor', length=10,
				width=3)

	# Decrease gap between maps
	for i in range(len(ax)):
		for a in ax[i:]:
			ax_loc = a.get_position()
			ax_loc.x0 -= 0.03
			ax_loc.x1 -= 0.03

			a.set_position(ax_loc)
			if hasattr(a, 'ax_dis'):
				a.ax_dis.set_position(ax_loc)

	for i, t in enumerate([r'$V_\mathrm{gas}$', r'$V_\mathrm{model}$', 
		r'$V_\mathrm{residuals} = V_\mathrm{gas} - V_\mathrm{model}$']):
		fig.text(ax[i].ax_dis.get_position().x0+0.02, 
			ax[i].ax_dis.get_position().y1-0.04, t, va='top', color='w', zorder=15)

	for a in ax[1:]:
		if hasattr(a, 'ax_dis'): 
			a.ax_dis.set_yticklabels([])
			a.ax_dis.set_ylabel('')

	ax_loc = ax[2].ax_dis.get_position()
	cax = fig.add_axes([ax_loc.x1+0.03, ax_loc.y0, 0.02, ax_loc.height])
	cbar = plt.colorbar(ax[0].cs, cax=cax)
	# cbar.ax.set_yticklabels([])
	fig.text(ax_loc.x1+0.08, 0.5, r'Mean velocity (km s$^{-1}$)', 
		rotation=270, verticalalignment='center')
	
	fig.savefig('%s/Documents/thesis/chapter5/ngc1316_inflow.png' % (
		cc.home_dir), dpi=300, bbox_inches='tight')



def BPT():
	from errors2_muse import get_dataCubeDirectory
	opt = 'pop'
	instrument = 'muse'
	Prefig(size=np.array((3, 2))*6, transparent=False)
	fig, ax = plt.subplots(2,3, sharey=True)
	
	analysis_dir = "%s/Data/muse/analysis" % (cc.base_dir)
	galaxiesFile = "%s/galaxies.txt" % (analysis_dir)
	x_cent_gals, y_cent_gals = np.loadtxt(galaxiesFile, unpack=True, 
		skiprows=1, usecols=(1,2), dtype=int)
	galaxy_gals = np.loadtxt(galaxiesFile, skiprows=1, usecols=(0,),
		dtype=str)

	for j, galaxy in enumerate(['ic1459','ngc1316']):
		i_gal = np.where(galaxy_gals==galaxy)[0][0]
		center = (x_cent_gals[i_gal], y_cent_gals[i_gal])

		D = Data(galaxy, instrument=instrument, opt=opt)
		opt = D.opt

		r = np.sqrt((D.xBar - center[0])**2 + (D.yBar - center[1])**2)
		for i, l in enumerate(['[NII]6583d', '[SII]6716', '[OI]6300d']):

			y = np.log10(D.e_line['[OIII]5007d'].flux/1.35
				/ D.e_line['Hbeta'].flux)
			x = np.log10(D.e_line[l].flux/D.e_line['Halpha'].flux)

			y_err = np.sqrt((D.e_line['[OIII]5007d'].flux.uncert/
				D.e_line['[OIII]5007d'].flux)**2 +
				(D.e_line['Hbeta'].flux.uncert/D.e_line['Hbeta'].flux)**2)\
				/ np.log(10)
			x_err = np.sqrt((D.e_line[l].flux.uncert/D.e_line[l].flux)**2 +
				(D.e_line['Halpha'].flux.uncert/D.e_line['Halpha'].flux)**2)\
				/ np.log(10)

			large_err = (x_err > 0.5) + (y_err > 0.5)

			Seyfert2_combined = np.ones(len(x)).astype(bool)
			LINER_combined = np.ones(len(x)).astype(bool)
			SF_combined = np.ones(len(x)).astype(bool)
			x_line1 = np.arange(-2.2, 1, 0.001)
			if l == '[SII]6716':
				ax[j, i].text(-0.7, 1.2, 'Seyfert 2')
				ax[j, i].text(-1.1, -0.2, 'Star-forming')
				ax[j, i].text(0.2, -0.5, 'LINER')

				Seyfert2 = ((0.72/(x - 0.32) + 1.30 < y) + (x > 0.32)) \
					* (1.89 * x + 0.76 < y) * ~large_err
				LINER = ((0.72/(x - 0.32) + 1.30 < y) + (x > 0.32)) \
					* (1.89 * x + 0.76 > y) * ~large_err
				SF = (y < 0.72/(x - 0.32) + 1.30) * (x < 0.32) * ~large_err

				y_line1 = 0.72/(x_line1 - 0.32) + 1.30
				m = y_line1 < 1
				ax[j, i].plot(x_line1[m], y_line1[m],'k')

				y_line2 = 1.89 * x_line1 + 0.76
				m = y_line2 > y_line1
				a = np.min(x_line1[m])
				x_line2 = np.arange(a, 0.60, 0.001)
				y_line2 = 1.89 * x_line2 + 0.76
				ax[j, i].plot(x_line2, y_line2, 'k')

				lab = r'$\log\left(\frac{[SII]\lambda\lambda6717,6731}{H\alpha}\right)$'
				ax[j, i].set_xlim([-1.2, 0.7])


			elif l == '[NII]6583d':
				x /= 1.34

				ax[j, i].text(-0.5, 1.2, 'Seyfert 2/LINER')
				ax[j, i].text(-1.9, -0.2, 'Star-forming')
				ax[j, i].text(-0.1, -0.75, 'Composite', rotation=280, va='center',
					ha='center')

				Seyfert2 = ((0.61/(x - 0.47) + 1.19 < y) + (x > 0.47)) \
					* ~large_err
				LINER = ((0.61/(x - 0.47) + 1.19 < y) + (x > 0.47)) \
					* ~large_err
				SF = (0.61/(x - 0.47) + 1.19 > y) * (x < 0.47) * ~large_err

				y_line1 = 0.61/(x_line1 - 0.47) + 1.19
				m = y_line1 < 1
				ax[j, i].plot(x_line1[m], y_line1[m],'k')

				y_line2 = 0.61/(x_line1 - 0.05) + 1.3
				m1 = y_line2 < y_line1
				a = np.min(x_line1[m1])
				x_line2 = np.arange(a, 0.60, 0.001)
				y_line2 = 0.61/(x_line2 - 0.05) + 1.3
				m2 = y_line2 < 1
				ax[j, i].plot(x_line2[m2], y_line2[m2],'k--')

				lab = r'$\log\left(\frac{[NII]\lambda6584}{H\alpha}\right)$'

				ax[j, i].set_xlim([-2, 1])
				# add_grids(ax[j, i], '[NII]','[OIII]', x_Ha=True)

			elif l == '[OI]6300d':
				x /= 1.33

				ax[j, i].text(-1.5, 1.2, 'Seyfert 2')
				ax[j, i].text(-2.0, -0.2, 'Star-forming')
				ax[j, i].text(-0.6, -0.5, 'LINER')

				Seyfert2 = ((y > 0.73/(x + 0.59) + 1.33) + (x > -0.59)) \
					* (y > 1.18 * x + 1.30) * ~large_err
				LINER = ((y > 0.73/(x + 0.59) + 1.33) + (x > -0.59)) \
					* (y < 1.18 * x + 1.30) * ~large_err
				SF = (y < 0.73/(x + 0.59) + 1.33) * (x < -0.59) * ~large_err

				y_line1 = 0.73/(x_line1 + 0.59) + 1.33
				m = y_line1 < 1
				ax[j, i].plot(x_line1[m], y_line1[m],'k')

				y_line2 = 1.18 * x_line1 + 1.30
				m = y_line2 > y_line1
				a = np.min(x_line1[m])
				x_line2 = np.arange(a, 0.60, 0.001)
				y_line2 = 1.18 * x_line2 + 1.30
				ax[j, i].plot(x_line2, y_line2, 'k')

				# ax[j, i].axvline(-0.59, ls='--', c='k')

				lab = r'$\log\left(\frac{[OI]\lambda6300}{H\alpha}\right)$'

				ax[j, i].set_xlim([-2.2, 0])

				# add_grids(ax[i], '[OI]', '[OIII]', x_Ha=True)

			ax[j, i].set_ylim([-1.2, 1.5])

			Seyfert2_combined *= Seyfert2
			LINER_combined *= LINER
			SF_combined *= SF

			myerrorbar(ax[j, i], x, y, xerr=x_err, yerr=y_err, marker='.', 
				color=r)

			# ax[1, i].set_xlabel(r'%sH$\alpha$)' % (lab), size='x-large')
			ax[1, i].set_xlabel(lab, size='x-large')

	y_loc = np.mean([ax[0,0].get_position().y0, ax[1,0].get_position().y1])
	fig.text(ax[0,0].get_position().x0-0.03, y_loc, 
		r'$\log\left(\frac{[OIII]\lambda5007}{H\beta}\right)$', va='center', ha='right', 
		rotation='vertical', size='x-large')
	for i, g in enumerate(['IC 1459', 'NGC 1316']):
		loc = np.mean([ax[i,0].get_position().y0, 
			ax[i,0].get_position().y1])
		print g
		fig.text(0.06, loc, g, va='center', ha='right', 
			rotation='vertical', size='xx-large')
	for a in ax[0,:]:
		a.set_xticklabels([])


	saveTo = '%s/Documents/thesis/chapter5/BPT.png' % (cc.home_dir)		
	fig.subplots_adjust(wspace=0,hspace=0)
	fig.savefig(saveTo, dpi=240)
	plt.close()


def SAURON():
	Prefig(size=(13,10))
	fig, ax = plt.subplots()

	galaxies = np.array(['eso443-g024', 'ic1459', 'ic1531', 'ic4296', 
		'ngc0612', 'ngc1399', 'ngc3100', 'ngc3557', 'ngc7075', 
		'pks0718-34'])
	str_galaxies = np.array(['ESO 443-G24', 'IC 1459', 'IC 1531', 'IC 4296', 
		'NGC 612', 'NGC 1399', 'NGC 3100', 'NGC 3557', 'NGC 7075', 
		'PKS 718-34'])

	# galaxies = ['ic1459']
	# D = Ds()

	for i, galaxy in enumerate(galaxies):
		# pickleFile = open('%s/Data/vimos/analysis/%s' % (cc.base_dir, galaxy) 
		# 	+ '/pop/pickled/dataObj.pkl')
		# D = pickle.load(pickleFile)
		# pickleFile.close()
		D = Data(galaxy, instrument='vimos', opt='pop')

		if all([l in D.e_components for l in ['[NI]d', 'Hbeta', 
			'[OIII]5007d']]):

			x = np.log10(D.e_line['[NI]d'].flux/D.e_line['Hbeta'].flux)
			x_err = np.sqrt((D.e_line['[NI]d'].flux.uncert/
				D.e_line['[NI]d'].flux)**2 +
				(D.e_line['Hbeta'].flux.uncert/D.e_line['Hbeta'].flux)**2)/\
				np.log(10)

			y = np.log10(D.e_line['[OIII]5007d'].flux/1.35
				/ D.e_line['Hbeta'].flux)
			y_err = np.sqrt((D.e_line['[OIII]5007d'].flux.uncert/
				D.e_line['[OIII]5007d'].flux)**2 + (
				D.e_line['Hbeta'].flux.uncert/D.e_line['Hbeta'].flux)**2) \
				/ np.log(10)

			ax.errorbar(x, y, xerr=x_err, yerr=y_err, label=str_galaxies[i],
				fmt='o', ms=9)

	ax.legend()
	ax.text(0, 1.2, 'Seyfert 2/LINER')
	ax.text(-1.9, -0.2, 'Star-forming')
	ax.text(-0.7, -0.75, 'Composite', rotation=280, va='center', ha='center')

	ax.set_xlabel(r'$\log\left(\frac{[NI]\lambda\lambda5197,5200}{H\beta}\right)$')
	ax.set_ylabel(r'$\log\left(\frac{[OIII]\lambda5007}{H\beta}\right)$')

	ax.set_xlim([-2., 1.])
	ax.set_ylim([-1.5, 1.5])
	xlim = ax.get_xlim()
	x_line = np.linspace(xlim[0], xlim[1], 100)
	y_line = 0.61/(x_line - 0.47) + 1.19

	m = y_line < 1
	plt.plot(log_NII_Ha_to_NI_Hb(x_line[m]), y_line[m], 'k')

	y_line2 = 0.61/(x_line - 0.05) + 1.3
	m1 = y_line2 < y_line
	a = np.min(x_line[m1])
	x_line2 = np.arange(a, 0.60, 0.001)
	y_line2 = 0.61/(x_line2 - 0.05) + 1.3
	m2 = y_line2 < 1
	ax.plot(log_NII_Ha_to_NI_Hb(x_line2[m2]), y_line2[m2],'k--')

	## Add MAPPINGS-III grids
	add_grids(ax, '[NI]', '[OIII]')
	ax.set_xlim([-2., 1.])
	ax.set_ylim([-1.5, 1.5])


	fig.savefig('%s/Documents/thesis/chapter5/SAURON.png' % (cc.home_dir), dpi=240,
		bbox_inches='tight')



def WHbN1():
	Prefig()
	fig, ax = plt.subplots()

	galaxies = np.array(['ic1459', 'ic1531', 'ic4296', 'ngc3100', 'ngc3557', 
		'ngc7075'])
	str_galaxies = np.array(['IC 1459', 'IC 1531', 'IC 4296', 'NGC 3100', 
		'NGC 3557', 'NGC 7075'])
	c = np.array(['b', 'orange', 'g', 'r', 'mediumorchid', 'saddlebrown'])

	# galaxies = ['ic1459']
	# D = Ds()

	for i, galaxy in enumerate(galaxies):
		print 'WHbN1:', galaxy
		D = Data(galaxy, instrument='vimos', opt='pop')

		if all([l in D.e_components for l in ['[NI]d', 'Hbeta']]):

			x = np.log10(D.e_line['[NI]d'].flux/D.e_line['Hbeta'].flux)
			x_err = np.sqrt((D.e_line['[NI]d'].flux.uncert/
				D.e_line['[NI]d'].flux)**2 +
				(D.e_line['Hbeta'].flux.uncert/D.e_line['Hbeta'].flux)**2)/\
				np.log(10)

			y = np.log10(D.e_line['Hbeta'].equiv_width)
			y_err = y.uncert

			m = (D.components['[NI]d'].equiv_width < 0.5) * (
				D.components['Hbeta'].equiv_width < EqW_Ha_to_EqW_Hb(0.5))

			ax.errorbar(x[~m], y[~m], xerr=x_err[~m], yerr=y_err[~m], 
				label=str_galaxies[i], fmt='o', ms=9, c=c[i])

			ax.errorbar(x[m], y[m], xerr=x_err[m], yerr=y_err[m], fmt='x', ms=9, 
				c=c[i])

	ax.legend(loc=3)

	ax.set_ylim([-1., 0.6])
	ax.set_xlim([-1.75, 0.75])
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	ax.axhline(np.log10(EqW_Ha_to_EqW_Hb(3)), ls=':', c='k')

	ax.plot([log_NII_Ha_to_NI_Hb(-0.4), log_NII_Ha_to_NI_Hb(-0.4)],
		[np.log10(EqW_Ha_to_EqW_Hb(3)), ylim[1]], ls=':', c='k')
	ax.plot([log_NII_Ha_to_NI_Hb(-0.4), xlim[1]], 
		[np.log10(EqW_Ha_to_EqW_Hb(6)), np.log10(EqW_Ha_to_EqW_Hb(6))], ls=':', 
		c='k')

	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_ylabel(r'log(EW(H$_\beta$)/$\AA$)')
	ax.set_xlabel(r'log [NI]$\lambda\lambda$5197,5200/H$\,\beta$')
	ax.text(-1.65, 0.45, 'Star Forming')
	ax.text(-0.95, 0.45, 'Strong AGN')
	ax.text(-0.95, 0.1, 'Weak AGN')
	ax.text(-1.65, -0.2,'Retired Galaxies')

	fig.savefig('%s/Documents/thesis/chapter5/WHbN1.png' % (cc.home_dir), dpi=240)


def ic4296_WHaN2():
	print 'WHaN2 for IC 4296'
	Prefig()
	fig, ax = plt.subplots()

	galaxy = 'ic4296'

	# galaxies = ['ic1459']
	# D = Ds()
	analysis_dir = "%s/Data/muse/analysis" % (cc.base_dir)
	galaxiesFile = "%s/galaxies.txt" % (analysis_dir)
	x_cent_gals, y_cent_gals = np.loadtxt(galaxiesFile, unpack=True, 
		skiprows=1, usecols=(1,2), dtype=int)
	galaxy_gals = np.loadtxt(galaxiesFile, skiprows=1, usecols=(0,),
		dtype=str)
	i_gal = np.where(galaxy_gals==galaxy)[0][0]
	center = (x_cent_gals[i_gal], y_cent_gals[i_gal])

	# pickleFile = open('%s/Data/muse/analysis/%s' % (cc.base_dir, galaxy) 
	# 	+ '/pop/pickled/dataObj.pkl')
	# D = pickle.load(pickleFile)
	# pickleFile.close()
	D = Data(galaxy, instrument='muse', opt='pop')

	D.sauron = True
	if all([l in D.e_components for l in ['[NII]6583d', 'Halpha']]):

		x = np.log10(D.e_line['[NII]6583d'].flux/1.34/D.e_line['Halpha'].flux)
		x_err = np.sqrt((D.e_line['[NII]6583d'].flux.uncert/
			D.e_line['[NII]6583d'].flux)**2 +
			(D.e_line['Halpha'].flux.uncert/D.e_line['Halpha'].flux)**2)/\
			np.log(10)

		y = np.log10(D.e_line['Halpha'].equiv_width)
		y_err = y.uncert

		r = np.sqrt((D.xBar-center[0])**2 + (D.yBar-center[1])**2)
		m = np.isfinite(x)*np.isfinite(y)*np.isfinite(x_err)*np.isfinite(y_err)*\
			np.isfinite(r)
		myerrorbar(ax, x[m], y[m], xerr=x_err[m], yerr=y_err[m], color=r[m], 
			marker='o')

		# myerrorbar(ax, x, y, xerr=x_err, yerr=y_err, color=r, marker='o')

	ax.set_ylim([-0.1, 1.2])
	ax.set_xlim([-1., 1.])
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	ax.axhline(np.log10(3), ls=':', c='k')

	ax.plot([-0.4, -0.4], [np.log10(3), ylim[1]], ls=':', c='k')
	ax.plot([-0.4, xlim[1]], [np.log10(6), np.log10(6)], ls=':', c='k')

	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_ylabel(r'log(EW(H$_\alpha$)/$\AA$)')
	ax.set_xlabel(r'log [NII]$\lambda$6548/H$\,\alpha$')
	ax.text(-0.95, 1., 'Star Forming')
	ax.text(-0.3, 1., 'Strong AGN')
	ax.text(-0.3, 0.65, 'Weak AGN')
	ax.text(-0.95, 0.3,'Retired Galaxies')

	fig.savefig('%s/Documents/thesis/chapter5/WHaN2.png' % (cc.home_dir), dpi=240)



def H_profile(instrument='vimos'):
	from matplotlib import ticker
	print '**************Need to correctly set seeing************'


	if instrument=='vimos':
		galaxies = np.array(['ic1459', 'ngc0612', 'ngc3100'])
		str_galaxies = np.array(['IC 1459', 'NGC 612', 'NGC 3100'])
		cols = (4, 5)
		line = 'Hbeta'
		res = 0.67 # arcsec/pix
	elif instrument=='muse':
		galaxies = np.array(['ic1459', 'ngc1316'])
		str_galaxies = np.array(['IC 1459', 'NGC 1316'])
		cols = (1, 2)
		line = 'Halpha'
		res = 0.2 # arcsec/pix

	Prefig(size=np.array((len(galaxies), 1))*7)
	fig, ax = plt.subplots(1, len(galaxies), sharey=True)

	analysis_dir = "%s/Data/%s/analysis" % (cc.base_dir, instrument)
	galaxiesFile = "%s/galaxies.txt" % (analysis_dir)
	x_cent_gals, y_cent_gals = np.loadtxt(galaxiesFile, unpack=True, 
		skiprows=1, usecols=cols, dtype=int)
	galaxy_gals = np.loadtxt(galaxiesFile, skiprows=1, usecols=(0,),
		dtype=str)


	for i, galaxy in enumerate(galaxies):
		print 'H profile:', galaxy
		D = Data(galaxy, instrument=instrument, opt='pop')

		if galaxy=='ic1459' and instrument=='vimos':
			seeing_sigma = np.mean([0.82,0.84,1.20,1.26,1.10,1.27]) * 2.5
			norm_r = 2.5
			norm_y = 0.04
		elif galaxy=='ngc0612':
			seeing_sigma = np.mean([0.72,0.78,1.45,1.46,1.08,1.11]) * 2.5
			norm_r = 2.5
			norm_y = 0.2
		elif galaxy=='ngc3100':
			seeing_sigma = np.mean([0.76,0.82,0.75,0.82,1.16,1.20]) * 2.5
			norm_r = 2
			norm_y = 0.15
		elif instrument=='muse':
			from errors2_muse import get_dataCubeDirectory
			f=fits.open(get_dataCubeDirectory(galaxy))
			seeing_sigma = np.mean([f[0].header['ESO TEL AMBI FWHM START'],
				f[0].header['ESO TEL AMBI FWHM START']]) * 2.5
			if galaxy=='ic1459':
				norm_r = 5
				norm_y = 0.005
			elif galaxy=='ngc1316':
				norm_r = 5
				norm_y = 0.015

		i_gal = np.where(galaxy_gals==galaxy)[0][0]
		center = (x_cent_gals[i_gal], y_cent_gals[i_gal])

		r = np.sqrt((D.xBar - center[0])**2 + (D.yBar - center[1])**2) * res
		if line in D.e_components:

			H = D.e_line[line].flux

			ax[i].errorbar(r, H/np.nanmax(H), yerr=H.uncert/np.nanmax(H), 
				fmt='.', color='k')

			o = np.argsort(r)

			lim = ax[i].get_xlim()
			x = np.arange(-seeing_sigma, lim[1], 0.001)

			y = 1/x**2
			y = ndimage.gaussian_filter1d(y, seeing_sigma) # convolve with seeing
			y = norm_y * norm_r**2 * y # normalised by eye

			ax[i].plot(x[x>=0], y[x>=0], zorder=10, color='r')
			ax[i].set_xlim(lim)

			ax[i].text(0.93*lim[1], 0.7, str_galaxies[i], ha='right')

			ax[i].set_ylim([-0.02, 1.1])

			ax[i].set_yscale('log')
			ax[i].tick_params(which='major', direction='in', length=10, 
				width=2)
			ax[i].tick_params(axis='y', which='minor', direction='in', 
				length=6, width=2)
			for axis in [ax[i].xaxis, ax[i].yaxis]:
				axis.set_major_formatter(
					ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

	if instrument == 'vimos':
		ax[0].set_ylabel(r'H$\,\beta$ normalised flux')
	if instrument == 'muse':
		ax[0].set_ylabel(r'H$\,\alpha$ normalised flux')
	for a in ax:
		a.set_xlabel('Radius (arcsec)')

	fig.subplots_adjust(wspace=0,hspace=0)
	fig.savefig('%s/Documents/thesis/chapter5/%s/%s_profile.png' % (
		cc.home_dir, instrument, line), dpi=240)
	plt.close('all')
	



if __name__=='__main__':
	# if 'home' in cc.device:
		# H_profile(instrument='vimos')

	ngc3100_NI_Hb()

		# WHbN1()
		
		# SAURON()

	plot(['ic1459', 'ngc0612', 'ngc3100'], 
		['IC 1459', 'NGC 612', 'NGC 3100'], 'kin', 'vimos')
	# elif cc.device == 'uni':
		# ic4296_WHaN2()

	# H_profile(instrument='muse')

	ngc1316_inflow()

	# BPT()

	plot(['ic1459', 'ngc1316'], ['IC 1459', 'NGC 1316'], 'kin', 'muse')
