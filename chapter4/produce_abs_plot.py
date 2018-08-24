from Bin2 import Data
import matplotlib.pyplot as plt 
import numpy as np 
from plot_velfield_nointerp import plot_velfield_nointerp
from prefig import Prefig
from checkcomp import checkcomp
cc = checkcomp()
from astropy.io import fits 
from plot_results import set_lims
from matplotlib import ticker


def plot(galaxies, str_galaxies, file_name, instrument='vimos', 
	debug=False):
	if instrument == 'muse2':
		instrument = 'muse'
		plot2 = True
	else: plot2 = False
	opt = 'pop'
	overplot={'CO':'w', 'radio':'g'}
	if instrument=='muse' and not plot2:
		Prefig(size=np.array((len(galaxies)*2, 6))*7)
		fig, axs = plt.subplots(6, len(galaxies)*2)
	elif plot2: 
		Prefig(size=np.array((len(galaxies)*2, 5))*7)
		fig, axs = plt.subplots(5, len(galaxies)*2)
	elif instrument=='vimos':
		Prefig(size=np.array((len(galaxies)*2, 7))*7)
		fig, axs = plt.subplots(7, len(galaxies)*2)

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

		if debug:
			from produce_plots import Ds 
			D = Ds()
		else:
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

		if instrument=='vimos':
			plots = [
				"absorption_line('G4300')",
				"absorption_line('Fe4383')",
				"absorption_line('Ca4455')",
				"absorption_line('Fe4531')",
				"absorption_line('H_beta')",
				"absorption_line('Fe5015')",
				"absorption_line('Mg_b')"
				]
		elif instrument=='muse' and not plot2:
			plots = [
				"absorption_line('H_beta')",
				"absorption_line('Fe5015')",
				"absorption_line('Mg_b')",
				"absorption_line('Fe5270')",
				"absorption_line('Fe5335')",
				"absorption_line('Fe5406')",
				]
		elif instrument=='muse' and plot2:
			plots = [
				"absorption_line('Fe5709')",
				"absorption_line('Fe5782')",
				"absorption_line('NaD')",
				"absorption_line('TiO1')",#remove_badpix=True)",
				"absorption_line('TiO2')"#,remove_badpix=True)"
				]


		for j, p in enumerate(plots):
			if any([l in p for l in ['H_beta','Ca4455','Fe5270','Fe5335',
				'Fe5406','Fe5709','Fe5782']]):
				vmin, vmax = 0.5, 3.5
			elif 'TiO1' in p:
				vmin, vmax = 0, 0.35
			elif 'Ti02' in p:
				vmin, vmax = 0, 0.1
			else:
				vmin, vmax = 3, 7


			if 'Mg_b' in p and galaxy in ['ngc0612', 'pks0718-34']:
				axs[j, 2*i].remove()
			else:
				axs[j, 2*i] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
					D.xBar, D.yBar, eval('D.'+p), header,  
					vmin=vmin, vmax=vmax, 
					cmap='inferno', flux_unbinned=D.unbinned_flux, 
					signal_noise=D.SNRatio, signal_noise_target=SN_target, 
					ax=axs[j, 2*i])
				if overplot:
					for o, color in overplot.iteritems():
						scale = 'log' if o == 'radio' else 'lin'
						add_(o, color, axs[j, 2*i], galaxy, nolegend=True, 
							scale=scale)
				
		if instrument=='vimos':
			plots = [
				"absorption_line('G4300',uncert=True)[1]",
				"absorption_line('Fe4383',uncert=True)[1]",
				"absorption_line('Ca4455',uncert=True)[1]",
				"absorption_line('Fe4531',uncert=True)[1]",
				"absorption_line('H_beta',uncert=True)[1]",
				"absorption_line('Fe5015',uncert=True)[1]",
				"absorption_line('Mg_b',uncert=True)[1]"
				]
		elif instrument=='muse' and not plot2:
			plots = [
				"absorption_line('H_beta',uncert=True)[1]",
				"absorption_line('Fe5015',uncert=True)[1]",
				"absorption_line('Mg_b',uncert=True)[1]",
				"absorption_line('Fe5270',uncert=True)[1]",
				"absorption_line('Fe5335',uncert=True)[1]",
				"absorption_line('Fe5406',uncert=True)[1]",
				]
		elif instrument=='muse' and plot2:
			plots = [
				"absorption_line('Fe5709',uncert=True)[1]",
				"absorption_line('Fe5782',uncert=True)[1]",
				"absorption_line('NaD',uncert=True)[1]",
				"absorption_line('TiO1',uncert=True)[1]",#remove_badpix=True)[1]",
				"absorption_line('TiO2',uncert=True)[1]"#,remove_badpix=True)[1]"
				]

		for j, p in enumerate(plots):
			if 'Mg_b' in p and galaxy in ['ngc0612', 'pks0718-34']:
				axs[j, 2*i+1].remove()
			else:
				vmax = 0.03 if 'TiO' in p else 0.5
				axs[j, 2*i+1] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
					D.xBar, D.yBar, eval('D.'+p), header,  
					vmin=0, vmax=vmax, 
					cmap='inferno', flux_unbinned=D.unbinned_flux, 
					signal_noise=D.SNRatio, signal_noise_target=SN_target, 
					ax=axs[j, 2*i+1])


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

	# Remove y axis labels from all except left column 
	if 'ngc0612' not in galaxies:
		for a in axs[:,1:].flatten():
			if hasattr(a, 'ax_dis'): 
				a.ax_dis.set_yticklabels([])
				a.ax_dis.set_ylabel('')
	else:
		for a in axs[:-1,1:].flatten():
			if hasattr(a, 'ax_dis'): 
				a.ax_dis.set_yticklabels([])
				a.ax_dis.set_ylabel('')
		if hasattr(axs[-1,-1], 'ax_dis'): 
			axs[-1,-1].ax_dis.set_yticklabels([])
			axs[-1,-1].ax_dis.set_ylabel('')


	# Remove x axis labels from all except lowest plots
	if instrument=='muse' or all(
		[g not in ['ngc0612', 'pks0718-34'] for g in galaxies]):
		for a in axs[:-1,:].flatten():
			if hasattr(a, 'ax_dis'): 
				a.ax_dis.set_xticklabels([])
				a.ax_dis.set_xlabel('')
	elif instrument=='vimos' and 'ngc0612' in galaxies:
		for a in axs[:-2,:].flatten():
			if hasattr(a, 'ax_dis'): 
				a.ax_dis.set_xticklabels([])
				a.ax_dis.set_xlabel('')
		for a in axs[-2,2:].flatten():
			if hasattr(a, 'ax_dis'): 
				a.ax_dis.set_xticklabels([])
				a.ax_dis.set_xlabel('')
	elif instrument=='vimos' and 'pks0718-34' in galaxies:
		for a in axs[:-2,:].flatten():
			if hasattr(a, 'ax_dis'): 
				a.ax_dis.set_xticklabels([])
				a.ax_dis.set_xlabel('')
		for a in axs[-2,:2].flatten():
			if hasattr(a, 'ax_dis'): 
				a.ax_dis.set_xticklabels([])
				a.ax_dis.set_xlabel('')

	# Create gap between galaxies
	for i in range(2, len(galaxies)*2, 2):
		for a in axs[:, i:i+2].flatten():
			ax_loc = a.get_position()
			ax_loc.x0 += i*0.01
			ax_loc.x1 += i*0.01

			a.set_position(ax_loc)
			if hasattr(a, 'ax_dis'):
				a.ax_dis.set_position(ax_loc)

	for i in range(1, len(galaxies)*2, 2):
		for a in axs[:, i].flatten():
			ax_loc = a.get_position()
			ax_loc.x0 -= 0.01
			ax_loc.x1 -= 0.01

			a.set_position(ax_loc)
			if hasattr(a, 'ax_dis'):
				a.ax_dis.set_position(ax_loc)
	

	# Add galaxy name labels
	if len(galaxies) == 1:
		fig.text(0.5, 0.9, str_galaxies[0], va='top', ha='center', 
			size='xx-large')
	elif len(galaxies) == 2:
		fig.text(0.33, 0.9, str_galaxies[0], va='top', ha='center', 
			size='xx-large')
		fig.text(0.72, 0.9, str_galaxies[1], va='top', ha='center', 
			size='xx-large')


	# Add plot title to lefthand side
	if instrument=='vimos':
		y_loc = np.mean([axs[0,0].get_position().y0, 
			axs[0,0].get_position().y1])
		fig.text(0.07, y_loc, 'G4300', va='center', ha='right', 
			rotation='vertical', size='xx-large')
		y_loc = np.mean([axs[1,0].get_position().y0, 
			axs[1,0].get_position().y1])
		fig.text(0.07, y_loc, 'Fe4383', va='center', ha='right',
			rotation='vertical', size='xx-large')
		y_loc = np.mean([axs[2,0].get_position().y0, 
			axs[2,0].get_position().y1])
		fig.text(0.07, y_loc, 'Ca4455', va='center', ha='right',
			rotation='vertical', size='xx-large')
		y_loc = np.mean([axs[3,0].get_position().y0, 
			axs[3,0].get_position().y1])
		fig.text(0.07, y_loc, 'Fe4531', va='center', ha='right',
			rotation='vertical', size='xx-large')
		y_loc = np.mean([axs[4,0].get_position().y0, 
			axs[4,0].get_position().y1])
		fig.text(0.07, y_loc, r'H$\beta$', va='center', ha='right',
			rotation='vertical', size='xx-large')
		y_loc = np.mean([axs[5,0].get_position().y0, 
			axs[5,0].get_position().y1])
		fig.text(0.07, y_loc, 'Fe5015', va='center', ha='right',
			rotation='vertical', size='xx-large')
		y_loc = np.mean([axs[6,0].get_position().y0, 
			axs[6,0].get_position().y1])
		fig.text(0.07, y_loc, r'Mg\,b', va='center', ha='right',
			rotation='vertical', size='xx-large')
	elif instrument=='muse' and not plot2:
		y_loc = np.mean([axs[0,0].get_position().y0, 
			axs[0,0].get_position().y1])
		fig.text(0.07, y_loc, r'H$_\beta$', va='center', ha='right', 
			rotation='vertical', size='xx-large')
		y_loc = np.mean([axs[1,0].get_position().y0, 
			axs[1,0].get_position().y1])
		fig.text(0.07, y_loc, 'Fe5015', va='center', ha='right',
			rotation='vertical', size='xx-large')
		y_loc = np.mean([axs[2,0].get_position().y0, 
			axs[2,0].get_position().y1])
		fig.text(0.07, y_loc, r'Mg\,b', va='center', ha='right',
			rotation='vertical', size='xx-large')
		y_loc = np.mean([axs[3,0].get_position().y0, 
			axs[3,0].get_position().y1])
		fig.text(0.07, y_loc, 'Fe5270', va='center', ha='right',
			rotation='vertical', size='xx-large')
		y_loc = np.mean([axs[4,0].get_position().y0, 
			axs[4,0].get_position().y1])
		fig.text(0.07, y_loc, 'Fe5335', va='center', ha='right',
			rotation='vertical', size='xx-large')
		y_loc = np.mean([axs[5,0].get_position().y0, 
			axs[5,0].get_position().y1])
		fig.text(0.07, y_loc, 'Fe5406', va='center', ha='right',
			rotation='vertical', size='xx-large')
	elif instrument=='muse' and plot2:
		y_loc = np.mean([axs[0,0].get_position().y0, 
			axs[0,0].get_position().y1])
		fig.text(0.07, y_loc, 'Fe5709', va='center', ha='right',
			rotation='vertical', size='xx-large')
		y_loc = np.mean([axs[1,0].get_position().y0, 
			axs[1,0].get_position().y1])
		fig.text(0.07, y_loc, 'Fe5782', va='center', ha='right',
			rotation='vertical', size='xx-large')
		y_loc = np.mean([axs[2,0].get_position().y0, 
			axs[2,0].get_position().y1])
		fig.text(0.07, y_loc, 'NaD', va='center', ha='right', 
			rotation='vertical', size='xx-large')
		y_loc = np.mean([axs[3,0].get_position().y0, 
			axs[3,0].get_position().y1])
		fig.text(0.07, y_loc, 'TiO1', va='center', ha='right',
			rotation='vertical', size='xx-large')
		y_loc = np.mean([axs[4,0].get_position().y0, 
			axs[4,0].get_position().y1])
		fig.text(0.07, y_loc, 'TiO2', va='center', ha='right',
			rotation='vertical', size='xx-large')


	# Add colorbar
	ax_loc = axs[0,3].get_position()
	ticks = ticker.MaxNLocator(nbins=4)

	cax = fig.add_axes([ax_loc.x1+0.05, ax_loc.y0, 0.02, ax_loc.height])

	# Left
	cbar = plt.colorbar(axs[0,0].cs, cax=cax, ticks=ticks)
	fig.text(ax_loc.x1+0.02, (ax_loc.y0+ax_loc.y1)/2, 
		r'Line Strength ($\AA$)', rotation=90, va='center', 
		ha='center')

	# Right
	cax2 = cax.twinx()
	cax2.set_ylim(axs[0,1].cs.get_clim())
	cax2.yaxis.set_major_locator(ticks)
	fig.text(ax_loc.x1+0.11, (ax_loc.y0+ax_loc.y1)/2, 
		r'Line Strength Uncertainty ($\AA$)', rotation=270, 
		va='center', ha='center')

	# Plots with alternative limits
	diff = ['Ca4455','H_beta','Fe5270','Fe5335','Fe5406','Fe5709','Fe5782']
	diff_str = ['Ca4455',r'H$_\beta$','Fe5270','Fe5335','Fe5406','Fe5709',
		'Fe5782']
	not_diff = ['G4300', 'Fe4283', 'Fe4531', 'Fe5015', 'Mg_b', 'NaD']
	not_diff_str = ['G4300', 'Fe4283', 'Fe4531', 'Fe5015', r'Mg\,b', 'NaD']
	diff_lims = [np.where([l in p for p in plots])[0][0] if 
		any([l in p for p in plots]) else np.nan 
		for l in diff]
	if any(~np.isnan(diff_lims)):
		top_loc = int(np.nanmin(diff_lims))
		ticks = ticker.MaxNLocator(nbins=4)

		# set location and label
		if top_loc==0:
			not_diff_lim = [r for r in range(len(plots)) 
				if float(r) not in diff_lims]
			top_loc = int(np.min(not_diff_lim))
			label = ', '.join([not_diff_str[n] for n in not_diff_lim])+'\n'
		else:
			label = ', '.join(np.array(diff_str)[~np.isnan(diff_lims)])+'\n'

		ax_loc = axs[top_loc,3].get_position()
		cax = fig.add_axes([ax_loc.x1+0.05, ax_loc.y0, 0.02, 
			ax_loc.height])
		# Left
		cbar = plt.colorbar(axs[top_loc,0].cs, cax=cax, ticks=ticks)
		fig.text(ax_loc.x1+0.02, (ax_loc.y0+ax_loc.y1)/2, 
			label + r'Line Strength ($\AA$)', rotation=90, 
			va='center', ha='center')

		# Right
		cax2 = cax.twinx()
		cax2.set_ylim(axs[top_loc,1].cs.get_clim())
		cax2.yaxis.set_major_locator(ticks)
		fig.text(ax_loc.x1+0.11, (ax_loc.y0+ax_loc.y1)/2, 
			label + r'Line Strength Uncertainty ($\AA$)', 
			rotation=270, va='center', ha='center')

	for index in ['TiO1', 'Ti02']:
		diff_lims = np.array([i if index in p else np.nan 
			for i, p in enumerate(plots)])
		if any(~np.isnan(diff_lims)):
			ticks = ticker.MaxNLocator(nbins=4)
			top_loc = int(diff_lims[~np.isnan(diff_lims)][0])
			ax_loc = axs[top_loc,3].get_position()
			cax = fig.add_axes([ax_loc.x1+0.05, ax_loc.y0, 0.02, 
				ax_loc.height])
			# Left
			cbar = plt.colorbar(axs[top_loc,0].cs, cax=cax, ticks=ticks)
			fig.text(ax_loc.x1+0.02, (ax_loc.y0+ax_loc.y1)/2, 
				index+' Line Strength (mag)', rotation=90, va='center', 
				ha='center')

			# Right
			cax2 = cax.twinx()
			cax2.set_ylim(axs[top_loc,1].cs.get_clim())
			cax2.yaxis.set_major_locator(ticks)
			fig.text(ax_loc.x1+0.11, (ax_loc.y0+ax_loc.y1)/2, 
				index+' Line Strength Uncertainty (mag)', rotation=270, 
				va='center', ha='center')

	ticks = ticker.MaxNLocator(nbins=4)


	if debug:
		fig.savefig('%s/%s.png' % (out_dir, 'test'), bbox_inches='tight',
			dpi=200)
	else:
		fig.savefig('%s/%s.png' % (out_dir, file_name), 
			bbox_inches='tight', dpi=200)
	plt.close('all')



if __name__=='__main__':
	# plot(['ic1459', 'ic4296'], ['IC 1459', 'IC 4296'], 'abs1',
	# 	instrument='muse')

	plot(['ngc1316', 'ngc1399'], ['NGC 1316', 'NGC 1399'], 'abs2',
		instrument='muse')

	# plot(['ic1459', 'ic4296'], ['IC 1459', 'IC 4296'], 'abs1b', 
	# 	instrument='muse2')

	plot(['ngc1316', 'ngc1399'], ['NGC 1316', 'NGC 1399'], 'abs2b', 
		instrument='muse2')

	# plot(['eso443-g024', 'ic1459'], ['ESO 443-G24', 'IC 1459'], 'abs1', 
	# 	instrument='vimos')

	# plot(['ic1531', 'ic4296'], ['IC 1531', 'IC 4296'], 'abs2', 
	# 	instrument='vimos')

	# plot(['ngc0612', 'ngc1399'], ['NGC 612', 'NGC 1399'], 'abs3', 
	# 	instrument='vimos')

	# plot(['ngc3100', 'ngc3557'], ['NGC 3100', 'NGC 3557'], 'abs4', 
	# 	instrument='vimos')

	# plot(['ngc7075', 'pks0718-34'], ['NGC7075', 'PKS 0718-34'], 'abs5', 
	# 	instrument='vimos')
