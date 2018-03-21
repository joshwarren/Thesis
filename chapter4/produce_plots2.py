# Routine to produce plots from Viva corrections.
from checkcomp import checkcomp
cc = checkcomp()
import numpy as np
if 'home' not in cc.device:
	import matplotlib # 20160202 JP to stop lack-of X-windows error
	matplotlib.use('Agg') # 20160202 JP to stop lack-of X-windows error
	import matplotlib.pyplot as plt 
else:
	import matplotlib.pyplot as plt 
from plot_velfield_nointerp import plot_velfield_nointerp
from prefig import Prefig
from Bin2 import Data
# import corner
from tools import myerrorbar
from astropy.io import fits
from scipy.ndimage import zoom


overplot={'CO':'c', 'radio':'g'}

def populations_corner(galaxy, instrument='vimos'):
	# D = Data(galaxy, instrument=instrument, opt='pop')

	b = 100
	dist = np.loadtxt('%s/Data/%s/analysis/%s/pop/pop/distribution/%i.dat' % (
		cc.base_dir, instrument, galaxy, b))
	corner.corner(dist, 
		labels=[r"$t (\mathrm{Gyr})$", r"$\left[\frac{Fe}{H}\right]$", 
		r"$\left[\frac{\alpha}{Fe}\right]$"], quantiles=[0.16, 0.5, 0.84], 
		show_titles=True, title_kwargs={"fontsize": 12})
	plt.show()




	# age = np.zeros(D2.number_of_bins)
	# met = np.zeros(D2.number_of_bins)
	# alp = np.zeros(D2.number_of_bins)
	# unc_age = np.zeros(D2.number_of_bins)
	# unc_met = np.zeros(D2.number_of_bins)
	# unc_alp = np.zeros(D2.number_of_bins)

	# if not debug:
	# 	for j in xrange(D2.number_of_bins):
	# 		ag, me, al = np.loadtxt('%s/pop/distribution/%i.dat' % (
	# 			vin_dir2, j), unpack=True)

	# 		for plot, unc_plot, pop in zip([age,met,alp],
	# 			[unc_age,unc_met,unc_alp], [ag,me,al]):

	# 			hist = np.histogram(pop, bins=40)
	# 			x = (hist[1][0:-1]+hist[1][1:])/2
	# 			hist = hist[0]
	# 			plot[j] = x[np.argmax(hist)]

	# 			gt_fwhm = hist >= np.max(hist)/2
	# 			unc_plot[j] = np.max(x[gt_fwhm]) - np.min(x[gt_fwhm])

def add_grid(ax, xline, yline, alpha=0.5, show_alpha=True, show_labels=True):
	age,Z_H,alpha_fe,HdA,HdF,CN1,CN2,Ca4227,G4300,HgA,HgF,Fe4383,Ca4455,Fe4531,\
		C24668,Hb,Fe5015,Mg1,Mg2,Mgb,Fe5270,Fe5335,Fe5406,Fe5709,Fe5782,NaD,TiO1,\
		TiO2=np.genfromtxt("%s/models/TMJ_SSPs/tmj.dat" % (cc.home_dir), 
		unpack=True, skip_header=35)

	MgFe = np.sqrt(Mgb*(0.72*Fe5270+0.28*Fe5335))

	x = eval(xline)
	y = eval(yline)

	x_grid = np.array([#x[(age==0.6) * (alpha_fe==0)], 
		x[(age==1.) * (alpha_fe==alpha)], 
		x[(age==5.) * (alpha_fe==alpha)], x[(age==15.) * (alpha_fe==alpha)]])
	y_grid = np.array([#y[(age==0.1) * (alpha_fe==0)], 
		y[(age==1.) * (alpha_fe==alpha)], 
		y[(age==5.) * (alpha_fe==alpha)], y[(age==15.) * (alpha_fe==alpha)]])

	for i in range(x_grid.shape[0]):
		ax.plot(x_grid[i,:], y_grid[i,:], 'k-')
	for j in range(x_grid.shape[1]):
		ax.plot(x_grid[:,j], y_grid[:,j], 'k--')

	ax.arrow(x_grid[0,0], y_grid[0,0], (x_grid[1,0]-x_grid[0,0])/3., 
		(y_grid[1,0]-y_grid[0,0])/3., ls='-', head_width=0.2, head_length=0.2,
		fc="k", ec="w", zorder=-1)
	ax.arrow(x_grid[0,0], y_grid[0,0], (x_grid[0,1]-x_grid[0,0])/3., 
		(y_grid[0,1]-y_grid[0,0])/3., ls='-', head_width=0.2, head_length=0.2,
		fc="k", ec="w", zorder=-1)

	if show_labels:
		ax.text(x_grid[0,0] + (x_grid[1,0]-x_grid[0,0])/3., 
			y_grid[0,0] + (y_grid[1,0]-y_grid[0,0])/3., 'Age', va='top', 
			ha='center')
		ax.text(x_grid[0,0] + (x_grid[0,1]-x_grid[0,0])/3. - 0.8, 
			y_grid[0,0] + (y_grid[0,1]-y_grid[0,0])/3. + 0.1, 'Metalicity', 
			va='bottom', ha='left')
	if show_alpha:
		ax.text(0.98,0.98, 'Alpha enhancement = %s'%(alpha), transform=ax.transAxes, 
			va='top', ha='right')

# plot each individual galaxy
def absorption_plots(galaxy, instrument='vimos'):
	print galaxy, instrument
	D = Data(galaxy, instrument=instrument, opt='pop')
	# if True:
	#  	import cPickle as pickle
	#  	pickle_file = '%s/Data/%s/analysis/%s/pop/pickled' % (cc.base_dir, 
	#  		instrument, galaxy)
	# 	pickleFile = open("%s/dataObj.pkl" % (pickle_file), 'rb')
	# 	D = pickle.load(pickleFile)
	# 	pickleFile.close()

	fig, ax = plt.subplots()
	# add_grid(ax, 'Hb', 'MgFe')
	add_grid(ax, 'Hb', 'Mgb')
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()

	# Subtract all fitted emission lines
	Hb, e_Hb = D.absorption_line('H_beta', uncert=True, nomask=True)
	Mg, e_Mg = D.absorption_line('Mgb', uncert=True, nomask=True)
	# Hb, e_Hb = D.absorption_line('H_beta', uncert=True, nomask=False)
	# Mg, e_Mg = D.absorption_line('Mgb', uncert=True, nomask=False)
	# r = np.sqrt((D.xBar - D.center[0])**2 + (D.yBar - D.center[1])**2)
	r = np.sqrt((D.xBar)**2 + (D.yBar)**2)

	# Hb = np.append(Hb, [np.nan, np.nan])
	# Mg = np.append(Mg, [np.nan, np.nan])
	# e_Hb = np.append(e_Hb, [0, 0])
	# e_Mg = np.append(e_Mg, [0, 0])
	if instrument == 'vimos':
		vmax = 20*np.sqrt(2)
	elif instrument == 'muse':
		vmax = 75*np.sqrt(2)

	m = ((e_Hb < 1) * (e_Mg < 1))
	m2 = np.zeros(len(Hb)).astype(bool)
	m2[::10] = True
	# m2[-2:] = True
	m *= m2

	# print r[m].min(), r[m].max()
	# if not any(m):
	# 	m = np.ones(len(e_Hb)).astype(bool)

	myerrorbar(ax, Hb[m], Mg[m], xerr=e_Hb[m], yerr=e_Mg[m], color=r[m], marker='.',
		colorbar=True, cmap='viridis_r', vmin=0, vmax=vmax)

	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	ax.set_xlabel(r'H$\,\beta$')
	ax.set_ylabel(r'Mg$\,$b')

	fig.savefig('%s/absorption_plots_%s.png' % (instrument, galaxy))

	plt.close('all')

def single_absorption_plot(instrument='vimos'):
	if instrument == 'vimos':
		galaxies = ['eso443-g024', 'ic1459', 'ic1531', 'ic4296', 'ngc1399', 
			'ngc3100', 'ngc3557', 'ngc7075'
		]
		str_galaxies = ['ESO 443-G24', 'IC 1459', 'IC 1531', 'IC 4296', 'NGC 1399',
			'NGC 3100', 'NGC 3557', 'NGC 7075'
			]
		Prefig(size=np.array((4, 2))*7)
		fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
		vmax = 20*np.sqrt(2)
		res = 0.67

	elif instrument == 'muse':
		galaxies = ['ic1459', 'ic4296', 'ngc1316', 'ngc1399']
		str_galaxies = ['IC 1459', 'IC 4296', 'NGC 1316', 'NGC 1399']
		Prefig(size=np.array((4, 2))*7)
		fig, ax = plt.subplots(1, 4, sharex=True, sharey=True)
		vmax = 75*np.sqrt(2)
		res = 0.2

	for i, a in enumerate(ax.flatten()):
		if i == 0:
			add_grid(a, 'Hb', 'Mgb', show_alpha=False)
		else:
			add_grid(a, 'Hb', 'Mgb', show_alpha=False, show_labels=False)
		
	xlim = ax[0,0].get_xlim()
	ylim = ax[0,0].get_ylim()



	for i, galaxy in enumerate(galaxies):
		D = Data(galaxy, instrument=instrument, opt='pop')

		Hb, e_Hb = D.absorption_line('H_beta', uncert=True, nomask=True)
		Mg, e_Mg = D.absorption_line('Mgb', uncert=True, nomask=True)
		r = np.sqrt((D.xBar)**2 + (D.yBar)**2) * res

		m = ((np.abs(e_Hb) < 1) * (np.abs(e_Mg) < 1))
		# m2 = np.zeros(len(Hb)).astype(bool)
		# m2[::10] = True
		# m *= m2
		myerrorbar(ax.flatten()[i], Hb[m], Mg[m], xerr=e_Hb[m], yerr=e_Mg[m], 
			color=r[m], marker='.', cmap='viridis_r', vmin=0, vmax=vmax)
		ax.flatten()[i].text(5,6.5, str_galaxies[i])


	fig.text(0.1, 0.5, r'H$\,\beta$', va='center', ha='right',
			rotation='vertical', size='xx-large')
	fig.text(0.52, 0.09, r'Mg$\,$b', va='top', ha='center', size='xx-large')

	ax[0,0].set_xlim(xlim)
	ax[0,0].set_ylim(ylim)

	ax_loc = ax[0,3].get_position()
	cax = fig.add_axes([ax_loc.x1+0.03, ax_loc.y0, 0.02, ax_loc.height])
	cbar = plt.colorbar(ax[0,0].sc, cax=cax)
	t = cbar.ax.text(2.0,0.5, 'Distance (arcsec)', rotation=270, 
		verticalalignment='center')
	# cbar.ax.set_yticklabels([])

	fig.subplots_adjust(wspace=0,hspace=0)
	fig.savefig('%s/absorption.png' % (instrument), bbox_inches='tight', dpi=200)
	plt.close('all')

def show_spectra(galaxy, bins, instrument='vimos', opt='kin'):
	print 'pPXF fit for ' + galaxy + '(' + instrument + ')'
	Prefig(size=(26,12))
	if instrument == 'vimos':
		ext = 0
		from plot_results import add_
		from errors2 import get_dataCubeDirectory
	elif instrument == 'muse':
		ext = 1
		from plot_results_muse import add_
		from errors2_muse import get_dataCubeDirectory

	D = Data(galaxy, instrument=instrument, opt=opt)

	a1 = plt.subplot(121)

	f = fits.open(get_dataCubeDirectory(galaxy))
	header = f[ext].header
	f.close()
	
	a1 = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, D.yBar, D.flux, 
		header, show_bin_num=False, flux_unbinned=D.unbinned_flux, ax=a1, 
		dots=bins)

	for o, color in overplot.iteritems():
			scale = 'log' if o == 'radio' else 'lin'
			add_(o, color, a1, galaxy, nolegend=True, scale=scale)

	for i, b in enumerate(bins):
		a = plt.subplot(len(bins)*100 + 20 + (i+1)*2)
		a.text(0.03, 0.8, str(i+1), color='k', size=25, transform=a.transAxes)
		b_obj = D.bin[b]
		plot_bin(a, D, b)
		if i != len(bins)-1:
			a.set_xticklabels([])
		else:
			a.set_xlabel(r'Wavelength ($\AA$)')

		if (i+1) == (len(bins)+1)/2:
			a.set_ylabel(r"Flux ($10^{-15}\,erg\,s^{-1}\,\AA^{-1}\,cm^{-2}$)",
				labelpad=10)
			a.ax2.set_ylabel('Residuals, Emission lines and Noise '
				+ r'($10^{-15}\,erg\,s^{-1}\,\AA^{-1}\,cm^{-2}$)',rotation=270,
				labelpad=50)


	plt.gcf().savefig('pPXF_fits_'+instrument+'_'+galaxy+'.png', 
		bbox_inches='tight', dpi=200)
	plt.close('all')

# Produce the indervidual spectra showing the pPXF fit
def plot_bin(ax, D, b):
	b = D.bin[b]
	ax2 = ax.twinx()

	scale = 1 if D.instrument == 'vimos' else 10**-5

	lam = b.lam
	spectrum = b.spectrum * scale
	bestfit = b.bestfit * scale
	resid = (spectrum - bestfit) * scale
	emission = np.nansum([b.e_line[l].spectrum for l in D.e_line_no_mask], 
		axis=0)*scale



	ax.plot(lam, spectrum, 'k', label='Data')
	ax.plot(lam, bestfit, 'r', label='Bestfit')

	ax2.plot(lam, resid, '.', color='LimeGreen', mec='LimeGreen', ms=4, 
		label='Residuals')
	ax2.plot(lam, emission, 'orange', label='Emission Lines')

	mn = np.min(bestfit)#[goodpixels])
	mx = np.max(bestfit)#[goodpixels])
	mn1 = np.max([mn-2*np.std(resid),#[self.goodpixels]), 
		np.min(resid)])#[self.goodpixels])])
	ax.set_ylim([mn1, mx] + np.array([-0.05, 0.05])*(mx-mn1))
	ax2.set_ylim(*ax.get_ylim() - mn)

	ax.ax2 = ax2

# Compare VIMOS and MUSE results
def compare():
	print 'Compare VIMOS and MUSE results'
	
	plots = ["components['stellar'].plot['sigma']", "components['Hbeta'].flux", 
		"components['[OIII]5007d'].flux", "absorption_line('Mg_b',uncert=True)"]
	str_plots = [r'$\sigma\ (km\,s^{-1})$', 
		r'F(H$\beta)\ (10^{-15}\,erg\,s^{-1}\,cm^{-2})$', 
		r'F(OIII) $(10^{-15}\,erg\,s^{-1}\,cm^{-2})$', 
		r'$Mg\,b\ (\AA)$']

	for galaxy in ['ic1459', 'ic4296', 'ngc1399']:
		print '     '+galaxy
		Prefig(size=np.array((2, 2))*7)
		fig, ax = plt.subplots(2,2, sharex=True)

		DV = Data(galaxy, instrument='vimos', opt='pop')
		DM = Data(galaxy, instrument='muse', opt='pop')

		DV_kin = Data(galaxy, instrument='vimos', opt='kin')
		DM_kin = Data(galaxy, instrument='muse', opt='kin')

		for i, p in enumerate(plots):
			try:
				if "absorption_line" in p:
					ab, err = eval('DM.'+p)
					ax.flatten()[i].errorbar(np.sqrt(DM.xBar**2+DM.yBar**2)*0.2, 
						ab, fmt='x', label='MUSE', c='b', yerr=err)
				elif "sigma" in p:
					ax.flatten()[i].errorbar(
						np.sqrt(DM_kin.xBar**2+DM_kin.yBar**2)*0.2, 
						eval('DM_kin.'+p), fmt='x', label='MUSE', c='b',
						yerr=eval('DM_kin.'+p+'.uncert'))
				else:
					# ax.flatten()[i].errorbar(np.sqrt(DM.xBar**2+DM.yBar**2)*0.2, 
					# 	eval('DM.'+p)/10**5/DM.n_spaxels_in_bin/0.2**2, 
					# 	fmt='x', c='b', label='MUSE',
					# 	yerr=eval('DM.'+p+'.uncert')/10**5/DM.n_spaxels_in_bin
					# 	/0.2**2)
					ax.flatten()[i].errorbar(np.sqrt(DM.xBar**2+DM.yBar**2)*0.2, 
						eval('DM.'+p)/10**5/0.2**2, 
						fmt='x', c='b', label='MUSE',
						yerr=eval('DM.'+p+'.uncert')/10**5
						/0.2**2)
			except:
				# pass
				print 'MUSE', p
			try:
				if "absorption_line" in p:
					ab, err = eval('DV.'+p)
					ax.flatten()[i].errorbar(np.sqrt(DV.xBar**2+DV.yBar**2)*0.67, 
						ab, fmt='.', label='VIMOS', yerr=err, c='r')
				elif "sigma" in p:
					ax.flatten()[i].errorbar(
						np.sqrt(DV_kin.xBar**2+DV_kin.yBar**2)*0.67, 
						eval('DV_kin.'+p), fmt='.', label='VIMOS', c='r',
						yerr=eval('DV_kin.'+p+'.uncert'))
				else:
					# ax.flatten()[i].errorbar(np.sqrt(DV.xBar**2+DV.yBar**2)*0.67, 
					# 	eval('DV.'+p)/DV.n_spaxels_in_bin/0.67**2, 
					# 	fmt='.', c='r', label='VIMOS',
					# 	yerr=eval('DV.'+p+'.uncert')/DV.n_spaxels_in_bin/0.67**2)
					ax.flatten()[i].errorbar(np.sqrt(DV.xBar**2+DV.yBar**2)*0.67, 
						eval('DV.'+p)/0.67**2, 
						fmt='.', c='r', label='VIMOS',
						yerr=eval('DV.'+p+'.uncert')/0.67**2)

			except:
				# pass		
				print 'VIMOS', p

		# ax[1,1].scatter(np.nan, np.nan, marker='.', 
		# 	label='VIMOS' + ' ' + galaxy.upper())
		# ax[1,1].scatter(np.nan, np.nan, marker='x', 
		# 	label='MUSE' + ' ' + galaxy.upper())
		for i, p in enumerate(plots):
			ax.flatten()[i].set_ylabel(str_plots[i])
		for a in ax[1,:]:
			a.set_xlabel('Radius (arcsec)')

		ax[1,1].set_ylim([3,9])
		# ax[1,1].axis('off')
		# h, l = ax[0,0].get_legend_handles_labels()
		# ax[1,1].legend(h,l)
		ax[1,1].legend()
		fig.tight_layout()
		fig.savefig('compare_'+galaxy+'.png', bbox_inches='tight', dpi=200)

	plt.close('all')

# Compare velocity fields
def compare_velfield():
	print 'Compare velocity fields'
	Prefig(size=(20,20))

	fig, ax = plt.subplots(3,3, sharex=True, sharey=True, 
		subplot_kw={'aspect':'equal'})

	galaxy_str = ['IC 1459', 'IC 4296', 'NGC 1399']
	for g, galaxy in enumerate(['ic1459', 'ic4296', 'ngc1399']):
		Dv = Data(galaxy, instrument='vimos', opt='kin')
		Dm = Data(galaxy, instrument='muse', opt='kin')

		from errors2 import get_dataCubeDirectory
		f = fits.open(get_dataCubeDirectory(galaxy))
		headerv = f[0].header
		f.close()
		from errors2_muse import get_dataCubeDirectory
		f = fits.open(get_dataCubeDirectory(galaxy))
		headerm = f[1].header
		f.close()

		offset={'RA':0.0014, 'DEC':0.0004}
		headerv['HIERARCH CCD1 ESO INS IFU RA'] += offset['RA']
		headerv['HIERARCH CCD1 ESO INS IFU DEC'] += offset['DEC']

		vv_temp = regrid(Dv.x, Dv.y, Dv.bin_num, Dv.components['stellar'].plot['vel'], 
			headerv, 10)

		vm = regrid(Dm.x, Dm.y, Dm.bin_num, Dm.components['stellar'].plot['vel'], 
			headerm, 3)
		
		out = np.zeros((51,51))
		for i in range(51):
			for j in range(51):
				vv = np.array(vm)*np.nan
				vv[i:400+i,j:400+j] = vv_temp
				out[i,j] = np.nansum(np.abs(vv-vm))

		w = np.where(out == np.min(out))
		i, j = w[0][0], w[1][0]
		vv = np.array(vm)*np.nan
		vv[i:400+i,j:400+j] = vv_temp

		from sauron_colormap import sauron

		from plot_results import set_lims
		vmin, vmax = set_lims(Dm.components['stellar'].plot['vel'])

		ax[0,g].imshow(vv, cmap=sauron, clim=[vmin, vmax], 
			extent=[-75*0.2, 75*0.2, -75*0.2, 75*0.2])
		res = np.array(vm - vv)
		ax[1,g].imshow(vm-np.nanmean(res), cmap=sauron, clim=[vmin, vmax],
			extent=[-75*0.2, 75*0.2, -75*0.2, 75*0.2])

		ax[2,g].imshow(res-np.nanmean(res), cmap=sauron, clim=[vmin, vmax],
			extent=[-75*0.2, 75*0.2, -75*0.2, 75*0.2])

		ax[0,g].set_title(galaxy_str[g], size='x-large')

	for a in ax.flatten():
		a.tick_params(top=True, bottom=True, left=True, right=True, 
			direction='in', which='major', length=10, width=3, labelsize='large')
		# a.tick_params(top=True, bottom=True, left=True, right=True, 
		# 	direction='in', which='minor', length=10, width=3)
	ax[2,1].set_xlabel(r'$\Delta$ RA (arcsec)')
	ax[1,0].set_ylabel(r'$\Delta$ Dec (arcsec)')

	fig.text(0.07, 0.75, 'VIMOS', va='center', ha='right', 
		rotation='vertical', size='x-large')
	fig.text(0.07, 0.5, 'MUSE', va='center', ha='right',
		rotation='vertical', size='x-large')
	fig.text(0.07, 0.23, 'MUSE - VIMOS', va='center', ha='right',
		rotation='vertical', size='x-large')

	fig.subplots_adjust(hspace=0.05, wspace=0.05)

	fig.savefig('compare_velfield.png', bbox_inches='tight', dpi=200)

def regrid(x, y, bin_num, vel, header, scale):
	# Much of this code is taken from plot_velfield_nointerp.py
	try:
		# VIMOS parameters
		header['CRVAL1'] = header['HIERARCH CCD1 ESO INS IFU RA']
		header['CRVAL2'] = header['HIERARCH CCD1 ESO INS IFU DEC']
		header['CD1_1'] = -abs(header['CDELT1']/(60**2))
		header['CD2_2'] = header['CDELT2']/(60**2)
	except KeyError:
		header['CD1_1'] = -abs(header['CD1_1'])
		#pass # MUSE has the correctly labelled headers

	x, y = x.astype(float), y.astype(float)
	from scipy.spatial import distance
	pixelSize = np.min(distance.pdist(np.column_stack([x, y])))

	xmin = 0
	xmax = header['NAXIS1']
	ymin = 0
	ymax = header['NAXIS2']

	nx = int(round((xmax - xmin)/pixelSize))
	ny = int(round((ymax - ymin)/pixelSize))
	img = np.full((nx, ny), np.nan)  # use nan for missing data
	j = np.round((x - xmin)/pixelSize).astype(int)-1
	k = np.round((y - ymin)/pixelSize).astype(int)-1
	img[j, k] = vel[bin_num]
	img = zoom(img, scale, order=0)
	return img

if __name__=='__main__':
	if 'home' in cc.device:
		# for galaxy in ['eso443-g024', 'ic1459', 'ic1531', 'ic4296', 
		# 	'ngc1399', 'ngc3100', 'ngc3557', 'ngc7075']:
		# 	# populations_corner('eso443-g024', instrument='vimos')
		# 	absorption_plots(galaxy, instrument='vimos')
		# single_absorption_plot(instrument='vimos')


		# compare()

		# show_spectra('ic1459', [0,100,400], instrument='vimos', opt='pop')
		# show_spectra('ic1459', [0,494,2522], instrument='muse', opt='pop')
		# show_spectra('ic4296', [0,100,630], instrument='vimos', opt='pop')
		# show_spectra('ic4296', [0,414,279], instrument='muse', opt='pop')

		compare_velfield()

	elif cc.device == 'uni':
		for galaxy in ['ic1459', 'ic4296', 'ngc1316', 'ngc1399']:
			# populations_corner('eso443-g024', instrument='muse')
			absorption_plots(galaxy, instrument='muse')