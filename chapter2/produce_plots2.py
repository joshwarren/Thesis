# Routine to produce plots for chapter2 required by the examiners. 
from checkcomp import checkcomp
cc = checkcomp()
import numpy as np
if 'home' not in cc.device:
	import matplotlib # 20160202 JP to stop lack-of X-windows error
	matplotlib.use('Agg') # 20160202 JP to stop lack-of X-windows error
	import matplotlib.pyplot as plt
else:
	import matplotlib.pyplot as plt 
from Bin2 import Data
from prefig import Prefig
from astropy.io import fits
from scipy.ndimage import zoom
from plot_velfield_nointerp import correct_header,plot_velfield_nointerp
from errors2_muse import get_dataCubeDirectory, run_ppxf, set_params
from plot_results import set_lims


# Copied from ../chapter4/produce_plots2.py
def regrid(x, y, bin_num, vel, header, scale):
	# Much of this code is taken from plot_velfield_nointerp.py
	header = correct_header(header)

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



def compare_hst(galaxy, instrument='vimos'):
	print 'Compare ' + instrument.upper() + ' ' + galaxy.upper() + ' to HST image'
	if instrument == 'vimos':
		ext = 0
		from errors2 import get_dataCubeDirectory
		scale = 1
		spectral_res = 0.7
	elif instrument == 'muse':
		ext = 1
		from errors2_muse import get_dataCubeDirectory
		scale = 10**-5
		spectral_res = 1.25
	Prefig()
	# D = Data(galaxy, instrument=instrument, opt='kin')

	# Load cube header
	f = fits.open(get_dataCubeDirectory(galaxy))
	cube_header = f[ext].header
	cube_header = correct_header(cube_header)
	lam = np.arange(cube_header['NAXIS3'])*cube_header['CD3_3'] \
		+ cube_header['CRVAL3']
	# img = np.trapz(f[ext].data, x=lam, axis=0) * scale 
	img = np.nansum(f[ext].data,axis=0) * scale * spectral_res
	f.close()

	# Load HST image and header
	f = fits.open(get_dataCubeDirectory(galaxy).hst)
	hst = f[1].data
	hst = hst[::-1, :]
	hst_header = f[1].header
	# changes into 10^-15 erg/s/cm^2/A
	# hst *= hst_header['PHOTFLAM']/10**-15 *1000 
	# hst *= hst_header['PHOTFLAM']/10**-15 *1729/0.159 
	hst *= 3.396e-18/10**-15 *1729 / 0.159
	# F555W bandpass is 1720A wide. 
	# EXPTIME is 1000 sec
	# 0.159 is the throughput fraction for F555W?
	# 3.396e-18 is the 'new' value of PHOTFLAM (2002) but our obs are 2009...
	f.close()

	if instrument == 'vimos':
		scale = 20
	elif instrument == 'muse':
		scale = 12
	# img = zoom(img, scale, order=0)
	# hst = zoom(hst, 3, order=0)
	# hst = zoom(hst, 1./scale, order=0)

	# hst = hst[53:93,60:100]
	print np.max(hst), np.nanmax(img), np.where(hst==np.nanmax(hst))
	# plt.imshow(hst-(img/np.nanmax(img)*np.max(hst)))
	s = img.shape
	r = np.sqrt((np.arange(s[0]).repeat(s[1])-s[0]/2)**2 
		+ (np.tile(np.arange(s[1]),s[0])-s[1]/2)**2)
	r_hst = np.sqrt((np.arange(hst.shape[0]).repeat(hst.shape[1])
			- np.where(hst==np.nanmax(hst[400:600, 400:600]))[0][0])**2 
		+ (np.tile(np.arange(hst.shape[1]),hst.shape[0])
			- np.where(hst==np.nanmax(hst[400:600, 400:600]))[1][0])**2)\
		* 3./scale
	m = r_hst < 70
	hst = hst.flatten()[m]
	r_hst = r_hst[m]
	# plt.imshow(r.reshape((40,40)))
	plt.scatter(r_hst.flatten(), hst.flatten(), marker='.', label='HST')
	plt.scatter(r.flatten(), img.flatten(),#/np.nanmax(img)*np.max(hst), 
		marker='.', label=instrument.upper())
	plt.legend()



	plt.show()



	djshadjk

	hst = hst[1000:2000, 1000:2000] # Crop HST image
	hst *= np.nanmax(img)/np.nanmax(hst)

	out = np.zeros((100,100))*np.nan
	for i in range(out.shape[0]):
		for j in range(out.shape[1]):
			out[i,j] = np.nanmean(np.abs(hst[
				hst.shape[0]/2 - img.shape[0]/2 - out.shape[0]/2 + i:
				hst.shape[0]/2 + img.shape[0]/2 - out.shape[0]/2 + i,
				hst.shape[1]/2 - img.shape[1]/2 - out.shape[1]/2 + j:
				hst.shape[1]/2 + img.shape[1]/2 - out.shape[1]/2 + j] - img))
	w = np.where(out == np.nanmin(out))

	res = np.array(hst) * np.nan
	res[hst.shape[0]/2 - img.shape[0]/2 - out.shape[0]/2 + w[0][0]:
		hst.shape[0]/2 + img.shape[0]/2 - out.shape[0]/2 + w[0][0],
		hst.shape[1]/2 - img.shape[1]/2 - out.shape[1]/2 + w[1][0]:
		hst.shape[1]/2 + img.shape[1]/2 - out.shape[1]/2 + w[1][0]] = \
		hst[hst.shape[0]/2 - img.shape[0]/2 - out.shape[0]/2 + w[0][0]:
		hst.shape[0]/2 + img.shape[0]/2 - out.shape[0]/2 + w[0][0],
		hst.shape[1]/2 - img.shape[1]/2 - out.shape[1]/2 + w[1][0]:
		hst.shape[1]/2 + img.shape[1]/2 - out.shape[1]/2 + w[1][0]] - img

	plt.imshow(res)

	plt.imshow(hst, alpha=0.8, vmax=400)

	plt.show()












	# print hst.shape, hst_header['CD2_2']*60**2
	# print img.shape, cube_header['CD2_2']*60**2

	# fig, ax = plt.subplots()
	# ax.imshow(img, extent=np.append(np.array([cube_header['NAXIS1'], 0]) 
	# 	* cube_header['CD1_1'] + cube_header['CRVAL1'], 
	# 	np.array([0, cube_header['NAXIS2']])* cube_header['CD2_2'] 
	# 	+ cube_header['CRVAL2']))

	# ax.imshow(np.log(hst[::-1,:]), 
	# 	extent=np.append(np.array([hst_header['NAXIS1'], 0]) 
	# 	* hst_header['CD1_1'] + hst_header['CRVAL1'], np.array([0,
	# 	hst_header['NAXIS2']])* hst_header['CD2_2'] + hst_header['CRVAL2']))

	# plt.show()








def test():
	a = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]).astype(float)
	# a = zoom(a, 0.5, order=1)
	from scipy.stats import binned_statistic_2d

	a, _, _, _ = binned_statistic_2d(np.tile(np.arange(4),4),
		np.arange(4).repeat(4),a.flatten(),bins=(2,2), statistic=np.nanmean)
	print a







def ngc1316_NaD_effect():
	bin = 918
	D = Data('ngc1316', instrument='muse', opt='kin')

	tessellation_File = "/Data/muse/analysis/ngc1316/pop_no_Na/"+\
		"setup/voronoi_2d_binning_output.txt"

	x,y,bin_num = np.loadtxt(tessellation_File, usecols=(0,1,2), \
	    unpack=True, skiprows=1).astype(int)#, dtype='int,int,int')

	n_bins = max(bin_num) + 1
	## Contains the order of the bin numbers in terms of index number.
	order = np.sort(bin_num)
	## ----------========= Reading the spectrum  ===============---------

	dataCubeDirectory = get_dataCubeDirectory('ngc1316')

	f = fits.open(dataCubeDirectory)
	galaxy_data, header = f[1].data, f[1].header
	galaxy_noise = f[2].data

	## write key parameters from header - can then be altered in future	
	CRVAL_spec = header['CRVAL3']
	CDELT_spec = header['CD3_3']
	## ----------============= Spatially Binning ===============---------
	spaxels_in_bin = np.where(bin_num == bin)[0]
	n_spaxels_in_bin = len(spaxels_in_bin)

	bin_lin = np.nansum(galaxy_data[:,x[spaxels_in_bin],y[spaxels_in_bin]], 
	    axis=1)/n_spaxels_in_bin
	bin_lin_noise = np.nansum(galaxy_noise[:,x[spaxels_in_bin],
	    y[spaxels_in_bin]]**2, axis=1)
	bin_lin_noise = np.sqrt(bin_lin_noise)/n_spaxels_in_bin

	params = set_params(opt='kin', set_range_star=np.array([2000, 5800]), reps=0)

	pp_No_Na = run_ppxf('ngc1316', np.array(bin_lin), np.array(bin_lin_noise), 
		CDELT_spec, CRVAL_spec, params)

	params = set_params(opt='kin', reps=0)

	pp = run_ppxf('ngc1316', np.array(bin_lin), np.array(bin_lin_noise), 
		CDELT_spec, CRVAL_spec, params)



	fig, ax = plt.subplots()

	ax.plot(pp.lam, pp.galaxy, 'k', label='Spectrum')
	ax.plot(pp.lam, pp.bestfit, 'r--', label='Bestfit including NaD')
	ax.plot(pp_No_Na.lam, pp_No_Na.bestfit, 'g--', label='Bestfit excluding NaD')

	# ax.set_xlim([5250,5450])
	# ax.set_ylim([0.000007, 0.000012])
	ax.legend()












def compare_hst2(galaxy, instrument='vimos'):
	print 'HST comparsion to %s for %s' % (instrument, galaxy)
	from scipy import ndimage
	if instrument == 'vimos':
		ext = 0
		from errors2 import get_dataCubeDirectory
		scale = 1
		spectral_res = 0.7
	elif instrument == 'muse':
		ext = 1
		from errors2_muse import get_dataCubeDirectory
		scale = 10**-5
		spectral_res = 1.25
	Prefig()
	fig, ax = plt.subplots()

	f = fits.open(get_dataCubeDirectory(galaxy).hst)
	hst = f[1].data
	hst = hst[::-1, :]
	# hst = hst[::-1, ::-1]
	hst_header = f[1].header
	f.close()
	if instrument=='vimos':
		if galaxy == 'ngc3557':
			hst = ndimage.gaussian_filter(hst, sigma=1)
		else:
			hst = ndimage.gaussian_filter(hst, sigma=2)
	elif instrument=='muse':
		hst = ndimage.gaussian_filter(hst, sigma=3)

	f = fits.open(get_dataCubeDirectory(galaxy))
	cube_header = f[ext].header
	cube_header = correct_header(cube_header)
	f.close()

	# hst *= 3.396e-18/10**-15 *1729 / 0.159

	size = 200
	s = hst.shape
	extent = np.array([-size/2.,size/2.])*hst_header['CD1_1']*60*60

	extent = np.array([-size/2.,size/2.])*np.sqrt(hst_header['OCD1_1']**2 + 
		hst_header['OCD1_2']**2) * 60**2

	D = Data(galaxy, instrument=instrument, opt='kin')

	vmin, vmax = set_lims(D.flux, positive=True)
	ax = plot_velfield_nointerp(D.x,D.y,D.bin_num,D.xBar,D.yBar, D.flux, cube_header,
		colorbar=False, cmap='gist_yarg', vmax=vmax, vmin=vmin, ax=ax, 
		galaxy=galaxy.upper()) #, flux_unbinned=D.unbinned_flux)

	RA_cent = (cube_header['NAXIS1'] - D.center[0] - cube_header['CRPIX1'] #+1
		) * -1 * abs(cube_header['CD1_1']) + cube_header['CRVAL1']
	dec_cent = (cube_header['NAXIS2'] - D.center[1] - cube_header['CRPIX2'] #+1
		) * abs(cube_header['CD2_2']) + cube_header['CRVAL2']

	xlim = ax.get_xlim()
	ylim = ax.get_ylim()

	from find_galaxy import find_galaxy # part of mge package: fits photometry
	f = find_galaxy(hst, quiet=True, plot=False)

	extent=np.append(
		(np.array([0, hst_header['NAXIS1']]) - f.ymed) * 
			hst_header['CD1_1'] + RA_cent, # hst_header['CRVAL1'],
		(np.array([0, hst_header['NAXIS2']]) - f.xmed) * 
			hst_header['CD2_2'] + dec_cent) # hst_header['CRVAL2'])

	ax.contour(-2.5*np.log10(hst/np.max(hst)), extent=extent, levels=np.arange(20),
		colors='r')

	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	fig.savefig('hst_%s_%s.png' % (instrument, galaxy), bbox_inches='tight', dpi=240)



if __name__=='__main__':
	# compare_hst('ic1459', instrument='vimos')
	# test()
	# ngc1316_NaD_effect()
	for g in ['ic1459', 'ic4296', 'ngc1399', 'ngc3557']:
		compare_hst2(g, instrument='vimos')
	for g in ['ic1459', 'ic4296', 'ngc1316', 'ngc1399']:
		compare_hst2(g, instrument='muse')
