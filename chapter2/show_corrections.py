# Routine to produce plots to demonstrate correction.py changes

import numpy as np
from astropy.io import fits
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import convolve2d
from tools import all_neighbours
from checkcomp import checkcomp
cc = checkcomp()
from prefig import Prefig 
import matplotlib.pyplot as plt
from plot_velfield_nointerp import plot_velfield_nointerp 
import matplotlib.gridspec as gridspec

Prefig(subplots=(1.5,1))
gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1, 1],
                       height_ratios=[2, 1]
                       )

ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[1, :])




# for galaxy in ['eso443-g024',
# 			'ic1459',
# 			'ic1531', 
# 			'ic4296',
# 			'ngc0612',
# 			'ngc1399',
# 			'ngc3100',
# 			'ngc3557',
# 			'ngc7075',
# 			'pks0718-34']:
for galaxy in ['ngc3557']:
	f = fits.open('%s/Data/vimos/cubes/%s.cube.combined.fits' % (cc.base_dir, 
		galaxy))

	
	# fig, ax = plt.figure()

	x, y = np.meshgrid(np.arange(40),np.arange(40))
	x, y = x.flatten(), y.flatten()

	image = np.nansum(f[0].data, axis=(0,))


	ax1 = plot_velfield_nointerp(x, y, np.arange(40**2), x, y,
		image.flatten(), f[0].header, cmap='hot', ax=ax1)

	ax1.ax_dis.text(10,10, 'Before',zorder=19, color='w')
	# ax[0].contour(image)


	c = fits.open('%s/Data/vimos/cubes/%s.cube.combined.corr.fits' % (
		cc.base_dir, galaxy))
	c_image = np.nansum(c[0].data, axis=(0,))

	ax2 = plot_velfield_nointerp(x, y, np.arange(40**2), x, y,
		c_image.flatten(), c[0].header, cmap='hot', ax=ax2)
	ax2.ax_dis.text(10,10, 'After', zorder=19, color='w')

	# ax[1].contour(c_image)
	ax2.ax_dis.set_yticklabels([])
	ax2.ax_dis.set_ylabel('')

	for a in [ax1, ax2]:
		if hasattr(a, 'ax_dis'):
			a.ax_dis.tick_params(top=True, bottom=True, left=True, 
				right=True, direction='in', which='major', length=15,
				width=2, labelsize='large', color='w')#, labeltop=True, 
				# labelbottom=False)
			a.ax_dis.tick_params(top=True, bottom=True, left=True, 
				right=True, direction='in', which='minor', length=7,
				width=2, color='w')
			# a.ax_dis.xaxis.set_label_position('top')

	# plt.show()


bin_x = 25#26
bin_y = 12#12
# for galaxy in ['eso443-g024',
# 			'ic1459',
# 			'ic1531', 
# 			'ic4296',
# 			'ngc0612',
# 			'ngc1399',
# 			'ngc3100',
# 			'ngc3557',
# 			'ngc7075',
# 			'pks0718-34']:
for galaxy in ['ngc3100']:
	f = fits.open('%s/Data/vimos/cubes/%s.cube.combined.fits' % (cc.base_dir, 
		galaxy))

	lam = np.arange(f[0].header['NAXIS3'])*f[0].header['CDELT3'] \
		+ f[0].header['CRVAL3']
	spec = f[0].data[:,bin_x,bin_y]

	# Prefig(size=(20,10))
	# fig,ax = plt.subplots()

	ax3.plot(lam, spec/np.median(spec)+3, 'k')
	ax3.text(4650, 3.5, 'Original spectrum')

	c = fits.open('%s/Data/vimos/cubes/%s.cube.combined.corr.fits' % (
		cc.base_dir, galaxy))
	c_spec = c[0].data[:,bin_x,bin_y]

	ax3.plot(lam, c_spec/np.median(c_spec), 'r')
	ax3.text(4650, 0.5, 'Corrected spectrum', color='r')
	
	median_spec = np.nanmedian(f[0].data[:, 
		[bin_x-1, bin_x-1, bin_x-1, bin_x, bin_x, bin_x+1, bin_x+1, bin_x+1], 
		[bin_y-1, bin_y, bin_y+1, bin_y-1, bin_y+1, bin_y-1, bin_y, bin_y+1]], 
		axis=1)

	correction = spec/median_spec

	y_new = lowess(correction, np.arange(f[0].header['NAXIS3']), 
		frac=150./f[0].header['NAXIS3'], it=2, return_sorted=False)

	ax3.plot(lam, y_new/np.median(y_new)+1.5, 'b')
	ax3.text(4650, 2, 'Correction spectrum', color='b')
	ax3.set_xlabel(r'Wavelength ($\AA$)')
	ax3.set_ylabel('Normalised Flux\n+ constant')

	fig = plt.gcf()
	fig.savefig('%s/Documents/thesis/chapter2/corr_image.png' %(cc.home_dir), 
		bbox_inches='tight', dpi=260)
	plt.close()

if False:
	tl=[]
	tr=[]
	lt=[]
	lb=[]
	rt=[]
	rb=[]
	bl=[]
	br=[]
	for galaxy in [#'eso443-g024',
		# 'ic1459',
		# 'ic1531', 
		# 'ic4296',
		# 'ngc0612',
		# 'ngc1399',
		# 'ngc3100',
		'ngc3557',
		# 'ngc7075',
		# 'pks0718-34'
		]:

		f = fits.open('%s/Data/vimos/cubes/%s.cube.combined.fits' % (
			cc.base_dir, galaxy))
		image = np.nansum(f[0].data, axis=0)

		# Q1.append(np.nansum(image[:20,:20]))
		# Q2.append(np.nansum(image[20:,:20]))
		# Q3.append(np.nansum(image[:20,20:]))
		# Q4.append(np.nansum(image[20:,20:]))

		# tl.append(np.nansum(image[:20,19]))
		# tr.append(np.nansum(image[:20,20]))
		# bl.append(np.nansum(image[20:,19]))
		# br.append(np.nansum(image[20:,20]))
		# lt.append(np.nansum(image[20,:20]))
		# lb.append(np.nansum(image[19,:20]))
		# rt.append(np.nansum(image[20,20:]))
		# rb.append(np.nansum(image[19,20:]))

		tl.append(image[15:20,19])
		tr.append(image[15:20,20])
		bl.append(image[20:25,19])
		br.append(image[20:25,20])
		lt.append(image[20,15:20])
		lb.append(image[19,15:20])
		rt.append(image[20,20:25])
		rb.append(image[19,20:25])


	tl=np.array(tl)
	tr=np.array(tr)
	lt=np.array(lt)
	lb=np.array(lb)
	rt=np.array(rt)
	rb=np.array(rb)
	bl=np.array(bl)
	br=np.array(br)



	print bl.shape, np.nanmax([tl,tr],axis=0).shape


	print 'top:', np.nanmean(np.abs(tl-tr)/np.nanmax([tl,tr],axis=0),axis=1)
	print 'bottom:', np.nanmean(np.abs(bl-br)/np.nanmax([bl,br],axis=0),axis=1)
	print 'left:', np.nanmean(np.abs(lt-lb)/np.nanmax([lt,lb],axis=0),axis=1)
	print 'right:', np.nanmean(np.abs(rt-rb)/np.nanmax([rt,rb],axis=0),axis=1)


	im = np.array(image)*0
	im[15:20,19] = tl[0]
	im[15:20,20] = tr[0]
	im[20:25,19] = bl[0]
	im[20:25,20] = br[0]
	plt.imshow(im)
	plt.show()

	print np.abs(tl-tr)/tl



	# x, y = np.meshgrid(np.arange(11, 29),np.arange(11,29))
	# x, y = x.flatten(), y.flatten()
	# rag = []
	# for bin_x,bin_y in zip(x,y):
	# 	median_spec = np.nanmedian(f[0].data[:, 
	# 		[bin_x-1, bin_x-1, bin_x-1, bin_x, bin_x, bin_x+1, bin_x+1, bin_x+1], 
	# 		[bin_y-1, bin_y, bin_y+1, bin_y-1, bin_y+1, bin_y-1, bin_y, bin_y+1]], 
	# 		axis=1)

	# 	correction = spec/median_spec

	# 	y_new = lowess(correction, np.arange(f[0].header['NAXIS3']), 
	# 		frac=150./f[0].header['NAXIS3'], it=2, return_sorted=False)

	# 	rag.append(np.ptp(y_new))

	# i = np.argmax(rag)
	# print galaxy, rag[i], x[i], y[i]


