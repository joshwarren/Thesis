# import cPickle as pickle
from Bin2 import Data
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


def plot(galaxies, str_galaxies):
	opt = 'pop'
	overplot={'CO':'w', 'radio':'g'}
	Prefig(size=np.array((4, 3))*10)
	fig, axs = plt.subplots(3, 4)#, sharex=True, sharey=True)
	out_dir = '%s/Documents/thesis/chapter5/vimos' % (cc.home_dir)


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
	# 		self.components = {'stellar':comp(),'[OIII]5007d':comp()}
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


	axs.flatten()[8].remove()
	axs.flatten()[11].remove()
	for j, i in enumerate([0,1,2,3,4,5,6,7,9,10]):
		galaxy = galaxies[j]
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

		# attr, vmin, vmax = np.loadtxt('%s/lims.txt' % (vin_dir), dtype=str, 
		# 	usecols=(0,1,2), skiprows=1, unpack=True)
		# vmin, vmax = vmin.astype(float), vmax.astype(float)


		vin_dir += '/%s/%s' % (galaxy, opt) 

		# pickle_file = '%s/pickled' % (vin_dir)
		# pickleFile = open("%s/dataObj.pkl" % (pickle_file), 'rb')
		# D = pickle.load(pickleFile)
		# pickleFile.close()
		D = Data(galaxy, instrument='vimos', opt=opt)

		f = fits.open(get_dataCubeDirectory(galaxy))
		header = f[0].header
		f.close()

		if '[OIII]5007d' in D.components.keys():
			axs.flatten()[i] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
				D.xBar, D.yBar, D.components['[OIII]5007d'].flux, header,  
				# vmin=vmin[attr==plots[0]], vmax=vmax[attr==plots[0]], 
				cmap=sauron, flux_unbinned=D.unbinned_flux, 
				signal_noise=D.e_line['[OIII]5007d'].amp_noise, 
				signal_noise_target=4, #galaxy_labelcolor='w',
				ax=axs.flatten()[i])#, galaxy=str_galaxies[j])

			fig.text(axs.flatten()[i].ax_dis.get_position().x0+0.02, 
				axs.flatten()[i].ax_dis.get_position().y1-0.02, galaxy.upper(), 
				va='top', color='w', zorder=15, size='large')

			if overplot:
				for o, color in overplot.iteritems():
					scale = 'log' if o == 'radio' else 'lin'
					add_(o, color, axs.flatten()[i], galaxy, nolegend=True, 
						scale=scale)
		else:
			pass

		# if galaxy == 'ngc0612':
		# 	Prefig()
		# 	fig2, ax2 = plt.subplots()
		# 	ax2 = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
		# 		D.xBar, D.yBar, D.components['Hbeta'].flux, header,  
		# 		# vmin=vmin[attr==plots[0]], vmax=vmax[attr==plots[0]], 
		# 		cmap=sauron, flux_unbinned=D.unbinned_flux, 
		# 		signal_noise=D.e_line['Hbeta'].amp_noise, 
		# 		signal_noise_target=4, galaxy_labelcolor='w',
		# 		galaxy=str_galaxies[j], ax=ax2)
		# 	if overplot:
		# 		for o, color in overplot.iteritems():
		# 			add_(o, color, ax2, galaxy, nolegend=True)

			
		# 	ax2.ax_dis.tick_params(top=True, bottom=True, left=True, 
		# 		right=True, direction='in', which='major', length=20,
		# 		width=3, labelsize='large')
		# 	ax2.ax_dis.tick_params(top=True, bottom=True, left=True, 
		# 		right=True, direction='in', which='minor', length=10,
		# 		width=3)
		# 	ax2.ax_dis.xaxis.label.set_size(22)
		# 	ax2.ax_dis.yaxis.label.set_size(22)

		# 	# Add colorbar
		# 	ax_loc = ax2.get_position()
		# 	cax = fig2.add_axes([ax_loc.x1+0.03, ax_loc.y0, 0.02, ax_loc.height])
		# 	cbar = plt.colorbar(ax2.cs, cax=cax)
		# 	cbar.ax.set_yticklabels([])

		# 	fig2.savefig('%s/ngc0612_Hb.png' % (out_dir))

	for a in axs.flatten()[[1,2,3,5,6,7,10]]:
		if hasattr(a, 'ax_dis'):
			a.ax_dis.tick_params(top=True, bottom=True, left=True, 
				right=True, direction='in', which='major', length=20,
				width=3, labelsize='large')
			a.ax_dis.tick_params(top=True, bottom=True, left=True, 
				right=True, direction='in', which='minor', length=10,
				width=3)
			a.ax_dis.xaxis.label.set_size(22)
			a.ax_dis.yaxis.label.set_size(22)

	for a in axs.flatten()[[1,2,3,5,6,7,10]]:
		if hasattr(a, 'ax_dis'): 
			a.ax_dis.set_yticklabels([])
			a.ax_dis.set_ylabel('')

	for a in axs.flatten()[[0,1,2,3,5,6]]:
		if hasattr(a, 'ax_dis'): 
			a.ax_dis.set_xticklabels([])
			a.ax_dis.set_xlabel('')

	# Add colorbar
	ax_loc = axs[0,3].get_position()
	cax = fig.add_axes([ax_loc.x1+0.03, ax_loc.y0, 0.02, ax_loc.height])
	cbar = plt.colorbar(axs[1,1].cs, cax=cax)
	cbar.ax.set_yticklabels([])



	fig.savefig('%s/Hb.png' % (out_dir), bbox_inches='tight', dpi=240)



if __name__=='__main__':
	plot(['eso443-g024', 'ic1459', 'ic1531', 'ic4296', 'ngc0612', 
		'ngc1399', 'ngc3100', 'ngc3557', 'ngc7075', 'pks0718-34'], 
		['ESO 443-G24', 'IC 1459', 'IC 1531', 'IC 4296', 'NGC 612',
		'NGC 1399', 'NGC 3100', 'NGC 3557', 'NGC 7075', 'PKS 718-34'])
