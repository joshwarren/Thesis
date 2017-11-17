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

def plot(galaxies, str_galaxies, file_name):
	opt = 'pop'
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




		vin_dir += '/%s/%s' % (galaxy, opt) 

		pickle_file = '%s/pickled' % (vin_dir)
		pickleFile = open("%s/dataObj.pkl" % (pickle_file), 'rb')
		D = pickle.load(pickleFile)
		pickleFile.close()

		vin_dir += '/pop'

		f = fits.open(get_dataCubeDirectory(galaxy))
		header = f[0].header
		f.close()

		


		age = np.zeros(D.number_of_bins)
		met = np.zeros(D.number_of_bins)
		alp = np.zeros(D.number_of_bins)
		unc_age = np.zeros(D.number_of_bins)
		unc_met = np.zeros(D.number_of_bins)
		unc_alp = np.zeros(D.number_of_bins)

		for j in xrange(D.number_of_bins):
			ag, me, al = np.loadtxt('%s/distribution/%i.dat' % (
				vin_dir, j), unpack=True)

			for plot, unc_plot, pop in zip([age,met,alp],
				[unc_age,unc_met,unc_alp], [ag,me,al]):

				hist = np.histogram(pop, bins=40)
				x = (hist[1][0:-1]+hist[1][1:])/2
				hist = hist[0]
				plot[j] = x[np.argmax(hist)]

				gt_fwhm = hist >= np.max(hist)/2
				unc_plot[j] = np.max(x[gt_fwhm]) - np.min(x[gt_fwhm])

		# Age
		axs[2*i,0] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
			D.xBar, D.yBar, age, header,  
			vmin=0, vmax=15, 
			cmap='gnuplot2', flux_unbinned=D.unbinned_flux, 
			signal_noise=D.SNRatio, signal_noise_target=SN_target, 
			ax=axs[2*i,0])
		if overplot:
			for o, color in overplot.iteritems():
				add_(o, color, axs[2*i,0], galaxy, nolegend=True)
		

		axs[2*i+1,0] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
			D.xBar, D.yBar, unc_age, header, vmin=0, vmax=15, 
			cmap='gnuplot2', flux_unbinned=D.unbinned_flux, 
			signal_noise=D.SNRatio, 
			signal_noise_target=SN_target, ax=axs[2*i+1,0])

		# Metalicity
		axs[2*i,1] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
			D.xBar, D.yBar, met, header, vmin=-2.25, vmax=0.67, 
			cmap='gnuplot2', 
			flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
			signal_noise_target=SN_target, ax=axs[2*i,1])
		if overplot:
			for o, color in overplot.iteritems():
				add_(o, color, axs[2*i,1], galaxy, nolegend=True)


		axs[2*i+1,1] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, 
			D.yBar, unc_met, header, vmin=0, vmax=0.67+2.25, cmap='gnuplot2', 
			flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
			signal_noise_target=SN_target, ax=axs[2*i+1,1])

		# Alpha
		axs[2*i,2] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar, 
			D.yBar, alp, header, vmin=-0.3, vmax=0.5, cmap='gnuplot2', 
			flux_unbinned=D.unbinned_flux, signal_noise=D.SNRatio, 
			signal_noise_target=SN_target, ax=axs[2*i,2])
		if overplot:
			for o, color in overplot.iteritems():
				add_(o, color, axs[2*i,2], galaxy, nolegend=True)


		axs[2*i+1,2] = plot_velfield_nointerp(D.x, D.y, D.bin_num, D.xBar,
			D.yBar, unc_alp, header, vmin=0, vmax=0.5+0.3, cmap='gnuplot2', 
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


	fig.text(0.24, 0.9, r'Age', va='top', ha='center', size='xx-large')
	fig.text(0.51, 0.9, r'Metalicity', va='top', ha='center', size='xx-large')
	fig.text(0.8, 0.9, r'Alpha-enhancement', va='top', ha='center', 
		size='xx-large')

	if len(galaxies) == 1:
		fig.text(0.07, 0.5, str_galaxies[0], va='center', ha='right', 
			rotation='vertical', size='xx-large')
	if len(galaxies) == 2:
		raise ValueError('Not yet coded in location of galaxy labels')
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
	cbar = plt.colorbar(axs[0,0].cs, cax=cax)
	cbar.ax.set_yticklabels([])

	fig.savefig('%s/%s.png' % (out_dir, file_name), bbox_inches='tight')



if __name__=='__main__':
	plot(['eso443-g024', 'ic1459', 'ic1531'], 
		['ESO 443-G24', 'IC 1459', 'IC 1531'], 'pop1')

	plot(['ic4296', 'ngc0612', 'ngc1399'], 
		['IC 4296', 'NGC 612', 'NGC 1399'], 'pop2')

	plot(['ngc3100', 'ngc3557', 'ngc7075'], 
		['NGC 3100', 'NGC 3557', 'NGC 7075'], 'pop3')

	plot(['pks0718-34'], ['PKS 0718-34'], 'pop4')
