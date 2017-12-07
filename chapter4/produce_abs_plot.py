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
from plot_results import set_lims


from Bin import myArray

def plot(galaxies, str_galaxies, file_name):
	opt = 'kin'
	overplot={'CO':'c', 'radio':'g'}
	Prefig(size=np.array((len(galaxies)*2, 7))*7)
	fig, axs = plt.subplots(7, len(galaxies)*2)#, sharex=True, sharey=True)
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

	# 	def absorption_line(self, p, uncert=False):
	# 		if uncert:
	# 			return [0,1,1,2], [0,1,1,2]
	# 		else:
	# 			return [0,1,1,2]

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

		vin_dir += '/pop'

		f = fits.open(get_dataCubeDirectory(galaxy))
		header = f[0].header
		f.close()

		plots = [
			"absorption_line('G4300')",
			"absorption_line('Fe4383')",
			"absorption_line('Ca4455')",
			"absorption_line('Fe4531')",
			"absorption_line('H_beta')",
			"absorption_line('Fe5015')",
			"absorption_line('Mg_b')"
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


			if 'Mg_b' not in p or galaxy not in ['ngc0612', 'pks0718-34']:
				axs[j, 2*i] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
					D.xBar, D.yBar, eval('D.'+p), header,  
					vmin=vmin, vmax=vmax, 
					cmap='inferno', flux_unbinned=D.unbinned_flux, 
					signal_noise=D.SNRatio, signal_noise_target=SN_target, 
					ax=axs[j, 2*i])
				if overplot:
					for o, color in overplot.iteritems():
						add_(o, color, axs[j, 2*i], galaxy, nolegend=True)
			else:
				axs[j, 2*i].remove()

		plots = [
			"absorption_line('G4300',uncert=True)[1]",
			"absorption_line('Fe4383',uncert=True)[1]",
			"absorption_line('Ca4455',uncert=True)[1]",
			"absorption_line('Fe4531',uncert=True)[1]",
			"absorption_line('H_beta',uncert=True)[1]",
			"absorption_line('Fe5015',uncert=True)[1]",
			"absorption_line('Mg_b',uncert=True)[1]"
			]

		for j, p in enumerate(plots):
			if 'Mg_b' not in p or galaxy not in ['ngc0612', 'pks0718-34']:
				axs[j, 2*i+1] = plot_velfield_nointerp(D.x, D.y, D.bin_num, 
					D.xBar, D.yBar, eval('D.'+p), header,  
					vmin=0, vmax=0.5, 
					cmap='inferno', flux_unbinned=D.unbinned_flux, 
					signal_noise=D.SNRatio, signal_noise_target=SN_target, 
					ax=axs[j, 2*i+1])
			else:
				axs[j, 2*i+1].remove()



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


	if all([g not in ['ngc0612', 'pks0718-34'] for g in galaxies]):
		for a in axs[:-1,:].flatten():
			if hasattr(a, 'ax_dis'): 
				a.ax_dis.set_xticklabels([])
				a.ax_dis.set_xlabel('')

	if 'ngc0612' in galaxies:
		for a in axs[:-2,:].flatten():
			if hasattr(a, 'ax_dis'): 
				a.ax_dis.set_xticklabels([])
				a.ax_dis.set_xlabel('')
		for a in axs[-2,2:].flatten():
			if hasattr(a, 'ax_dis'): 
				a.ax_dis.set_xticklabels([])
				a.ax_dis.set_xlabel('')

	if 'pks0718-34' in galaxies:
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
	


	if len(galaxies) == 1:
		fig.text(0.5, 0.9, str_galaxies[0], va='top', ha='center', size='xx-large')
	elif len(galaxies) == 2:
		fig.text(0.33, 0.9, str_galaxies[0], va='top', ha='center', size='xx-large')
		fig.text(0.72, 0.9, str_galaxies[1], va='top', ha='center', size='xx-large')



	fig.text(0.07, 0.83, 'G4300', va='center', ha='right', 
		rotation='vertical', size='xx-large')
	fig.text(0.07, 0.72, 'Fe4383', va='center', ha='right',
		rotation='vertical', size='xx-large')
	fig.text(0.07, 0.61, 'Ca4455', va='center', ha='right',
		rotation='vertical', size='xx-large')
	fig.text(0.07, 0.5, 'Fe4531', va='center', ha='right',
		rotation='vertical', size='xx-large')
	fig.text(0.07, 0.39, r'H$_\beta$', va='center', ha='right',
		rotation='vertical', size='xx-large')
	fig.text(0.07, 0.27, 'Fe5015', va='center', ha='right',
		rotation='vertical', size='xx-large')
	fig.text(0.07, 0.16, r'Mg$_b$', va='center', ha='right',
		rotation='vertical', size='xx-large')

	# Add colorbar
	ax_loc = axs[0,3].get_position()
	cax = fig.add_axes([ax_loc.x1+0.03, ax_loc.y0, 0.02, ax_loc.height])
	cbar = plt.colorbar(axs[0,0].cs, cax=cax)
	cbar.ax.set_yticklabels([])

	fig.savefig('%s/%s.png' % (out_dir, file_name), bbox_inches='tight',
		dpi=60)



if __name__=='__main__':
	plot(['eso443-g024', 'ic1459'], ['ESO 443-G24', 'IC 1459'], 'abs1')

	plot(['ic1531', 'ic4296'], ['IC 1531', 'IC 4296'], 'abs2')

	plot(['ngc0612', 'ngc1399'], ['NGC 612', 'NGC 1399'], 'abs3')

	plot(['ngc3100', 'ngc3557'], ['NGC 3100', 'NGC 3557'], 'abs4')

	plot(['ngc7075', 'pks0718-34'], ['NGC7075', 'PKS 0718-34'], 'abs5')
