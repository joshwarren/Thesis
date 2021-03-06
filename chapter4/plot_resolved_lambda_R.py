import numpy as np 
from astropy.io import fits
from checkcomp import checkcomp
cc = checkcomp()
if cc.remote:
	import matplotlib # 20160202 JP to stop lack-of X-windows error
	matplotlib.use('Agg') # 20160202 JP to stop lack-of X-windows error
	import matplotlib.pyplot as plt # used for plotting
else:
	import matplotlib.pyplot as plt # used for plotting
from errors2 import get_dataCubeDirectory
from classify import get_R_e
from prefig import Prefig
Prefig()

def label(galaxy):
	if 'ic' in galaxy:
		return 'IC ' + galaxy[2:]
	elif galaxy == 'ngc0612':
		return 'NGC 612'
	elif 'ngc' in galaxy:
		return 'NGC ' + galaxy[3:]
	elif galaxy == 'pks0718-34':
		return 'PKS 718-34'
	elif galaxy == 'eso443-g024':
		return 'ESO 443-G24'

def plot_lambda_R():
	fig, ax = plt.subplots()
	FR = ['ic1531', 'ngc0612', 'ngc1316', 'ngc3100', 'ngc3557', 
		'pks0718-34']
	for galaxy in ['eso443-g024',
				'ic1459',
				'ic1531', 
				'ic4296',
				'ngc0612',
				'ngc1399',
				'ngc3100',
				'ngc3557',
				'ngc7075',
				'pks0718-34']:
		opt = 'kin'

		lam_R_file = '%s/Data/vimos/analysis/%s/%s/lambda_R.txt' % (
			cc.base_dir, galaxy, opt)
		r, lam_R = np.loadtxt(lam_R_file, unpack=True)
		r /= get_R_e(galaxy)

		if galaxy in FR:
			ax.plot(r, lam_R, 'r')
		else:
			ax.plot(r, lam_R, 'r--')

		ax.text(r[-1], lam_R[-1], label(galaxy), color='r', 
			ha='center', va='bottom', size='x-small')

	for galaxy in ['ic1459', 'ic4296', 'ngc1316', 'ngc1399']:
		opt = 'pop_no_Na' if galaxy == 'ngc1316' else 'kin'

		lam_R_file = '%s/Data/muse/analysis/%s/%s/lambda_R.txt' % (
			cc.base_dir, galaxy, opt)
		r, lam_R = np.loadtxt(lam_R_file, unpack=True)
		# print r
		# print get_R_e(galaxy)
		# asdas
		# r /= get_R_e(galaxy)

		if galaxy in FR:
			ax.plot(r, lam_R, 'b')
		else:
			ax.plot(r, lam_R, 'b--')

		if galaxy == 'ngc1399':
			va = 'top'
		else: 
			va='bottom'
		ax.text(r[-1], lam_R[-1], label(galaxy), color='b', 
			ha='center', va=va, size='x-small')

	ax.set_xlabel(r'Radius (R$_e$)')
	ax.set_ylabel(r'$\lambda_R$')

	fig.savefig('%s/Documents/thesis/chapter4/lambda_R.png' % (cc.home_dir), dpi=240)

if __name__=='__main__':
	plot_lambda_R()






















