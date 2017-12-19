# Routine to plot comparison between absorption measurements are 2.5 and 3A. 

from checkcomp import checkcomp
cc = checkcomp()
if 'home' not in cc.device:
	import matplotlib # 20160202 JP to stop lack-of X-windows error
	matplotlib.use('Agg') # 20160202 JP to stop lack-of X-windows error
import cPickle as pickle
import matplotlib.pyplot as plt 
import numpy as np
from prefig import Prefig
Prefig(size=(10,10))


c = 299792.458 # speed of light in km/s

galaxy='ic1459'
opt='pop'

pickleFile = open('/Data/muse/analysis/%s/%s/pickled/dataObj.pkl' % (
	galaxy, opt))
D = pickle.load(pickleFile)
pickleFile.close()


fig, ax = plt.subplots()#subplot_kw={'aspect':'equal'})
for l in ['H_beta', 'Fe5015', 'Mg_b', 'Fe5270', 'Fe5335', 'Fe5406', 
	'Fe5709', 'Fe5782', 'NaD', 'TiO1', 'TiO2']:

	l_25 = D.absorption_line(l, res=2.5, instrument='muse')
	l_3 = D.absorption_line(l, res=3.0, instrument='muse')
	if l == 'H_beta':
		l = r'H\,$\beta$'
	if l == 'Mg_b':
		l = 'Mg\,b'
	if l == 'TiO1':
		l = 'TiO1 (mag)'
	if l == 'TiO2':
		l = 'TiO2 (mag)'
	ax.scatter(l_25, (l_3-l_25)/l_25, marker='.', label=l)
	print l, np.nanmean(np.abs(l_3-l_25)/np.max([l_3,l_25],axis=0)), \
		np.nanstd(np.abs(l_3-l_25)/np.max([l_3,l_25],axis=0))
ax.legend()
# lim = ax.get_xlim()
# ax.plot([0,10],[0,10], 'k', zorder=-1)
# ax.set_xlim(lim)
# ax.set_ylim(lim)
ax.axhline(0, c='k', zorder=10)
ax.axhline(0.01, c='k', ls=':', zorder=10)
ax.axhline(-0.01, c='k', ls=':', zorder=10)
ax.legend()
ax.set_xlabel(r'$I_\mathrm{2.5 \AA}$')
ax.set_ylabel(r'$(I_\mathrm{3.0 \AA} - I_\mathrm{2.5 \AA})\,/\,I_\mathrm{2.5 \AA}$')

fig.savefig('%s/Documents/thesis/chapter2/compare_resolutions.png' % (
	cc.home_dir), bbox_inches='tight', dpi=240)
