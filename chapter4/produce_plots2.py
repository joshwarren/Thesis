# Routine to produce plots from Viva corrections.
from checkcomp import checkcomp
cc = checkcomp()
import numpy as np
if 'home' not in cc.device:
	import matplotlib # 20160202 JP to stop lack-of X-windows error
	matplotlib.use('Agg') # 20160202 JP to stop lack-of X-windows error
import matplotlib.pyplot as plt 
from prefig import Prefig
from Bin2 import Data
import corner
from tools import myerrorbar

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

def add_grid(ax, xline, yline, alpha=0.5):
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









def absorption_plots(galaxy, instrument='vimos'):
	D = Data(galaxy, instrument=instrument, opt='pop')

	fig, ax = plt.subplots()
	# add_grid(ax, 'Hb', 'MgFe')
	add_grid(ax, 'Hb', 'Mgb')
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()


	Hb, e_Hb = D.absorption_line('H_beta', uncert=True)
	Mg, e_Mg = D.absorption_line('Mgb', uncert=True)
	r = np.sqrt((D.xBar - D.center[0])**2 + (D.yBar - D.center[1])**2)

	myerrorbar(ax, Hb, Mg, xerr=e_Hb, yerr=e_Mg, color=r, marker='.')

	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	plt.show()

	














if __name__=='__main__':
	# populations_corner('eso443-g024', instrument='vimos')
	absorption_plots('ic1459', instrument='vimos')