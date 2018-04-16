# Comparison between MUSE and VIMOS derived values as required by examiners


from prefig import Prefig
import matplotlib.pyplot as plt
from Bin2 import Data
import numpy as np

# Compare VIMOS and MUSE results
def ic1459():
	print 'Compare VIMOS and MUSE emission line results for IC1459'
	
	plots = ["components['[OIII]5007d'].flux", "components['Hbeta'].flux"]
	str_plots = [r'F(OIII) $(10^{-15}\,erg\,s^{-1}\,cm^{-2})$', 
		r'F(H$\beta)\ (10^{-15}\,erg\,s^{-1}\,cm^{-2})$']

	Prefig(size=np.array((2, 1.2))*7)
	fig, ax = plt.subplots(1,2, sharex=True)

	ax[0].text(0.1, 0.9, 'IC 1459', transform=ax[0].transAxes)
	
	for i, p in enumerate(plots):
		print '     '+ p
		
	
		DV = Data('ic1459', instrument='vimos', opt='pop')
		DM = Data('ic1459', instrument='muse', opt='pop')

			
		r = np.sqrt((DM.xBar - DM.center[0])**2 + (DM.yBar - DM.center[1])**2)*0.2
			
		ax[i].errorbar(r, eval('DM.'+p)/10**5/0.2**2, fmt='x', c='b', label='MUSE',
			yerr=eval('DM.'+p+'.uncert')/10**5)
		
		r = np.sqrt((DV.xBar - DV.center[0])**2 + (DV.yBar - DV.center[1])**2)*0.67
			
		ax[i].errorbar(r, eval('DV.'+p)/0.67**2, fmt='.', c='r', label='VIMOS', 
			yerr=eval('DV.'+p+'.uncert')/0.67**2)

		fig.tight_layout()
		ax[i].set_ylabel(str_plots[i])
		ax[i].set_xlabel('Radius (arcsec)')
		# ax[0,1].xaxis.set_tick_params(labelbottom=True)

		# ax[1,1].axis('off')
		# h, l = ax[0,0].get_legend_handles_labels()
		# ax[1,1].legend(h,l, bbox_to_anchor=(0,0.88), loc='upper left')
		ax[1].legend()

		fig.savefig('compare_ic1459.png', bbox_inches='tight', dpi=200)

	plt.close('all')


def ic4296():
	print 'Compare VIMOS and MUSE emission line results for IC4296'
	
	plots = "components['[OIII]5007d'].flux"
	str_plots = r'F(OIII) $(10^{-15}\,erg\,s^{-1}\,cm^{-2})$'
	

	Prefig(size=np.array((1, 1))*7)

	fig, ax = plt.subplots()

	ax.text(0.1, 0.9, 'IC 4296', transform=ax.transAxes)

	DV = Data('ic4296', instrument='vimos', opt='pop')
	DM = Data('ic4296', instrument='muse', opt='pop')

		
	r = np.sqrt((DM.xBar - DM.center[0])**2 + (DM.yBar - DM.center[1])**2)*0.2
		
	ax.errorbar(r, eval('DM.'+plots)/10**5/0.2**2, fmt='x', c='b', label='MUSE',
		yerr=eval('DM.'+plots+'.uncert')/10**5)
	
	r = np.sqrt((DV.xBar - DV.center[0])**2 + (DV.yBar - DV.center[1])**2)*0.67
		
	ax.errorbar(r, eval('DV.'+plots)/0.67**2, fmt='.', c='r', label='VIMOS', 
		yerr=eval('DV.'+plots+'.uncert')/0.67**2)

	fig.tight_layout()
	ax.set_ylabel(str_plots)
	ax.set_xlabel('Radius (arcsec)')
	ax.set_xlim([0,7])
	# ax[0,1].xaxis.set_tick_params(labelbottom=True)

	# ax[1,1].axis('off')
	# h, l = ax[0,0].get_legend_handles_labels()
	# ax[1,1].legend(h,l, bbox_to_anchor=(0,0.88), loc='upper left')
	ax.legend()

	fig.savefig('compare_ic4296.png', bbox_inches='tight', dpi=200)

	plt.close('all')


if __name__=='__main__':
	ic1459()
	ic4296()