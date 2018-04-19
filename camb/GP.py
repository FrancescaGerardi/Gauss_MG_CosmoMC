from sklearn.gaussian_process import GaussianProcessRegressor
import sys
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import matplotlib.pyplot as plt
import argparse



parser = argparse.ArgumentParser(description='Input parameters from CAMB')
parser.add_argument('--inired',metavar='zini',  type=float, nargs='+',
                   help='initial redshift')
parser.add_argument('--endred',metavar='zend',  type=float, nargs='+',
                   help='end redshift')
parser.add_argument('--ODEsteps',metavar='ODEsteps',  type=int, nargs='+',
                   help='number of steps for the ODE solver')
parser.add_argument('--redshifts',metavar='z',  type=float, nargs='*',default=[],
                   help='values of redshifts')
parser.add_argument('--eos',metavar='wde',  type=float, nargs='*',default=[],
                   help='equation of state')
parser.add_argument('--l',metavar='l',  type=float, nargs='+',
                   help='correlation length')
parser.add_argument('--lb',metavar='l',  type=float, nargs='*',default=[],
                   help='correlation length bins')
# parser.add_argument('--outfile', nargs='+', type=argparse.FileType('w'),default=sys.stdout)
parser.add_argument('--outfile', nargs='+', type=str ,default=sys.stdout)
args = parser.parse_args()

#print args.outfile
#print args.inired[0], args.endred[0], args.ODEsteps[0]
#print args.redshifts                                   valori in mezzo ai bin
#print args.eos
#print args.l[0]
#print args.outfile[0]



#defining the baseline -1
base = lambda x: -1+x-x

#Training points
inired = args.inired[0]
endred = args.endred[0]
ODEsteps = args.ODEsteps[0]
#z_edge = np.array(args.redshifts) #NH redshift at edge of each bin
wde = np.array(args.eos)
lc = args.l[0]  #correlation lenght costante
l = np.array(args.lb)
filename = args.outfile[0]

z = np.array(args.redshifts)#z_edge[:-1] + np.diff(z_edge)/2 #NH z is now the redshift in the middle of each bin

nb=len(z)

for i in range (nb):
	wde[i]= wde[i]+ base(z[i])


bounds=np.zeros(nb)
bounds[0]=inired
for i in range(1,nb):
	bounds[i] = z[i]
#print bounds


#array da plottare alla fine, entro cui rientrano tutti i training points
red_sampl=np.array([])
wde_pr=np.array([])

#CONTATORI PER IL BINNING IN CORRELATION LENGHT,  a prescindere si chiedono tante correlation lenghts quanti sono i bin - 1 !!
n=len(l)  #numero di bin in lunghezza di correlazione
nsamp=int(ODEsteps/n)
ni = np.zeros(n)
tot=int(0)
for i in range (0,n):
	ni[i] = nsamp
	tot = tot + ni[i]
i=-1
while (tot<ODEsteps):
	i=i+1	
	ni[i]=ni[i]+1
	tot = tot + 1

redc=np.zeros(2)
wdec=np.zeros(2)

#inizio ciclofor per processi gaussiani
for i in range (0,n):
	redc[0]=z[i]
	redc[1]=z[i+1]
#	print redc
	wdec[0]=wde[i]
	wdec[1]=wde[i+1]
#	print wdec
	gp=GaussianProcessRegressor(kernel=RBF(l[i], (l[i],l[i]))).fit(redc[:, np.newaxis], wdec - base(redc)) 
	j=int(ni[i])
	z_sampling = np.linspace(bounds[i], bounds[i+1], j+1)
#	print z_sampling
	w_pred, sigma = gp.predict(z_sampling[:, np.newaxis], return_std=True)
	w_pred = w_pred + base(z_sampling)
#	print w_pred

	z_sampling=z_sampling[:j]	
	red_sampl=np.concatenate([red_sampl,z_sampling])

	w_pred=w_pred[:j]	
	wde_pr=np.concatenate([wde_pr,w_pred])

print red_sampl
	
#plotting
plt.plot(red_sampl,wde_pr,'.')
plt.show()
#save file
np.savetxt(filename, np.array([red_sampl, wde_pr]).T, fmt="%15.8e")

exit()
