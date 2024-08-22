################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
from ising2d import walker
################################

# run the Monte carlo
def mcrun(walker,steps,seed):

   np.random.seed(seed)

   etemp = np.zeros(steps)
   mtemp = np.zeros(steps)

   for i in range(steps):

      walker.move()
      etemp[i] = walker.energy()/N**2.
      mtemp[i] = walker.magnetization()/N**2.

   return etemp, mtemp

#################################################################
gpts = 30 #temp grid
N = 16 #lattice sites
steps = 56250 #time steps/thrd
thrds = 8

temps = np.linspace(1.5,4.5,gpts)

results = np.zeros(thrds)

energies = np.zeros(gpts)
magnetzn = np.zeros(gpts)

for j in range(gpts):

   print("Temp = %.2f" % temps[j])

   lattice = walker(N,temps[j])

   pool = Pool(processes=thrds)
 
   #initialize the parallel processes
   results = [pool.apply_async(mcrun,args=(lattice,steps,os.getpid())) for n in range(thrds)]

   for result in results:

      output = result.get()

      etemp, mtemp = output[0],output[1]

      energies[j] += np.mean(etemp[50000:])
      magnetzn[j] += np.abs(np.mean(mtemp[50000:]))

energies = energies/thrds
magnetzn = magnetzn/thrds

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(24,16))

ax1.scatter(temps,energies,s=50,color='darkblue')
ax2.scatter(temps,magnetzn,s=50,color='orangered')

ax1.set_xlabel(r'$T$ ($k_B^{-1}$)',fontsize=25)
ax1.set_ylabel(r'$E/N$ ($k_B^{-1})$',fontsize=25)
ax1.tick_params(axis='both',direction='in',length=4,width=1,labelsize=20)

ax2.set_xlabel(r'$T$ ($k_B^{-1}$)',fontsize=25)
ax2.set_ylabel(r'$|M|/N$',fontsize=25)
ax2.tick_params(axis='both',direction='in',length=4,width=1,labelsize=20)

plt.tight_layout()
plt.savefig('parallel-results.pdf',bbox_inches='tight')
plt.show()
