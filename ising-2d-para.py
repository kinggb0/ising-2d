################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool
import os
################################
class walker:

   #Initialize the walker
   def __init__(self,npart,kt):

      self.npart = npart #number of particles
      self.kt = kt #kB T

      #initialize the random spins with <s_i>= +/- 1
      self.config = 2 * np.random.randint(2, size=(npart,npart)) - 1

   #sum spins for nearest neighbors to lattice site (i,j) w/ PBC
   def neighbors(self,i,j):

      return self.config[(i+1)%self.npart,j] + self.config[(i-1)%self.npart,j] \
             +self.config[i,(j+1)%self.npart] + self.config[i,(j-1)%self.npart]

   #Compute the move using Metropolis Hastings
   def move(self):
    
      #pick random lattice site 
      i = np.random.randint(0,self.npart)
      j = np.random.randint(0,self.npart)

      spin = self.config[i,j]
   
      #compute nearest neighbor spin sum
      sumnb = self.neighbors(i,j)      

      #the energy cost to flip spin at (i,j)
      de = 2.*spin*sumnb

      #flip (i,j) if it lowers the energy
      if (de < 0):
         spin *= -1
      #accept it with probability given by Boltzmann weight
      elif ( np.random.uniform() <= np.exp(-de/self.kt) ):
         spin *= -1

      #update the configuration
      self.config[i,j] = spin

      return

   #compute the energy of the system E = -\sum_{<ij>} s_is_j 
   def energy(self):

      energy = 0.

      for i in range(self.npart):
         for j in range(self.npart): 
            sumnb = self.neighbors(i,j)
            energy += -self.config[i,j]*sumnb
      
      return energy/4.

   #compute the total magnetization M = \sum_i s_i
   def magnetization(self):
      return np.sum(self.config)

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
steps = 12500 #time steps/thrd
wait = 1 #time between checking moves
thrds = 8

temps = np.linspace(1.5,4.5,gpts)

results = np.zeros(thrds)

energies = np.zeros(gpts)
magnetzn = np.zeros(gpts)

for j in range(gpts):

   print("Temp = %.2f" % temps[j])

   lattice = walker(N,temps[j])

   pool = Pool(processes=thrds)
 
   results = [pool.apply_async(mcrun,args=(lattice,steps,os.getpid())) for n in range(thrds)]

   for result in results:

      output = result.get()

      etemp, mtemp = output[0],output[1]

      energies[j] += np.mean(etemp[6750:])
      magnetzn[j] += np.abs(np.mean(mtemp[6750:]))

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
