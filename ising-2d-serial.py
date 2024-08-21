################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
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

#################################################################
gpts = 100 #temp grid
N = 16 #lattice sites
steps = 40000 #time steps
wait = 1 #time between checking moves

energies = np.zeros(steps)

lattice = walker(N,2.)

for i in range(steps):

   lattice.move()
   energies[i] = lattice.energy()/N**2.

fig=plt.figure(figsize=(12,8))
plt.scatter(np.arange(1,steps+1,1),energies)
plt.show()
