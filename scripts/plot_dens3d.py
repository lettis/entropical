#!/usr/bin/env python3

import pandas as pd
import sys
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

dens = pd.read_table(sys.argv[1]
                   , sep=" "
                   , skipinitialspace=True)
dens = np.array(dens)
dens = dens.reshape(int(dens.size/4), 4)

ax.scatter(xs=dens[:,0]
         , ys=dens[:,1]
         , zs=dens[:,2]
         , c=plt.cm.inferno(dens[:,3]/max(dens[:,3]))
         , marker='.'
         , s=50)
plt.show()

