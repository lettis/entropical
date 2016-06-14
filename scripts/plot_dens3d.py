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

Axes3D.scatter(dens[:,0]
             , dens[:,1]
             , dens[:,2]
             , color=plt.cm.inferno(dens[:,3]/max(dens[:,3]))
             , depthshade=True
             , marker='.'
             , s=0.3)
plt.show()

