#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

dens = pd.read_table(sys.argv[1]
                   , sep=" "
                   , skipinitialspace=True)
dens = np.array(dens)
dens = dens.reshape(int(dens.size/3), 3)

plt.scatter(dens[:,0]
          , dens[:,1]
          , color=plt.cm.inferno(dens[:,2]/max(dens[:,2]))
          , marker='.'
          , s=0.3)
plt.show()

