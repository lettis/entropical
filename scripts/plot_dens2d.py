#!/usr/bin/env python3

from default_definitions import *

dens = to_array(sys.argv[1])
dens = dens.reshape(int(dens.size/3), 3)

plt.scatter(dens[:,0]
          , dens[:,1]
          , color=plt.cm.Greys(dens[:,2]/max(dens[:,2]))
          , marker='.')
plt.show()

