import pickle
import numpy as np
from datetime import datetime
import sys
import pandas as pd

sys.path.append('/users/dylananderson/Documents/projects/gencadeClimate/')
slpPath = '/users/dylananderson/documents/data/prmsl/'
wisPath = '/users/dylananderson/documents/data/WIS_ST63218/'
wlPath = '/users/dylananderson/documents/data/frfWaterLevel/'
from weatherTypes import weatherTypes

startTime = [2023, 8, 1]
# endTime = [1980,12,31]
# endTime = [2023, 8, 31]
endTime = [2023, 9, 30]

import plotting

from metOcean import getMetOcean

# # point hope
# lonLeft = -166.8
# lonRight = -166.6
# latBot = 68.4
# latTop = 68.6

# # Wevok
# lonLeft = -166.1
# lonRight = -165.9
# latBot = 69.2
# latTop = 69.3

# # Deering
# lonLeft = -162.8
# lonRight = -162.6
# latBot = 66.2
# latTop = 66.3

# # Wales North
# lonLeft = -168.6
# lonRight = -168.4
# latBot = 65.9
# latTop = 66.1

# # point lay
# lonLeft = -163.8
# lonRight = -163.7
# latBot = 69.7
# latTop = 69.8
#
# # Kivalina
# lonLeft = -165.1
# lonRight = -164.9
# latBot = 67.4
# latTop = 67.6

# # Shishmaref
# lonLeft = -166.6
# lonRight = -166.4
# latBot = 66.4
# latTop = 66.6

# # Wainwright
# lonLeft = -160.6
# lonRight = -160.4
# latBot = 70.7
# latTop = 70.8

# # Utqiagvik
# lonLeft = -157.3
# lonRight = -157.2
# latBot = 71.45
# latTop = 71.55

# # Drew Point
# lonLeft = -154.1
# lonRight = -153.9
# latBot = 71.2
# latTop = 71.3

# # Purdoe Bay
# lonLeft = -148.3
# lonRight = -148.2
# latBot = 70.7
# latTop = 70.8

# Barter Island
lonLeft = -143.8
lonRight = -143.7
latBot = 70.4
latTop = 70.6


# # FRF
# lonLeft = -75.6
# lonRight = -75.4
# latBot = 36.2
# latTop = 36.3


# stAugMET = getMetOcean(shoreNormal=90,lonLeft=lonLeft, lonRight=lonRight, latBot=latBot, latTop=latTop, wlPath=wlPath, wisPath=wisPath, startTime=startTime, endTime=endTime)
# stAugMET.getERA5Waves()
metOcean = getMetOcean(shoreNormal=0,lonLeft=lonLeft, lonRight=lonRight, latBot=latBot, latTop=latTop, wlPath=wlPath, wisPath=wisPath, startTime=startTime, endTime=endTime)
metOcean.getERA5WavesAndWinds(printToScreen=True)
# plotting.plotOceanConditions(struct=purdhoeWaves)
metOcean.getERA5Bathymetry(printToScreen=True)

#
import pickle
outdict = {}
outdict['metOcean'] = metOcean
# outdict['tyndallMET'] = tyndallMET
# outdict['stAugSLP'] = stAugSLP
outdict['endTime'] = endTime
outdict['startTime'] = startTime
outdict['lonLeft'] = lonLeft
outdict['lonRight'] = lonRight
outdict['latBot'] = latBot
outdict['latTop'] = latTop

# with open('walesNorthWavesWindsTemps168pt5by66.pickle', 'wb') as f:
# with open('deeringWavesWindsTemps162pt75by66pt25.pickle', 'wb') as f:
# with open('wevokWavesWindsTemps166by69pt25.pickle', 'wb') as f:
# with open('barterIslandWavesWindsTemps143pt75by70pt5.pickle', 'wb') as f:
# with open('purdoeBayWavesWindsTemps148pt25by70pt75.pickle', 'wb') as f:
# with open('drewPointWavesWindsTemps154by71pt25.pickle', 'wb') as f:
# with open('utqiagvikWavesWindsTemps157pt25by71pt5.pickle', 'wb') as f:
# with open('wainwrightWavesWindsTemps160pt5by70pt75.pickle', 'wb') as f:
# with open('shishmarefWavesWindsTemps166pt5by66pt5.pickle', 'wb') as f:
# with open('kivalina2WavesWindsTemps165by67pt5.pickle', 'wb') as f:
# with open('pointLayWavesWindsTemps164by70.pickle', 'wb') as f:
# with open('pointHopeWavesWindsTemps166pt75by68pt5.pickle', 'wb') as f:
# with open('frfWavesWindsTemps75pt5by36pt25.pickle', 'wb') as f:
#     pickle.dump(outdict, f)



import matplotlib.pyplot as plt
plt.figure()
plt.plot(metOcean.timeWave,metOcean.Hs)
