import sys
sys.path.append('/users/dylananderson/Documents/projects/gencadeClimate/')

from weatherTypes import weatherTypes


startTime = [2023, 1, 1]
# endTime = [1940,3,30]
endTime = [2023, 3, 30]
lonLeft = 0
lonRight = 360
latBot = 60
latTop = 90
purdhoeSLP = weatherTypes(lonLeft=lonLeft, lonRight=lonRight, latBot=latBot, latTop=latTop, startTime=startTime, endTime=endTime)

purdhoeSLP.extractERA5(printToScreen=True)
import plotting
plotting.plotSlpExample(struct=purdhoeSLP)

import pyproj

lon = purdhoeSLP.xGrid
lat = purdhoeSLP.yGrid
spNC = pyproj.Proj("EPSG:3411")
spE, spN = spNC(lon, lat)
