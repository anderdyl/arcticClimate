import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean


with open(r"iceData25ClustersMediumishMin150withDayNumRG.pickle", "rb") as input_file:
   iceData = pickle.load(input_file)

num_clusters = iceData['num_clusters']
kma_order = iceData['kma_order']
km = iceData['kmSorted']
xFlat = iceData['xFlat']
points = iceData['points']
x = iceData['x']
y = iceData['y']
group_size = iceData['group_size']

plt.style.use('dark_background')

# plotting the EOF patterns
fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(5, 5)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for hh in range(25):
    #p1 = plt.subplot2grid((6,6),(c1,c2))
    ax = plt.subplot(gs1[hh],projection=ccrs.NorthPolarStereo(central_longitude=-45))
    num = kma_order[hh]
    # num = kma_orderOG[hh]

    # spatialField = kmOG[(hh - 1), :]# / 100 - np.nanmean(SLP_C, axis=0) / 100
    # spatialField = km[(hh - 1), :]# / 100 - np.nanmean(SLP_C, axis=0) / 100
    spatialField = km[(hh), :]# / 100 - np.nanmean(SLP_C, axis=0) / 100

    linearField = np.ones((np.shape(xFlat)))
    linearField[points] = spatialField
    rectField = linearField.reshape(75, 85)

    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    # gl = ax.gridlines(draw_labels=True)
    extent = [-9.97, 168.35, 30.98, 34.35]
    # ax.pcolormesh(x[25:155], y[125:235], rectField, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[40:135], y[140:230], rectField, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[45:135], y[140:230], rectField, cmap=cmocean.cm.ice)
    ax.pcolormesh(x[65:150], y[160:235], rectField, cmap=cmocean.cm.ice)

    ax.set_xlim([-2203000, -110000])
    ax.set_ylim([0, 1870000])
    ax.plot(-1950000,1250000,marker='o',color='red',markersize=5)
    # gl.xlabels_top = False
    # gl.ylabels_left = False
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    # ax.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
    # ax.set_title('{} days'.format(group_size[num]))
    # ax.set_title('{} days'.format(group_size[hh]))

    c2 += 1
    if c2 == 6:
        c1 += 1
        c2 = 0





