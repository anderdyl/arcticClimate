from scipy.io.matlab.mio5_params import mat_struct
from datetime import datetime, date, timedelta
import random
import itertools
import operator
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from scipy.stats import norm, genpareto, t
from scipy.special import ndtri  # norm inv
import matplotlib.dates as mdates
from scipy.stats import  genextreme, gumbel_l, spearmanr, norm, weibull_min
from scipy.spatial import distance
import xarray as xr
import os
import numpy as np
import datetime
from netCDF4 import Dataset
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec
import pickle
from scipy.io.matlab.mio5_params import mat_struct
from datetime import datetime, date, timedelta
import scipy.io as sio

with open(r"gevCopulaSims100000pointHope.pickle", "rb") as input_file:
    gevCopulaSimsInput = pickle.load(input_file)
gevCopulaSims = gevCopulaSimsInput['gevCopulaSims']

with open(r"normalizedWaveHydrographsPointHope.pickle", "rb") as input_file:
   normalizedWaveHydrographs = pickle.load(input_file)
normalizedHydros = normalizedWaveHydrographs['normalizedHydros']
bmuDataMin = normalizedWaveHydrographs['bmuDataMin']
bmuDataMax = normalizedWaveHydrographs['bmuDataMax']
bmuDataStd = normalizedWaveHydrographs['bmuDataStd']
bmuDataNormalized = normalizedWaveHydrographs['bmuDataNormalized']


with open(r"waveHydrographsPointHope.pickle", "rb") as input_file:
   waveHydrographs = pickle.load(input_file)
hydros = waveHydrographs['hydros']

with open(r"hydrographCopulaDataPointHope.pickle", "rb") as input_file:
   hydrographCopulaData = pickle.load(input_file)
copulaData = hydrographCopulaData['copulaData']

with open(r"historicalDataPointHope.pickle", "rb") as input_file:
   historicalData = pickle.load(input_file)
grouped = historicalData['grouped']
groupLength = historicalData['groupLength']
bmuGroup = historicalData['bmuGroup']
timeGroup = historicalData['timeGroup']

with open(r"dwts49ClustersArctic2023.pickle", "rb") as input_file:
   historicalDWTs = pickle.load(input_file)

order = historicalDWTs['kma_order']

lon = historicalDWTs['lon']
lat = historicalDWTs['lat']
km = historicalDWTs['km']
SLP = historicalDWTs['SLP']
X_in = historicalDWTs['X_in']
Y_in = historicalDWTs['Y_in']
sea_nodes = historicalDWTs['sea_nodes']
group_size = historicalDWTs['group_size']
numDWTs=49
# plt.style.use('dark_background')

dwtcolors = cm.rainbow(np.linspace(0, 1, 49))
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(int(np.sqrt(49)), int(np.sqrt(49)))
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.plt.figure()
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for hh in range(49):
    #ax = plt.subplot2grid((3,3),(c1,c2),projection=ccrs.NorthPolarStereo(central_longitude=-45))
    #ax = plt.subplot2grid((3,3),(c1,c2))#,projection=ccrs.NorthPolarStereo(central_longitude=-45))
    ax = plt.subplot(gs1[hh],projection=ccrs.NorthPolarStereo(central_longitude=-45))
    num = order[hh]

    # # m = Basemap(projection='merc',llcrnrlat=-40,urcrnrlat=55,llcrnrlon=255,urcrnrlon=375,lat_ts=10,resolution='c')
    m = Basemap(projection='npstere', boundinglat=50, lon_0=180, resolution='l')

    cx,cy =m(lon,lat)
    m.drawcoastlines()

    # spatialField = np.multiply(EOFs[hh,0:(len(Xsea))],np.sqrt(variance[hh]))
    # spatialField = Km_slp[(hh), :] / 100 - np.nanmean(SLP, axis=0) / 100
    spatialField = km[(num), 0:6529] / 100 - np.nanmean(SLP, axis=0) / 100

    rectField = np.ones((np.shape(X_in))) * np.nan
    for tt in range(len(sea_nodes)):
        rectField[sea_nodes[tt]] = spatialField[tt]

    m.fillcontinents()#color=dwtcolors[hh])

    clevels = np.arange(-35,35,1)

    #ax.pcolormesh(cx, cy, rectField)#, cmap=cmocean.cm.ice)
    # CS = m.contourf(cx, cy, rectField, clevels, vmin=-24, vmax=24, cmap=cm.RdBu_r, shading='gouraud')
    CS = ax.pcolormesh(cx, cy, rectField, vmin=-24, vmax=24, cmap=cm.RdBu_r)#, cmap=cmocean.cm.ice)

    ax.set_xlim([1911000, 7428000])
    ax.set_ylim([-392400, 6986000])
    # ax.set_xlim([np.min(cx)-1000, np.max(cx)-1000])
    # ax.set_ylim([np.min(cy)-1000, np.max(cy)-1000])
    # #tx, ty = m(320, -30)
    # ax.text(np.min(cx)+(np.max(cx)-np.min(cx))/3.2*2, np.min(cy)+(np.max(cy)-np.min(cy))/9, '{}'.format(group_size[num]))

    #ax.set_title('{}'.format(group_size[num]))

    c2 += 1
    if c2 == 6:
        c1 += 1
        c2 = 0

    if plotIndx < 8:
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    counter = counter + 1
    if plotIndy < 8:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1





dwtcolors = cm.rainbow(np.linspace(0, 1, numDWTs))
#plt.style.use('dark_background')

dist_space = np.linspace(0, 5, 100)
fig = plt.figure(figsize=(10,10))
gs2 = gridspec.GridSpec(7, 7)
# gs1 = gridspec.GridSpec(int(np.sqrt(49)), int(np.sqrt(49)))
gs2.update(wspace=0.00, hspace=0.00) # set the spacing between axes.plt.figure()
colorparam = np.zeros((numDWTs,))
counter = 0
plotIndx = 0
plotIndy = 0
for xx in range(numDWTs):
    dwtInd = xx
    #dwtInd = order[xx]
    #dwtInd = newOrder[xx]

    #ax = plt.subplot2grid((6, 5), (plotIndx, plotIndy), rowspan=1, colspan=1)
    ax = plt.subplot(gs2[xx])

    # normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))
    normalize = mcolors.Normalize(vmin=.25, vmax=4)

    ax.set_xlim([0, 5])
    ax.set_ylim([0, 1])
    #data = dwtHs[dwtInd]

    data2 = np.array([sub[0] for sub in copulaData[dwtInd]])
    wldata2 = np.array([sub[5] for sub in copulaData[dwtInd]])
    finder = np.where(np.isnan(data2))
    findreal = np.where(np.isreal(data2))
    dataNan = data2[finder]
    data3 = data2[~np.isnan(data2)]
    wldata3 = wldata2[~np.isnan(data2)]
    data = data3[~np.isnan(wldata3)]

    if dwtInd == 32:
        data2 = np.array([sub[0] for sub in copulaData[dwtInd]])
        data = data2[~np.isnan(data2)]

    if len(data) > 0:
        kde = gaussian_kde(data)
        colorparam[counter] = np.nanmean(data)
        colormap = cm.Reds
        color = colormap(normalize(colorparam[counter]))
        ax.plot(dist_space, kde(dist_space), linewidth=2, color=color)
        ax.spines['bottom'].set_color([0.5, 0.5, 0.5])
        ax.spines['top'].set_color([0.5, 0.5, 0.5])
        ax.spines['right'].set_color([0.5, 0.5, 0.5])
        ax.spines['left'].set_color([0.5, 0.5, 0.5])
        # ax.text(1.8, 1, np.round(colorparam*100)/100, fontweight='bold')

    else:
        ax.spines['bottom'].set_color([0.3, 0.3, 0.3])
        ax.spines['top'].set_color([0.3, 0.3, 0.3])
        ax.spines['right'].set_color([0.3, 0.3, 0.3])
        ax.spines['left'].set_color([0.3, 0.3, 0.3])

    if plotIndx < 7:

        ax.yaxis.set_ticklabels([])
    if plotIndx < 6:
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
    else:
        ax.xaxis.set_ticks([0, 2, 4])
        ax.xaxis.set_ticklabels(['0','2','4'])
        ax.set_xlabel('Hs (m)')

    if plotIndy == 0:
        ax.set_ylabel('Probability')
    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    if plotIndx == 9 and plotIndy == 0:
        ax.yaxis.set_ticklabels([])

    #ax.set_title('{} / {} / {}'.format(len(data), len(data3),len(dataNan)))
    counter = counter + 1
    if plotIndy < 6:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1
    print(plotIndy, plotIndx)

plt.show()
s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
s_map.set_array(colorparam)
fig.subplots_adjust(right=0.86)
cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
cbar = fig.colorbar(s_map, cax=cbar_ax)
cbar.set_label('Mean Hs (m)')












dist_space = np.linspace(1, 10, 120)
fig = plt.figure(figsize=(10,10))
gs2 = gridspec.GridSpec(7, 7)

colorparam = np.zeros((numDWTs,))
counter = 0
plotIndx = 0
plotIndy = 0
for xx in range(numDWTs):
    dwtInd = xx
    #dwtInd = order[xx]
    #dwtInd = newOrder[xx]

    #ax = plt.subplot2grid((6, 5), (plotIndx, plotIndy), rowspan=1, colspan=1)
    ax = plt.subplot(gs2[xx])

    # normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))
    normalize = mcolors.Normalize(vmin=2, vmax=8.0)

    ax.set_xlim([1, 10])
    ax.set_ylim([0, 0.6])
    #data = dwtHs[dwtInd]

    data2 = np.array([sub[2] for sub in copulaData[dwtInd]])
    wldata2 = np.array([sub[5] for sub in copulaData[dwtInd]])
    finder = np.where(np.isnan(data2))
    findreal = np.where(np.isreal(data2))
    dataNan = data2[finder]
    data3 = data2[~np.isnan(data2)]
    wldata3 = wldata2[~np.isnan(data2)]
    data = data3[~np.isnan(wldata3)]



    if len(data) > 0:
        kde = gaussian_kde(data)
        colorparam[counter] = np.nanmean(data)
        colormap = cm.Reds
        color = colormap(normalize(colorparam[counter]))
        ax.plot(dist_space, kde(dist_space), linewidth=1, color=color)
        ax.spines['bottom'].set_color([0.5, 0.5, 0.5])
        ax.spines['top'].set_color([0.5, 0.5, 0.5])
        ax.spines['right'].set_color([0.5, 0.5, 0.5])
        ax.spines['left'].set_color([0.5, 0.5, 0.5])
        # ax.text(1.8, 1, np.round(colorparam*100)/100, fontweight='bold')

    else:
        ax.spines['bottom'].set_color([0.3, 0.3, 0.3])
        ax.spines['top'].set_color([0.3, 0.3, 0.3])
        ax.spines['right'].set_color([0.3, 0.3, 0.3])
        ax.spines['left'].set_color([0.3, 0.3, 0.3])

    if plotIndx < 4:
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    if plotIndx == 9 and plotIndy == 0:
        ax.yaxis.set_ticklabels([])

    ax.set_title('{} / {} / {}'.format(len(data), len(data3),len(dataNan)))
    counter = counter + 1
    if plotIndy < 4:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1
    print(plotIndy, plotIndx)

plt.show()
s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
s_map.set_array(colorparam)
fig.subplots_adjust(right=0.86)
cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
cbar = fig.colorbar(s_map, cax=cbar_ax)
cbar.set_label('Max Tp (s)')








dist_space = np.linspace(0, 360, 360)
fig = plt.figure(figsize=(10,10))
gs2 = gridspec.GridSpec(7, 7)

colorparam = np.zeros((numDWTs,))
counter = 0
plotIndx = 0
plotIndy = 0
for xx in range(numDWTs):
    dwtInd = xx
    #dwtInd = order[xx]
    #dwtInd = newOrder[xx]

    #ax = plt.subplot2grid((6, 5), (plotIndx, plotIndy), rowspan=1, colspan=1)
    ax = plt.subplot(gs2[xx])

    # normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))
    normalize = mcolors.Normalize(vmin=2, vmax=8.0)

    ax.set_xlim([0, 360])
    ax.set_ylim([0, 0.015])
    #data = dwtHs[dwtInd]


    data2 = np.array([sub[4] for sub in copulaData[dwtInd]])
    wldata2 = np.array([sub[5] for sub in copulaData[dwtInd]])
    finder = np.where(np.isnan(data2))
    findreal = np.where(np.isreal(data2))
    dataNan = data2[finder]
    data3 = data2[~np.isnan(data2)]
    wldata3 = wldata2[~np.isnan(data2)]
    data = data3[~np.isnan(wldata3)]



    if len(data) > 0:
        kde = gaussian_kde(data)
        colorparam[counter] = np.nanmean(data)
        colormap = cm.Reds
        color = colormap(normalize(colorparam[counter]))
        ax.plot(dist_space, kde(dist_space), linewidth=1, color=color)
        ax.spines['bottom'].set_color([0.5, 0.5, 0.5])
        ax.spines['top'].set_color([0.5, 0.5, 0.5])
        ax.spines['right'].set_color([0.5, 0.5, 0.5])
        ax.spines['left'].set_color([0.5, 0.5, 0.5])
        # ax.text(1.8, 1, np.round(colorparam*100)/100, fontweight='bold')

    else:
        ax.spines['bottom'].set_color([0.3, 0.3, 0.3])
        ax.spines['top'].set_color([0.3, 0.3, 0.3])
        ax.spines['right'].set_color([0.3, 0.3, 0.3])
        ax.spines['left'].set_color([0.3, 0.3, 0.3])

    if plotIndx < 4:
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    if plotIndx == 9 and plotIndy == 0:
        ax.yaxis.set_ticklabels([])

    ax.set_title('{} / {} / {}'.format(len(data), len(data3),len(dataNan)))
    counter = counter + 1
    if plotIndy < 4:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1
    print(plotIndy, plotIndx)

plt.show()
s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
s_map.set_array(colorparam)
fig.subplots_adjust(right=0.86)
cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
cbar = fig.colorbar(s_map, cax=cbar_ax)
cbar.set_label('Mean Dm (deg)')











dist_space = np.linspace(-0.25, 2.25, 100)
fig = plt.figure(figsize=(10,10))
gs2 = gridspec.GridSpec(7, 7)

colorparam = np.zeros((numDWTs,))
counter = 0
plotIndx = 0
plotIndy = 0
for xx in range(numDWTs):
    dwtInd = xx
    #dwtInd = order[xx]
    #dwtInd = newOrder[xx]

    #ax = plt.subplot2grid((6, 5), (plotIndx, plotIndy), rowspan=1, colspan=1)
    ax = plt.subplot(gs2[xx])

    # normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))
    normalize = mcolors.Normalize(vmin=-1, vmax=2.0)

    ax.set_xlim([-1,2.5])
    ax.set_ylim([0, 2])
    #data = dwtHs[dwtInd]


    data2 = np.array([sub[0] for sub in copulaData[dwtInd]])
    wldata2 = np.array([sub[5] for sub in copulaData[dwtInd]])

    finder = np.where(np.isnan(data2))
    findreal = np.where(np.isreal(data2))
    dataNan = data2[finder]
    data3 = data2[~np.isnan(data2)]

    wldata3 = wldata2[~np.isnan(data2)]
    data = wldata3[~np.isnan(wldata3)]



    if len(data) > 0:
        kde = gaussian_kde(data)
        colorparam[counter] = np.nanmean(data)
        colormap = cm.Reds
        color = colormap(normalize(colorparam[counter]))
        ax.plot(dist_space, kde(dist_space), linewidth=1, color=color)
        ax.spines['bottom'].set_color([0.5, 0.5, 0.5])
        ax.spines['top'].set_color([0.5, 0.5, 0.5])
        ax.spines['right'].set_color([0.5, 0.5, 0.5])
        ax.spines['left'].set_color([0.5, 0.5, 0.5])
        # ax.text(1.8, 1, np.round(colorparam*100)/100, fontweight='bold')

    else:
        ax.spines['bottom'].set_color([0.3, 0.3, 0.3])
        ax.spines['top'].set_color([0.3, 0.3, 0.3])
        ax.spines['right'].set_color([0.3, 0.3, 0.3])
        ax.spines['left'].set_color([0.3, 0.3, 0.3])

    if plotIndx < 4:
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    if plotIndx == 9 and plotIndy == 0:
        ax.yaxis.set_ticklabels([])

    ax.set_title('{} / {} / {}'.format(len(data), len(data3),len(dataNan)))
    counter = counter + 1
    if plotIndy < 4:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1
    print(plotIndy, plotIndx)

plt.show()
s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
s_map.set_array(colorparam)
fig.subplots_adjust(right=0.86)
cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
cbar = fig.colorbar(s_map, cax=cbar_ax)
cbar.set_label('water level (m)')










plt.style.use('default')
subset = [41,48,3,23,17]
# plotting the EOF patterns
fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(5, 5)
# gs1.update(wspace=0.25, hspace=0.25) # set the spacing between axes.
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0

for hh in range(5):
    ax = plt.subplot(gs1[hh],projection=ccrs.NorthPolarStereo(central_longitude=-45))

    # num = np.where(order == subset[hh])[0][0]
    num = order[subset[hh]]
    # # m = Basemap(projection='merc',llcrnrlat=-40,urcrnrlat=55,llcrnrlon=255,urcrnrlon=375,lat_ts=10,resolution='c')
    m = Basemap(projection='npstere', boundinglat=50, lon_0=180, resolution='l')

    cx,cy =m(lon,lat)
    m.drawcoastlines()

    # spatialField = np.multiply(EOFs[hh,0:(len(Xsea))],np.sqrt(variance[hh]))
    # spatialField = Km_slp[(hh), :] / 100 - np.nanmean(SLP, axis=0) / 100
    spatialField = km[(num), 0:6529] / 100 - np.nanmean(SLP, axis=0) / 100

    rectField = np.ones((np.shape(X_in))) * np.nan
    for tt in range(len(sea_nodes)):
        rectField[sea_nodes[tt]] = spatialField[tt]

    m.fillcontinents()#color=dwtcolors[hh])

    clevels = np.arange(-35,35,1)

    #ax.pcolormesh(cx, cy, rectField)#, cmap=cmocean.cm.ice)
    # CS = m.contourf(cx, cy, rectField, clevels, vmin=-24, vmax=24, cmap=cm.RdBu_r, shading='gouraud')
    CS = ax.pcolormesh(cx, cy, rectField, vmin=-24, vmax=24, cmap=cm.RdBu_r)#, cmap=cmocean.cm.ice)

    # ax.set_xlim([2111000, 6828000])
    # ax.set_ylim([-392400, 6986000])
    ax.set_xlim([1711000, 7728000])
    ax.set_ylim([-222400, 5986000])

    m.fillcontinents(color=dwtcolors[num])


plotIndx = 0
plotIndy = 0
for xx in range(5):

    dwtInd = subset[xx]
    dist_space = np.linspace(0, 6.6, 100)

    ax = plt.subplot(gs1[xx + 5])
    normalize = mcolors.Normalize(vmin=.25, vmax=4)

    ax.set_xlim([-.1, 6.5])
    ax.set_ylim([0, 0.9])
    #data = dwtHs[dwtInd]
    synData = gevCopulaSims[dwtInd][:,0]

    data2 = np.array([sub[0] for sub in copulaData[dwtInd]])
    wldata2 = np.array([sub[5] for sub in copulaData[dwtInd]])
    finder = np.where(np.isnan(data2))
    findreal = np.where(np.isreal(data2))
    dataNan = data2[finder]
    data3 = data2[~np.isnan(data2)]
    wldata3 = wldata2[~np.isnan(data2)]
    data = data3[~np.isnan(wldata3)]

    if dwtInd == 32:
        data2 = np.array([sub[0] for sub in copulaData[dwtInd]])
        data = data2[~np.isnan(data2)]

    if len(data) > 0:
        kde = gaussian_kde(data)
        colorparam[counter] = np.nanmean(data)
        colormap = cm.Reds
        color = colormap(normalize(colorparam[counter]))
        # ax.plot(dist_space, kde(dist_space), linewidth=2, color=color)
        ax.plot(dist_space, kde(dist_space), linewidth=2, color='k')

        # kde2 = gaussian_kde(synData)
        # ax.plot(dist_space, kde2(dist_space),'--', linewidth=2, color=[0.5,0.5,0.5])


        ax.spines['bottom'].set_color([0.5, 0.5, 0.5])
        ax.spines['top'].set_color([0.5, 0.5, 0.5])
        ax.spines['right'].set_color([0.5, 0.5, 0.5])
        ax.spines['left'].set_color([0.5, 0.5, 0.5])
        # ax.text(1.8, 1, np.round(colorparam*100)/100, fontweight='bold')

    else:
        ax.spines['bottom'].set_color([0.3, 0.3, 0.3])
        ax.spines['top'].set_color([0.3, 0.3, 0.3])
        ax.spines['right'].set_color([0.3, 0.3, 0.3])
        ax.spines['left'].set_color([0.3, 0.3, 0.3])


    ax.xaxis.set_ticks([0, 2.5, 5])
    ax.xaxis.set_ticklabels(['0','2.5','5'])
    ax.set_xlabel('Hs (m)')

    if plotIndy == 0:
        ax.set_ylabel('Probability')
    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    if plotIndx == 9 and plotIndy == 0:
        ax.yaxis.set_ticklabels([])
    counter = counter + 1
    if plotIndy < 6:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1
    print(plotIndy, plotIndx)


#
# plotIndx = 0
# plotIndy = 0
# for xx in range(5):
#
#     dwtInd = subset[xx]
#     dist_space = np.linspace(0, 6.6, 100)
#
#     ax = plt.subplot(gs1[xx + 5])
#     normalize = mcolors.Normalize(vmin=.25, vmax=4)
#
#     ax.set_xlim([-.1, 6.5])
#     ax.set_ylim([0, 0.9])
#     #data = dwtHs[dwtInd]
#
#     data2 = np.array([sub[0] for sub in copulaData[dwtInd]])
#     wldata2 = np.array([sub[5] for sub in copulaData[dwtInd]])
#     finder = np.where(np.isnan(data2))
#     findreal = np.where(np.isreal(data2))
#     dataNan = data2[finder]
#     data3 = data2[~np.isnan(data2)]
#     wldata3 = wldata2[~np.isnan(data2)]
#     data = data3[~np.isnan(wldata3)]
#
#     if dwtInd == 32:
#         data2 = np.array([sub[0] for sub in copulaData[dwtInd]])
#         data = data2[~np.isnan(data2)]
#
#     if len(data) > 0:
#         kde = gaussian_kde(data)
#         colorparam[counter] = np.nanmean(data)
#         colormap = cm.Reds
#         color = colormap(normalize(colorparam[counter]))
#         # ax.plot(dist_space, kde(dist_space), linewidth=2, color=color)
#         ax.plot(dist_space, kde(dist_space), linewidth=2, color='k')
#
#         ax.spines['bottom'].set_color([0.5, 0.5, 0.5])
#         ax.spines['top'].set_color([0.5, 0.5, 0.5])
#         ax.spines['right'].set_color([0.5, 0.5, 0.5])
#         ax.spines['left'].set_color([0.5, 0.5, 0.5])
#         # ax.text(1.8, 1, np.round(colorparam*100)/100, fontweight='bold')
#
#     else:
#         ax.spines['bottom'].set_color([0.3, 0.3, 0.3])
#         ax.spines['top'].set_color([0.3, 0.3, 0.3])
#         ax.spines['right'].set_color([0.3, 0.3, 0.3])
#         ax.spines['left'].set_color([0.3, 0.3, 0.3])
#
#
#     ax.xaxis.set_ticks([0, 2.5, 5])
#     ax.xaxis.set_ticklabels(['0','2.5','5'])
#     ax.set_xlabel('Max Hs (m)')
#
#     if plotIndy == 0:
#         ax.set_ylabel('Probability')
#     ax.yaxis.set_ticklabels([])
#     ax.yaxis.set_ticks([])
#     counter = counter + 1
#     if plotIndy < 6:
#         plotIndy = plotIndy + 1
#     else:
#         plotIndy = 0
#         plotIndx = plotIndx + 1
#     print(plotIndy, plotIndx)
#
#
# plotIndx = 0
# plotIndy = 0
# for xx in range(5):
#
#     dwtInd = subset[xx]
#     dist_space = np.linspace(0, 6.6, 100)
#
#     ax = plt.subplot(gs1[xx + 10])
#     normalize = mcolors.Normalize(vmin=.25, vmax=4)
#
#     ax.set_xlim([-.1, 6.5])
#     ax.set_ylim([0, 1.2])
#     #data = dwtHs[dwtInd]
#
#     data2 = np.array([sub[1] for sub in copulaData[dwtInd]])
#     wldata2 = np.array([sub[5] for sub in copulaData[dwtInd]])
#     finder = np.where(np.isnan(data2))
#     findreal = np.where(np.isreal(data2))
#     dataNan = data2[finder]
#     data3 = data2[~np.isnan(data2)]
#     wldata3 = wldata2[~np.isnan(data2)]
#     data = data3[~np.isnan(wldata3)]
#
#     if dwtInd == 32:
#         data2 = np.array([sub[0] for sub in copulaData[dwtInd]])
#         data = data2[~np.isnan(data2)]
#
#     if len(data) > 0:
#         kde = gaussian_kde(data)
#         colorparam[counter] = np.nanmean(data)
#         colormap = cm.Reds
#         color = colormap(normalize(colorparam[counter]))
#         # ax.plot(dist_space, kde(dist_space), linewidth=2, color=color)
#         ax.plot(dist_space, kde(dist_space), linewidth=2, color='k')
#
#         ax.spines['bottom'].set_color([0.5, 0.5, 0.5])
#         ax.spines['top'].set_color([0.5, 0.5, 0.5])
#         ax.spines['right'].set_color([0.5, 0.5, 0.5])
#         ax.spines['left'].set_color([0.5, 0.5, 0.5])
#         # ax.text(1.8, 1, np.round(colorparam*100)/100, fontweight='bold')
#
#     else:
#         ax.spines['bottom'].set_color([0.3, 0.3, 0.3])
#         ax.spines['top'].set_color([0.3, 0.3, 0.3])
#         ax.spines['right'].set_color([0.3, 0.3, 0.3])
#         ax.spines['left'].set_color([0.3, 0.3, 0.3])
#
#
#     ax.xaxis.set_ticks([0, 2.5, 5])
#     ax.xaxis.set_ticklabels(['0','2.5','5'])
#     ax.set_xlabel('Min Hs (m)')
#     ax.yaxis.set_ticklabels([])
#     ax.yaxis.set_ticks([])
#     if plotIndy == 0:
#         ax.set_ylabel('Probability')
#
#     counter = counter + 1
#     if plotIndy < 6:
#         plotIndy = plotIndy + 1
#     else:
#         plotIndy = 0
#         plotIndx = plotIndx + 1
#     print(plotIndy, plotIndx)
#




plotIndx = 0
plotIndy = 0
for xx in range(5):

    dwtInd = subset[xx]
    dist_space = np.linspace(0, 6.6, 100)

    ax = plt.subplot(gs1[xx + 10])
    normalize = mcolors.Normalize(vmin=.25, vmax=4)

    ax.set_xlim([-.1, 6.5])
    ax.set_ylim([-.1, 6.5])
    #data = dwtHs[dwtInd]

    data1 = gevCopulaSims[dwtInd][0:2000,0]#np.array([sub[0] for sub in copulaData[dwtInd]])
    data2 = gevCopulaSims[dwtInd][0:2000,1]#np.array([sub[1] for sub in copulaData[dwtInd]])

    finder = np.where(np.isnan(data2))
    findreal = np.where(np.isreal(data2))
    dataNan = data2[finder]
    data3 = data2[~np.isnan(data2)]
    data3b = data1[~np.isnan(data2)]

    if dwtInd == 32:
        data2 = np.array([sub[0] for sub in copulaData[dwtInd]])
        data = data2[~np.isnan(data2)]

    if len(data) > 0:

        xy = np.vstack([data3, data3b])
        z = gaussian_kde(xy)(xy)
        ax.scatter(data3, data3b, 5, c=z)

        # kde = gaussian_kde(data)
        # colorparam[counter] = np.nanmean(data)
        # colormap = cm.Reds
        # color = colormap(normalize(colorparam[counter]))
        # # ax.plot(dist_space, kde(dist_space), linewidth=2, color=color)
        # ax.plot(dist_space, kde(dist_space), linewidth=2, color='k')

        ax.spines['bottom'].set_color([0.5, 0.5, 0.5])
        ax.spines['top'].set_color([0.5, 0.5, 0.5])
        ax.spines['right'].set_color([0.5, 0.5, 0.5])
        ax.spines['left'].set_color([0.5, 0.5, 0.5])
        # ax.text(1.8, 1, np.round(colorparam*100)/100, fontweight='bold')

    else:
        ax.spines['bottom'].set_color([0.3, 0.3, 0.3])
        ax.spines['top'].set_color([0.3, 0.3, 0.3])
        ax.spines['right'].set_color([0.3, 0.3, 0.3])
        ax.spines['left'].set_color([0.3, 0.3, 0.3])


    ax.xaxis.set_ticks([0, 2.5, 5])
    ax.xaxis.set_ticklabels(['0','2.5','5'])
    ax.set_xlabel('Min Hs (m)')
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
    if plotIndy == 0:
        ax.set_ylabel('Max Hs (m)')
        ax.yaxis.set_ticks([0, 2.5, 5])
        ax.yaxis.set_ticklabels(['0','2.5','5'])
    counter = counter + 1
    if plotIndy < 6:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1
    print(plotIndy, plotIndx)




fig10 = plt.figure(figsize=(10,5))
ax100 = plt.subplot2grid((1,2),(0,0))
dwtInd = subset[1]
data1 = hydros[dwtInd][87]['hs']#np.array([sub[0] for sub in copulaData[dwtInd]])
ax100.plot(np.linspace(0,24,24),data1,color='k',linewidth=2,label='Observations')
ax100.plot([0,24],[np.max(data1),np.max(data1)],'--',color='purple',linewidth=2,label='Max of Hydrograph')
ax100.plot([0,24],[np.min(data1),np.min(data1)],'--',color='blue',linewidth=2,label='Min of Hydrograph')
ax100.set_ylim([0,7])
ax100.set_xlabel('Hours')
ax100.set_ylabel('Hs (m)')
ax100.set_title('Historical Storm Hydrograph')
plt.legend()
ax101 = plt.subplot2grid((1,2),(0,1))
dwtInd = subset[1]
data2 = (data1-np.min(data1))/np.max((data1-np.min(data1)))#normalizedHydros[dwtInd][87]['hsNorm']#np.array([sub[0] for sub in copulaData[dwtInd]])
ax101.plot(np.linspace(0,1,24),data2,color='k',linewidth=2)
ax101.set_xlabel('Normalized Time')
ax101.set_ylabel('Normalized Hs')
ax101.set_title('Normalized Hydrograph')
    #
    # ax.xaxis.set_ticks([0, 2.5, 5])
    # ax.xaxis.set_ticklabels(['0','2.5','5'])
    # ax.set_xlabel('Min Hs (m)')
    # ax.yaxis.set_ticklabels([])
    # ax.yaxis.set_ticks([])
    # if plotIndy == 0:
    #     ax.set_ylabel('Max Hs (m)')
    #     ax.yaxis.set_ticks([0, 2.5, 5])
    #     ax.yaxis.set_ticklabels(['0','2.5','5'])
    # counter = counter + 1
    # if plotIndy < 6:
    #     plotIndy = plotIndy + 1
    # else:
    #     plotIndy = 0
    #     plotIndx = plotIndx + 1
    # print(plotIndy, plotIndx)



