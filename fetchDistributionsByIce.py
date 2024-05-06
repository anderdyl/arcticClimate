
import numpy as np
import datetime
from netCDF4 import Dataset
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec
from scipy.io.matlab.mio5_params import mat_struct
from datetime import datetime, date, timedelta
import scipy.io as sio
import cmocean
from matplotlib import gridspec
import cartopy.crs as ccrs
import pickle

with open(r"fetchAreaAttempt1.pickle", "rb") as input_file:
   fetchPickle = pickle.load(input_file)
fetchTotalConcentration = fetchPickle['fetchTotalConcentration']
kiloArea = fetchPickle['kiloArea']
slpWaves = fetchPickle['slpWaves']
dayTime = fetchPickle['dayTime']

# outputDWTs['fetchTotalConcentration'] = fetchTotalConcentration
# outputDWTs['slpWaves'] = slpWaves
# outputDWTs['dayTime'] = dayTime
# outputDWTs['gapFilledIce'] = gapFilledIce
# outputDWTs['mask'] = mask
# outputDWTs['xEdge'] = xEdge
# outputDWTs['yEdge'] = yEdge
# outputDWTs['kiloArea'] = kiloArea
# outputDWTs['wavesYearly'] = wavesYearly
# outputDWTs['iceYearly'] = iceYearly
# outputDWTs['xMeshSmallx'] = xMeshSmallx
# outputDWTs['yMeshSmallx'] = yMeshSmallx
# outputDWTs['tupVerts'] = tupVerts
# outputDWTs['points2'] = points2
# outputDWTs['x'] = x
# outputDWTs['y'] = y
# outputDWTs['xSmall'] = xSmall
# outputDWTs['ySmall'] = ySmall
# outputDWTs['myDates'] = myDates
# outputDWTs['waves'] = waves

# with open(r"iceData25ClustersMediumishMin150withDayNumRG.pickle", "rb") as input_file:

with open(r"iceData36ClustersDayNumRGAdjusted.pickle", "rb") as input_file:
   historicalDWTs = pickle.load(input_file)

order = historicalDWTs['kma_order']

bmus = historicalDWTs['bmus_final']
time = historicalDWTs['dayTime']
timeArray = np.array(time)
kma_order = historicalDWTs['kma_order']
kmSorted = historicalDWTs['kmSorted']
xFlat = historicalDWTs['xFlat']
points = historicalDWTs['points']
group_size = historicalDWTs['group_size']
x = historicalDWTs['x']
y = historicalDWTs['y']




fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(6, 6)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for hh in range(36):
    ax = plt.subplot(gs1[hh],projection=ccrs.NorthPolarStereo(central_longitude=-45))
    num = kma_order[hh]

    # spatialField = kmOG[(hh - 1), :]# / 100 - np.nanmean(SLP_C, axis=0) / 100
    spatialField = kmSorted[(hh), :]# / 100 - np.nanmean(SLP_C, axis=0) / 100
    linearField = np.ones((np.shape(xFlat))) * np.nan
    linearField[points] = spatialField
    rectField = linearField.reshape(75, 85)

    # Inverted orientation
    ax.pcolormesh(y[160:235],x[65:150], np.fliplr(rectField.T), cmap=cmocean.cm.ice)
    ax.set_ylim([-2323000, 300000])
    ax.set_xlim([-200000, 2130000])
    ax.text(1230000/3,30000,'{} days'.format(group_size[num]))

    temp = np.fliplr(rectField.T)
    tempInd = np.where((np.isnan(temp)))
    temp = temp*np.nan
    temp[tempInd] = 0.5
    temp2 = temp
    temp[64:,0:25] = np.nan
    ax.pcolormesh(y[160:235],x[65:150], temp, vmin=0, vmax=1, cmap='Greys')

    c2 += 1
    if c2 == 6:
        c1 += 1
        c2 = 0








import datetime as dt
numDWTs=36
fetchLengths = []
waveHeights = []
for hh in range(numDWTs):
    dwtInd = np.where((bmus==hh))
    fetchBin = []
    waveBin = []
    fetchBin.append(np.asarray(fetchTotalConcentration)[dwtInd[0]])
    waveBin.append(slpWaves['wh_all'][dwtInd[0]])
    fetchLengths.append(np.concatenate(fetchBin,axis=0))
    waveHeights.append(np.concatenate(waveBin,axis=0))


dwtcolors = cm.rainbow(np.linspace(0, 1, numDWTs))
plt.style.use('dark_background')
dist_space = np.linspace(0, 1600000/1000, 1000)
fig = plt.figure(figsize=(10,10))
gs2 = gridspec.GridSpec(6, 6)

colorparam = np.zeros((numDWTs,))
counter = 0
plotIndx = 0
plotIndy = 0
for xx in range(numDWTs):
    dwtInd = xx
    ax = plt.subplot(gs2[xx])
    normalize = mcolors.Normalize(vmin=50000/1000, vmax=1500000/1000)
    data2 = fetchLengths[xx]
    data = data2[~np.isnan(data2)]/1000*kiloArea

    if len(data) > 2:
        kde = gaussian_kde(data)
        colorparam[counter] = np.nanmean(data)
        colormap = cm.Reds
        color = colormap(normalize(colorparam[counter]))
        ax.plot(dist_space, kde(dist_space), linewidth=1, color=color)
        ax.spines['bottom'].set_color([0.5, 0.5, 0.5])
        ax.spines['top'].set_color([0.5, 0.5, 0.5])
        ax.spines['right'].set_color([0.5, 0.5, 0.5])
        ax.spines['left'].set_color([0.5, 0.5, 0.5])

    else:
        ax.spines['bottom'].set_color([0.3, 0.3, 0.3])
        ax.spines['top'].set_color([0.3, 0.3, 0.3])
        ax.spines['right'].set_color([0.3, 0.3, 0.3])
        ax.spines['left'].set_color([0.3, 0.3, 0.3])

    if plotIndx < 5:

        ax.yaxis.set_ticklabels([])
    if plotIndx < 4:
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
    else:
        ax.xaxis.set_ticks([200, 800, 1400])
        ax.xaxis.set_ticklabels(['200','800','1400'])
        ax.set_xlabel('Fetch Area (km^2)')
    if plotIndy == 0:
        ax.set_ylabel('Probability')

    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    if plotIndx == 9 and plotIndy == 0:
        ax.yaxis.set_ticklabels([])

    # ax.set_title('{} / {} / {}'.format(len(data), len(data3),len(dataNan)))
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
cbar.set_label('Wave Gen Area (km^2)')



dwtcolors = cm.rainbow(np.linspace(0, 1, numDWTs))
plt.style.use('dark_background')
dist_space = np.linspace(0, 1800000/1000, 1000)
fig = plt.figure(figsize=(10,10))
gs2 = gridspec.GridSpec(6, 6)

colorparam = np.zeros((numDWTs,))
counter = 0
plotIndx = 0
plotIndy = 0
for xx in range(numDWTs):
    dwtInd = xx

    ax = plt.subplot(gs2[xx])

    fetchTemp = fetchLengths[xx]/1000*kiloArea
    waveTemp = waveHeights[xx]
    # data = data2[~np.isnan(data2)]
    ax.set_xlim([0, 2100000/1000])
    ax.set_ylim([0, 6])
    if len(fetchTemp) > 2:
        # kde = gaussian_kde(data)
        # colorparam[counter] = np.nanmean(data)
        # colormap = cm.Reds
        # color = colormap(normalize(colorparam[counter]))
        # ax.plot(dist_space, kde(dist_space), linewidth=1, color=color)
        ax.plot(fetchTemp, waveTemp,'.', color='white')

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

    if plotIndx < 5:

        ax.yaxis.set_ticklabels([])
    if plotIndx < 4:
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
    else:
        ax.xaxis.set_ticks([200, 800, 1400])
        ax.xaxis.set_ticklabels(['200','800','1400'])
        ax.set_xlabel('Fetch (km)')
    if plotIndy == 0:
        ax.set_ylabel('Hs (m)')

    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    if plotIndx == 9 and plotIndy == 0:
        ax.yaxis.set_ticklabels([])

    # ax.set_title('{} / {} / {}'.format(len(data), len(data3),len(dataNan)))
    counter = counter + 1
    if plotIndy < 4:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1
    print(plotIndy, plotIndx)

plt.show()

from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d

def Empirical_CDF(x):
    '''
    Returns empirical cumulative probability function at x.
    '''

    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)

    return cdf

def Empirical_ICDF(x, p):
    '''
    Returns inverse empirical cumulative probability function at p points
    '''

    # TODO: build in functionality for a fill_value?

    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)

    # interpolate KDE CDF to get support values 
    fint = interp1d(
        cdf, x,
        fill_value=(np.nanmin(x), np.nanmax(x)),
        #fill_value=(np.min(x), np.max(x)),
        bounds_error=False
    )
    return fint(p)

import statsmodels.api as sm

def ksdensity_CDF(x):
    '''
    Kernel smoothing function estimate.
    Returns cumulative probability function at x.
    '''

    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()

    # interpolate KDE CDF at x position (kde.support = x)
    fint = interp1d(kde.support, kde.cdf)

    return fint(x)

def ksdensity_ICDF(x, p):
    '''
    Returns Inverse Kernel smoothing function at p points
    '''

    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()

    # interpolate KDE CDF to get support values 
    fint = interp1d(kde.cdf, kde.support)

    # ensure p inside kde.cdf
    p[p<np.min(kde.cdf)] = kde.cdf[0]
    p[p>np.max(kde.cdf)] = kde.cdf[-1]

    return fint(p)


import random
import datetime as dt
numDWTs=36
fetchSims = []
for hh in range(numDWTs):
    dwtInd = np.where((bmus==hh))
    fetchBin = []
    obsFetch = np.asarray(fetchTotalConcentration)[dwtInd[0]]
    data = obsFetch[~np.isnan(obsFetch)]/1000*kiloArea
    if len(data) > 2:
        uSim = [random.uniform(0,1) for ff in range(100000)]
        simFetch = ksdensity_ICDF(data,np.asarray(uSim))
        fetchSims.append(simFetch)
    else:
        fetchSims.append(np.nan)

# s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
# s_map.set_array(colorparam)
# fig.subplots_adjust(right=0.86)
# cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
# cbar = fig.colorbar(s_map, cax=cbar_ax)
# cbar.set_label('Fetch Length (km)')
#



dwtcolors = cm.rainbow(np.linspace(0, 1, numDWTs))
#plt.style.use('dark_background')

plt.style.use('dark_background')
dist_space = np.linspace(0, 1800000/1000, 1000)
fig = plt.figure(figsize=(10,10))
gs2 = gridspec.GridSpec(6, 65)

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
    normalize = mcolors.Normalize(vmin=50000/1000, vmax=1500000/1000)

    # ax.set_xlim([0, 5])
    # ax.set_ylim([0, 1])
    #data = dwtHs[dwtInd]

    data2 = fetchSims[xx]
    if np.all(np.isnan(data2)):
        data = np.array([0])
    else:
        data = data2[~np.isnan(data2)]

    if len(data) > 2:
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

    if plotIndx < 5:

        ax.yaxis.set_ticklabels([])
    if plotIndx < 4:
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
    else:
        ax.xaxis.set_ticks([200, 800, 1400])
        ax.xaxis.set_ticklabels(['200','800','1400'])
        ax.set_xlabel('Fetch (km)')
    if plotIndy == 0:
        ax.set_ylabel('Probability')

    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    if plotIndx == 9 and plotIndy == 0:
        ax.yaxis.set_ticklabels([])

    # ax.set_title('{} / {} / {}'.format(len(data), len(data3),len(dataNan)))
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
cbar.set_label('Wave Gen Area (km^2)')




gevCopulaSimsPickle = 'ice36FetchSims.pickle'
outputgevCopulaSims = {}
outputgevCopulaSims['fetchSims'] = fetchSims

with open(gevCopulaSimsPickle,'wb') as f:
    pickle.dump(outputgevCopulaSims, f)
