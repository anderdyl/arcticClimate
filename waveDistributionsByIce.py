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
import cmocean


#
# with open(r"normalizedWaveHydrographsHope.pickle", "rb") as input_file:
#    normalizedWaveHydrographs = pickle.load(input_file)
# normalizedHydros = normalizedWaveHydrographs['normalizedHydros']
# bmuDataMin = normalizedWaveHydrographs['bmuDataMin']
# bmuDataMax = normalizedWaveHydrographs['bmuDataMax']
# bmuDataStd = normalizedWaveHydrographs['bmuDataStd']
# bmuDataNormalized = normalizedWaveHydrographs['bmuDataNormalized']
#
#
# with open(r"waveHydrographsHope.pickle", "rb") as input_file:
#    waveHydrographs = pickle.load(input_file)
# hydros = waveHydrographs['hydros']
#
# with open(r"hydrographCopulaDataHope.pickle", "rb") as input_file:
#    hydrographCopulaData = pickle.load(input_file)
# copulaData = hydrographCopulaData['copulaData']
#
# with open(r"historicalDataHope.pickle", "rb") as input_file:
#    historicalData = pickle.load(input_file)
# grouped = historicalData['grouped']
# groupLength = historicalData['groupLength']
# bmuGroup = historicalData['bmuGroup']
# timeGroup = historicalData['timeGroup']
#
# with open(r"dwts49ClustersArctic2y2022.pickle", "rb") as input_file:
#    historicalDWTs = pickle.load(input_file)
#
# order = historicalDWTs['kma_order']

import datetime as dt

def datenum_to_date(datenum):
    """
    Convert Matlab datenum into Python date.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return dt.datetime.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           - timedelta(days=366)
           #+ timedelta(hours=int(hours)) \
           #+ timedelta(minutes=int(minutes)) \
           #+ timedelta(seconds=round(seconds)) \

# define some constants
epoch = dt.datetime(1970, 1, 1)
matlab_to_epoch_days = 719529  # days from 1-1-0000 to 1-1-1970
matlab_to_epoch_seconds = matlab_to_epoch_days * 24 * 60 * 60

def matlab_to_datetime(matlab_date_num_seconds):
    # get number of seconds from epoch
    from_epoch = matlab_date_num_seconds - matlab_to_epoch_seconds

    # convert to python datetime
    return epoch + dt.timedelta(seconds=from_epoch)

import mat73
waveData = mat73.loadmat('era5_waves_pthope.mat')
# wlData = mat73.loadmat('pthope_tides_for_dylan.mat')
wlData = mat73.loadmat('updated_env_for_dylan.mat')

wlArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in wlData['time']]
waveArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in waveData['time_all']]


allWLs = wlData['wl']
wlsInd = np.where(allWLs < -2.5)
allWLs[wlsInd] = np.zeros((len(wlsInd),))


res = allWLs#wlData['wl']
wlHycom = allWLs#wlData['wl']
tWl = np.array(wlArrayTime)
waveDates = [datenum_to_date(t) for t in waveData['time_all']]
tWave = np.array(waveArrayTime)
hsCombined = waveData['wh_all']
tpCombined = waveData['tp_all']
dmCombined = waveData['mwd_all']
waveNorm = dmCombined


with open(r"iceData25ClustersMediumishMin150withDayNumRG.pickle", "rb") as input_file:
   historicalDWTs = pickle.load(input_file)

order = historicalDWTs['kma_order']

bmus = historicalDWTs['bmus_corrected']
time = historicalDWTs['dayTime']
timeArray = np.array(time)
kma_order = historicalDWTs['kma_order']
kmSorted = historicalDWTs['kmSorted']
xFlat = historicalDWTs['xFlat']
points = historicalDWTs['points']
group_size = historicalDWTs['group_size']
x = historicalDWTs['x']
y = historicalDWTs['y']


from matplotlib import gridspec
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


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
    spatialField = kmSorted[(hh), :]# / 100 - np.nanmean(SLP_C, axis=0) / 100
    linearField = np.ones((np.shape(xFlat))) * np.nan
    linearField[points] = spatialField
    rectField = linearField.reshape(75, 85)

    # ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    # gl = ax.gridlines(draw_labels=True)
    # extent = [-9.97, 168.35, 30.98, 34.35]

    # ax.pcolormesh(x[25:155], y[125:235], rectField, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[40:135], y[140:230], rectField, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[45:135], y[140:230], rectField, cmap=cmocean.cm.ice)

    # #  this is the proper orientation
    # ax.pcolormesh(x[65:150], y[160:235], rectField, cmap=cmocean.cm.ice)
    # ax.set_xlim([-2323000, 0])
    # ax.set_ylim([0, 2330000])
    # ax.text(-2323000/3*2,2100000,'{} days'.format(group_size[num]))

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

    # gl.xlabels_top = False
    # gl.ylabels_left = False
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    # ax.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
    # ax.set_title('{} days'.format(group_size[num]))

    c2 += 1
    if c2 == 6:
        c1 += 1
        c2 = 0



asdfg

numDWTs=25
waveHieghts = []
for hh in range(numDWTs):
    dwtInd = np.where((bmus==hh))
    waveBin = []
    for qq in range(len(dwtInd[0])):
        dayInd = np.where((tWave >= dt.datetime(time[dwtInd[0][qq]].year, time[dwtInd[0][qq]].month,time[dwtInd[0][qq]].day,0,0,0)) &
                          (tWave <= dt.datetime(timeArray[dwtInd[0][qq]].year, timeArray[dwtInd[0][qq]].month,
                                               timeArray[dwtInd[0][qq]].day,23,0,0)))
        waveBin.append(hsCombined[dayInd[0]])
    waveHieghts.append(np.concatenate(waveBin,axis=0))



dwtcolors = cm.rainbow(np.linspace(0, 1, numDWTs))
#plt.style.use('dark_background')

plt.style.use('dark_background')
dist_space = np.linspace(0, 5, 100)
fig = plt.figure(figsize=(10,10))
gs2 = gridspec.GridSpec(5, 5)

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
    normalize = mcolors.Normalize(vmin=.5, vmax=3.0)

    ax.set_xlim([0, 5])
    ax.set_ylim([0, 1])
    #data = dwtHs[dwtInd]

    data2 = waveHieghts[xx]
    data = data2[~np.isnan(data2)]

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

    if plotIndx < 5:

        ax.yaxis.set_ticklabels([])
    if plotIndx < 4:
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





