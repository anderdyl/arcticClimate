
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

with open(r"fetchLengthAttempt1.pickle", "rb") as input_file:
   fetchPickle = pickle.load(input_file)
fetch = fetchPickle['fetch']
fetchFiltered = fetchPickle['fetchFiltered']
slpWaves = fetchPickle['slpWaves']
dayTime = fetchPickle['dayTime']
fetch = fetchPickle['fetch']


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






import random

fetchFiltered2 = np.nan * np.zeros((len(fetch),))
c = 0
for qq in range(len(fetch)):
    # if len(fetch[qq]) == 0:
    #     fetchFiltered[qq] = np.nan
    if np.all(np.isnan(fetch[qq])):
        if dayTime[qq].month == 8 or dayTime[qq].month == 9:
            fetchFiltered2[qq] = 2250000 + random.random() * 100000 - 50000
        else:
            fetchFiltered2[qq] = np.array([np.nan])
    else:
        # print('{}'.format(qq))
        if len(fetch[qq]) >= 1:

            if len(fetch[qq]) == 2:
                # if fetch[qq][0] == 0:
                #     fetchFiltered[qq] = fetch[qq][1]
                # else:
                if fetch[qq][1] < 1000000:
                    fetchFiltered2[qq] = fetch[qq][1]
                else:
                    fetchFiltered2[qq] = fetch[qq][0]
            elif len(fetch[qq]) == 3:
                if fetch[qq][1] < 100000:
                    fetchFiltered2[qq] = fetch[qq][2]
                elif fetch[qq][0] < 20000:
                    fetchFiltered2[qq] = fetch[qq][1]
                else:
                    fetchFiltered2[qq] = fetch[qq][0]
            elif len(fetch[qq]) == 4:
                if fetch[qq][0] < 150000:
                    fetchFiltered2[qq] = fetch[qq][2]
                else:
                    fetchFiltered2[qq] = fetch[qq][0]
            elif len(fetch[qq]) == 5:
                if fetch[qq][1] < 100000:
                    fetchFiltered2[qq] = fetch[qq][2]
                else:
                    fetchFiltered2[qq] = fetch[qq][0]
            else:
                # print('we should be here')
                fetchFiltered2[qq] = fetch[qq][0]

            if dayTime[qq].month == 9:# or dayTime[qq].month == 9:
                if fetch[qq][0] < 50000:
                    fetchFiltered2[qq] = fetch[qq][1]

            if slpWaves['wh_all'][qq] == 0:
                fetchFiltered2[qq] = np.array([np.nan])

            if fetch[qq][0] > 600000:
                fetchFiltered2[qq] = fetch[qq][0]

        elif len(fetch[qq]) == 0:
            # print('hey hi yo')
            if dayTime[qq].month == 8 or dayTime[qq].month == 9:
                fetchFiltered2[qq] = 2000000 + random.random()*10000-5000
            else:
                fetchFiltered2[qq] = np.array([np.nan])

    if slpWaves['wh_all'][qq] > 0:
        if np.isnan(fetchFiltered2[qq]):
            print('so we have waves but fetch {} times on {}'.format(c,dayTime[qq]))
            print(fetch[qq])
            c = c +1

plt.figure()
plt.plot(dayTime,fetchFiltered2)


plt.style.use('dark_background')
from scipy import signal as sg
plt.figure()
iceYearly = np.nan*np.ones((42,365))
c = 0
ax = plt.subplot2grid((2,1),(0,0))

for hh in range(42):
    temp = fetchFiltered2[c:c+365]
    temp2 = sg.medfilt(temp,5)
    iceYearly[hh,:]=temp2
    c = c + 365
    ax.plot(dayTime[0:365],temp2/1000,alpha=0.5)

badInd = np.where(np.isnan(iceYearly))
iceYearly[badInd] = 0
ax.plot(dayTime[0:365],np.nanmean(iceYearly,axis=0)/1000,color='white',linewidth=2,label='Average Fetch')
ax.set_ylabel('Fetch (km)')

ax.xaxis.set_ticks([datetime(1979,1,1),datetime(1979,2,1),datetime(1979,3,1),datetime(1979,4,1),datetime(1979,5,1),datetime(1979,6,1),
                    datetime(1979,7,1),datetime(1979,8,1),datetime(1979,9,1),datetime(1979,10,1),datetime(1979,11,1),datetime(1979,12,1)])
ax.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
# ax.set_xlabel('Fetch (km)')
# ax.set_title('Average Fetch Length to Sea Ice')



wavesYearly = np.nan*np.ones((42,365))
c = 0
ax2 = plt.subplot2grid((2,1),(1,0))

for hh in range(42):
    temp2 = slpWaves['wh_all'][c:c+365]
    #temp2 = sg.medfilt(temp,5)
    wavesYearly[hh,:]=temp2
    c = c + 365
    ax2.plot(dayTime[0:365],temp2/1000,alpha=0.5)


badIndW = np.where(np.isnan(wavesYearly))
wavesYearly[badIndW] = 0
ax2.plot(dayTime[0:365],np.nanmean(wavesYearly,axis=0)/1000,color='white',linewidth=2,label='Average Fetch')
ax2.set_ylabel('Hs (m)')

ax2.xaxis.set_ticks([datetime(1979,1,1),datetime(1979,2,1),datetime(1979,3,1),datetime(1979,4,1),datetime(1979,5,1),datetime(1979,6,1),
                    datetime(1979,7,1),datetime(1979,8,1),datetime(1979,9,1),datetime(1979,10,1),datetime(1979,11,1),datetime(1979,12,1)])
ax2.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
# ax.set_xlabel('Fetch (km)')
# ax2.set_title('Average Fetch Length to Sea Ice')


sdfg


dwtPickle = 'fetchLengthAttempt2.pickle'
outputDWTs = {}
outputDWTs['fetch'] = fetch
outputDWTs['fetchFiltered'] = fetchFiltered2
outputDWTs['slpWaves'] = slpWaves
outputDWTs['dayTime'] = dayTime
outputDWTs['iceYearly'] = iceYearly

with open(dwtPickle,'wb') as f:
    pickle.dump(outputDWTs, f)


import datetime as dt
numDWTs=25
fetchLengths = []
waveHeights = []
for hh in range(numDWTs):
    dwtInd = np.where((bmus==hh))
    fetchBin = []
    waveBin = []
    #for qq in range(len(dwtInd[0])):
        # dayInd = np.where((tWave >= dt.datetime(time[dwtInd[0][qq]].year, time[dwtInd[0][qq]].month,time[dwtInd[0][qq]].day,0,0,0)) &
        #                   (tWave <= dt.datetime(timeArray[dwtInd[0][qq]].year, timeArray[dwtInd[0][qq]].month,
        #                                        timeArray[dwtInd[0][qq]].day,23,0,0)))
        # dayInd = np.where((dayTime == ))
        # fetchBin.append(hsCombined[dayInd[0]])
    fetchBin.append(fetchFiltered[dwtInd[0]])
    waveBin.append(slpWaves['wh_all'][dwtInd[0]])

    fetchLengths.append(np.concatenate(fetchBin,axis=0))
    waveHeights.append(np.concatenate(waveBin,axis=0))



dwtcolors = cm.rainbow(np.linspace(0, 1, numDWTs))
#plt.style.use('dark_background')

plt.style.use('dark_background')
dist_space = np.linspace(0, 1800000/1000, 1000)
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
    normalize = mcolors.Normalize(vmin=50000/1000, vmax=1500000/1000)

    # ax.set_xlim([0, 5])
    # ax.set_ylim([0, 1])
    #data = dwtHs[dwtInd]

    data2 = fetchLengths[xx]
    data = data2[~np.isnan(data2)]/1000

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
cbar.set_label('Fetch Length (km)')






dwtcolors = cm.rainbow(np.linspace(0, 1, numDWTs))
#plt.style.use('dark_background')

plt.style.use('dark_background')
dist_space = np.linspace(0, 1800000/1000, 1000)
fig = plt.figure(figsize=(10,10))
gs2 = gridspec.GridSpec(5, 5)

colorparam = np.zeros((numDWTs,))
counter = 0
plotIndx = 0
plotIndy = 0
for xx in range(numDWTs):
    dwtInd = xx

    ax = plt.subplot(gs2[xx])

    fetchTemp = fetchLengths[xx]/1000
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
numDWTs=25
fetchSims = []
for hh in range(numDWTs):
    dwtInd = np.where((bmus==hh))
    fetchBin = []
    obsFetch = fetchFiltered[dwtInd[0]]
    data = obsFetch[~np.isnan(obsFetch)]/1000
    if len(data) > 2:
        uSim = [random.uniform(0,1) for hh in range(1000)]
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
cbar.set_label('Fetch Length (km)')

