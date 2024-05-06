
import datetime
import pickle
from scipy.io.matlab.mio5_params import mat_struct
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from datetime import timedelta
import cmocean
import sys
import subprocess
if 'darwin' in sys.platform:
    print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
    subprocess.Popen('caffeinate')
# define some constants
epoch = dt.datetime(1970, 1, 1)
matlab_to_epoch_days = 719529  # days from 1-1-0000 to 1-1-1970
matlab_to_epoch_seconds = matlab_to_epoch_days * 24 * 60 * 60

def matlab_to_datetime(matlab_date_num_seconds):
    # get number of seconds from epoch
    from_epoch = matlab_date_num_seconds - matlab_to_epoch_seconds

    # convert to python datetime
    return epoch + dt.timedelta(seconds=from_epoch)

def dateDay2datetimeDate(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [dt.date(int(d[0]), int(d[1]), int(d[2])) for d in d_vec]

def dateDay2datetime(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [dt.datetime(int(d[0]), int(d[1]), int(d[2])) for d in d_vec]

import datetime as dt
def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return dt.datetime.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           + timedelta(hours=int(hours)) \
           + timedelta(minutes=int(minutes)) \
           + timedelta(seconds=round(seconds)) \
           - timedelta(days=366)


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




import scipy.io as sio

def ReadMatfile(p_mfile):
    'Parse .mat file to nested python dictionaries'

    def RecursiveMatExplorer(mstruct_data):
        # Recursive function to extrat mat_struct nested contents

        if isinstance(mstruct_data, mat_struct):
            # mstruct_data is a matlab structure object, go deeper
            d_rc = {}
            for fn in mstruct_data._fieldnames:
                d_rc[fn] = RecursiveMatExplorer(getattr(mstruct_data, fn))
            return d_rc

        else:
            # mstruct_data is a numpy.ndarray, return value
            return mstruct_data

    # base matlab data will be in a dict
    mdata = sio.loadmat(p_mfile, squeeze_me=True, struct_as_record=False)
    mdata_keys = [x for x in mdata.keys() if x not in
                  ['__header__','__version__','__globals__']]

    # use recursive function
    dout = {}
    for k in mdata_keys:
        dout[k] = RecursiveMatExplorer(mdata[k])
    return dout

def dateDay2datetimeDate(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [date(int(d[0]), int(d[1]), int(d[2])) for d in d_vec]

def dateDay2datetime(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [datetime(int(d[0]), int(d[1]), int(d[2])) for d in d_vec]




# with open(r"dwtsAll6TCTracksClusters.pickle", "rb") as input_file:
# with open(r"dwts25ClustersArctic2.pickle", "rb") as input_file:
# with open(r"dwts49ClustersArctic2y2022.pickle", "rb") as input_file:
# with open(r"dwts64ClustersArcticRGy2022.pickle", "rb") as input_file:
with open(r"dwts49ClustersArcticRGy2022.pickle", "rb") as input_file:
# with open(r"dwts81ClustersArcticRGy2022.pickle", "rb") as input_file:
    historicalDWTs = pickle.load(input_file)

numClusters = 49
timeDWTs = historicalDWTs['SLPtime']
# outputDWTs['slpDates'] = slpDates
bmus = historicalDWTs['bmus_corrected']

bmus_dates = dateDay2datetimeDate(timeDWTs)
bmus_dates_months = np.array([d.month for d in bmus_dates])
bmus_dates_days = np.array([d.day for d in bmus_dates])


slpCentroids = historicalDWTs['sorted_centroids']
slpEOFsub = historicalDWTs['EOFsub']
SLP = historicalDWTs['SLP']
slpMean = np.mean(SLP,axis=0)
slpStd = np.std(SLP,axis=0)
slpPCs = historicalDWTs['PCsub']


with open(r"iceData36ClustersDayNumRGFinalized.pickle", "rb") as input_file:
# with open(r"iceData25ClustersMediumishMin150withDayNumRG.pickle", "rb") as input_file:
   historicalICEs = pickle.load(input_file)

orderIce = historicalICEs['kma_order']
bmusIce = historicalICEs['bmus_final'][31:]


timeIce = historicalICEs['dayTime'][31:]
timeArrayIce = np.array(timeIce)#
data_a_ice = historicalICEs['data_a'][31:,:]
iceEOFsub = historicalICEs['EOFsub']
iceMean = historicalICEs['iceMean']
iceStd = historicalICEs['iceStd']
gapFilledIce = historicalICEs['gapFilledIce']
iceX = historicalICEs['x']
iceY = historicalICEs['y']
icePoints = historicalICEs['points']
iceXflat = historicalICEs['xFlat']
icePCs = historicalICEs['PCsub'][31:,:]

iceCentroids = np.zeros((36,26))
for qq in range(int(np.max(bmusIce)+1)):
    posc = np.where((bmusIce == qq))
    iceCentroids[qq,:] = np.mean(data_a_ice[posc[0],:],axis=0)





with open(r"fetchAreaAttempt1.pickle", "rb") as input_file:
   fetchPickle = pickle.load(input_file)
kiloArea = fetchPickle['kiloArea']
slpWaves = fetchPickle['slpWaves'][31:]
dayTime = fetchPickle['dayTime'][31:]
fetchTotalConcentration = np.asarray(fetchPickle['fetchTotalConcentration'])[31:]/1000*kiloArea


import mat73
waveData = mat73.loadmat('era5_waves_pthope.mat')
# wlData = mat73.loadmat('pthope_tides_for_dylan.mat')
wlData = mat73.loadmat('updated_env_for_dylan.mat')

wlArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in wlData['time']]
waveArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in waveData['time_all']]


wh_all = np.hstack((waveData['wh_all'],np.nan*np.zeros(3624,)))
tp_all = np.hstack((waveData['tp_all'],np.nan*np.zeros(3624,)))
dm_all = np.hstack((waveData['mwd_all'],np.nan*np.zeros(3624,)))

st = dt.datetime(2022, 1, 1)
end = dt.datetime(2022,6,1)
from dateutil.relativedelta import relativedelta
step = relativedelta(hours=1)
hourTime = []
while st < end:
    hourTime.append(st)#.strftime('%Y-%m-%d'))
    st += step

time_all = np.hstack((waveArrayTime,hourTime))



hsCombined = wh_all
nanInd = np.where((hsCombined==0))
hsCombined[nanInd] = np.nan * np.ones((len(nanInd)))
tpCombined = tp_all
nanInd = np.where((tpCombined==0))
tpCombined[nanInd] = np.nan * np.ones((len(nanInd)))

dmCombined = dm_all
nanInd = np.where((dmCombined==0))
dmCombined[nanInd] = np.nan * np.ones((len(nanInd)))

# waveNorm = wavesInput['waveNorm']
# wlFRF = wavesInput['wlFRF']
# tFRF = wavesInput['tWl']
# resFRF = wavesInput['res']
import pandas as pd
data = np.array([hsCombined[24*31:],tpCombined[24*31:],dmCombined[24*31:]])
ogdf = pd.DataFrame(data=data.T, index=time_all[24*31:], columns=["hs", "tp", "dm"])
year = np.array([tt.year for tt in time_all[24*31:]])
ogdf['year'] = year
month = np.array([tt.month for tt in time_all[24*31:]])
ogdf['month'] = month

dailyMaxHs = ogdf.resample("d")['hs'].max()
dailyMeanHs = ogdf.resample("d")['hs'].mean()
dailyMeanTp = ogdf.resample("d")['tp'].mean()

iceCumSumPercent = np.cumsum(historicalICEs['nPercent'])[0:10]
slpCumSumPercent = np.cumsum(historicalDWTs['nPercent'])[0:50]



dwtPairCentroids = np.hstack((slpPCs[:,0:50],icePCs[:,0:9]))#,dailyMeanHs))#,dailyMeanTp))
dwtPairCentroids = np.concatenate((dwtPairCentroids,np.array(dailyMeanHs.values)[:,None]),axis=1)
dwtPairCentroids = np.concatenate((dwtPairCentroids,np.array(dailyMeanTp.values)[:,None]),axis=1)


#                 dwtPairCentroids[c,:] = np.hstack((slpCentroids[hh,0:10],iceCentroids[ff,0:2],0,0))#,iceCentroids[ff,23:25],dailyMeanHs))

#
#
#
# expandedBMU = np.nan * np.zeros((len(bmus),))
# dwtPairs = []
# # dwtPairCentroids = np.zeros((36*49,26+26))
# dwtPairCentroids = np.zeros((36*49,10+2+2))
#
# totalDays = np.zeros((36*49,))
# wavePair = []
# wavePairNoNans = []
# avgWavePair = []
# c = 0
# for ff in range(36):
#     for hh in range(49):
#         dwtPairs.append([ff,hh])
#         # dwtPairCentroids[c,:] = np.hstack((slpCentroids[hh,0:26],iceCentroids[ff]))
#         # dwtPairCentroids[c, :] = np.hstack((slpCentroids[hh, 0:10], iceCentroids[ff, 0:2]))
#         ind = np.where((bmusIce == ff) & (bmus == hh))
#         expandedBMU[ind[0]] = int(c)
#         totalDays[c] = len(ind[0])
#         # waveTemp = dailyMaxHs[ind[0]].values
#         waveTemp = dailyMeanHs[ind[0]].values
#         waveTpTemp = dailyMeanTp[ind[0]].values
#         wavePair.append(waveTemp)
#         avgWavePair.append(np.nanmean(waveTemp))
#         if np.isreal(np.nanmean(waveTemp)):
#             if np.isnan(np.nanmean(waveTemp)):
#                 dwtPairCentroids[c,:] = np.hstack((slpCentroids[hh,0:10],iceCentroids[ff,0:2],0,0))#,iceCentroids[ff,23:25],dailyMeanHs))
#             else:
#                 withNans = np.where((np.isnan(waveTemp)))
#                 waveTemp[withNans] = 0
#                 waveTpTemp[withNans] = 0
#                 dwtPairCentroids[c,:] = np.hstack((slpCentroids[hh,0:10],iceCentroids[ff,0:2],np.nanmean(waveTemp),np.nanmean(waveTpTemp)))#,iceCentroids[ff,23:25],dailyMeanHs))
#         else:
#             dwtPairCentroids[c, :] = np.hstack(
#                 (slpCentroids[hh, 0:10], iceCentroids[ff, 0:2], 0, 0))  # ,iceCentroids[ff,23:25],dailyMeanHs))
#
#         wavePairNoNans.append(len(np.where((waveTemp > 0))[0]))
#         c = c+1
#
#

dwtPercent = historicalDWTs['nPercent'][0:50]
dwtPercent = (dwtPercent / dwtPercent[0])

icePercent = historicalICEs['nPercent'][0:9]
icePercent = icePercent / icePercent[0]

nPercents = np.hstack((dwtPercent,icePercent,1,1))

whereNan = np.where(np.isnan(dwtPairCentroids))
dwtPairCentroids[whereNan] = 0

dwtPairCentroidsMean = np.mean(dwtPairCentroids,axis=0)
dwtPairCentroidsStd = np.std(dwtPairCentroids,axis=0)

dwtPairCentroidsNorm = (dwtPairCentroids[:,:] - dwtPairCentroidsMean) / dwtPairCentroidsStd * nPercents

# onlyObserved = np.asarray([int(ff) for ff in np.unique(expandedBMU)])
#
# dwtPairCentroidsNormObserved = dwtPairCentroidsNorm[onlyObserved,:]


from sklearn.cluster import KMeans

# import time as timeP
# import mpl_toolkits.mplot3d.axes3d as p3
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.datasets import make_swiss_roll
# # Compute clustering
# print("Compute unstructured hierarchical clustering...")
# st = timeP.time()
# numClusters = 30
# X = dwtPairCentroidsNormObserved
#
#
# # Define the structure A of the data. Here a 10 nearest neighbors
# from sklearn.neighbors import kneighbors_graph
# connectivity = kneighbors_graph(X, n_neighbors=25, include_self=False)
#
# ward = AgglomerativeClustering(n_clusters=numClusters, connectivity=connectivity, linkage='ward').fit(X)
# elapsed_time = timeP.time() - st
# label = ward.labels_
# print("Elapsed time: %.2fs" % elapsed_time)
# print("Number of points: %i" % label.size)
#
#
# # #############################################################################
# # Plot result
# fig = plt.figure()
# #ax = p3.Axes3D(fig)
# #ax.view_init(7, -80)
# ax = plt.subplot2grid((2,2),(0,0),rowspan=2,colspan=2)
# for l in np.unique(label):
#     # ax.scatter(X[label == l, 0], X[label == l, 1],color=plt.cm.jet(np.float(l) / np.max(label + 1)),s=20, edgecolor='k')
#     ax.scatter(X[label == l, 0], X[label == l, 1],s=20, edgecolor='k')
#
# plt.title('Without connectivity constraints (time %.2fs)' % elapsed_time)
#
#
#
# # groupsize
# _, group_size = np.unique(ward.labels_, return_counts=True)
#
# # groups
# d_groups = {}
# dwtSize = []
# for k in range(numClusters):
#     dwtSize.append(len(np.where(ward.labels_ == k)[0]))
#     d_groups['{0}'.format(k)] = np.where(ward.labels_ == k)
#
#
# # centroids = np.zeros((numClusters, CPCs.shape[1]))
# # for k in range(numClusters):
# #     #print(PCsub[d_groups['{0}'.format(k)],0:1])
# #     centroids[k,:] = np.mean(CPCs[d_groups['{0}'.format(k)],:], axis=1) #/var_explained
#
# bmus = ward.labels_
#
# allClustersWard = []
# for qq in range(30):
#     # num = onlyObserved[qq]
#     finderInd = np.where((ward.labels_ == qq))
#     clusterList = []
#     for pp in range(len(finderInd[0])):
#         num = onlyObserved[finderInd[0][pp]]
#         secondFinder = np.where((expandedBMU == num))
#         # secondFinder = np.where((expandedBMU == finderInd[0][pp]))
#         clusterList.append(secondFinder)
#     # allClusters.append(np.array(clusterList).flatten())
#     allClustersWard.append(np.column_stack(clusterList).ravel())
#
# group_Dwt_s = np.asarray([len(yy) for yy in allClustersWard])

# CAN WE PUT A GUIDED REGRESSION IN HERE FOR THE PRESENCE OF WAVES?

numClusters = 100
min_group_size = 50
#  KMeans
keep_iter = True
count_iter = 0
while keep_iter:
    # n_init: number of times KMeans runs with different centroids seeds
    kma = KMeans(n_clusters=numClusters, n_init=1000).fit(dwtPairCentroidsNorm)

    #  check minimun group_size
    group_keys, group_size = np.unique(kma.labels_, return_counts=True)

    # sort output
    group_k_s = np.column_stack([group_keys, group_size])
    group_k_s = group_k_s[group_k_s[:, 0].argsort()]  # sort by cluster num



    # allClusters = []
    # for qq in range(numClusters):
    #     # num = onlyObserved[qq]
    #     finderInd = np.where((kma.labels_ == qq))
    #     clusterList = []
    #     for pp in range(len(finderInd[0])):
    #         num = onlyObserved[finderInd[0][pp]]
    #         secondFinder = np.where((expandedBMU == num))
    #         # secondFinder = np.where((expandedBMU == finderInd[0][pp]))
    #         clusterList.append(secondFinder)
    #     # allClusters.append(np.array(clusterList).flatten())
    #     allClusters.append(np.column_stack(clusterList).ravel())

    # group_Dwt_s = np.asarray([len(yy) for yy in allClusters])

    if not min_group_size:
        keep_iter = False

    else:
        # keep iterating?
        keep_iter = np.where(group_k_s[:, 1] < min_group_size)[0].any()
        # keep_iter = np.where(group_k_s < min_group_size)[0].any()

        count_iter += 1
        # log kma iteration
        print('KMA iteration info:')
        for rr in group_k_s:
            print('  cluster: {0}, size: {1}'.format(rr[0], rr[1]))
        print('Try again: ', keep_iter)
        print('Total attemps: ', count_iter)
        print()
        # # log kma iteration
        # print('KMA iteration info:')
        # for rr in range(len(group_k_s)):
        #     print('  cluster: {0}, size: {1}'.format(rr, group_k_s[rr]))
        # print('Try again: ', keep_iter)
        # print('Total attemps: ', count_iter)
        # print()



print('KMA iteration info:')
for rr in range(len(group_k_s)):
    print('  cluster: {0}, size: {1}'.format(rr, group_k_s[rr]))
print('Total attemps: ', count_iter)
print()

print('Total clustered days: {}'.format(np.sum(group_k_s)))

# groups
d_groups = {}
for k in range(numClusters):
    d_groups['{0}'.format(k)] = np.where(kma.labels_ == k)
# centroids
#



asdfg

# slpClusterCentroids = np.dot(kma.cluster_centers_[:,0:10], slpEOFsub[0:10,0:len(slpStd)])
# iceClusterCentroids = np.dot(kma.cluster_centers_[:,10:15], iceEOFsub[0:5,:])
# slpClusterCentroids = np.dot(kma.cluster_centers_[:,0:32], slpEOFsub[0:32,0:len(slpStd)])
# iceClusterCentroids = np.dot(kma.cluster_centers_[:,32:41], iceEOFsub[0:9,:])
slpClusterCentroids = np.dot(kma.cluster_centers_[:,0:10], slpEOFsub[0:10,0:len(slpStd)])
iceClusterCentroids = np.dot(kma.cluster_centers_[:,10:12], iceEOFsub[0:2,:])
# # km, x and var_centers
kmIce = np.multiply(iceClusterCentroids,np.tile(iceStd, (numClusters, 1))) + np.tile(iceMean, (numClusters, 1))
kmSlp = np.multiply(slpClusterCentroids,np.tile(slpStd, (numClusters, 1))) + np.tile(slpMean, (numClusters, 1))

#
# # # sort kmeans
# kma_order = np.argsort(np.mean(-kmIce, axis=1))
kma_order = np.argsort(kma.cluster_centers_[:,10])

#
# # sort kmeans
# kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, num_clusters)
#
# bmus_corrected = np.zeros((len(kma.labels_),), ) * np.nan
# for i in range(num_clusters):
#     posc = np.where(kma.labels_ == kma_order[i])
#     bmus_corrected[posc] = i
#
# # reorder centroids
# sorted_cenEOFs = kma.cluster_centers_[kma_order, :]
# sorted_centroids = centroids[kma_order, :]
#
# repmatDesviacion = np.tile(iceStd, (num_clusters,1))
# repmatMedia = np.tile(iceMean, (num_clusters,1))
# Km = np.multiply(sorted_centroids,repmatDesviacion) + repmatMedia
#
#




import cartopy.crs as ccrs

from matplotlib import gridspec
plt.style.use('default')
# plotting the EOF patterns
fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(10, 10)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for hh in range(numClusters):
    ax = plt.subplot(gs1[hh],projection=ccrs.NorthPolarStereo(central_longitude=-45))
    num = kma_order[hh]
    tempInd = allClusters[num] #np.where(kma.labels_== hh)

    if len(tempInd) > 0:
        spatialField = np.nanmean(gapFilledIce[tempInd], axis=0)

        linearField = np.ones((np.shape(iceXflat))) * np.nan
        linearField[icePoints] = spatialField
        rectField = linearField.reshape(65, 70)

        ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
        extent = [-9.97, 168.35, 30.98, 34.35]
        ax.pcolormesh(iceX[70:140], iceY[165:230], rectField, cmap=cmocean.cm.ice)

        ax.set_xlim([-2220000, -275000])
        ax.set_ylim([50000, 1850000])




slpLon = historicalDWTs['lon']
slpLat = historicalDWTs['lat']
X_in = historicalDWTs['X_in']
sea_nodes = historicalDWTs['sea_nodes']

from mpl_toolkits.basemap import Basemap
# import basemap as Basemap

import matplotlib.cm as cm
dwtcolors = cm.rainbow(np.linspace(0, 1,110))
# plotting the SLP patterns
fig2 = plt.figure(figsize=(11,10))
gs1 = gridspec.GridSpec(10, 10)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.plt.figure()
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for hh in range(numClusters):

    ax = plt.subplot(gs1[hh])
    num = kma_order[hh]
    tempInd = allClusters[num] #np.where(kma.labels_== hh)
    if len(tempInd) > 0:

        m = Basemap(projection='npstere', boundinglat=50, lon_0=180, resolution='l')
        cx,cy =m(slpLon,slpLat)
        m.drawcoastlines()

        # spatialField = np.multiply(EOFs[hh,0:(len(Xsea))],np.sqrt(variance[hh]))
        # spatialField = Km_slp[(hh), :] / 100 - np.nanmean(SLP, axis=0) / 100
        # spatialField = kmSorted[(hh), 0: (len(Xsea))] / 100 - np.nanmean(SLP, axis=0) / 100
        spatialField = np.nanmean(SLP[tempInd], axis=0) / 100 - np.nanmean(SLP, axis=0) / 100

        rectField = np.ones((np.shape(X_in))) * np.nan
        for tt in range(len(sea_nodes)):
            rectField[sea_nodes[tt]] = spatialField[tt]

        m.fillcontinents(color=dwtcolors[hh])

        clevels = np.arange(-25,25,1)

        #ax.pcolormesh(cx, cy, rectField)#, cmap=cmocean.cm.ice)
        CS = m.contourf(cx, cy, rectField, clevels, vmin=-20, vmax=20, cmap=cm.RdBu_r)#, shading='gouraud')

        ax.set_xlim([np.min(cx)+10000, np.max(cx)+10000])
        ax.set_ylim([np.min(cy)+10000, np.max(cy)+10000])
        #tx, ty = m(320, -30)
        ax.text(np.min(cx)+(np.max(cx)-np.min(cx))/3.2*2, np.min(cy)+(np.max(cy)-np.min(cy))/9, '{}'.format(group_Dwt_s[num]))

    #ax.set_title('{}'.format(group_size[num]))

    # c2 += 1
    # if c2 == 9:
    #     c1 += 1
    #     c2 = 0
    #
    # if plotIndx < 9:
    #     ax.xaxis.set_ticks([])
    #     ax.xaxis.set_ticklabels([])
    # if plotIndy > 0:
    #     ax.yaxis.set_ticklabels([])
    #     ax.yaxis.set_ticks([])
    # counter = counter + 1
    # if plotIndy < 9:
    #     plotIndy = plotIndy + 1
    # else:
    #     plotIndy = 0
    #     plotIndx = plotIndx + 1


dayOfYear = np.array([hh.timetuple().tm_yday for hh in dayTime])  # returns 1 for January 1st


# plotting the EOF patterns
fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(10, 10)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for qq in range(numClusters):
    ax = plt.subplot(gs1[qq])#,projection=ccrs.NorthPolarStereo(central_longitude=-45))
    # tempInd = np.where(bmus_corrected==hh)
    num = kma_order[qq]
    tempInd = allClusters[num]
    if len(tempInd) > 0:

        # tempTime = np.asarray(dayTime)[tempInd[0]]
        # tempMonths = [temp.month for temp in tempTime]
        #
        # ax.hist(tempMonths,bins = [-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5])
        # ax.set_xlim([0,13])
        tempDayOf = dayOfYear[tempInd]
        ax.hist(tempDayOf,bins = np.arange(0,367))
        ax.set_xlim([0,367])
        ax.text(10,1,'{}'.format(len(tempDayOf)))






# plotting the EOF patterns
fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(10, 10)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for qq in range(numClusters):
    ax = plt.subplot(gs1[qq])#,projection=ccrs.NorthPolarStereo(central_longitude=-45))
    # tempInd = np.where(bmus_corrected==hh)
    num = kma_order[qq]
    tempInd = allClusters[num]
    if len(tempInd) > 0:

        # tempTime = np.asarray(dayTime)[tempInd[0]]
        # tempMonths = [temp.month for temp in tempTime]
        #
        # ax.hist(tempMonths,bins = [-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5])
        # ax.set_xlim([0,13])
        tempDayOf = dayOfYear[tempInd]
        tempHsOf = dailyMeanHs[tempInd]

        # ax.hist(tempDayOf,bins = np.arange(0,367))
        ax.scatter(tempDayOf,tempHsOf)

        ax.set_xlim([0,367])
        ax.text(10,1,'{}'.format(len(tempDayOf)))
        ax.set_ylim([0,5])





bmus_corrected = np.zeros((len(dayOfYear),), ) * np.nan

for qq in range(numClusters):
    ax = plt.subplot(gs1[qq])#,projection=ccrs.NorthPolarStereo(central_longitude=-45))
    # tempInd = np.where(bmus_corrected==hh)
    num = kma_order[qq]
    tempInd = allClusters[num]
    bmus_corrected[tempInd] = int(qq)#*np.ones(len(tempInd))








waveData = mat73.loadmat('era5_waves_pthope.mat')
# wlData = mat73.loadmat('pthope_tides_for_dylan.mat')
wlData = mat73.loadmat('updated_env_for_dylan.mat')

wlArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in wlData['time']]
waveArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in waveData['time_all']]


wh_all = np.hstack((waveData['wh_all'],np.nan*np.zeros(3624,)))
tp_all = np.hstack((waveData['tp_all'],np.nan*np.zeros(3624,)))
dm_all = np.hstack((waveData['mwd_all'],np.nan*np.zeros(3624,)))

st = dt.datetime(2022, 1, 1)
end = dt.datetime(2022,6,1)
from dateutil.relativedelta import relativedelta
step = relativedelta(hours=1)
hourTime = []
while st < end:
    hourTime.append(st)#.strftime('%Y-%m-%d'))
    st += step

time_all = np.hstack((waveArrayTime,hourTime))



tWave = np.array(time_all)[24*31:]
hsCombined = wh_all[24*31:]#waveData['wh_all']
tpCombined = tp_all[24*31:]#waveData['tp_all']
dmCombined = dm_all[24*31:]#waveData['mwd_all']


waveNorm = dmCombined



def hourlyVec2datetime(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [datetime(d[0], d[1], d[2], d[3]) for d in d_vec]

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)





def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return datetime.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           + timedelta(hours=int(hours)) \
           + timedelta(minutes=int(minutes)) \
           + timedelta(seconds=round(seconds)) \
           - timedelta(days=366)





dt = datetime(1979, 2, 1)
end = datetime(2022, 6, 1)
step = timedelta(days=1)
midnightTime = []
while dt < end:
    midnightTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step


# bmus = newBmus[0:len(midnightTime)]
bmus = bmus_corrected#newBmus[0:len(midnightTime)]

# newBmus
# bmus = bmus[0:len(midnightTime)]
import itertools
import operator
grouped = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(bmus), key=operator.itemgetter(1))]

groupLength = np.asarray([len(i) for i in grouped])
bmuGroup = np.asarray([bmus[i[0]] for i in grouped])
timeGroup = [np.asarray(midnightTime)[i] for i in grouped]

# fetchGroup = np.asarray([fetchFiltered[i[0]] for i in grouped])

startTimes = [i[0] for i in timeGroup]
endTimes = [i[-1] for i in timeGroup]

hydrosInds = np.unique(bmuGroup)



bmusRange = np.arange(0,(numClusters))
hydros = list()
c = 0
for p in bmusRange:#range(len(np.unique(bmuGroup))):

    tempInd = p#hydrosInds[p]
    print('working on bmu = {}'.format(tempInd))
    index = np.where((bmuGroup==tempInd))[0][:]
    if len(index) == 0:
        print('no waves here')
        tempList = list()
    else:

        print('should have at least {} days in it'.format(len(index)))
        subLength = groupLength[index]
        m = np.ceil(np.sqrt(len(index)))
        tempList = list()


        for i in range(len(index)):
            st = startTimes[index[i]]
            et = endTimes[index[i]] + timedelta(days=1)

            # tempFetch = fetchGroup[index[i]]#np.array([fetchFiltered[index[i]]])

            if et > datetime(2022,5,31):
                print('done up to 2022/5/31')
            else:
                waveInd = np.where((tWave < et) & (tWave >= st))
                # waterInd = np.where((tWl < et) & (tWl >= st))
                if len(waveInd[0]) > 0:
                    newTime = startTimes[index[i]]

                    tempHydroLength = subLength[i]
                    counter = 0
                    while tempHydroLength > 1:
                        #randLength = random.randint(1,2)
                        etNew = newTime + timedelta(days=1)
                        waveInd = np.where((tWave < etNew) & (tWave >= newTime))
                        fetchInd = np.where((np.asarray(dayTime) == newTime))

                        # waterInd = np.where((tWl < etNew) & (tWl >= st))
                        deltaDays = et-etNew
                        #print('After first cut off = {}'.format(deltaDays.days))
                        c = c + 1
                        counter = counter + 1
                        if counter > 15:
                            print('we have split 15 times')

                        tempDict = dict()
                        tempDict['time'] = tWave[waveInd[0]]
                        tempDict['numDays'] = subLength[i]
                        tempDict['hs'] = hsCombined[waveInd[0]]
                        tempDict['tp'] = tpCombined[waveInd[0]]
                        tempDict['dm'] = waveNorm[waveInd[0]]
                        tempDict['fetch'] = fetchTotalConcentration[fetchInd[0][0]]#tempFetch[0]
                        # tempDict['res'] = res[waterInd[0]]
                        # tempDict['wl'] = wlHycom[waterInd[0]]
                        tempDict['cop'] = np.asarray([np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                                                      np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                                                      np.nanmean(waveNorm[waveInd[0]]),fetchTotalConcentration[fetchInd[0][0]]])#, np.nanmean(res[waterInd[0]])])
                        tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                        tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                        tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                        tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                        tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                        # tempDict['ssMean'] = np.nanmean(res[waterInd[0]])
                        tempList.append(tempDict)
                        tempHydroLength = tempHydroLength-1
                        newTime = etNew
                    else:
                        waveInd = np.where((tWave < et) & (tWave >= newTime))
                        fetchInd = np.where((np.asarray(dayTime) == newTime))

                        # waterInd = np.where((tWl < et) & (tWl >= etNew7))
                        c = c + 1
                        tempDict = dict()
                        tempDict['time'] = tWave[waveInd[0]]
                        tempDict['numDays'] = subLength[i]
                        tempDict['hs'] = hsCombined[waveInd[0]]
                        tempDict['tp'] = tpCombined[waveInd[0]]
                        tempDict['dm'] = waveNorm[waveInd[0]]
                        tempDict['fetch'] = fetchTotalConcentration[fetchInd[0][0]]#tempFetch[0]

                        # tempDict['res'] = res[waterInd[0]]
                        # tempDict['wl'] = wlHycom[waterInd[0]]
                        tempDict['cop'] = np.asarray(
                            [np.nanmin(hsCombined[waveInd[0]]),
                             np.nanmax(hsCombined[waveInd[0]]),
                             np.nanmin(tpCombined[waveInd[0]]),
                             np.nanmax(tpCombined[waveInd[0]]),
                             np.nanmean(
                                 waveNorm[waveInd[0]]),fetchTotalConcentration[fetchInd[0][0]]])  # ,
                        # np.nanmean(res[waterInd[0]])])
                        tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                        tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                        tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                        tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                        tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                        # tempDict['ssMean'] = np.nanmean(res[waterInd[0]])
                        tempList.append(tempDict)
    hydros.append(tempList)





### TODO: make a copula fit for each of the 70 DWTs
copulaData = list()
copulaDataNoNaNs = list()
for i in bmusRange:#(len(np.unique(bmuGroup))):
    tempHydros = hydros[i]

    dataCop = []
    dataCop2 = []
    for kk in range(len(tempHydros)):
        # if np.isnan(tempHydros[kk]['ssMean']):
        #     print('skipping a Nan in storm surge')
        # else:
        # dataCop.append(list([tempHydros[kk]['hsMax'],tempHydros[kk]['hsMin'],tempHydros[kk]['tpMax'],
        #                      tempHydros[kk]['tpMin'],tempHydros[kk]['dmMean'],tempHydros[kk]['ssMean'],kk]))
        dataCop.append(list([tempHydros[kk]['hsMax'],tempHydros[kk]['hsMin'],tempHydros[kk]['tpMax'],
                             tempHydros[kk]['tpMin'],tempHydros[kk]['dmMean'],tempHydros[kk]['fetch'],len(tempHydros[kk]['time']),kk]))
        # dataCop.append(list([tempHydros[kk]['hsMax'],tempHydros[kk]['hsMin'],tempHydros[kk]['tpMax'],
        #                      tempHydros[kk]['tpMin'],tempHydros[kk]['dmMean'],tempHydros[kk]['ssMean'],kk]))
        # if np.isnan(tempHydros[kk]['ssMean']):
        #     print('no storm surge')
        #     if i == 32:
        #         if np.isnan(tempHydros[kk]['hsMax']):
        #             print('skipping 32 with no waves')
        #         else:
        #             dataCop2.append(list([tempHydros[kk]['hsMax'], tempHydros[kk]['hsMin'], tempHydros[kk]['tpMax'],
        #                               tempHydros[kk]['tpMin'], tempHydros[kk]['dmMean'], 0, kk]))
        if np.isnan(tempHydros[kk]['hsMax']):
            print('no waves')

        else:
            dataCop2.append(list([tempHydros[kk]['hsMax'], tempHydros[kk]['hsMin'], tempHydros[kk]['tpMax'],
                                 tempHydros[kk]['tpMin'], tempHydros[kk]['dmMean'],tempHydros[kk]['fetch'], len(tempHydros[kk]['time']), kk]))
    copulaData.append(dataCop)
    copulaDataNoNaNs.append(dataCop2)



bmuDataNormalized = list()
bmuDataMin = list()
bmuDataMax = list()
bmuDataStd = list()
for i in bmusRange:#range(len(np.unique(bmuGroup))):
    temporCopula = np.asarray(copulaData[i])
    if len(temporCopula) == 0:
        bmuDataNormalized.append(np.vstack((0, 0)).T)
        bmuDataMin.append([0, 0])
        bmuDataMax.append([0, 0])
        bmuDataStd.append([0, 0])
    else:
        dataHs = np.array([sub[0] for sub in copulaData[i]])
        data = temporCopula[~np.isnan(dataHs)]
        data2 = data[~np.isnan(data[:,0])]
        if len(data2) == 0:
            print('woah, no waves here bub')
        #     data2 = data
        #     data2[:,5] = 0
            bmuDataNormalized.append(np.vstack((0, 0)).T)
            bmuDataMin.append([0, 0])
            bmuDataMax.append([0, 0])
            bmuDataStd.append([0, 0])
        else:

            maxHs = np.nanmax(data2[:,0])
            minHs = np.nanmin(data2[:,0])
            stdHs = np.nanstd(data2[:,0])
            hsNorm = (data2[:,0] - minHs) / (maxHs-minHs)
            maxTp = np.nanmax(data2[:,1])
            minTp = np.nanmin(data2[:,1])
            stdTp = np.nanstd(data2[:,1])
            tpNorm = (data2[:,1] - minTp) / (maxTp-minTp)
            # maxDm = np.nanmax(data2[:,4])
            # minDm = np.nanmin(data2[:,4])
            # stdDm = np.nanstd(data2[:,4])
            # dmNorm = (data2[:,4] - minDm) / (maxDm-minDm)
            # maxSs = np.nanmax(data2[:,5])
            # minSs = np.nanmin(data2[:,5])
            # stdSs = np.nanstd(data2[:,5])
            # ssNorm = (data2[:,5] - minSs) / (maxSs-minSs)
        # maxDm = np.nanmax(np.asarray(copulaData[i])[:,4])
        # minDm = np.nanmin(np.asarray(copulaData[i])[:,4])
        # stdDm = np.nanstd(np.asarray(copulaData[i])[:,4])
        # dmNorm = (np.asarray(copulaData[i])[:,4] - minDm) / (maxDm-minDm)
        # maxSs = np.nanmax(np.asarray(copulaData[i])[:,5])
        # minSs = np.nanmin(np.asarray(copulaData[i])[:,5])
        # stdSs = np.nanstd(np.asarray(copulaData[i])[:,5])
        # ssNorm = (np.asarray(copulaData[i])[:,5] - minSs) / (maxSs-minSs)
            bmuDataNormalized.append(np.vstack((hsNorm,tpNorm)).T)
            bmuDataMin.append([minHs,minTp])
            bmuDataMax.append([maxHs,maxTp])
            bmuDataStd.append([stdHs,stdTp])





normalizedHydros = list()
for i in bmusRange: #range(len(np.unique(bmuGroup))):
    tempHydros = hydros[i]
    tempList = list()
    for mm in range(len(tempHydros)):
        # if np.isnan(tempHydros[mm]['ssMean']):
        #     print('no storm surge')
        #     if i == 32:
        #         if np.isnan(tempHydros[mm]['hsMin']):
        #             print('skipping 32 with no waves')
        #         else:
        #             tempDict = dict()
        #             tempDict['hsNorm'] = (tempHydros[mm]['hs'] - tempHydros[mm]['hsMin']) / (
        #                         tempHydros[mm]['hsMax'] - tempHydros[mm]['hsMin'])
        #             tempDict['tpNorm'] = (tempHydros[mm]['tp'] - tempHydros[mm]['tpMin']) / (
        #                         tempHydros[mm]['tpMax'] - tempHydros[mm]['tpMin'])
        #             tempDict['timeNorm'] = np.arange(0, 1, 1 / len(tempHydros[mm]['time']))[0:len(tempDict['hsNorm'])]
        #             tempDict['dmNorm'] = (tempHydros[mm]['dm']) - tempHydros[mm]['dmMean']
        #             tempDict['ssNorm'] = np.zeros((len(tempDict['hsNorm']),))
        #             tempList.append(tempDict)
        if np.isnan(tempHydros[mm]['hsMin']):
            print('no waves')

        else:
            tempDict = dict()
            tempDict['hsNorm'] = (tempHydros[mm]['hs'] - tempHydros[mm]['hsMin']) / (tempHydros[mm]['hsMax']- tempHydros[mm]['hsMin'])
            tempDict['tpNorm'] = (tempHydros[mm]['tp'] - tempHydros[mm]['tpMin']) / (tempHydros[mm]['tpMax']- tempHydros[mm]['tpMin'])
            tempDict['timeNorm'] = np.arange(0,1,1/len(tempHydros[mm]['time']))[0:len(tempDict['hsNorm'])]
            tempDict['dmNorm'] = (tempHydros[mm]['dm']) - tempHydros[mm]['dmMean']
            # tempDict['ssNorm'] = (tempHydros[mm]['res']) - tempHydros[mm]['ssMean']
            tempList.append(tempDict)
    normalizedHydros.append(tempList)




import pickle

normHydrosPickle = 'normalizedWaveHydrographsHope2Dist49.pickle'
outputHydrosNorm = {}
outputHydrosNorm['normalizedHydros'] = normalizedHydros
outputHydrosNorm['bmuDataMin'] = bmuDataMin
outputHydrosNorm['bmuDataMax'] = bmuDataMax
outputHydrosNorm['bmuDataStd'] = bmuDataStd
outputHydrosNorm['bmuDataNormalized'] = bmuDataNormalized

with open(normHydrosPickle,'wb') as f:
    pickle.dump(outputHydrosNorm, f)


hydrosPickle = 'waveHydrographsReprojected81.pickle'
outputHydros = {}
outputHydros['hydros'] = hydros
with open(hydrosPickle,'wb') as f:
    pickle.dump(outputHydros, f)

copPickle = 'hydrographCopulaDataRepreojected81.pickle'
outputCopula = {}
outputCopula['copulaData'] = copulaData
outputCopula['copulaDataNoNaNs'] = copulaDataNoNaNs
with open(copPickle,'wb') as f:
    pickle.dump(outputCopula, f)


historicalPickle = 'historicalDataReprojected81.pickle'
outputHistorical = {}
outputHistorical['grouped'] = grouped
outputHistorical['groupLength'] = groupLength
outputHistorical['bmuGroup'] = bmuGroup
outputHistorical['timeGroup'] = timeGroup

with open(historicalPickle,'wb') as f:
    pickle.dump(outputHistorical, f)

from datetime import datetime, date, timedelta
import random


with open(r"dwt49RGHistoricalSimulations100withTemp.pickle", "rb") as input_file:
   dwtFutureSimulations = pickle.load(input_file)
evbmus_sim = dwtFutureSimulations['evbmus_sim']
dates_sim = dwtFutureSimulations['dates_sim']

with open(r"ice36FinalFutureSimulations1000.pickle", "rb") as input_file:
    iceSimulations = pickle.load(input_file)
iceSims = iceSimulations['evbmus_sim']
iceDates = iceSimulations['dates_sim']

with open(r"processedWaveWaterLevels49DWTs36IWTs2022.pickle", "rb") as input_file:
   onOffData = pickle.load(input_file)
percentWaves = onOffData['percentWaves']
percentIce = onOffData['percentIce']


# with open(r"iceData36ClustersDayNumRGAdjusted.pickle", "rb") as input_file:

with open(r"iceData36ClustersDayNumRGFinalized.pickle", "rb") as input_file:
   iceDWTs = pickle.load(input_file)



with open(r"ice36FetchSims.pickle", "rb") as input_file:
   fwtchSimsPickle = pickle.load(input_file)
fetchSims = fwtchSimsPickle['fetchSims']


iceOnOff = []
iceArea = []
years = np.arange(1979,2022)
# lets start in January and assume that we are always beginning with an iced ocean
for hh in range(100):
    simOfIce = iceSims[:,hh]
    yearsStitched = []
    yearsIceStitched = []
    timeStitched = []
    tempControl = 0

    for tt in range(len(years)):
        yearInd = np.where((np.array(iceDates) >= datetime(years[tt],1,1)) & (np.array(iceDates) < datetime(years[tt]+1,1,1)))
        timeSubset = np.array(iceDates)[yearInd]
        tempOfSim = simOfIce[yearInd]
        groupTemp = [[e[0] for e in d[1]] for d in
                            itertools.groupby(enumerate(tempOfSim), key=operator.itemgetter(1))]
        timeGroupTemp = [timeSubset[d] for d in groupTemp]
        bmuGroupIce = (np.asarray([tempOfSim[i[0]] for i in groupTemp]))
        withinYear = 0

        # lets a
        wavesStitched = []
        iceStitched = []
        for qq in range(len(groupTemp)):
            tempRand = np.random.uniform(size=len(groupTemp[qq]))
            randIceSize = [random.randint(0, 9999) for xx in range(len(tempRand))]
            iceDetails = fetchSims[int(bmuGroupIce[qq]-1)][randIceSize]
            if tempControl == 0:
                # lets enter the next group assuming that waves were previously off
                sortedRand = np.sort(tempRand)            # so we have a bunch of probabilities between 0 and 1
                finder = np.where((sortedRand > percentIce[int(bmuGroupIce[qq]-1)]))     # if greater than X then waves are ON
                # arbitrary, but lets say waves have to be turned on for 3 days before we say, sure the ocean broke up
                waveDays = np.zeros((len(sortedRand),))
                if timeGroupTemp[qq][0].month > 3:
                    if len(finder[0]) > 3:
                        if timeGroupTemp[qq][0].month > 10:
                            if (len(finder[0])/len(sortedRand))>0.7:
                                waveDays[finder[0]] = np.ones((len(finder[0])))  # flip those days that are waves to ON
                                tempControl = 1
                            else:
                                print('prevented wave from turning on Sim {}'.format(hh))
                        else:
                            waveDays[finder[0]] = np.ones((len(finder[0])))       # flip those days that are waves to ON
                            tempControl = 1
                    if timeGroupTemp[qq][0].month > 7 and timeGroupTemp[qq][-1].month < 9:
                        waveDays = np.ones((len(sortedRand),))
                        tempControl = 1
                wavesStitched.append(waveDays)
                # add iceDetails here
                if timeGroupTemp[qq][0].month < 9:
                    iceStitched.append(np.sort(iceDetails))
                else:
                    iceStitched.append(np.sort(iceDetails)[::-1])

            else:
                # now we're entering with waves ON and want to turn them OFF
                sortedRand = np.sort(tempRand)#[::-1]        # again a bunch of probabilities between 0 and 1
                finder = np.where((sortedRand > percentWaves[int(bmuGroupIce[qq]-1)]))  # if greater than X then waves are OFF
                # arbitrary, but again, waves need to be off for 3 days to really freeze up the ocean
                waveDays = np.ones((len(sortedRand),))
                if timeGroupTemp[qq][-1].month < 8 or timeGroupTemp[qq][0].month > 9:
                    if len(finder[0]) > 1:
                        waveDays[finder[0]] = np.zeros((len(finder[0])))
                        tempControl = 0
                if timeGroupTemp[qq][0].month > 1 and timeGroupTemp[qq][-1].month < 5:
                    waveDays = np.zeros((len(sortedRand),))
                    tempControl = 0

                if len(wavesStitched) > 0:
                    countOfStitched = np.sum(np.concatenate(wavesStitched,axis=0))
                    if timeGroupTemp[qq][0].month > 5 and countOfStitched > 7:
                        waveDays = np.ones((len(sortedRand),))

                wavesStitched.append(waveDays)
                if timeGroupTemp[qq][0].month < 9:
                    iceStitched.append(np.sort(iceDetails))
                else:
                    iceStitched.append(np.sort(iceDetails)[::-1])

        yearsStitched.append(wavesStitched)
        yearsIceStitched.append(iceStitched)
    iceOnOff.append(np.concatenate([x for xs in yearsStitched for x in xs]).ravel())
    iceArea.append(np.concatenate([x for xs in yearsIceStitched for x in xs]).ravel())


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


startIce = np.array([8,9,10,12,14,16,18,20,22,24,26,29,32,35,39,43,47,51,55,60,65,70,75,80,85,90,95,100,106,112])
endIce = np.array([10,12,13,14,15,16,17,16,15,14,13,14,15,16,16,17,18,19,20,16,17,15,14,20,22,16,18,22,26])
iceAreaSmooth = []
for n in range(100):
    print('smoothing ice area simulation {}'.format(n))
    tempIceArea = iceArea[n]
    smoothedIce2 = running_mean(tempIceArea, 60)
    contIce = np.hstack((startIce,smoothedIce2,endIce))
    iceAreaSmooth.append(contIce)

#
#
#
# newOrderIce = np.array([0,0,0,1,0,
#                         0,0,0,0,0,
#                         0,0,0,0,1,
#                         0,1,1,1,1,
#                         1,1,1,1,1])
#

dt = datetime(1979,6, 1)
end = datetime(2022, 6, 1)
# dt = datetime(2021,6, 1)
# end = datetime(2121, 6, 1)
step = timedelta(days=1)
midnightTime = []
while dt < end:
    midnightTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step


bmus_dates_months = np.array([d.month for d in midnightTime])

groupedList = list()
groupLengthList = list()
bmuGroupList = list()
iceOnOffGroupList = list()
iceAreaGroupList = list()
iceSlpGroupList = list()
timeGroupList = list()
for hh in range(100):
    print('breaking up hydrogrpahs for simulation {}'.format(hh))
    bmus = evbmus_sim[:,hh]
    iceBmus = iceOnOff[hh]
    actualIceBmus = iceSims[:,hh]
    iceAreaBmus = iceAreaSmooth[hh]

    tempBmusGroup = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(bmus), key=operator.itemgetter(1))]

    bmusICESLP = np.zeros((len(actualIceBmus),), ) * np.nan



    for i in range(2):
        posc = np.where(newOrderIce == i)
        for hh in range(len(posc[0])):
            posc2 = np.where(actualIceBmus == posc[0][hh])
            bmusICESLP[posc2] = i



    groupedList.append(tempBmusGroup)
    groupLengthList.append(np.asarray([len(i) for i in tempBmusGroup]))
    bmuGroupList.append(np.asarray([bmus[i[0]] for i in tempBmusGroup]))
    iceOnOffGroupList.append([iceBmus[i] for i in tempBmusGroup])
    iceAreaGroupList.append([iceAreaBmus[i] for i in tempBmusGroup])
    iceSlpGroupList.append([bmusICESLP[i] for i in tempBmusGroup])
    timeGroupList.append([bmus_dates_months[i] for i in tempBmusGroup])



### TODO: use the historical to develop a new chronology

numRealizations = 100


simBmuChopped = []
simBmuLengthChopped = []
simBmuGroupsChopped = []
simIceGroupsChopped = []
simIceSlpGroupsChopped = []
simIceAreaGroupsChopped = []
simTimeGroupsChopped = []
for pp in range(numRealizations):

    print('working on realization #{}'.format(pp))
    bmuGroup = bmuGroupList[pp]
    groupLength = groupLengthList[pp]
    grouped = groupedList[pp]
    iceGroup = iceOnOffGroupList[pp]
    iceSlpGroup = iceSlpGroupList[pp]
    iceAreaGroup = iceAreaGroupList[pp]
    timeGroup = timeGroupList[pp]

    simGroupLength = []
    simGrouped = []
    simBmu = []
    simIceGroupLength = []
    simIceGrouped = []
    simIceBmu = []
    simIceSlpGrouped = []
    simIceAreaGrouped = []
    simTimeGrouped = []
    for i in range(len(groupLength)):
        # if np.remainder(i,10000) == 0:
        #     print('done with {} hydrographs'.format(i))
        tempGrouped = grouped[i]
        tempBmu = int(bmuGroup[i])
        tempIceGrouped = iceGroup[i]
        tempIceSlpGrouped = iceSlpGroup[i]
        tempTimeGrouped = timeGroup[i]
        tempIceAreaGrouped = iceAreaGroup[i]
        remainingDays = groupLength[i]
        counter = 0
        # if groupLength[i] < 2:
        #     simGroupLength.append(int(groupLength[i]))
        #     simGrouped.append(grouped[i])
        #     simIceGrouped.append(iceGroup[i])
        #     simIceSlpGrouped.append(iceSlpGroup[i])
        #     simTimeGrouped.append(timeGroup[i])
        #     simBmu.append(tempBmu)
        # else:
        #     counter = 0
        while (len(grouped[i]) - counter) > 1:
                # print('we are in the loop with remainingDays = {}'.format(remainingDays))
                # random days between 3 and 5
            randLength = 1#random.randint(1, 2)
                # add this to the record
            simGroupLength.append(int(randLength))
                # simGrouped.append(tempGrouped[0:randLength])
                # print('should be adding {}'.format(grouped[i][counter:counter+randLength]))
            simGrouped.append(grouped[i][counter:counter+randLength])
            simIceGrouped.append(iceGroup[i][counter:counter+randLength])
            simIceSlpGrouped.append(iceSlpGroup[i][counter:counter+randLength])
            simTimeGrouped.append(timeGroup[i][counter:counter+randLength])
            simIceAreaGrouped.append(iceAreaGroup[i][counter:counter+randLength])
            simBmu.append(tempBmu)
                # remove those from the next step
                # tempGrouped = np.delete(tempGrouped,np.arange(0,randLength))
                # do we continue forward
            remainingDays = remainingDays - randLength
            counter = counter + randLength

            # if (len(grouped[i]) - counter) > 0:
        simGroupLength.append(int((len(grouped[i]) - counter)))
                # simGrouped.append(tempGrouped[0:])
        simGrouped.append(grouped[i][counter:])
        simIceGrouped.append(iceGroup[i][counter:])
        simIceSlpGrouped.append(iceSlpGroup[i][counter:])
        simTimeGrouped.append(timeGroup[i][counter:])
        simIceAreaGrouped.append(iceAreaGroup[i][counter:])

        simBmu.append(tempBmu)

    simBmuLengthChopped.append(np.asarray(simGroupLength))
    simBmuGroupsChopped.append(simGrouped)
    simIceGroupsChopped.append(simIceGrouped)
    simBmuChopped.append(np.asarray(simBmu))
    simIceSlpGroupsChopped.append(simIceSlpGrouped)
    simIceAreaGroupsChopped.append(simIceAreaGrouped)
    simTimeGroupsChopped.append(simTimeGrouped)





simsChoppedPickle = 'simulations100Chopped49_2Dist.pickle'
outputSimsChopped = {}
outputSimsChopped['simBmuLengthChopped'] = simBmuLengthChopped
outputSimsChopped['simBmuGroupsChopped'] = simBmuGroupsChopped
outputSimsChopped['simBmuChopped'] = simBmuChopped
outputSimsChopped['simIceGroupsChopped'] = simIceGroupsChopped
outputSimsChopped['simIceSlpGroupsChopped'] = simIceSlpGroupsChopped
outputSimsChopped['simTimeGroupsChopped'] = simTimeGroupsChopped
outputSimsChopped['simIceAreaGroupsChopped'] = simIceAreaGroupsChopped

with open(simsChoppedPickle,'wb') as f:
    pickle.dump(outputSimsChopped, f)

