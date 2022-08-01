import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import numpy as np
import pandas as pd
import os
import datetime
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
# import mda
from scipy.spatial import distance_matrix

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import datetime
from dateutil.relativedelta import relativedelta
import random
import xarray as xr
import geopy.distance
from dipy.segment.metric import Metric
from dipy.segment.metric import ResampleFeature
from dipy.segment.clustering import QuickBundles
from mpl_toolkits.basemap import Basemap

def dateDay2datetime(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''
    return [datetime.datetime(d[0], d[1], d[2]) for d in d_vec]

def dateDay2datetimeDate(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''
    return [datetime.date(d[0], d[1], d[2]) for d in d_vec]


def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color


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


def dt2cal(dt):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.

    Parameters
    ----------
    dt : datetime64 array (...)
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 7)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """

    # allocate output
    out = np.empty(dt.shape + (7,), dtype="u4")
    # decompose calendar floors
    Y, M, D, h, m, s = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = Y + 1970 # Gregorian Year
    out[..., 1] = (M - Y) + 1 # month
    out[..., 2] = (D - M) + 1 # dat
    out[..., 3] = (dt - D).astype("m8[h]") # hour
    out[..., 4] = (dt - h).astype("m8[m]") # minute
    out[..., 5] = (dt - m).astype("m8[s]") # second
    out[..., 6] = (dt - s).astype("m8[us]") # microsecond
    return out



def sort_cluster_gen_corr_end(centers, dimdim):
    '''
    SOMs alternative
    '''
    # TODO: DOCUMENTAR.

    # get dimx, dimy
    dimy = np.floor(np.sqrt(dimdim)).astype(int)
    dimx = np.ceil(np.sqrt(dimdim)).astype(int)

    if not np.equal(dimx*dimy, dimdim):
        # TODO: RAISE ERROR
        pass

    dd = distance_matrix(centers, centers)
    qx = 0
    sc = np.random.permutation(dimdim).reshape(dimy, dimx)

    # get qx
    for i in range(dimy):
        for j in range(dimx):

            # row F-1
            if not i==0:
                qx += dd[sc[i-1,j], sc[i,j]]

                if not j==0:
                    qx += dd[sc[i-1,j-1], sc[i,j]]

                if not j+1==dimx:
                    qx += dd[sc[i-1,j+1], sc[i,j]]

            # row F
            if not j==0:
                qx += dd[sc[i,j-1], sc[i,j]]

            if not j+1==dimx:
                qx += dd[sc[i,j+1], sc[i,j]]

            # row F+1
            if not i+1==dimy:
                qx += dd[sc[i+1,j], sc[i,j]]

                if not j==0:
                    qx += dd[sc[i+1,j-1], sc[i,j]]

                if not j+1==dimx:
                    qx += dd[sc[i+1,j+1], sc[i,j]]

    # test permutations
    q=np.inf
    go_out = False
    for i in range(dimdim):
        if go_out:
            break

        go_out = True

        for j in range(dimdim):
            for k in range(dimdim):
                if len(np.unique([i,j,k]))==3:

                    u = sc.flatten('F')
                    u[i] = sc.flatten('F')[j]
                    u[j] = sc.flatten('F')[k]
                    u[k] = sc.flatten('F')[i]
                    u = u.reshape(dimy, dimx, order='F')

                    f=0
                    for ix in range(dimy):
                        for jx in range(dimx):

                            # row F-1
                            if not ix==0:
                                f += dd[u[ix-1,jx], u[ix,jx]]

                                if not jx==0:
                                    f += dd[u[ix-1,jx-1], u[ix,jx]]

                                if not jx+1==dimx:
                                    f += dd[u[ix-1,jx+1], u[ix,jx]]

                            # row F
                            if not jx==0:
                                f += dd[u[ix,jx-1], u[ix,jx]]

                            if not jx+1==dimx:
                                f += dd[u[ix,jx+1], u[ix,jx]]

                            # row F+1
                            if not ix+1==dimy:
                                f += dd[u[ix+1,jx], u[ix,jx]]

                                if not jx==0:
                                    f += dd[u[ix+1,jx-1], u[ix,jx]]

                                if not jx+1==dimx:
                                    f += dd[u[ix+1,jx+1], u[ix,jx]]

                    if f<=q:
                        q = f
                        sc = u

                        if q<=qx:
                            qx=q
                            go_out=False

    return sc.flatten('F')

# SLPs = ReadMatfile('/media/dylananderson/Elements/pointHope/slpsPolarGrid2v4y2022.mat')
# SLPs = ReadMatfile('/media/dylananderson/Elements1/pointHope/slpsPolarGrid2v2.mat')
#
import mat73

SLPs = mat73.loadmat('/media/dylananderson/Elements/pointHope/slpsPolarGrid2v4y2022.mat')

X_in = SLPs['X_in']
Y_in = SLPs['Y_in']
Xsea = SLPs['Xsea']
Ysea = SLPs['Ysea']
SLP = SLPs['SLPsea']
GRD = SLPs['GRDsea']
SLPtime = SLPs['time']
inSea = SLPs['in']
onCoast = SLPs['on']

#
# asdfg
#
# import mat73
# import datetime as dt
# # define some constants
# epoch = dt.datetime(1970, 1, 1)
# matlab_to_epoch_days = 719529  # days from 1-1-0000 to 1-1-1970
# matlab_to_epoch_seconds = matlab_to_epoch_days * 24 * 60 * 60
#
# def matlab_to_datetime(matlab_date_num_seconds):
#     # get number of seconds from epoch
#     from_epoch = matlab_date_num_seconds - matlab_to_epoch_seconds
#
#     # convert to python datetime
#     return epoch + dt.timedelta(seconds=from_epoch)
#
# waveData = mat73.loadmat('era5_waves_pthope.mat')
#
#
# waveArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in waveData['time_all']]
#
# # wlData = mat73.loadmat('updated_env_for_dylan.mat')
# #
# # wlArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in wlData['time']]
#
#
# import cdsapi
#
# c = cdsapi.Client()
#
# c.retrieve(
#     'reanalysis-era5-single-levels',
#     {
#         'product_type': 'reanalysis',
#         'variable': [
#             'mean_wave_direction', 'peak_wave_period', 'significant_height_of_combined_wind_waves_and_swell',
#         ],
#         'year': '2022',
#         'month': '07',
#         'day': '01',
#         'time': [
#             '00:00', '01:00', '02:00',
#             '03:00', '04:00', '05:00',
#             '06:00', '07:00', '08:00',
#             '09:00', '10:00', '11:00',
#             '12:00', '13:00', '14:00',
#             '15:00', '16:00', '17:00',
#             '18:00', '19:00', '20:00',
#             '21:00', '22:00', '23:00',
#         ],
#         'format': 'netcdf',
#         'area': [
#             68.6, -167.1, 68.4,
#             -166.9,
#         ],
#     },
#     'download1.nc')
#
#
#
#
#
#
#
# import xarray as xr
#
# ds2 = xr.open_dataset('/home/dylananderson/Downloads/era5wavesJanToMay2022.nc')
# df2 = ds2.to_dataframe()
#
# wh_all = np.hstack((waveData['wh_all'],np.nan*np.zeros(3624,)))
# tp_all = np.hstack((waveData['tp_all'],np.nan*np.zeros(3624,)))
#
# st = dt.datetime(2022, 1, 1)
# # end = dt.datetime(2021,12,31)
# end = dt.datetime(2022,6,1)
# step = relativedelta(hours=1)
# hourTime = []
# while st < end:
#     hourTime.append(st)#.strftime('%Y-%m-%d'))
#     st += step
#
# time_all = np.hstack((waveArrayTime,hourTime))
# # swh = ds2['swh'][0,0,:,:]
#
#
# st = dt.datetime(1979, 2, 1)
# # end = dt.datetime(2021,12,31)
# end = dt.datetime(2022,6,1)
# step = relativedelta(days=1)
# dayTime = []
# while st < end:
#     dayTime.append(st)#.strftime('%Y-%m-%d'))
#     st += step
#
#
# # data = np.vstack((np.asarray(waveArrayTime),waveData['wh_all']))
# # data = np.vstack((data,waveData['tp_all']))
# # df = pd.DataFrame(data.T,columns=['time','wh_all','tp_all'])
#
#
#
# # data = np.vstack((waveData['wh_all'],waveData['tp_all']))
# # df = pd.DataFrame(data=data.T,index=np.asarray(waveArrayTime),columns=['wh_all','tp_all'])
# data = np.vstack((wh_all,tp_all))
# df = pd.DataFrame(data=data.T,index=np.asarray(time_all),columns=['wh_all','tp_all'])
#
# means = df.groupby(pd.Grouper(freq='1D')).mean()
#
#
# waves = means.loc['1979-1-1':'2022-5-31']
#
# # myDates = [dt.datetime.strftime(dt.datetime(year[hh],month[hh],day[hh]), "%Y-%m-%d") for hh in range(len(year))]
# # iceTime = [dt.datetime(np.array(year)[i],np.array(month)[i],np.array(day)[i]) for i in range(len(year))]
# myDates = [dt.datetime.strftime(dayTime[hh], "%Y-%m-%d") for hh in range(len(dayTime))]
#
# slpWaves = waves[waves.index.isin(myDates)]
#
# slpWaves = slpWaves.fillna(0)
#
#
import datetime as dt
st = dt.datetime(1979, 2, 1)
end = dt.datetime(2022,6,1)
step = relativedelta(days=1)
dayTime = []
while st < end:
    dayTime.append(st)#.strftime('%Y-%m-%d'))
    st += step
# iceDay
dayOfYear = np.array([hh.timetuple().tm_yday for hh in dayTime])  # returns 1 for January 1st
dayOfYearSine = np.sin(2*np.pi/366*dayOfYear)
dayOfYearCosine = np.cos(2*np.pi/366*dayOfYear)


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import linear_model

SlpGrd = np.hstack((SLP,GRD))
SlpGrdMean = np.mean(SlpGrd,axis=0)
SlpGrdStd = np.std(SlpGrd,axis=0)
SlpGrdNorm = (SlpGrd[:,:] - SlpGrdMean) / SlpGrdStd
SlpGrdNorm[np.isnan(SlpGrdNorm)] = 0
# 95% repres
repres = 0.951

# principal components analysis
ipca = PCA(n_components=min(SlpGrdNorm.shape[0], SlpGrdNorm.shape[1]))
PCs = ipca.fit_transform(SlpGrdNorm)
EOFs = ipca.components_
variance = ipca.explained_variance_
nPercent = variance / np.sum(variance)
APEV = np.cumsum(variance) / np.sum(variance) * 100.0
nterm = np.where(APEV <= repres * 100)[0][-1]

PCsub = PCs[:, :nterm - 1]
EOFsub = EOFs[:nterm - 1, :]

PCsub_std = np.std(PCsub, axis=0)
PCsub_norm = np.divide(PCsub, PCsub_std)

X = PCsub_norm  #  predictor

# PREDICTAND: WAVES data
#wd = np.array([xds_WAVES[vn].values[:] for vn in name_vars]).T
# wd = np.vstack((iceWaves['wh_all'],iceWaves['tp_all'],np.multiply(iceWaves['wh_all']**2,iceWaves['tp_all'])**(1.0/3))).T
wd = np.vstack((dayOfYearSine,dayOfYearCosine)).T

wd_std = np.nanstd(wd, axis=0)
wd_norm = np.divide(wd, wd_std)

Y = wd_norm  # predictand

# Adjust
[n, d] = Y.shape
X = np.concatenate((np.ones((n, 1)), X), axis=1)

clf = linear_model.LinearRegression(fit_intercept=True)
Ymod = np.zeros((n, d)) * np.nan
for i in range(d):
    clf.fit(X, Y[:, i])
    beta = clf.coef_
    intercept = clf.intercept_
    Ymod[:, i] = np.ones((n,)) * intercept
    for j in range(len(beta)):
        Ymod[:, i] = Ymod[:, i] + beta[j] * X[:, j]

# de-scale
Ym = np.multiply(Ymod, wd_std)



#import cartopy.crs as ccrs
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#[XR, YR] = np.meshgrid(XRs, YRs)
sea_nodes = []
for qq in range(len(Xsea)):
    sea_nodes.append(np.where((X_in == Xsea[qq]) & (Y_in == Ysea[qq])))



lat = SLPs['lat']
lon = SLPs['lon']
# plotting the EOF patterns
plt.figure()
c1 = 0
c2 = 0
for hh in range(9):
    #ax = plt.subplot2grid((3,3),(c1,c2),projection=ccrs.NorthPolarStereo(central_longitude=-45))
    ax = plt.subplot2grid((3,3),(c1,c2))#,projection=ccrs.NorthPolarStereo(central_longitude=-45))
    # m = Basemap(projection='merc',llcrnrlat=-40,urcrnrlat=55,llcrnrlon=255,urcrnrlon=375,lat_ts=10,resolution='c')
    m = Basemap(projection='npstere', boundinglat=50, lon_0=180, resolution='l')

    cx,cy =m(lon,lat)
    m.drawcoastlines()

    spatialField = np.multiply(EOFs[hh,0:(len(Xsea))],np.sqrt(variance[hh]))

    # linearField = np.ones((np.shape(X_in.flatten()))) * np.nan
    # linearField[indexIn[0]] = spatialField
    # rectField = linearField.reshape(96,91)

    rectField = np.ones((np.shape(X_in))) * np.nan
    for tt in range(len(sea_nodes)):
        rectField[sea_nodes[tt]] = spatialField[tt]

    ax.pcolormesh(cx, cy, rectField)#, cmap=cmocean.cm.ice)

    ax.set_xlim([np.min(cx)+10000, np.max(cx)+10000])
    ax.set_ylim([np.min(cy)+10000, np.max(cy)+10000])
    ax.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
    c2 += 1
    if c2 == 3:
        c1 += 1
        c2 = 0



# KMA Regression Guided
num_clusters = 81
repres = 0.95
alpha = 0.3
min_size = None  # any int will activate group_min_size iteration
min_group_size=50

# xds_KMA = KMA_regression_guided(
#     xds_PCA, xds_Yregres, num_clusters,
#     repres, alpha, min_size
# )
# print(xds_KMA)


'''
 KMeans Classification for PCA data: regression guided

 xds_PCA:
     (n_components, n_components) PCs
     (n_components, n_features) EOFs
     (n_components, ) variance
 xds_Yregres:
     (time, vars) Ym
 num_clusters
 repres
 '''


Y = Ym
# min_group_size=60

# append Yregres data to PCs
data = np.concatenate((PCsub, Y), axis=1)
data_std = np.std(data, axis=0)
data_mean = np.mean(data, axis=0)

#  normalize but keep PCs weigth
data_norm = np.ones(data.shape) * np.nan
for i in range(PCsub.shape[1]):
    data_norm[:, i] = np.divide(data[:, i] - data_mean[i], data_std[0])
for i in range(PCsub.shape[1], data.shape[1]):
    data_norm[:, i] = np.divide(data[:, i] - data_mean[i], data_std[i])

# apply alpha (PCs - Yregress weight)
data_a = np.concatenate(
    ((1 - alpha) * data_norm[:, :nterm],
     alpha * data_norm[:, nterm:]),
    axis=1
)

#  KMeans
keep_iter = True
count_iter = 0
while keep_iter:
    # n_init: number of times KMeans runs with different centroids seeds
    kma = KMeans(n_clusters=num_clusters, n_init=100).fit(data_a)

    #  check minimun group_size
    group_keys, group_size = np.unique(kma.labels_, return_counts=True)

    # sort output
    group_k_s = np.column_stack([group_keys, group_size])
    group_k_s = group_k_s[group_k_s[:, 0].argsort()]  # sort by cluster num

    if not min_group_size:
        keep_iter = False

    else:
        # keep iterating?
        keep_iter = np.where(group_k_s[:, 1] < min_group_size)[0].any()
        count_iter += 1

        # log kma iteration
        print('KMA iteration info:')
        for rr in group_k_s:
            print('  cluster: {0}, size: {1}'.format(rr[0], rr[1]))
        print('Try again: ', keep_iter)
        print('Total attemps: ', count_iter)
        print()

# groups
d_groups = {}
for k in range(num_clusters):
    d_groups['{0}'.format(k)] = np.where(kma.labels_ == k)
# TODO: STORE GROUPS WITHIN OUTPUT DATASET



# # centroids = np.dot(kma.cluster_centers_[0:nterm-1], EOFsub)
# centroids = np.dot(kma.cluster_centers_[:,0:nterm-1], EOFsub)

# # centroids
# centroids = np.zeros((num_clusters, data.shape[1]))
# for k in range(num_clusters):
#     centroids[k, :] = np.mean(data[d_groups['{0}'.format(k)], :], axis=1)

centroids = np.zeros((num_clusters, EOFsub.shape[1]))#PCsub.shape[1]))
for k in range(num_clusters):
    centroids[k, :] = np.dot(np.mean(PCsub[d_groups['{0}'.format(k)],:], axis=1),EOFsub)

# # km, x and var_centers
km = np.multiply(centroids,np.tile(SlpGrdStd, (num_clusters, 1))) + np.tile(SlpGrdMean, (num_clusters, 1))
# km = np.multiply(centroids,np.tile(iceStd, (num_clusters, 1))) + np.tile(iceMean, (num_clusters, 1))


# kma_order = np.arange(0,26)
kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, num_clusters)
# kma_order = np.flipud(np.argsort(group_size))
# sort kmeans
#kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, num_clusters)

bmus = kma.labels_
bmus_corrected = np.zeros((len(kma.labels_),), ) * np.nan
for i in range(num_clusters):
    posc = np.where(kma.labels_ == kma_order[i])
    bmus_corrected[posc] = i

# reorder centroids
sorted_cenEOFs = kma.cluster_centers_[kma_order, :]
sorted_centroids = centroids[kma_order, :]



kmSorted = np.multiply(sorted_centroids,np.tile(SlpGrdStd, (num_clusters, 1))) + np.tile(SlpGrdMean, (num_clusters, 1))

# # removing the last one
# lastInd = np.where(bmus_corrected==26)
# bmus_corrected[lastInd] = 20
# secondToLastInd = np.where(bmus_corrected==25)
# bmus_corrected[secondToLastInd] = 21
#
# kmSorted = kmSorted[0:25,:]
# sorted_centroids = sorted_centroids[0:25,:]
# sorted_cenEOFs = sorted_cenEOFs[0:25,:]



import matplotlib.cm as cm
dwtcolors = cm.rainbow(np.linspace(0, 1,num_clusters))
#tccolors = np.flipud(cm.gray(np.linspace(0,1,7)))
#dwtcolors = np.vstack((etcolors,tccolors[1:,:]))


# plotting the EOF patterns
fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(int(np.sqrt(num_clusters)), int(np.sqrt(num_clusters)))
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.plt.figure()
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for hh in range(num_clusters):
    #ax = plt.subplot2grid((3,3),(c1,c2),projection=ccrs.NorthPolarStereo(central_longitude=-45))
    #ax = plt.subplot2grid((3,3),(c1,c2))#,projection=ccrs.NorthPolarStereo(central_longitude=-45))
    ax = plt.subplot(gs1[hh])
    num = kma_order[hh]

    # m = Basemap(projection='merc',llcrnrlat=-40,urcrnrlat=55,llcrnrlon=255,urcrnrlon=375,lat_ts=10,resolution='c')
    m = Basemap(projection='npstere', boundinglat=50, lon_0=180, resolution='l')

    cx,cy =m(lon,lat)
    m.drawcoastlines()

    # spatialField = np.multiply(EOFs[hh,0:(len(Xsea))],np.sqrt(variance[hh]))
    # spatialField = Km_slp[(hh), :] / 100 - np.nanmean(SLP, axis=0) / 100
    spatialField = kmSorted[(hh), 0: (len(Xsea))] / 100 - np.nanmean(SLP, axis=0) / 100
    rectField = np.ones((np.shape(X_in))) * np.nan
    for tt in range(len(sea_nodes)):
        rectField[sea_nodes[tt]] = spatialField[tt]

    m.fillcontinents(color=dwtcolors[hh])

    clevels = np.arange(-35,35,1)

    #ax.pcolormesh(cx, cy, rectField)#, cmap=cmocean.cm.ice)
    CS = m.contourf(cx, cy, rectField, clevels, vmin=-24, vmax=24, cmap=cm.RdBu_r, shading='gouraud')

    ax.set_xlim([np.min(cx)+10000, np.max(cx)+10000])
    ax.set_ylim([np.min(cy)+10000, np.max(cy)+10000])
    #tx, ty = m(320, -30)
    ax.text(np.min(cx)+(np.max(cx)-np.min(cx))/3.2*2, np.min(cy)+(np.max(cy)-np.min(cy))/9, '{}'.format(group_size[num]))

    #ax.set_title('{}'.format(group_size[num]))

    c2 += 1
    if c2 == 9:
        c1 += 1
        c2 = 0

    if plotIndx < 9:
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    counter = counter + 1
    if plotIndy < 9:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1






import pickle

dwtPickle = 'dwts81ClustersArcticRGy2022.pickle'
outputDWTs = {}
outputDWTs['APEV'] = APEV
outputDWTs['EOFs'] = EOFs
outputDWTs['EOFsub'] = EOFsub
outputDWTs['GRD'] = GRD
outputDWTs['PCA'] = PCA
outputDWTs['PCs'] = PCs
outputDWTs['PCsub'] = PCsub
outputDWTs['SLP'] = SLP
outputDWTs['SLPtime'] = SLPtime
outputDWTs['X_in'] = X_in
outputDWTs['lon'] = lon
outputDWTs['lat'] = lat
outputDWTs['Xsea'] = Xsea
outputDWTs['Ysea'] = Ysea
outputDWTs['Y_in'] = Y_in
outputDWTs['bmus_corrected'] = bmus_corrected
outputDWTs['centroids'] = centroids
outputDWTs['d_groups'] = d_groups
outputDWTs['group_size'] = group_size
outputDWTs['ipca'] = ipca
outputDWTs['km'] = km
outputDWTs['kma'] = kma
outputDWTs['kma_order'] = kma_order
outputDWTs['nPercent'] = nPercent
outputDWTs['nterm'] = nterm
outputDWTs['num_clusters'] = num_clusters
outputDWTs['sea_nodes'] = sea_nodes
outputDWTs['sorted_cenEOFs'] = sorted_cenEOFs
outputDWTs['sorted_centroids'] = sorted_centroids
outputDWTs['variance'] = variance

with open(dwtPickle,'wb') as f:
    pickle.dump(outputDWTs, f)