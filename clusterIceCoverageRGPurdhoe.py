import numpy as np
import matplotlib.pyplot as plt
import os

import cmocean
import pandas as pd

from scipy.spatial import distance_matrix
import sys
import subprocess

if 'darwin' in sys.platform:
    print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
    subprocess.Popen('caffeinate')

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



iceSubset = []
year = []
month = []
day = []

grabYears = np.arange(1979,2024)
counter = 0
for ff in range(len(grabYears)):

    icedir = '/users/dylananderson/Documents/data/ice/nsidc0051/daily/{}'.format(grabYears[ff])
    # icedir = '/media/dylananderson/Elements/iceData/nsidc0051/daily/{}'.format(grabYears[ff])
    files = os.listdir(icedir)
    files.sort()
    files_path = [os.path.abspath(icedir) for x in os.listdir(icedir)]

    print('working on {}'.format(grabYears[ff]))

    for hh in range(len(files)):
        if not files[hh].startswith('.'):
            #infile='/media/dylananderson/Elements/iceData/nsidc0051/monthly/nt_198901_f08_v1.1_n.bin'
            timeFile = files[hh].split('_')[1]
            infile = os.path.join(files_path[hh],files[hh])
            fr=open(infile,'rb')
            hdr=fr.read(300)
            ice=np.fromfile(fr,dtype=np.uint8)
            iceReshape=ice.reshape(448,304)
            #Convert to the fractional parameter range of 0.0 to 1.0
            iceDivide = iceReshape/250.
            #mask all land and missing values
            iceMasked=np.ma.masked_greater(iceDivide,1.0)
            fr.close()
            #Show ice concentration
            #plt.figure()
            # plt.imshow(iceMasked[125:235,25:155])
            #plt.imshow(np.flipud(iceMasked[125:235,25:155].T))

            year.append(int(timeFile[0:4]))
            month.append(int(timeFile[4:6]))
            day.append(int(timeFile[6:8]))
            #iceSubset.append(iceMasked[125:235,25:155])
            iceSubset.append(iceMasked[165:230,70:140])


    #     import cartopy.crs as ccrs
    # #
    #     fig=plt.figure(figsize=(6, 6))
    # # #Here is the proj4 string: Proj.4 Projection: +proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6 356889.449 +units=m +no_defs
    # # proj4params = {'proj=stere': 'stere', 'lat_0': 90, 'lat_ts': 70, 'lon_0': -45, 'k': 1, 'x_0': 0, 'y_0': 0, 'a': 6378273,
    # #                'b': 6356889.449, 'units': 'm'}
    # # #ax = plt.axes(projection=ccrs.Stereographic(proj4params))
    # #
    #     from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    #     ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
    #
    #     # ax.coastlines(resolution='110m',linewidth=0.5)
    #     ax.set_extent([-180,180,50,90],crs=ccrs.PlateCarree())
    #     gl = ax.gridlines(draw_labels=True)
    #     #set ice extent from Polar Stereographic Projection and Grid document
    #     extent=[-9.97,168.35,30.98,34.35]
    #     #ax.imshow(ice,cmap=plt.cm.Blues, vmin=1,vmax=100,extent=extent,transform=ccrs.PlateCarree())
    #     dx = dy = 25000
    #     x = np.arange(-3850000, +3750000, +dx)
    #     y = np.arange(+5850000, -5350000, -dy)
    #     # ax.pcolormesh(x[125:235],y[25:155],iceReshape[125:235,25:155])
    #     ax.pcolormesh(x[25:155],y[125:235],iceMasked[125:235,25:155],cmap=cmocean.cm.ice)
        # ax.set_xlim([-3223000,0])
        # ax.set_ylim([0,2730000])
        # gl.xlabels_top = False
        # gl.ylabels_left = False
        # gl.xformatter = LONGITUDE_FORMATTER
        # gl.yformatter = LATITUDE_FORMATTER
        # ax.set_title('{}/{}/{}'.format(int(timeFile[0:4]),int(timeFile[4:6]),int(timeFile[6:8])))
        # # gl.xlabel_style = {'size': 15, 'color': 'gray'}
        # # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
        # if counter < 10:
        #     plt.savefig('/media/dylananderson/Elements/icePlots/frame0000'+str(counter)+'.png')
        # elif counter < 100:
        #     plt.savefig('/media/dylananderson/Elements/icePlots/frame000'+str(counter)+'.png')
        # elif counter < 1000:
        #     plt.savefig('/media/dylananderson/Elements/icePlots/frame00'+str(counter)+'.png')
        # elif counter < 10000:
        #     plt.savefig('/media/dylananderson/Elements/icePlots/frame0'+str(counter)+'.png')
        # else:
        #     plt.savefig('/media/dylananderson/Elements/icePlots/plots2/frame'+str(counter)+'.png')
        # plt.close()
        # counter = counter+1



# geomorphdir = '/media/dylananderson/Elements/icePlots/'
#
# files2 = os.listdir(geomorphdir)
#
# files2.sort()
#
# files_path2 = [os.path.join(geomorphdir,x) for x in os.listdir(geomorphdir)]
#
# files_path2.sort()
#
# import cv2
#
# frame = cv2.imread(files_path2[0])
# height, width, layers = frame.shape
# forcc = cv2.VideoWriter_fourcc(*'XVID')
# video = cv2.VideoWriter('iceVisualized.avi', forcc, 36, (width, height))
# for image in files_path2:
#     video.write(cv2.imread(image))
# cv2.destroyAllWindows()
# video.release()

# allIce = np.ones((len(iceSubset),12000))
# allIce = np.ones((len(iceSubset),10925))
allIce = np.ones((len(iceSubset),65*70))

for qq in range(len(iceSubset)):
    temp = np.ma.MaskedArray.flatten(iceSubset[qq])
    allIce[qq,:] = temp

del iceSubset

onlyIce = np.copy(allIce)
test3 = onlyIce[0,:]#.reshape(110,130)
getLand = np.where(test3 == 1.016)
getCircle = np.where(test3 == 1.004)
getCoast = np.where(test3 == 1.012)
badInds1 = np.hstack((getLand[0],getCircle[0]))
badInds = np.hstack((badInds1,getCoast[0]))

onlyIceLessLand = np.delete(onlyIce,badInds,axis=1)
del onlyIce
del allIce
dx = dy = 25000
x = np.arange(-3850000, +3750000, +dx)
y = np.arange(+5850000, -5350000, -dy)
# xAll = x[125:235]
# yAll = y[25:155]
xAll = x[165:230]
yAll = y[70:140]
xMesh,yMesh = np.meshgrid(xAll,yAll)
xFlat = xMesh.flatten()
yFlat = yMesh.flatten()

xPoints = np.delete(xFlat,badInds,axis=0)
yPoints = np.delete(yFlat,badInds,axis=0)
pointsAll = np.arange(0,len(xFlat))
points = np.delete(pointsAll,badInds,axis=0)


# TODO: LETS PUT AN ICE PATTERN ON EVERY DAY AND THEN ADD WAVES ON TO IT
import datetime as dt
from dateutil.relativedelta import relativedelta

iceTime = [dt.datetime(np.array(year)[i],np.array(month)[i],np.array(day)[i]) for i in range(len(year))]

st = dt.datetime(1979, 1, 1)
# end = dt.datetime(2021,12,31)
end = dt.datetime(2023,10,3)
step = relativedelta(days=1)
dayTime = []
while st < end:
    dayTime.append(st)#.strftime('%Y-%m-%d'))
    st += step



daysDaysWithNoIce = [x for x in dayTime if x not in iceTime]
ind_dict = dict((k,i) for i,k in enumerate(dayTime))
inter = set(daysDaysWithNoIce).intersection(dayTime)
indices = [ ind_dict[x] for x in inter ]
indices.sort()

daysDaysWithIce = [x for x in dayTime if x in iceTime]
ind_dict = dict((k,i) for i,k in enumerate(dayTime))
inter2 = set(daysDaysWithIce).intersection(dayTime)
indices2 = [ ind_dict[x] for x in inter2]
indices2.sort()

gapFilledIce = np.nan * np.ones((len(dayTime),len(points)))
gapFilledIce[indices,:] = onlyIceLessLand[0:len(indices)]
gapFilledIce[indices2,:] = onlyIceLessLand
#gapFilledIce = gapFilledIce[0:15856]
# iceTime = [np.array(np.array(year)[i],np.array(month)[i],np.array(day)[i]) for i in range(len(year))]

# iceDay
dayOfYear = np.array([hh.timetuple().tm_yday for hh in dayTime])  # returns 1 for January 1st
dayOfYearSine = np.sin(2*np.pi/366*dayOfYear)
dayOfYearCosine = np.cos(2*np.pi/366*dayOfYear)


import pickle
with open(r"fetchAreaAttempt1.pickle", "rb") as input_file:
   fetchPickle = pickle.load(input_file)
fetchTotalConcentration = fetchPickle['fetchTotalConcentration']
kiloArea = fetchPickle['kiloArea']
slpWaves = fetchPickle['slpWaves']
dayTime = fetchPickle['dayTime']

iceConcentration = np.asarray(fetchTotalConcentration)



#
# gapFilledIce = gapFilledIce[0:-214]
# dayOfYearCosine = dayOfYearCosine[0:-214]
# dayOfYearSine = dayOfYearSine[0:-214]
# dayTime = dayTime[0:-214]

# plt.figure()
# plt.plot(dayTime,dayOfYearSine)
# plt.plot(dayTime,dayOfYearCosine)
# import cartopy.crs as ccrs
# fig=plt.figure(figsize=(6, 6))
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
# spatialField = gapFilledIce[15693]  # / 100 - np.nanmean(SLP_C, axis=0) / 100
# linearField = np.ones((np.shape(xFlat))) * np.nan
# linearField[points] = spatialField
# rectField = linearField.reshape(75, 85)
# ax.set_extent([-180,180,50,90],crs=ccrs.PlateCarree())
# gl = ax.gridlines(draw_labels=True)
# extent=[-9.97,168.35,30.98,34.35]
# dx = dy = 25000
# x = np.arange(-3850000, +3750000, +dx)
# y = np.arange(+5850000, -5350000, -dy)
# ax.pcolormesh(x[55:140], y[150:225], rectField, cmap=cmocean.cm.ice)
# ax.set_xlim([-3223000,0])
# ax.set_ylim([0,2730000])
# gl.xlabels_top = False
# gl.ylabels_left = False
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# ax.set_title('{}/{}/{}'.format(int(timeFile[0:4]),int(timeFile[4:6]),int(timeFile[6:8])))


# ax.pcolormesh(,y[25:155],iceReshape[125:235,25:155])

# onlyIceLessLand = np.delete(onlyIce,getLand,axis=1)
# test4 = onlyIceLessLand[0,:]
# getCircle = np.where(test4 == 1.004)
# onlyIceLessLandLessCircle = np.delete(onlyIceLessLand,getCircle,axis=1)
# test5 = onlyIceLessLandLessCircle[0,:]
# getCoast = np.where(test5 == 1.012)
# onlyIceLessLandLessCircleLessCoast = np.delete(onlyIceLessLandLessCircle,getCircle,axis=1)






#
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
#
#
#
# # data = np.vstack((np.asarray(waveArrayTime),waveData['wh_all']))
# # data = np.vstack((data,waveData['tp_all']))
# # df = pd.DataFrame(data.T,columns=['time','wh_all','tp_all'])
#
# data = np.vstack((waveData['wh_all'],waveData['tp_all']))
# df = pd.DataFrame(data=data.T,index=np.asarray(waveArrayTime),columns=['wh_all','tp_all'])
# means = df.groupby(pd.Grouper(freq='1D')).mean()
#
#
# waves = means.loc['1979-1-1':'2021-12-31']
#
# # myDates = [dt.datetime.strftime(dt.datetime(year[hh],month[hh],day[hh]), "%Y-%m-%d") for hh in range(len(year))]
# # iceTime = [dt.datetime(np.array(year)[i],np.array(month)[i],np.array(day)[i]) for i in range(len(year))]
# myDates = [dt.datetime.strftime(dayTime[hh], "%Y-%m-%d") for hh in range(len(dayTime))]
#
# iceWaves = waves[waves.index.isin(myDates)]
#
# iceWaves = iceWaves.fillna(0)



# iceMean = np.mean(onlyIceLessLand,axis=0)
# iceStd = np.std(onlyIceLessLand,axis=0)
# iceNorm = (onlyIceLessLand[:,:] - iceMean) / iceStd
# iceNorm[np.isnan(iceNorm)] = 0
iceMean = np.mean(gapFilledIce,axis=0)
iceStd = np.std(gapFilledIce,axis=0)
iceNorm = (gapFilledIce[:,:] - iceMean) / iceStd
iceNorm[np.isnan(iceNorm)] = 0
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import linear_model


# 95% repres
repres = 0.951

# principal components analysis
ipca = PCA(n_components=min(iceNorm.shape[0], iceNorm.shape[1]))
PCs = ipca.fit_transform(iceNorm)
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
# wd = np.vstack((dayOfYearSine,dayOfYearCosine,iceConcentration)).T
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




import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# plotting the EOF patterns
plt.figure()
c1 = 0
c2 = 0
for hh in range(9):
    ax = plt.subplot2grid((3,3),(c1,c2),projection=ccrs.NorthPolarStereo(central_longitude=-45))
    spatialField = np.multiply(EOFs[hh,0:len(xPoints)],np.sqrt(variance[hh]))
    linearField = np.ones((np.shape(xFlat))) * np.nan
    linearField[points] = spatialField
    # rectField = linearField.reshape(110,130)
    # rectField = linearField.reshape(100,120)
    rectField = linearField.reshape(65,70)

    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True)
    extent = [-9.97, 168.35, 30.98, 34.35]
    # ax.pcolormesh(x[25:155], y[125:235], rectField)#, cmap=cmocean.cm.ice)
    ax.pcolormesh(x[70:140], y[165:230], rectField)#, cmap=cmocean.cm.ice)

    ax.set_xlim([-3223000, 0])
    ax.set_ylim([0, 2730000])
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
    c2 += 1
    if c2 == 3:
        c1 += 1
        c2 = 0





# KMA Regression Guided
num_clusters = 49
repres = 0.90
alpha = 0.4
min_size = None  # any int will activate group_min_size iteration
min_group_size=30

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






with open(r"iceData36ClustersDayNumRGAdjusted.pickle", "rb") as input_file:
   iceDWTs = pickle.load(input_file)

kma = iceDWTs['kma']
num_clusters = 49
EOFsub = iceDWTs['EOFsub']
PCsub = iceDWTs['PCsub']
iceStd = iceDWTs['iceStd']
iceMean = iceDWTs['iceMean']
group_size = iceDWTs['group_size']
dayOfYear = iceDWTs['dayOfYear']

with open(r"fetchAreaAttempt1.pickle", "rb") as input_file:
   fetchPickle = pickle.load(input_file)
fetchTotalConcentration = fetchPickle['fetchTotalConcentration']
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
km = np.multiply(centroids,np.tile(iceStd, (num_clusters, 1))) + np.tile(iceMean, (num_clusters, 1))
# km = np.multiply(centroids,np.tile(iceStd, (num_clusters, 1))) + np.tile(iceMean, (num_clusters, 1))

# kma_order = np.arange(0,26)
kma_order = np.flipud(np.argsort(group_size))
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

kmSorted = np.multiply(sorted_centroids,np.tile(iceStd, (num_clusters, 1))) + np.tile(iceMean, (num_clusters, 1))




bmus2 = bmus_corrected



tempInd = np.where(bmus2 == 48)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 0) & (np.asarray(tempTime) <= 300))
bmus2[tempInd[0][lateInds]] = 36
lateInds = np.where((np.asarray(tempTime) >= 300) & (np.asarray(tempTime) <= 370))
bmus2[tempInd[0][lateInds]] = 3

lastInd = np.where(bmus2==47)
bmus2[lastInd] = 41

# lastInd = np.where(bmus2==44)
# bmus2[lastInd] = 33

lastInd = np.where(bmus2==46)
bmus2[lastInd] = 36
lastInd = np.where(bmus2==45)
bmus2[lastInd] = 36
lastInd = np.where(bmus2==38)
bmus2[lastInd] = 37

tempInd = np.where(bmus2 == 15)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) <= 200))
bmus2[tempInd[0][lateInds]] = 12

tempInd = np.where(bmus2 == 11)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 110) & (np.asarray(tempTime) <= 210))
bmus2[tempInd[0][lateInds]] = 5

tempInd = np.where(bmus2 == 13)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 110) & (np.asarray(tempTime) <= 230))
bmus2[tempInd[0][lateInds]] = 27

tempInd = np.where(bmus2 == 6)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 234) & (np.asarray(tempTime) <= 320))
bmus2[tempInd[0][lateInds]] = 47

tempInd = np.where(bmus2 == 1)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 54) & (np.asarray(tempTime) <= 190))
bmus2[tempInd[0][lateInds]] = 48

tempInd = np.where(bmus2 == 16)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 170) & (np.asarray(tempTime) <= 234))
bmus2[tempInd[0][lateInds]] = 8

tempInd = np.where(bmus2 == 22)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 170) & (np.asarray(tempTime) <= 240))
bmus2[tempInd[0][lateInds]] = 8

# rearange between 19, 26, 33, 32

tempInd = np.where(bmus2 == 32)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 219) & (np.asarray(tempTime) <= 289))
bmus2[tempInd[0][lateInds]] = 26

tempInd = np.where(bmus2 == 26)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 170) & (np.asarray(tempTime) <= 220))
bmus2[tempInd[0][lateInds]] = 32


tempInd = np.where(bmus2 == 31)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 268) & (np.asarray(tempTime) <= 304))
bmus2[tempInd[0][lateInds]] = 25

tempInd = np.where(bmus2 == 25)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 209) & (np.asarray(tempTime) <= 269))
bmus2[tempInd[0][lateInds]] = 31


tempInd = np.where(bmus2 == 34)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 258) & (np.asarray(tempTime) <= 300))
bmus2[tempInd[0][lateInds]] = 41
lateInds = np.where((np.asarray(tempTime) >= 0) & (np.asarray(tempTime) <= 257))
bmus2[tempInd[0][lateInds]] = 29

tempInd = np.where(bmus2 == 20)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 268) & (np.asarray(tempTime) <= 370))
bmus2[tempInd[0][lateInds]] = 30

lastInd = np.where(bmus2==18)
bmus2[lastInd] = 15

lastInd = np.where(bmus2==40)
bmus2[lastInd] = 17

lastInd = np.where(bmus2==39)
bmus2[lastInd] = 35

tempInd = np.where(bmus2 == 44)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 220) & (np.asarray(tempTime) <= 254))
bmus2[tempInd[0][lateInds]] = 37
lateInds = np.where((np.asarray(tempTime) >= 254) & (np.asarray(tempTime) <= 320))
bmus2[tempInd[0][lateInds]] = 30

lastInd = np.where(bmus2==47)
bmus2[lastInd] = 26

tempInd = np.where(bmus2 == 4)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 220))
bmus2[tempInd[0][lateInds]] = 3
lateInds = np.where((np.asarray(tempTime) <= 70))
bmus2[tempInd[0][lateInds]] = 48


tempInd = np.where(bmus2 == 3)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 105) & (np.asarray(tempTime) <= 200))
bmus2[tempInd[0][lateInds]] = 48
lateInds = np.where((np.asarray(tempTime) >= 29) & (np.asarray(tempTime) <= 110))
bmus2[tempInd[0][lateInds]] = 2

tempInd = np.where(bmus2 == 8)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 256))
bmus2[tempInd[0][lateInds]] = 16



tempInd = np.where(bmus2 == 10)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 324) & (np.asarray(tempTime) <= 400))
bmus2[tempInd[0][lateInds]] = 17
lateInds = np.where((np.asarray(tempTime) >= 261) & (np.asarray(tempTime) <= 324))
bmus2[tempInd[0][lateInds]] = 31

lastInd = np.where(bmus2==42)
bmus2[lastInd] = 28

lastInd = np.where(bmus2==43)
bmus2[lastInd] = 20

tempInd = np.where(bmus2 == 28)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) <= 255))
bmus2[tempInd[0][lateInds]] = 35

tempInd = np.where(bmus2 == 35)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 252))
bmus2[tempInd[0][lateInds]] = 28


tempInd = np.where(bmus2 == 41)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 286) & (np.asarray(tempTime) <= 400))
bmus2[tempInd[0][lateInds]] = 30
lateInds = np.where((np.asarray(tempTime) >= 0) & (np.asarray(tempTime) <= 286))
bmus2[tempInd[0][lateInds]] = 36


tempInd = np.where(bmus2 == 31)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) <= 225))
bmus2[tempInd[0][lateInds]] = 32

tempInd = np.where(bmus2 == 33)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) <= 225))
bmus2[tempInd[0][lateInds]] = 31
lateInds = np.where((np.asarray(tempTime) >= 268))
bmus2[tempInd[0][lateInds]] = 25

lastInd = np.where(bmus2==33)
bmus2[lastInd] = 23

tempInd = np.where(bmus2 == 11)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) <= 225))
bmus2[tempInd[0][lateInds]] = 2


tempInd = np.where(bmus2 == 31)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) <= 245))
bmus2[tempInd[0][lateInds]] = 24





areaPerBin = np.zeros((num_clusters,))
for i in range(num_clusters):
    posc = np.where(bmus2 == i)
    areaPerBin[i] = np.mean(np.asarray(fetchTotalConcentration)[posc])


kma_orderNew = (np.argsort(areaPerBin))


bmus_final = np.zeros((len(kma.labels_),), ) * np.nan
for i in range(36):
    posc = np.where(bmus2 == kma_orderNew[i])
    bmus_final[posc] = i



tempInd = np.where(bmus_final == 15)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) <= 200))
bmus2[tempInd[0][lateInds]] = 11
lateInds = np.where((np.asarray(tempTime) >= 330))
bmus_final[tempInd[0][lateInds]] = 11

tempInd = np.where(bmus_final == 10)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 200))
bmus_final[tempInd[0][lateInds]] = 12


tempInd = np.where(bmus_final == 32)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) <= 228))
bmus_final[tempInd[0][lateInds]] = 31

tempInd = np.where(bmus_final == 27)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) <= 221))
bmus_final[tempInd[0][lateInds]] = 28

tempInd = np.where(bmus_final == 18)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) >= 320))
bmus_final[tempInd[0][lateInds]] = 20

tempInd = np.where(bmus_final == 35)
tempTime = dayOfYear[tempInd[0]]
lateInds = np.where((np.asarray(tempTime) <= 200))
bmus_final[tempInd[0][lateInds]] = 23





from matplotlib import gridspec
numDWTs=36
fetchLengths = []
for hh in range(numDWTs):
    dwtInd = np.where((bmus_final==hh))
    fetchBin = []
    fetchBin.append(np.asarray(fetchTotalConcentration)[dwtInd[0]])
    fetchLengths.append(np.concatenate(fetchBin,axis=0))


from scipy.stats.kde import gaussian_kde
import matplotlib.cm as cm
import matplotlib.colors as mcolors

dwtcolors = cm.rainbow(np.linspace(0, 1, numDWTs))
#plt.style.use('dark_background')

plt.style.use('dark_background')
dist_space = np.linspace(0, 1600000/1000, 1000)
fig = plt.figure(figsize=(10,10))
gs2 = gridspec.GridSpec(7, 7)

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
        # ax.text(1.8, 1, np.round(colorparam*100)/100, fontweight='bold')

    else:
        ax.spines['bottom'].set_color([0.3, 0.3, 0.3])
        ax.spines['top'].set_color([0.3, 0.3, 0.3])
        ax.spines['right'].set_color([0.3, 0.3, 0.3])
        ax.spines['left'].set_color([0.3, 0.3, 0.3])
    ax.yaxis.set_ticklabels([])

plt.show()
s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
s_map.set_array(colorparam)
fig.subplots_adjust(right=0.86)
cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
cbar = fig.colorbar(s_map, cax=cbar_ax)
cbar.set_label('Wave Gen Area (km^2)')

# plotting the EOF patterns
fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(6, 6)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for qq in range(36):
    ax = plt.subplot(gs1[qq])#,projection=ccrs.NorthPolarStereo(central_longitude=-45))
    # tempInd = np.where(bmus_corrected==hh)
    tempInd = np.where(bmus_final==qq)
    # tempTime = np.asarray(dayTime)[tempInd[0]]
    # tempMonths = [temp.month for temp in tempTime]
    #
    # ax.hist(tempMonths,bins = [-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5])
    # ax.set_xlim([0,13])
    tempDayOf = dayOfYear[tempInd[0]]
    ax.hist(tempDayOf,bins = np.arange(0,367))
    ax.set_xlim([0,367])
    ax.text(10,1,'{}'.format(len(tempDayOf)))








#
# # # removing the last one
# lastInd = np.where(bmus2==35)
# bmus2[lastInd] = 28
#
# secondToLastInd = np.where(bmus2==34)
# bmus2[secondToLastInd] = 25
#
# thirdToLastInd = np.where(bmus2==33)
# bmus2[thirdToLastInd] = 23
#
# fourthToLastInd = np.where(bmus2==32)
# bmus2[fourthToLastInd] = 27
#
# fifthToLastInd = np.where(bmus2==31)
# bmus2[fifthToLastInd] = 27
#
# sixthToLastInd = np.where(bmus2==30)
# bmus2[sixthToLastInd] = 29
#
# seventhToLastInd = np.where(bmus2==29)
# bmus2[seventhToLastInd] = 26
#
# eighthToLastInd = np.where(bmus2==28)
# bmus2[eighthToLastInd] = 25
#
# hh = 27
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) > 10))
# bmus2[tempInd[0][lateInds]] = 17
# earlyInds = np.where((np.asarray(tempMonths) < 7))
# bmus2[tempInd[0][earlyInds]] = 20
#
# tenthToLastInd = np.where(bmus2==27)
# bmus2[tenthToLastInd] = 25
#
# hh = 3
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# earlyInds = np.where((np.asarray(tempMonths) < 3))
# bmus2[tempInd[0][earlyInds]] = 2
#
# hh = 2
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==3))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 4
# lateInds2 = np.where((np.asarray(tempMonths) ==4))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds2]] = 4
# lateInds3 = np.where((np.asarray(tempMonths) ==5))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds3]] = 4
# earlyInds = np.where((np.asarray(tempMonths) ==6))# & (np.asarray(tempMonths) ==2))
# bmus2[tempInd[0][earlyInds]] = 4
#
#
# hh = 5
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==6))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 4
# lateInds2 = np.where((np.asarray(tempMonths) ==5))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds2]] = 4
# lateInds3 = np.where((np.asarray(tempMonths) ==3))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds3]] = 7
# earlyInds = np.where((np.asarray(tempMonths) ==1))# & (np.asarray(tempMonths) ==2))
# bmus2[tempInd[0][earlyInds]] = 3
# earlyInds2 = np.where((np.asarray(tempMonths) ==2))# & (np.asarray(tempMonths) ==2))
# bmus2[tempInd[0][earlyInds2]] = 3
#
#
# hh = 6
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==6))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 8
# lateInds2 = np.where((np.asarray(tempMonths) ==7))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds2]] = 8
#
#
# hh = 7
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==11))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 5
# lateInds2 = np.where((np.asarray(tempMonths) ==12))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds2]] = 5
# lateInds3 = np.where((np.asarray(tempMonths) ==1))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds3]] = 2
# lateInds4 = np.where((np.asarray(tempMonths) ==2))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds4]] = 2
# lateInds5 = np.where((np.asarray(tempMonths) ==3))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds5]] = 0
#
# hh = 3
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==1))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 2
# lateInds2 = np.where((np.asarray(tempMonths) ==2))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds2]] = 2
#
#
# hh = 8
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==10))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 6
# lateInds2 = np.where((np.asarray(tempMonths) ==11))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds2]] = 6
#
#
# hh = 9
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==12))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 3
# earlyInds = np.where((np.asarray(tempMonths) ==7))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][earlyInds]] = 10
#
# hh = 12
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==10))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 9
#
# hh = 4
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==3))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 0
#
# hh = 13
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==12))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 11
# lateInds1 = np.where((np.asarray(tempMonths) ==1))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds1]] = 2
# lateInds2 = np.where((np.asarray(tempMonths) ==7))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds2]] = 8
# lateInds3 = np.where((np.asarray(tempMonths) ==8))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds3]] = 8
#
#
# hh = 15
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==10))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 17
#
#
#
# hh = 17
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==8))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 15
# lateInds1 = np.where((np.asarray(tempMonths) ==9))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds1]] = 15
#
# hh = 18
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==12))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 11
# lateInds1 = np.where((np.asarray(tempMonths) ==7))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds1]] = 12
#
# hh = 20
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds1 = np.where((np.asarray(tempMonths) ==10))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds1]] = 17
#
# hh = 21
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==6))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 20
# lateInds1 = np.where((np.asarray(tempMonths) ==10))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds1]] = 17
#
# hh = 23
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==11))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 17
# lateInds1 = np.where((np.asarray(tempMonths) ==10))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds1]] = 17
#
# hh = 25
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==10))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 18
# lateInds1 = np.where((np.asarray(tempMonths) ==9))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds1]] = 18
# lateInds2 = np.where((np.asarray(tempMonths) ==8))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds2]] = 18
# lateInds3 = np.where((np.asarray(tempMonths) ==7))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds3]] = 22
#
# hh = 26
# tempInd = np.where(bmus2 == hh)
# tempTime = np.asarray(dayTime)[tempInd[0]]
# tempMonths = [temp.month for temp in tempTime]
# lateInds = np.where((np.asarray(tempMonths) ==11))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds]] = 17
# lateInds1 = np.where((np.asarray(tempMonths) ==10))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds1]] = 17
# lateInds2 = np.where((np.asarray(tempMonths) ==9))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds2]] = 21
# lateInds3 = np.where((np.asarray(tempMonths) ==8))# & (np.asarray(tempMonths) ==5))
# bmus2[tempInd[0][lateInds3]] = 21
#


#
#

# secondToLastInd = np.where(bmus_corrected==25)
# bmus_corrected[secondToLastInd] = 21
#
# kmSorted = kmSorted[0:25,:]
# sorted_centroids = sorted_centroids[0:25,:]
# sorted_cenEOFs = sorted_cenEOFs[0:25,:]
# #
# # #
# # # CAN WE PUT A GUIDED REGRESSION IN HERE FOR THE PRESENCE OF WAVES?
# #
# # asdf
# num_clustersOG = 16
# PCsubOG = PCs[:, :nterm+1]
# EOFsubOG = EOFs[:nterm+1, :]
# kmaOG = KMeans(n_clusters=num_clustersOG, n_init=2000).fit(PCsubOG)
# # groupsize
# _, group_sizeOG = np.unique(kmaOG.labels_, return_counts=True)
# # groups
# d_groupsOG = {}
# for k in range(num_clusters):
#     d_groupsOG['{0}'.format(k)] = np.where(kmaOG.labels_ == k)
# # centroids
# centroidsOG = np.dot(kmaOG.cluster_centers_, EOFsubOG)
# # km, x and var_centers
# kmOG = np.multiply(centroidsOG,np.tile(iceStd, (num_clustersOG, 1))) + np.tile(iceMean, (num_clustersOG, 1))
#
# # # sort kmeans
# # kma_order = np.argsort(np.mean(-km, axis=1))
#
# # sort kmeans
# kma_orderOG = sort_cluster_gen_corr_end(kmaOG.cluster_centers_, num_clustersOG)
#
# bmus_correctedOG = np.zeros((len(kmaOG.labels_),), ) * np.nan
# for i in range(num_clustersOG):
#     posc = np.where(kmaOG.labels_ == kma_orderOG[i])
#     bmus_correctedOG[posc] = i
#
# # reorder centroids
# sorted_cenEOFsOG = kmaOG.cluster_centers_[kma_orderOG, :]
# sorted_centroidsOG = centroidsOG[kma_orderOG, :]
#
#
# repmatDesviacionOG = np.tile(iceStd, (num_clustersOG,1))
# repmatMediaOG = np.tile(iceMean, (num_clustersOG,1))
# KmOG = np.multiply(sorted_centroidsOG,repmatDesviacionOG) + repmatMediaOG
# # # km = np.multiply(centroids,np.tile(iceStd, (num_clusters, 1))) + np.tile(iceMean, (num_clusters, 1))
#




from matplotlib import gridspec
plt.style.use('default')
# plotting the EOF patterns
fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(6, 6)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for hh in range(36):
    #p1 = plt.subplot2grid((6,6),(c1,c2))
    ax = plt.subplot(gs1[hh],projection=ccrs.NorthPolarStereo(central_longitude=-45))
    num = kma_order[hh]
    # num = kma_orderOG[hh]
    tempInd = np.where(bmus_final==hh)

    # spatialField = kmOG[(hh - 1), :]# / 100 - np.nanmean(SLP_C, axis=0) / 100
    #spatialField = kmSorted[(hh), :]# / 100 - np.nanmean(SLP_C, axis=0) / 100
    spatialField = np.nanmean(gapFilledIce[tempInd[0]], axis=0)

    linearField = np.ones((np.shape(xFlat))) * np.nan
    linearField[points] = spatialField
    rectField = linearField.reshape(65, 70)

    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    # gl = ax.gridlines(draw_labels=True)
    extent = [-9.97, 168.35, 30.98, 34.35]
    # ax.pcolormesh(x[25:155], y[125:235], rectField, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[40:135], y[140:230], rectField, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[45:135], y[140:230], rectField, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[65:150], y[160:235], rectField, cmap=cmocean.cm.ice)
    ax.pcolormesh(x[70:140], y[165:230], rectField, cmap=cmocean.cm.ice)

    ax.set_xlim([-2220000, -275000])
    ax.set_ylim([50000, 1850000])
    # gl.xlabels_top = False
    # gl.ylabels_left = False
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    # ax.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
    #ax.set_title('{} days'.format(len(tempInd[0])))

    c2 += 1
    if c2 == 6:
        c1 += 1
        c2 = 0





asdfghj


import pickle

dwtPickle = 'iceData36ClustersDayNumRGFinalized.pickle'
outputDWTs = {}
outputDWTs['APEV'] = APEV
outputDWTs['EOFs'] = EOFs
outputDWTs['EOFsub'] = EOFsub
outputDWTs['bmus2'] = bmus2
outputDWTs['bmus_final'] = bmus_final

outputDWTs['PCA'] = PCA
outputDWTs['PCs'] = PCs
outputDWTs['PCsub'] = PCsub
outputDWTs['onlyIceLessLand'] = onlyIceLessLand
outputDWTs['iceMean'] = iceMean
outputDWTs['iceStd'] = iceStd
outputDWTs['iceNorm'] = iceNorm
outputDWTs['year'] = year
outputDWTs['month'] = month
outputDWTs['day'] = day
outputDWTs['xMesh'] = xMesh
outputDWTs['xMesh'] = yMesh
outputDWTs['xFlat'] = xFlat
outputDWTs['yFlat'] = yFlat
outputDWTs['x'] = x
outputDWTs['y'] = y
outputDWTs['xPoints'] = xPoints
outputDWTs['yPoints'] = yPoints
outputDWTs['points'] = points
outputDWTs['bmus_corrected'] = bmus_corrected
outputDWTs['centroids'] = centroids
outputDWTs['d_groups'] = d_groups
outputDWTs['group_size'] = group_size
outputDWTs['ipca'] = ipca
outputDWTs['km'] = km
outputDWTs['kma'] = kma
outputDWTs['kma_order'] = kma_order
outputDWTs['kma_orderNew'] = kma_orderNew

outputDWTs['xAll'] = xAll
outputDWTs['yAll'] = yAll
outputDWTs['nPercent'] = nPercent
outputDWTs['nterm'] = nterm
outputDWTs['num_clusters'] = num_clusters
outputDWTs['getLand'] = getLand
outputDWTs['getCircle'] = getCircle
outputDWTs['getCoast'] = getCoast
outputDWTs['sorted_cenEOFs'] = sorted_cenEOFs
outputDWTs['sorted_centroids'] = sorted_centroids
outputDWTs['variance'] = variance
outputDWTs['gapFilledIce'] = gapFilledIce
# outputDWTs['iceWaves'] = iceWaves
outputDWTs['iceTime'] = iceTime
outputDWTs['dayTime'] = dayTime
outputDWTs['repres'] = repres
outputDWTs['alpha'] = alpha
outputDWTs['min_group_size'] = min_group_size
outputDWTs['wd'] = wd
outputDWTs['wd_std'] = wd_std
outputDWTs['wd_norm'] = wd_norm
outputDWTs['Y'] = Y
outputDWTs['dayOfYear'] = dayOfYear
outputDWTs['kmSorted'] = kmSorted
outputDWTs['data_a'] = data_a
outputDWTs['data_norm'] = data_norm
outputDWTs['data_std'] = data_std
outputDWTs['data_mean'] = data_mean
outputDWTs['data'] = data

with open(dwtPickle,'wb') as f:
    pickle.dump(outputDWTs, f)



timeAsArray = np.array(dayTime)
plt.figure()
ax1 = plt.subplot2grid((2,1),(0,0))
for qq in range(len(np.unique(bmus_corrected))):
    getBMUS = np.where((bmus_corrected == qq))
    temp = bmus_corrected[getBMUS]
    tempTime = timeAsArray[getBMUS]
    ax1.plot(np.array(dayTime)[getBMUS[0]],qq*np.ones((len(temp),)),'.')



#
# import cartopy.crs as ccrs
# fig=plt.figure(figsize=(6, 6))
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
# # ax.coastlines(resolution='110m',linewidth=0.5)
# ax.set_extent([-180,180,50,90],crs=ccrs.PlateCarree())
# gl = ax.gridlines(draw_labels=True)
# #set ice extent from Polar Stereographic Projection and Grid document
# extent=[-9.97,168.35,30.98,34.35]
# #ax.imshow(ice,cmap=plt.cm.Blues, vmin=1,vmax=100,extent=extent,transform=ccrs.PlateCarree())
# dx = dy = 25000
# x = np.arange(-3850000, +3750000, +dx)
# y = np.arange(+5850000, -5350000, -dy)
# # ax.pcolormesh(x[125:235],y[25:155],iceReshape[125:235,25:155])
# test3 = allIce[5,:].reshape(110,130)
# getLand = np.where(test3 == 1.016)
# test3[getLand] = 2.0
# getCircle = np.where(test3 == 1.004)
# test3[getCircle] = 3.0
# getCircle2 = np.where(test3 == 1.012)
# test3[getCircle2] = 4.0
# ax.pcolormesh(x[25:155],y[125:235],test3,cmap=cmocean.cm.ice)
# ax.set_xlim([-3223000,0])
# ax.set_ylim([0,2730000])
# gl.xlabels_top = False
# gl.ylabels_left = False
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# # ax.set_title('{}/{}/{}'.format(int(timeFile[0:4]),int(timeFile[4:6]),int(timeFile[6:8])))
