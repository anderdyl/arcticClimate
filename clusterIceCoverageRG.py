import numpy as np
import matplotlib.pyplot as plt
import os

import cmocean
import pandas as pd

from scipy.spatial import distance_matrix

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

grabYears = np.arange(1979,2022)
counter = 0
for ff in range(len(grabYears)):

    icedir = '/media/dylananderson/Elements/iceData/nsidc0051/daily/{}'.format(grabYears[ff])
    files = os.listdir(icedir)
    files.sort()
    files_path = [os.path.abspath(icedir) for x in os.listdir(icedir)]

    print('working on {}'.format(grabYears[ff]))

    for hh in range(len(files)):
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
        iceSubset.append(iceMasked[150:225,55:140])


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
allIce = np.ones((len(iceSubset),75*85))

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
xAll = x[150:225]
yAll = y[55:140]
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
end = dt.datetime(2022,1,1)
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

# iceTime = [np.array(np.array(year)[i],np.array(month)[i],np.array(day)[i]) for i in range(len(year))]



# ax.pcolormesh(,y[25:155],iceReshape[125:235,25:155])

# onlyIceLessLand = np.delete(onlyIce,getLand,axis=1)
# test4 = onlyIceLessLand[0,:]
# getCircle = np.where(test4 == 1.004)
# onlyIceLessLandLessCircle = np.delete(onlyIceLessLand,getCircle,axis=1)
# test5 = onlyIceLessLandLessCircle[0,:]
# getCoast = np.where(test5 == 1.012)
# onlyIceLessLandLessCircleLessCoast = np.delete(onlyIceLessLandLessCircle,getCircle,axis=1)

import mat73
import datetime as dt
# define some constants
epoch = dt.datetime(1970, 1, 1)
matlab_to_epoch_days = 719529  # days from 1-1-0000 to 1-1-1970
matlab_to_epoch_seconds = matlab_to_epoch_days * 24 * 60 * 60

def matlab_to_datetime(matlab_date_num_seconds):
    # get number of seconds from epoch
    from_epoch = matlab_date_num_seconds - matlab_to_epoch_seconds

    # convert to python datetime
    return epoch + dt.timedelta(seconds=from_epoch)

waveData = mat73.loadmat('era5_waves_pthope.mat')


waveArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in waveData['time_all']]




# data = np.vstack((np.asarray(waveArrayTime),waveData['wh_all']))
# data = np.vstack((data,waveData['tp_all']))
# df = pd.DataFrame(data.T,columns=['time','wh_all','tp_all'])

data = np.vstack((waveData['wh_all'],waveData['tp_all']))
df = pd.DataFrame(data=data.T,index=np.asarray(waveArrayTime),columns=['wh_all','tp_all'])
means = df.groupby(pd.Grouper(freq='1D')).mean()


waves = means.loc['1979-1-1':'2021-12-31']

# myDates = [dt.datetime.strftime(dt.datetime(year[hh],month[hh],day[hh]), "%Y-%m-%d") for hh in range(len(year))]
# iceTime = [dt.datetime(np.array(year)[i],np.array(month)[i],np.array(day)[i]) for i in range(len(year))]
myDates = [dt.datetime.strftime(dayTime[hh], "%Y-%m-%d") for hh in range(len(dayTime))]

iceWaves = waves[waves.index.isin(myDates)]

iceWaves = iceWaves.fillna(0)



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
wd = np.vstack((iceWaves['wh_all'],iceWaves['tp_all'],np.multiply(iceWaves['wh_all']**2,iceWaves['tp_all'])**(1.0/3))).T

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
    rectField = linearField.reshape(75,85)

    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True)
    extent = [-9.97, 168.35, 30.98, 34.35]
    # ax.pcolormesh(x[25:155], y[125:235], rectField)#, cmap=cmocean.cm.ice)
    ax.pcolormesh(x[55:140], y[150:225], rectField)#, cmap=cmocean.cm.ice)

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
num_clusters = 36
repres = 0.95
alpha = 0.05
min_size = None  # any int will activate group_min_size iteration
min_group_size=60

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
km = np.multiply(centroids,np.tile(iceStd, (num_clusters, 1))) + np.tile(iceMean, (num_clusters, 1))
# km = np.multiply(centroids,np.tile(iceStd, (num_clusters, 1))) + np.tile(iceMean, (num_clusters, 1))

# sort kmeans
kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, num_clusters)

bmus_corrected = np.zeros((len(kma.labels_),), ) * np.nan
for i in range(num_clusters):
    posc = np.where(kma.labels_ == kma_order[i])
    bmus_corrected[posc] = i

# reorder centroids
sorted_cenEOFs = kma.cluster_centers_[kma_order, :]
sorted_centroids = centroids[kma_order, :]
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

# plotting the EOF patterns
fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(6, 6)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for hh in range(num_clusters):
    #p1 = plt.subplot2grid((6,6),(c1,c2))
    ax = plt.subplot(gs1[hh],projection=ccrs.NorthPolarStereo(central_longitude=-45))
    num = kma_order[hh]
    # num = kma_orderOG[hh]

    # spatialField = kmOG[(hh - 1), :]# / 100 - np.nanmean(SLP_C, axis=0) / 100
    spatialField = km[(hh), :]# / 100 - np.nanmean(SLP_C, axis=0) / 100
    linearField = np.ones((np.shape(xFlat))) * np.nan
    linearField[points] = spatialField
    rectField = linearField.reshape(75, 85)

    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    # gl = ax.gridlines(draw_labels=True)
    extent = [-9.97, 168.35, 30.98, 34.35]
    # ax.pcolormesh(x[25:155], y[125:235], rectField, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[40:135], y[140:230], rectField, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[45:135], y[140:230], rectField, cmap=cmocean.cm.ice)
    ax.pcolormesh(x[55:140], y[150:225], rectField, cmap=cmocean.cm.ice)

    ax.set_xlim([-3223000, 0])
    ax.set_ylim([0, 2730000])
    # gl.xlabels_top = False
    # gl.ylabels_left = False
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    # ax.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
    ax.set_title('{} days'.format(group_size[num]))

    c2 += 1
    if c2 == 6:
        c1 += 1
        c2 = 0





import pickle

dwtPickle = 'iceData36ClustersMid70.pickle'
outputDWTs = {}
outputDWTs['APEV'] = APEV
outputDWTs['EOFs'] = EOFs
outputDWTs['EOFsub'] = EOFsub
# outputDWTs['Km'] = Km
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
outputDWTs['iceWaves'] = iceWaves
outputDWTs['iceTime'] = iceTime
outputDWTs['dayTime'] = dayTime
outputDWTs['repres'] = repres
outputDWTs['alpha'] = alpha
outputDWTs['min_group_size'] = min_group_size
outputDWTs['wd'] = wd
outputDWTs['wd_std'] = wd_std
outputDWTs['wd_norm'] = wd_norm
outputDWTs['Y'] = Y

with open(dwtPickle,'wb') as f:
    pickle.dump(outputDWTs, f)





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
