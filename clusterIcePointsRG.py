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

iceConc = []
iceSubset = []
year = []
month = []
day = []
icePattern = []


# # Basically Point Hope?
# yStart = 182
# yEnd = 448-263
# xStart = 70
# xEnd = 304-231
# yStartConc = 168
# yEndConc = 448-236
# xStartConc = 65
# xEndConc = 304-195

# # Point Lay
# yStart = 185
# yEnd = 448-257
# xStart = 75
# xEnd = 304-223
# yStartConc = 170
# yEndConc = 448-240
# xStartConc = 70
# xEndConc = 304-180

# # Wevok
# yStart = 184
# yEnd = 448-259
# xStart = 73
# xEnd = 304-225
# yStartConc = 170
# yEndConc = 448-240
# xStartConc = 70
# xEndConc = 304-180

# # Wainwright
# yStart = 195
# yEnd = 448-247
# xStart = 78
# xEnd = 304-220
# yStartConc = 170
# yEndConc = 448-240
# xStartConc = 70
# xEndConc = 304-180

# # Shishmaref
# yStart = 177
# yEnd = 448-266
# xStart = 66
# xEnd = 304-233
# yStartConc = 164
# yEndConc = 448-250
# xStartConc = 60
# xEndConc = 304-205

# # Kivalina
# yStart = 179
# yEnd = 448-263
# xStart = 67
# xEnd = 304-231
# yStartConc = 164
# yEndConc = 448-250
# xStartConc = 60
# xEndConc = 304-205


# # Deering
# yStart = 183
# yEnd = 448-261
# xStart = 61
# xEnd = 304-238
# yStartConc = 164
# yEndConc = 448-250
# xStartConc = 60
# xEndConc = 304-205


# # Wales
# yStart = 175
# yEnd = 448-268
# xStart = 66
# xEnd = 304-233
# yStartConc = 164
# yEndConc = 448-250
# xStartConc = 60
# xEndConc = 304-205

# Utqiagvik
yStart = 200
yEnd = 448-242
xStart = 77
xEnd = 304-219
yStartConc = 173
yEndConc = 448-228
xStartConc = 70
xEndConc = 304-180

# # whole region
yStartWR = 155
yEndWR = 448-198
xStartWR = 60
xEndWR = 304-125

endYear = 2024
grabYears = np.arange(1978,endYear)
counter = 0
for ff in range(len(grabYears)):

    icedir = '/users/dylananderson/Documents/data/ice/nsidc0051/daily/{}'.format(grabYears[ff])
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
            iceDivide = iceReshape/250

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
            # iceSubset.append(iceMasked[165:250,70:195])
            iceExample=np.flipud(iceMasked)
            icePattern.append(np.flipud(iceMasked[yStartWR:yEndWR,xStartWR:xEndWR]))
            # iceConc.append(np.flipud(iceMasked[yStartConc:yEndConc,xStartConc:xEndConc]))
            iceConc.append(np.flipud(iceMasked[yStartConc:yEndConc,xStartConc:xEndConc]))

            # iceSubset.append(np.flipud(iceMasked[yStart:yEnd,xStart:xEnd]))
            iceSubset.append(np.flipud(iceMasked[yStart:yEnd,xStart:xEnd]))


# allIce = np.ones((len(iceSubset),185*225))
# iceSubset.append(iceMasked[165:250, 70:175])

#
allIcePattern = np.ones((len(icePattern),(yEndWR-yStartWR)*(xEndWR-xStartWR)))
allIce = np.ones((len(iceSubset),(yEnd-yStart)*(xEnd-xStart)))
allIceConc = np.ones((len(iceSubset),(yEndConc-yStartConc)*(xEndConc-xStartConc)))

# iceSubset.append(iceMasked[165:250, 70:175])

for qq in range(len(iceSubset)):
    temp = np.ma.MaskedArray.flatten(iceSubset[qq])
    allIce[qq,:] = temp
    temp2 = np.ma.MaskedArray.flatten(icePattern[qq])
    allIcePattern[qq,:] = temp2
    temp3 = np.ma.MaskedArray.flatten(iceConc[qq])
    allIceConc[qq,:] = temp3

del iceSubset
del icePattern
del iceConc

onlyIce = np.copy(allIce)
test3 = onlyIce[0,:]#.reshape(110,130)
getLand = np.where(test3 == 1.016)
getCircle = np.where(test3 == 1.004)
getCoast = np.where(test3 == 1.012)
badInds1 = np.hstack((getLand[0],getCircle[0]))
badInds = np.hstack((badInds1,getCoast[0]))
onlyIceLessLand = np.delete(onlyIce,badInds,axis=1)#np.copy(allIce)


onlyIceConc = np.copy(allIceConc)
test3 = onlyIceConc[0,:]#.reshape(110,130)
getLand = np.where(test3 == 1.016)
getCircle = np.where(test3 == 1.004)
getCoast = np.where(test3 == 1.012)
badInds1 = np.hstack((getLand[0],getCircle[0]))
badIndsConc = np.hstack((badInds1,getCoast[0]))
onlyIceConcLessLand = np.delete(onlyIceConc,badIndsConc,axis=1)

onlyIceWR = np.copy(allIcePattern)
test3 = onlyIceWR[0,:]#.reshape(110,130)
getLand = np.where(test3 == 1.016)
getCircle = np.where(test3 == 1.004)
getCoast = np.where(test3 == 1.012)
badInds1 = np.hstack((getLand[0],getCircle[0]))
badIndsWR = np.hstack((badInds1,getCoast[0]))

onlyIcePatternLessLand = np.delete(onlyIceWR,badIndsWR,axis=1)


del onlyIce
del allIceConc
del allIce
del allIcePattern

dx = dy = 25000
x = np.arange(-3850000, +3750000, +dx)
y = np.arange(5850000, -5350000,-dy)

xAll = x[xStart:xEnd]
yAll = y[yStart:yEnd]
xConc = x[xStartConc:xEndConc]
yConc = y[yStartConc:yEndConc]
xWR = x[xStartWR:xEndWR]
yWR = y[yStartWR:yEndWR]

xMesh,yMesh = np.meshgrid(xAll,yAll)
xFlat = xMesh.flatten()
yFlat = yMesh.flatten()
xMeshConc,yMeshConc = np.meshgrid(xConc,yConc)
xFlatConc = xMeshConc.flatten()
yFlatConc = yMeshConc.flatten()
xMeshWR,yMeshWR = np.meshgrid(xWR,yWR)
xFlatWR = xMeshWR.flatten()
yFlatWR = yMeshWR.flatten()

xPoints = np.delete(xFlat,badInds,axis=0)
yPoints = np.delete(yFlat,badInds,axis=0)
pointsAll = np.arange(0,len(xFlat))
points = np.delete(pointsAll,badInds,axis=0)

xPointsConc = np.delete(xFlatConc,badIndsConc,axis=0)
yPointsConc = np.delete(yFlatConc,badIndsConc,axis=0)
pointsAllConc = np.arange(0,len(xFlatConc))
pointsConc = np.delete(pointsAllConc,badIndsConc,axis=0)


xPointsWR = np.delete(xFlatWR,badIndsWR,axis=0)
yPointsWR = np.delete(yFlatWR,badIndsWR,axis=0)
pointsAllWR = np.arange(0,len(xFlatWR))
pointsWR = np.delete(pointsAllWR,badIndsWR,axis=0)



plt.style.use('default')
import cartopy.crs as ccrs
fig=plt.figure(figsize=(6, 6))

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ax = plt.subplot2grid((3,3),(0,0),rowspan=2,colspan=3)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
# spatialField = onlyIceLessLand[-10]  # / 100 - np.nanmean(SLP_C, axis=0) / 100
# linearField = np.ones((np.shape(xFlat))) * np.nan
# linearField[points] = spatialField
# # rectField = linearField.reshape(85, 105)
# rectField = np.flipud(linearField.reshape((yEnd-yStart),(xEnd-xStart)))
# # cs = ax.coastlines(resolution='110m', linewidth=0.5)
# # iceInd = np.where(spatialField < 1)
# ax.set_extent([-180,180,50,90],crs=ccrs.PlateCarree())
# gl = ax.gridlines(draw_labels=True)
# extent=[-9.97,168.35,30.98,34.35]
# dx = dy = 25000
# x = np.arange(-3850000, +3750000, +dx)
# # y = np.arange(-5350000, +5850000, dy)
# y = np.arange(5850000, -5350000,-dy)
# ax.pcolormesh(x[xStart:xEnd], y[yStart:yEnd], rectField, cmap=cmocean.cm.ice,vmin=0,vmax=1)
spatialField3 = onlyIceLessLand[365*4+30]
spatialField2 = onlyIceConcLessLand[365*4+30]
spatialField = onlyIcePatternLessLand[365*4+30]  # / 100 - np.nanmean(SLP_C, axis=0) / 100

linearField3 = np.ones((np.shape(xFlat))) * np.nan
linearField2 = np.ones((np.shape(xFlatConc))) * np.nan
linearField = np.ones((np.shape(xFlatWR))) * np.nan

linearField[pointsWR] = spatialField
linearField2[pointsConc] = spatialField2
linearField3[points] = spatialField3

# rectField = linearField.reshape(85, 105)
rectField = np.flipud(linearField.reshape((yEndWR-yStartWR),(xEndWR-xStartWR)))
rectField2 = np.flipud(linearField2.reshape((yEndConc-yStartConc),(xEndConc-xStartConc)))
rectField3 = np.flipud(linearField3.reshape((yEnd-yStart),(xEnd-xStart)))

# cs = ax.coastlines(resolution='110m', linewidth=0.5)
# iceInd = np.where(spatialField < 1)
ax.set_extent([-180,180,50,90],crs=ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True)
extent=[-9.97,168.35,30.98,34.35]
dx = dy = 25000
x = np.arange(-3850000, +3750000, +dx)
# y = np.arange(-5350000, +5850000, dy)
y = np.arange(5850000, -5350000,-dy)
ax.pcolormesh(x[xStartWR:xEndWR], y[yStartWR:yEndWR], rectField, cmap=cmocean.cm.ice,vmin=0,vmax=1)
ax.pcolormesh(x[xStartConc:xEndConc], y[yStartConc:yEndConc], rectField2, cmap=cmocean.cm.ice_r,vmin=0,vmax=1)
# ax.pcolormesh(x[xStart:xEnd], y[yStart:yEnd], rectField3, cmap=cmocean.cm.ice,vmin=0,vmax=1)
ax.pcolormesh(x[xStart:xEnd], y[yStart:yEnd], rectField3,cmap=cmocean.cm.thermal,vmin=0,vmax=1)



# xAll = x
# yAll = y
# ax.pcolormesh(x[55:140], y[150:225], rectField, cmap=cmocean.cm.ice)
# ax.pcolormesh(x, y, np.flipud(rectField), cmap=cmocean.cm.ice)
# ax.pcolormesh(x[xStart:xEnd], y[yStart:yEnd], rectField, cmap=cmocean.cm.ice,vmin=0,vmax=1)
# ax.pcolormesh(x, y, rectField, cmap=plt.cm.Blues)

ax.set_xlim([-2822300,2420000])
ax.set_ylim([-2100000,2673000])
gl.xlabels_top = False
gl.ylabels_left = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
# ax.set_title('{}/{}/{}'.format(int(timeFile[0:4]),int(timeFile[4:6]),int(timeFile[6:8])))



import xarray as xr
from scipy.io.matlab.mio5_params import mat_struct
import scipy.io as sio
from datetime import datetime, timedelta, date
import numpy as np
from time_operations import xds2datetime as x2d
from time_operations import xds_reindex_daily as xr_daily
from time_operations import xds_common_dates_daily as xcd_daily
import pickle
from dateutil.relativedelta import relativedelta
from alr import ALR_WRP



def dateDay2datetimeDate(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [date(d[0], d[1], d[2]) for d in d_vec]

def dateDay2datetime(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [datetime(int(d[0]), int(d[1]), int(d[2])) for d in d_vec]


import datetime as dt

iceTime = [dt.datetime(np.array(year)[i],np.array(month)[i],np.array(day)[i]) for i in range(len(year))]

import datetime as dt
from dateutil.relativedelta import relativedelta
st = dt.datetime(1978, 10, 26)
end = dt.datetime(2023,11,14)
step = relativedelta(days=1)
dayTime = []
while st < end:
    dayTime.append(st)
    st += step




daysDaysWithNoIceP = [x for x in dayTime if x not in iceTime]
ind_dict = dict((k,i) for i,k in enumerate(dayTime))
inter = set(daysDaysWithNoIceP).intersection(dayTime)
indices = [ ind_dict[x] for x in inter ]
indices.sort()

daysDaysWithIceP = [x for x in dayTime if x in iceTime]
ind_dict = dict((k,i) for i,k in enumerate(dayTime))
inter2 = set(daysDaysWithIceP).intersection(dayTime)
indices2 = [ ind_dict[x] for x in inter2]
indices2.sort()

badWinter = np.where((np.asarray(dayTime) > dt.datetime(1987,12,1)) & ((np.asarray(dayTime) < dt.datetime(1988,1,13))))
replaceWinter = np.where((np.asarray(dayTime) == dt.datetime(1988,1,13)))

badSummer = np.where((np.asarray(dayTime) > dt.datetime(1992,6,9)) & ((np.asarray(dayTime) < dt.datetime(1992,7,20))))
replaceSummer = np.where((np.asarray(dayTime) > dt.datetime(1993,6,9)) & ((np.asarray(dayTime) < dt.datetime(1993,7,20))))

gapFilledIceP = np.nan * np.ones((len(dayTime),len(pointsWR)))
gapFilledIceP[indices,:] = onlyIcePatternLessLand[0:len(indices)]
gapFilledIceP[indices2,:] = onlyIcePatternLessLand
gapFilledIceP[badWinter[0],:] = gapFilledIceP[replaceWinter[0],:]
gapFilledIceP[badSummer[0],:] = gapFilledIceP[replaceSummer[0],:]


gapFilledConc = np.nan * np.ones((len(dayTime),len(pointsConc)))
gapFilledConc[indices,:] = onlyIceConcLessLand[0:len(indices)]
gapFilledConc[indices2,:] = onlyIceConcLessLand
gapFilledConc[badWinter[0],:] = gapFilledConc[replaceWinter[0],:]
gapFilledConc[badSummer[0],:] = gapFilledConc[replaceSummer[0],:]


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
gapFilledIce[badSummer[0],:] = gapFilledIce[replaceSummer[0],:]

#gapFilledIce = gapFilledIce[0:15856]
# iceTime = [np.array(np.array(year)[i],np.array(month)[i],np.array(day)[i]) for i in range(len(year))]
del onlyIceLessLand
del onlyIceConcLessLand
del onlyIcePatternLessLand

# iceDay
dayOfYear = np.array([hh.timetuple().tm_yday for hh in dayTime])  # returns 1 for January 1st
dayOfYearSine = np.sin(2*np.pi/366*dayOfYear)
dayOfYearCosine = np.cos(2*np.pi/366*dayOfYear)

dayConc = np.mean(gapFilledConc,axis=1)
areaBelowPt5 =[]
for qq in range(len(dayConc)):
    areaBelowPt5.append(len(np.where(gapFilledConc[qq,:] < 0.5)[0]))


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

areaBelow = np.concatenate((np.array(areaBelowPt5)[0:2],moving_average(np.array(areaBelowPt5),5),np.array(areaBelowPt5)[-2:]))


meanIce = np.mean(gapFilledIce,axis=1)
meanIce[badWinter] = 0.99
iceDigitized = np.digitize(meanIce,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1])
# iceDigitized = np.digitize(meanIce,[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.1])


zeroBin = np.where((iceDigitized==0))

zeroBinIce = gapFilledIceP[zeroBin[0],:]




iceMean = np.mean(zeroBinIce,axis=0)
iceStd = np.std(zeroBinIce,axis=0)
iceNorm = (zeroBinIce[:,:] - iceMean) / iceStd
iceNorm[np.isnan(iceNorm)] = 0

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# principal components analysis
ipca = PCA(n_components=min(iceNorm.shape[0], iceNorm.shape[1]))
PCs = ipca.fit_transform(iceNorm)
EOFs = ipca.components_
variance = ipca.explained_variance_
nPercent = variance / np.sum(variance)
APEV = np.cumsum(variance) / np.sum(variance) * 100.0
nterm = np.where(APEV <= 0.95 * 100)[0][-1]








import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# plotting the EOF patterns
plt.figure()
c1 = 0
c2 = 0
for hh in range(9):
    ax = plt.subplot2grid((3,3),(c1,c2),projection=ccrs.NorthPolarStereo(central_longitude=-45))
    spatialField = np.multiply(EOFs[hh,0:len(xPointsWR)],np.sqrt(variance[hh]))
    linearField = np.ones((np.shape(xFlatWR))) * np.nan
    linearField[pointsWR] = spatialField
    # rectField = linearField.reshape(110,130)
    # rectField = linearField.reshape(75,85)
    rectField = np.flipud(linearField.reshape((yEndWR-yStartWR),(xEndWR-xStartWR)))

    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True)
    extent = [-9.97, 168.35, 30.98, 34.35]
    # ax.pcolormesh(x[25:155], y[125:235], rectField)#, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[55:140], y[150:225], rectField)#, cmap=cmocean.cm.ice)
    ax.pcolormesh(x[xStartWR:xEndWR], y[yStartWR:yEndWR], rectField)#, cmap=cmocean.cm.ice)

    # ax.set_xlim([-3223000, 0])
    # ax.set_ylim([0, 2730000])
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
    c2 += 1
    if c2 == 3:
        c1 += 1
        c2 = 0



# CAN WE PUT A GUIDED REGRESSION IN HERE FOR THE PRESENCE OF WAVES?


PCsub = PCs[:, :nterm+1]
EOFsub = EOFs[:nterm+1, :]



PCsub_std = np.std(PCsub, axis=0)
PCsub_norm = np.divide(PCsub, PCsub_std)

X = PCsub_norm  #  predictor

# PREDICTAND: WAVES data
#wd = np.array([xds_WAVES[vn].values[:] for vn in name_vars]).T
# wd = np.vstack((iceWaves['wh_all'],iceWaves['tp_all'],np.multiply(iceWaves['wh_all']**2,iceWaves['tp_all'])**(1.0/3))).T
# wd = np.vstack((dayOfYearSine,dayOfYearCosine,iceConcentration)).T
wd = np.vstack((dayOfYearSine[zeroBin[0]],dayOfYearCosine[zeroBin[0]],areaBelow[zeroBin[0]])).T

wd_std = np.nanstd(wd, axis=0)
wd_norm = np.divide(wd, wd_std)

Y = wd_norm  # predictand

# Adjust
[n, d] = Y.shape
X = np.concatenate((np.ones((n, 1)), X), axis=1)
from sklearn import linear_model

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


# kma = KMeans(n_clusters=num_clusters, n_init=2000).fit(PCsub)
# # groupsize
# _, group_size = np.unique(kma.labels_, return_counts=True)
#

# KMA Regression Guided
num_clusters = 9
repres = 0.90
alpha = 0.4
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
# kma_order = np.flipud(np.argsort(group_size))
kma_order = np.argsort(np.mean(-km, axis=1))
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





from matplotlib import gridspec

# plotting the EOF patterns
fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(int(np.sqrt(num_clusters)), int(np.sqrt(num_clusters)))
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

    spatialField = kmSorted[(hh), :]# / 100 - np.nanmean(SLP_C, axis=0) / 100
    linearField = np.ones((np.shape(xFlatWR))) * np.nan
    linearField[pointsWR] = spatialField
    # rectField = linearField.reshape(110, 130)
    # rectField = linearField.reshape(75,85)
    # rectField = linearField.reshape(85,95)
    rectField = np.flipud(linearField.reshape((yEndWR-yStartWR),(xEndWR-xStartWR)))

    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    # gl = ax.gridlines(draw_labels=True)
    extent = [-9.97, 168.35, 30.98, 34.35]
    # ax.pcolormesh(x[25:155], y[125:235], rectField, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[55:140], y[150:225], rectField, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[55:150], y [150:235], rectField, cmap=cmocean.cm.ice)
    ax.pcolormesh(x[xStartWR:xEndWR], y[yStartWR:yEndWR], rectField, cmap=cmocean.cm.ice)

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





zeroBmus = bmus



iceTime = dayTime

iceOrder = np.arange(0,num_clusters+9)
iceDateTimes = iceTime

bmus = iceDigitized + num_clusters-1
bmus[zeroBin[0]] = zeroBmus

OGbmus = bmus

fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(int(np.sqrt(len(np.unique(bmus)))+1), int(np.sqrt(len(np.unique(bmus)))))
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
c1 = 0
c2 = 0

for hh in range(len(np.unique(bmus))):
    ax = plt.subplot(gs1[hh])
    #num = kma_order[hh]
    finder = np.where(bmus==hh)
    ax.hist(dayOfYear[finder])
    ax.set_xlim([0,365])
    # ax.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
    ax.xaxis.set_ticks([1, 92, 183, 275])
    ax.xaxis.set_ticklabels(['Jan', 'Apr',  'Jul', 'Oct'])

fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(int(np.sqrt(len(np.unique(bmus)))+1), int(np.sqrt(len(np.unique(bmus)))))
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
c1 = 0
c2 = 0

for hh in range(len(np.unique(bmus))):
    ax = plt.subplot(gs1[hh])
    #num = kma_order[hh]
    finder = np.where(bmus==hh)
    ax.hist(areaBelow[finder])
    ax.set_xlim([0,1200])
    # ax.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
    # ax.xaxis.set_ticks([1, 92, 183, 275])
    # ax.xaxis.set_ticklabels(['Jan', 'Apr',  'Jul', 'Oct'])
#
# asfg
#
# import sys
# sys.path.insert(0, '/Users/dylananderson/Documents/projects/gencadeClimate/')
#
# with open(r"pointLayWavesWindsTemps164by70.pickle", "rb") as input_file:
#     pointLay = pickle.load(input_file)
#
#
# hs = pointLay['metOcean'].Hs
# tp = pointLay['metOcean'].Tp
# dm = pointLay['metOcean'].Dm
#
# tWave = np.array([datetime.utcfromtimestamp((qq - np.datetime64(0, 's')) / np.timedelta64(1, 's'))for qq in pointLay['metOcean'].timeWave])
# tC = np.array(tWave)
#
#
#
# # from dateutil.relativedelta import relativedelta
# # st = dt.datetime(startTime[0], startTime[1], startTime[2])
# # # end = dt.datetime(2021,12,31)
# # end = dt.datetime(endTime[0],endTime[1]+1,1)
# # step = relativedelta(hours=1)
# # hourTime = []
# # while st < end:
# #     hourTime.append(st)#.strftime('%Y-%m-%d'))
# #     st += step
# #
# #
# dt = datetime(1978, 11, 1)
# end = datetime(2023, 11, 1)
# step = timedelta(days=1)
# midnightTime = []
# while dt < end:
#     midnightTime.append(dt)#.strftime('%Y-%m-%d'))
#     dt += step
#
#
# # bmus = newBmus[0:len(midnightTime)]
#
# # newBmus
# # bmus = bmus[0:len(midnightTime)]
#
# import itertools
# import operator
#
# grouped = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(bmus), key=operator.itemgetter(1))]
#
# groupLength = np.asarray([len(i) for i in grouped])
# bmuGroup = np.asarray([bmus[i[0]] for i in grouped])
# timeGroup = [np.asarray(midnightTime)[i] for i in grouped]
#
# # fetchGroup = np.asarray([fetchFiltered[i[0]] for i in grouped])
#
# startTimes = [i[0] for i in timeGroup]
# endTimes = [i[-1] for i in timeGroup]
#
# hydrosInds = np.unique(bmuGroup)
#
#
# hydros = list()
# c = 0
# for p in range(len(np.unique(bmuGroup))):
#
#     tempInd = p+1#hydrosInds[p]
#     print('working on bmu = {}'.format(tempInd))
#     index = np.where((bmuGroup==tempInd))[0][:]
#
#     #print('should have at least {} days in it'.format(len(index)))
#     subLength = groupLength[index]
#     m = np.ceil(np.sqrt(len(index)))
#     tempList = list()
#
#     for i in range(len(index)):
#         st = startTimes[index[i]]
#         et = endTimes[index[i]] + timedelta(days=1)
#
#         waveInd = np.where((tWave < et) & (tWave >= st))
#         if len(waveInd[0]) > 0:
#             newTime = startTimes[index[i]]
#             tempHydroLength = subLength[i]
#             counter = 0
#             while tempHydroLength > 1:
#                 #randLength = random.randint(1,2)
#                 etNew = newTime + timedelta(days=1)
#                 waveInd = np.where((tWave < etNew) & (tWave >= newTime))
#                 fetchInd = np.where((np.asarray(dayTime) == newTime))
#                 deltaDays = et-etNew
#                 c = c + 1
#                 # counter = counter + 1
#                 # if counter > 15:
#                 #     print('we have split 15 times')
#
#                 tempDict = dict()
#                 tempDict['time'] = tWave[waveInd[0]]
#                 tempDict['numDays'] = subLength[i]
#                 tempDict['hs'] = hs[waveInd[0]]
#                 # tempDict['tp'] = tpCombined[waveInd[0]]
#                 # tempDict['dm'] = waveNorm[waveInd[0]]
#                 tempDict['fetch'] = dayConc[fetchInd[0][0]]#tempFetch[0]
#                         # tempDict['res'] = res[waterInd[0]]
#                         # tempDict['wl'] = wlHycom[waterInd[0]]
#                 # tempDict['cop'] = np.asarray([np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
#                 #                                       np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
#                 #                                       np.nanmean(waveNorm[waveInd[0]]),fetchTotalConcentration[fetchInd[0][0]]])#, np.nanmean(res[waterInd[0]])])
#                 # tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
#                 # tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
#                 # tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
#                 # tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
#                 # tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
#                 # # tempDict['ssMean'] = np.nanmean(res[waterInd[0]])
#                 tempList.append(tempDict)
#                 tempHydroLength = tempHydroLength-1
#                 newTime = etNew
#             else:
#                 waveInd = np.where((tWave < et) & (tWave >= newTime))
#                 fetchInd = np.where((np.asarray(dayTime) == newTime))
#
#                 # waterInd = np.where((tWl < et) & (tWl >= etNew7))
#                 c = c + 1
#                 tempDict = dict()
#                 tempDict['time'] = tWave[waveInd[0]]
#                 tempDict['numDays'] = subLength[i]
#                 tempDict['hs'] = hs[waveInd[0]]
#                 # tempDict['tp'] = tpCombined[waveInd[0]]
#                 # tempDict['dm'] = waveNorm[waveInd[0]]
#                 tempDict['fetch'] = dayConc[fetchInd[0][0]]#tempFetch[0]
#
#                         # # tempDict['res'] = res[waterInd[0]]
#                         # # tempDict['wl'] = wlHycom[waterInd[0]]
#                         # tempDict['cop'] = np.asarray(
#                         #     [np.nanmin(hsCombined[waveInd[0]]),
#                         #      np.nanmax(hsCombined[waveInd[0]]),
#                         #      np.nanmin(tpCombined[waveInd[0]]),
#                         #      np.nanmax(tpCombined[waveInd[0]]),
#                         #      np.nanmean(
#                         #          waveNorm[waveInd[0]]),fetchTotalConcentration[fetchInd[0][0]]])  # ,
#                         # # np.nanmean(res[waterInd[0]])])
#                         # tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
#                         # tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
#                         # tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
#                         # tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
#                         # tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
#                         # # tempDict['ssMean'] = np.nanmean(res[waterInd[0]])
#                 tempList.append(tempDict)
#     hydros.append(tempList)
#


# bmus = bmus[218:-154]+1
# bmus_dates = iceDateTimes[218:-154]#[151:-214]


bmus = bmus[6:-14]+1
bmus_dates = iceDateTimes[6:-14]#[151:-214]



xds_KMA_fit = xr.Dataset(
    {
        'bmus':(('time',), bmus),
    },
    coords = {'time': bmus_dates}#[datetime(r[0],r[1],r[2]) for r in timeDWTs]
)


#
#
# with open(r"awtPCs.pickle", "rb") as input_file:
#     historicalAWTs = pickle.load(input_file)
#
# # with open(r"ensoPCs2022.pickle", "rb") as input_file:
# #     historicalAWTs = pickle.load(input_file)
#
# dailyPC1 = historicalAWTs['dailyPC1']
# dailyPC2 = historicalAWTs['dailyPC2']
# dailyPC3 = historicalAWTs['dailyPC3']
# dailyDates = historicalAWTs['dailyDates']
# awt_bmus = historicalAWTs['awt_bmus']
# annualTime = historicalAWTs['annualTime']
# dailyAWT = historicalAWTs['dailyAWT']
#
# # AWT: PCs (Generated with copula simulation. Annual data, parse to daily)
# xds_PCs_fit = xr.Dataset(
#     {
#         'PC1': (('time',), dailyPC1),
#         'PC2': (('time',), dailyPC2),
#         #'PC3': (('time',), dailyPC3),
#     },
#     coords = {'time': [datetime(r[0],r[1],r[2]) for r in dailyDates]}
# )
# # reindex annual data to daily data
# xds_PCs_fit = xr_daily(xds_PCs_fit, datetime(1979,6,1),datetime(2021,5,31))



#
# with open(r"observedArcticTemp.pickle", "rb") as input_file:
#    historicalTemp = pickle.load(input_file)
#
# timeTemp = historicalTemp['dailyDate'][151:]
# arcticTemp = historicalTemp['arcticTemp'][151:]
#
# xds_Temp_fit = xr.Dataset(
#     {
#         'temp': (('time',), arcticTemp),
#     },
#     coords = {'time': timeTemp}
# )
#
# # reindex to daily data after 1979-01-01 (avoid NaN)
# xds_Temp_fit = xr_daily(xds_Temp_fit, datetime(1979, 6, 1),datetime(2022,5,31))


# Loading historical Arctic Temperatures for 1979 to 2022
with open(r"predictedArctic1978to2023.pickle", "rb") as input_file:
   arcticTemperatures = pickle.load(input_file)

timeTemp = arcticTemperatures['futureDate']
arcticTemp = np.array(arcticTemperatures['futureSims'][0])

xds_Temp_fit = xr.Dataset(
    {
        'temp': (('time',), arcticTemp),
    },
    coords = {'time': timeTemp}
)

# reindex to daily data after 1979-01-01 (avoid NaN)
# xds_Temp_fit = xr_daily(xds_Temp_fit, datetime(1979, 6, 1),datetime(2023,5,31))
# xds_Temp_fit = xr_daily(xds_Temp_fit, datetime(1979, 6, 1),datetime(2023,5,31))
xds_Temp_fit = xr_daily(xds_Temp_fit, datetime(1978, 11, 1),datetime(2023,10,31))



# --------------------------------------
# Mount covariates matrix

# available data:
# model fit: xds_KMA_fit, xds_MJO_fit, xds_PCs_fit
# model sim: xds_MJO_sim, xds_PCs_sim

# covariates: FIT
# d_covars_fit = xcd_daily([xds_PCs_fit, xds_Temp_fit, xds_KMA_fit])
d_covars_fit = xcd_daily([xds_Temp_fit, xds_KMA_fit])
#
# # # # # PCs covar
# cov_PCs = xds_PCs_fit.sel(time=slice(d_covars_fit[0],d_covars_fit[-1]))
# cov_1 = cov_PCs.PC1.values.reshape(-1,1)
# cov_2 = cov_PCs.PC2.values.reshape(-1,1)
# # cov_3 = cov_PCs.PC3.values.reshape(-1,1)

# MJO covars
cov_Temp = xds_Temp_fit.sel(time=slice(d_covars_fit[0],d_covars_fit[-1]))
cov_1 = cov_Temp.temp.values.reshape(-1,1)

# join covars and norm.
# cov_T = np.hstack((cov_1, cov_2, cov_3))#, cov_4))
#cov_T = np.hstack((cov_1))
cov_T = cov_1
cov_T_mean = np.mean(cov_T,axis=0)
cov_T_std = np.std(cov_T,axis=0)
#cov_T_std = np.array(cov_T_std[0])
# multCovT = np.array([0.31804979/0.31804979, 0.16031134/0.31804979, 0.12182678/0.31804979, 0.09111769/0.31804979, 1, 1])
# multCovT = np.array([0.4019148/0.4019148, 0.11355852/0.4019148, 0.10510168/0.4019148, 1])
multCovT = np.array([1])
# multCovT = np.array([0.4019148/0.4019148, 0.11355852/0.4019148, 1])

covTNorm = np.divide(np.subtract(cov_T,cov_T_mean),cov_T_std)
covTNormalize = np.multiply(covTNorm,multCovT)

# covTSimNorm = np.divide(np.subtract(cov_T_sim,cov_T_mean),cov_T_std)
# covTSimNormalize = np.multiply(covTSimNorm,multCovT)

# KMA related covars starting at KMA period
# i0 = d_covars_fit.index(x2d(xds_KMA_fit.time[0]))
i0 = d_covars_fit.index(datetime(int(xds_KMA_fit.time.dt.year[0]),int(xds_KMA_fit.time.dt.month[0]),int(xds_KMA_fit.time.dt.day[0])))

# cov_KMA = cov_T[i0:,:]
cov_KMA = cov_T[i0:]

d_covars_fit = d_covars_fit[i0:]

# generate xarray.Dataset
# cov_names = ['PC1', 'PC2', 'PC3', 'Temp']
cov_names = ['Temp']
# cov_names = ['PC1', 'PC2', 'Temp']

xds_cov_fit = xr.Dataset(
    {
        'cov_values': (('time','cov_names'), covTNormalize),
    },
    coords = {
        'time': d_covars_fit,
        'cov_names': cov_names,
    }
)


# --------------------------------------
# Autoregressive Logistic Regression

# available data:
# model fit: xds_KMA_fit, xds_cov_sim, num_clusters
# model sim: xds_cov_sim, sim_num, sim_years

# use bmus inside covariate time frame
xds_bmus_fit = xds_KMA_fit.sel(
    time=slice(d_covars_fit[0], d_covars_fit[-1])
)


# Autoregressive logistic wrapper
num_clusterstoALR = num_clusters+9
sim_num = 100
fit_and_save = True # False for loading
p_test_ALR = '/users/dylananderson/documents/data/pointHope/testIceALR/'

# ALR terms
# d_terms_settings = {
#     'mk_order'  : 2,
#     'constant' : True,
#     'long_term' : False,
#     'seasonality': (True, [2]),
#     'covariates': (True, xds_cov_fit),
# }

# ALR terms
d_terms_settings = {
    'mk_order'  : 2,
    'constant' : True,
    'long_term' : True,
    'seasonality': (True, [2]),
    'covariates': (True, xds_cov_fit),
}
# Autoregressive logistic wrapper
ALRW = ALR_WRP(p_test_ALR)
ALRW.SetFitData(
    num_clusterstoALR, xds_bmus_fit, d_terms_settings)

ALRW.FitModel(max_iter=20000)


#p_report = op.join(p_data, 'r_{0}'.format(name_test))

ALRW.Report_Fit() #'/media/dylananderson/Elements/NC_climate/testALR/r_Test', terms_fit==False)








# ALR model simulations
sim_years = 45
# start simulation at PCs available data
# d1 = x2d(xds_cov_fit.time[0])
d1 = datetime(int(xds_cov_fit.time.dt.year[0]),int(xds_cov_fit.time.dt.month[0]),int(xds_cov_fit.time.dt.day[0]))
d2 = datetime(d1.year+sim_years, d1.month, d1.day)
dates_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]

# print some info
print('ALR model fit   : {0} --- {1}'.format(
    d_covars_fit[0], d_covars_fit[-1]))
print('ALR model sim   : {0} --- {1}'.format(
    dates_sim[0], dates_sim[-1]))

# launch simulation
xds_ALR = ALRW.Simulate(
    sim_num, dates_sim[0:-2], xds_cov_fit)

dates_simHist = dates_sim[0:-2]





# Save results for matlab plot
evbmus_simHist = xds_ALR.evbmus_sims.values
#
# p_mat_output = ('/users/dylananderson/Documents/data/pointHope/testIceALR/testOfIcePercentileClassificationSynTemp_y{0}s{1}.h5'.format(
#         sim_years, sim_num))
# import h5py
# with h5py.File(p_mat_output, 'w') as hf:
#     hf['bmusim'] = evbmus_simHist
#     # hf['probcum'] = evbmus_probcum
#     hf['dates'] = np.vstack(
#         ([d.year for d in dates_simHist],
#         [d.month for d in dates_simHist],
#         [d.day for d in dates_simHist])).T
#
#
# # samplesPickle = 'iceOnlyTemp36DWTsSimulations100.pickle'
# samplesPickle = 'iceTempWithPercentileClassificationSynTempSimulations100.pickle'
#
# outputSamples = {}
# outputSamples['evbmus_sim'] = evbmus_simHist
# # outputSamples['evbmus_probcum'] = evbmus_probcum
# outputSamples['sim_years'] = sim_num
# outputSamples['dates_sim'] = dates_simHist
#
# with open(samplesPickle,'wb') as f:
#     pickle.dump(outputSamples, f)





# FUTURE ARCTIC AIR TEMPERATURE
futureArcticTime = arcticTemperatures['futureDate']
futureArcticTemp = arcticTemperatures['futureSims']
# futureArcticTemp = arcticTemperatures['alternateFutureSims']



sim_num = 10
diffSims = 100
evbmus_sim = np.nan*np.ones((len(futureArcticTime)-366-366,sim_num*diffSims))
c = 0
for simIndex in range(diffSims):

    print('working on large-scale climate simulation #{}'.format(simIndex))

    simTemp = np.array(arcticTemperatures['futureSims'][simIndex])

    xds_Temp_sim = xr.Dataset(
        {
            'temp': (('time',), simTemp),
        },
        coords={'time': futureArcticTime}
    )

    # reindex to daily data after 1979-01-01 (avoid NaN)
    xds_Temp_sim = xr_daily(xds_Temp_sim, datetime(1979, 6, 2), datetime(2075, 5, 31))
    #
    # swtBMUsim = swtBMUS[simIndex][0:100]#[0:len(awt_bmus)]
    # swtPC1sim = swtPC1[simIndex][0:100]#[0:len(awt_bmus)]
    # swtPC2sim = swtPC2[simIndex][0:100]#[0:len(awt_bmus)]
    # #swtPC3sim = swtPC3[simIndex][0:100]#[0:len(awt_bmus)]
    #
    # # trainingDates = mjoDatesSim#[datetime(r[0],r[1],r[2]) for r in dailyDates]
    # trainingDates = [datetime(r[0],r[1],r[2]) for r in simDailyDatesMatrix]
    # dailyAWTsim = np.ones((len(trainingDates),))
    # dailyPC1sim = np.ones((len(trainingDates),))
    # dailyPC2sim = np.ones((len(trainingDates),))
    # #dailyPC3sim = np.ones((len(trainingDates),))
    #
    # dailyDatesSWTyear = np.array([r[0] for r in simDailyDatesMatrix])
    # dailyDatesSWTmonth = np.array([r[1] for r in simDailyDatesMatrix])
    # dailyDatesSWTday = np.array([r[2] for r in simDailyDatesMatrix])
    # normPC1 = swtPC1sim
    # normPC2 = swtPC2sim
    # #normPC3 = swtPC3sim
    #
    # for i in range(len(swtBMUsim)):
    #     sSeason = np.where((simDailyDatesMatrix[:, 0] == swtDatesSim[i].year) & (
    #                 simDailyDatesMatrix[:, 1] == swtDatesSim[i].month) & (simDailyDatesMatrix[:, 2] == 1))
    #     ssSeason = np.where((simDailyDatesMatrix[:, 0] == swtDatesSim[i].year + 1) & (
    #                 simDailyDatesMatrix[:, 1] == swtDatesSim[i].month) & (simDailyDatesMatrix[:, 2] == 1))
    #
    #     dailyAWTsim[sSeason[0][0]:ssSeason[0][0] + 1] = swtBMUsim[i] * dailyAWT[sSeason[0][0]:ssSeason[0][0] + 1]
    #     dailyPC1sim[sSeason[0][0]:ssSeason[0][0] + 1] = normPC1[i] * np.ones(
    #         len(dailyAWT[sSeason[0][0]:ssSeason[0][0] + 1]), )
    #     dailyPC2sim[sSeason[0][0]:ssSeason[0][0] + 1] = normPC2[i] * np.ones(
    #         len(dailyAWT[sSeason[0][0]:ssSeason[0][0] + 1]), )
    #     #dailyPC3sim[sSeason[0][0]:ssSeason[0][0] + 1] = normPC3[i] * np.ones(
    #         #len(dailyAWT[sSeason[0][0]:ssSeason[0][0] + 1]), )
    #
    # xds_PCs_sim = xr.Dataset(
    #     {
    #         'PC1': (('time',), dailyPC1sim),
    #         'PC2': (('time',), dailyPC2sim),
    #         #'PC3': (('time',), dailyPC3sim),
    #     },
    #     # coords = {'time': [datetime(r.year,r.month,r.day) for r in swtDatesSim]}
    #     coords = {'time': [datetime(r[0], r[1], r[2]) for r in simDailyDatesMatrix]}
    # )
    #
    # # reindex annual data to daily data
    # xds_PCs_sim = xr_daily(xds_PCs_sim,datetime(2021,6,2),datetime(2050,5,31))
    # # xds_PCs_sim = xr_daily(xds_PCs_sim,datetime(1979,6,1),datetime(2021,5,31))
    #


    d_covars_sim = xcd_daily([xds_Temp_sim])#,xds_PCs_sim])
    # cov_PCs_sim = xds_PCs_sim.sel(time=slice(d_covars_sim[0],d_covars_sim[-1]))
    # cov_1_sim = cov_PCs_sim.PC1.values.reshape(-1,1)
    # cov_2_sim = cov_PCs_sim.PC2.values.reshape(-1,1)
    # #cov_3_sim = cov_PCs_sim.PC3.values.reshape(-1,1)
    # #cov_4_sim = cov_PCs_sim.PC4.values.reshape(-1,1)
    cov_Temp_sim = xds_Temp_sim.sel(time=slice(d_covars_sim[0],d_covars_sim[-1]))
    cov_1_sim = cov_Temp_sim.temp.values.reshape(-1,1)
    #cov_6_sim = cov_MJOs_sim.rmm2.values.reshape(-1,1)
    # cov_T_sim = np.hstack((cov_1_sim, cov_2_sim, cov_3_sim))#, cov_5_sim, cov_6_sim))
    cov_T_sim = cov_1_sim#, cov_5_sim, cov_6_sim))

    covTSimNorm = np.divide(np.subtract(cov_T_sim,np.mean(cov_T_sim,axis=0)),np.std(cov_T_sim,axis=0))
    # covTSimNorm = np.divide(np.subtract(cov_T_sim,cov_T_mean),cov_T_std)
    covTSimNormalize = np.multiply(covTSimNorm,multCovT)


    # generate xarray.Dataset
    xds_cov_sim = xr.Dataset(
        {
            'cov_values': (('time','cov_names'), covTSimNormalize),
        },
        coords = {
            'time': d_covars_sim,
            'cov_names': cov_names,
        }
    )



    # --------------------------------------
    # Autoregressive Logistic Regression

    # available data:
    # model fit: xds_KMA_fit, xds_cov_sim, num_clusters
    # model sim: xds_cov_sim, sim_num, sim_years


    #p_report = op.join(p_data, 'r_{0}'.format(name_test))

    #ALRW.Report_Fit() #'/media/dylananderson/Elements/NC_climate/testALR/r_Test', terms_fit==False)


    # # ALR model simulations
    # sim_years = 100
    # # start simulation at PCs available data
    # d1 = x2d(xds_cov_sim.time[0])
    # d2 = datetime(d1.year+sim_years, d1.month, d1.day)
    # dates_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]
    # dates_sim = dates_sim[0:-2]
    # ALR model simulations
    sim_years = 71+25
    # start simulation at PCs available data
    # d1 = x2d(xds_cov_sim.time[0])
    d1 = datetime(int(xds_cov_sim.time.dt.year[0]), int(xds_cov_sim.time.dt.month[0]), int(xds_cov_sim.time.dt.day[0]))

    d2 = datetime(d1.year+sim_years, d1.month, d1.day)
    dates_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]
    dates_sim = dates_sim[0:-2]
    # print some info

    # print('ALR model sim   : {0} --- {1}'.format(
    #     dates_sim[0], dates_sim[-1]))
    print('ALR model sim   : {0} --- {1}'.format(
        d_covars_fit[0], d_covars_fit[-1]))

    # launch simulation
    xds_ALR = ALRW.Simulate(
        sim_num, dates_sim, xds_cov_sim)

    # dates_sim = dates_sim[0:-2]


    # Save results for matlab plot
    evbmus_sim[:,c:c+10] = xds_ALR.evbmus_sims.values
    c = c + 10
    # evbmus_probcum = xds_ALR.evbmus_probcum.values



del simTemp
del onlyIceWR
del PCA
del KMeans
del PCs
del EOFs
del arcticTemperatures

# p_mat_output = ('/media/dylananderson/Elements/NC_climate/testALR/testFutureAnnual_y{0}s{1}.h5'.format(
#         sim_years, sim_num))
# import h5py
# with h5py.File(p_mat_output, 'w') as hf:
#     hf['bmusim'] = evbmus_sim
#     # hf['probcum'] = evbmus_probcum
#     hf['dates'] = np.vstack(
#         ([d.year for d in dates_sim],
#         [d.month for d in dates_sim],
#         [d.day for d in dates_sim])).T
#

# samplesPickle = 'ice18FutureSimulations100PointLay.pickle'
# outputSamples = {}
# outputSamples['evbmus_sim'] = evbmus_sim
# outputSamples['sim_years'] = sim_num
# outputSamples['dates_sim'] = dates_sim
# outputSamples['futureTemp'] = futureArcticTemp
# outputSamples['futureArcticTime'] = futureArcticTime
#
#
# with open(samplesPickle,'wb') as f:
#     pickle.dump(outputSamples, f)
#







def GenOneYearDaily(yy=1981, month_ini=1):
   'returns one generic year in a list of datetimes. Daily resolution'

   dp1 = datetime(yy, month_ini, 2)
   dp2 = dp1 + timedelta(days=364)

   return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]


def GenOneSeasonDaily(yy=1981, month_ini=1):
   'returns one generic year in a list of datetimes. Daily resolution'

   dp1 = datetime(yy, month_ini, 1)
   dp2 = dp1 + timedelta(3*365/12)

   return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]



import matplotlib.pyplot as plt



bmus_dates_months = np.array([d.month for d in dates_sim])
bmus_dates_days = np.array([d.day for d in dates_sim])


# generate perpetual year list
list_pyear = GenOneYearDaily(month_ini=6)
m_plot = np.zeros((num_clusterstoALR, len(list_pyear))) * np.nan
num_sim=1
# sort data
for i, dpy in enumerate(list_pyear):
   _, s = np.where(
      [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)]
   )
   b = evbmus_sim[s,:]
   # b = bmus[s]
   b = b.flatten()

   for j in range(num_clusterstoALR):
      _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!
      # _, bb = np.where([(j == b)])  # j+1 starts at 1 bmus value!
      m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)

import matplotlib.cm as cm
dwtcolors = cm.rainbow(np.linspace(0, 1, num_clusterstoALR))
# tccolors = np.flipud(cm.autumn(np.linspace(0,1,21)))
# dwtcolors = np.vstack((etcolors,tccolors[1:,:]))




fig = plt.figure(figsize=(10,4))
ax = plt.subplot2grid((1,1),(0,0))
# plot stacked bars
bottom_val = np.zeros(m_plot[1, :].shape)
for r in range(num_clusterstoALR):
   row_val = m_plot[r, :]
   ax.bar(list_pyear, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
   # store bottom
   bottom_val += row_val

import matplotlib.dates as mdates
# customize  axis
months = mdates.MonthLocator()
monthsFmt = mdates.DateFormatter('%b')
ax.set_xlim(list_pyear[0], list_pyear[-1])
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.set_ylim(0, 100)
ax.set_ylabel('Probability')


bmus_dates_monthsHist = np.array([d.month for d in dates_simHist])
bmus_dates_daysHist = np.array([d.day for d in dates_simHist])


# generate perpetual year list
list_pyear = GenOneYearDaily(month_ini=6)
m_plot = np.zeros((num_clusterstoALR, len(list_pyear))) * np.nan
num_sim=1
# sort data
for i, dpy in enumerate(list_pyear):
   _, s = np.where(
      [(bmus_dates_monthsHist == dpy.month) & (bmus_dates_daysHist == dpy.day)]
   )
   print(s)

   b = evbmus_simHist[s,0]
   # b = bmus[s]
   b = b.flatten()
   print(b)
   for j in range(num_clusterstoALR):
      _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!
      # _, bb = np.where([(j == b)])  # j+1 starts at 1 bmus value!

      m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)

import matplotlib.cm as cm
# etcolors = cm.viridis(np.linspace(0, 1, 70-20))
# tccolors = np.flipud(cm.autumn(np.linspace(0,1,21)))
# dwtcolors = np.vstack((etcolors,tccolors[1:,:]))

fig = plt.figure(figsize=(10,4))
ax = plt.subplot2grid((1,1),(0,0))
# plot stacked bars
bottom_val = np.zeros(m_plot[1, :].shape)
for r in range(num_clusterstoALR):
   row_val = m_plot[r, :]
   ax.bar(list_pyear, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
   # store bottom
   bottom_val += row_val

import matplotlib.dates as mdates
# customize  axis
months = mdates.MonthLocator()
monthsFmt = mdates.DateFormatter('%b')
ax.set_xlim(list_pyear[0], list_pyear[-1])
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
# ax.set_ylim(0, 100)
ax.set_ylim(0, 1)
ax.set_ylabel('Probability')







# generate perpetual year list
list_pyear = GenOneYearDaily(month_ini=6)
m_plot = np.zeros((num_clusterstoALR, len(list_pyear))) * np.nan
num_sim=1
# sort data
for i, dpy in enumerate(list_pyear):
   _, s = np.where([(bmus_dates_monthsHist == dpy.month) & (bmus_dates_daysHist == dpy.day)])
   # b = evbmus_sim[s,:]
   b = bmus[s]
   b = b.flatten()

   for j in range(num_clusterstoALR):
      _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!
      # _, bb = np.where([(j == b)])  # j+1 starts at 1 bmus value!

      m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)

# plt.style.use('dark_background')
fig = plt.figure(figsize=(10,4))
ax = plt.subplot2grid((1,1),(0,0))
# plot stacked bars
bottom_val = np.zeros(m_plot[1, :].shape)
for r in range(num_clusterstoALR):
   row_val = m_plot[r, :]
   ax.bar(
      list_pyear, row_val, bottom=bottom_val,
      width=1, color=np.array([dwtcolors[r]]))

   # store bottom
   bottom_val += row_val

import matplotlib.dates as mdates

# customize  axis
months = mdates.MonthLocator()
monthsFmt = mdates.DateFormatter('%b')

ax.set_xlim(list_pyear[0], list_pyear[-1])
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.set_ylim(0, 1)
ax.set_ylabel('Probability')

#
# from matplotlib import gridspec
#
#
# #
# # Lets get complicated...
# # a grid, 8 x 4 for the 8 SWTs and the 4 seasons?
# # generate perpetual seasonal list
# fig = plt.figure()
# gs = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.15)
#
# # onlyDo = [0,1,2,3,4,5]
# onlyDo = [0,5]
#
# c = 0
# for m in np.unique(onlyDo):
#
#     list_pSeason = GenOneYearDaily(month_ini=6)
#     m_plot = np.zeros((70, len(list_pSeason))) * np.nan
#     num_clusters=70
#     num_sim=1
#     # Identify on the data that occurs in this awt
#     awtIndex = np.where(dailyAWT[0:-2] == m)
#     subsetMonths = bmus_dates_months[awtIndex]
#     subsetDays = bmus_dates_days[awtIndex]
#     subsetBmus = bmus[awtIndex]
#     #subsetBmus = evbmus_sim[awtIndex[0], :]
#     # sort data
#
#     for i, dpy in enumerate(list_pSeason):
#         _, s = np.where([(subsetMonths == dpy.month) & (subsetDays == dpy.day)])
#         # _, s = np.where([(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)])
#         # b = evbmus_sim[s,:]
#         # b = bmus[s]
#         b = subsetBmus[s]
#         b = b.flatten()
#
#         for j in range(num_clusters):
#             _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!
#
#             m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)
#
#     ax = plt.subplot(gs[c])
#     # plot stacked bars
#     bottom_val = np.zeros(m_plot[1, :].shape)
#     for r in range(num_clusters):
#         row_val = m_plot[r, :]
#         ax.bar(list_pSeason, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
#
#         # store bottom
#         bottom_val += row_val
#     # customize  axis
#     months = mdates.MonthLocator()
#     monthsFmt = mdates.DateFormatter('%b')
#     ax.set_xlim(list_pSeason[0], list_pSeason[-1])
#     ax.xaxis.set_major_locator(months)
#     ax.xaxis.set_major_formatter(monthsFmt)
#     #ax.set_ylim(0, 100)
#     ax.set_ylabel('')
#     c = c + 1
#

# dailyMWT = dailyMWT[0:-2]
#
# # evbmus_sim = evbmus_sim - 1
# # bmus = bmus + 1
# fig = plt.figure()
# gs = gridspec.GridSpec(9, 4, wspace=0.1, hspace=0.15)
# c = 0
# for awt in np.unique(awt_bmus):
#
#     ind = np.where((dailyMWT == awt))[0][:]
#     timeSubDays = bmus_dates_days[ind]
#     timeSubMonths = bmus_dates_months[ind]
#     a = evbmus_sim[ind,:]
#
#     monthsIni = [3,6,9,12]
#     for m in monthsIni:
#
#         list_pSeason = GenOneSeasonDaily(month_ini=m)
#         m_plot = np.zeros((70, len(list_pSeason))) * np.nan
#         num_clusters=70
#         num_sim=1
#         # sort data
#         for i, dpy in enumerate(list_pSeason):
#             _, s = np.where([(timeSubMonths == dpy.month) & (timeSubDays == dpy.day)])
#             b = a[s,:]
#             # b = bmus[s]
#             b = b.flatten()
#             if len(b) > 0:
#                 for j in range(num_clusters):
#                     _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!
#                     # _, bb = np.where([(j == b)])  # j starts at 0 bmus value!
#
#                     m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)
#
#
#         # indNan = np.where(np.isnan(m_plot))[0][:]
#         # if len(indNan) > 0:
#         #     m_plot[indNan] = np.ones((len(indNan),))
#         #m_plot = m_plot[1:,:]
#         ax = plt.subplot(gs[c])
#         # plot stacked bars
#         bottom_val = np.zeros(m_plot[1, :].shape)
#         for r in range(num_clusters):
#             row_val = m_plot[r, :]
#             indNan = np.where(np.isnan(row_val))[0][:]
#             if len(indNan) > 0:
#                 row_val[indNan] = 0
#             ax.bar(list_pSeason, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
#
#             # store bottom
#             bottom_val += row_val
#         # customize  axis
#         months = mdates.MonthLocator()
#         monthsFmt = mdates.DateFormatter('%b')
#         ax.set_xlim(list_pSeason[0], list_pSeason[-1])
#         ax.xaxis.set_major_locator(months)
#         ax.xaxis.set_major_formatter(monthsFmt)
#         ax.set_ylim(0, 100)
#         ax.set_ylabel('')
#         c = c + 1
#

# sim_num=100

# Lets make a plot comparing probabilities in sim vs. historical
probH = np.nan*np.ones((num_clusterstoALR,))
probS = np.nan * np.ones((sim_num,num_clusterstoALR))

for h in np.unique(bmus):
    findH = np.where((bmus == h))[0][:]
    probH[int(h-1)] = len(findH)/len(bmus)
    # probH[int(h)] = len(findH)/len(bmus)

    for s in range(sim_num):
        findS = np.where((evbmus_sim[:,s] == h))[0][:]
        probS[s,int(h-1)] = len(findS)/len(bmus)
        # probS[s,int(h)] = len(findS)/len(bmus)



plt.figure()
# plt.plot(probH,np.mean(probS,axis=0),'.')
# plt.plot([0,0.03],[0,0.03],'.--')
ax = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
for i in range(num_clusterstoALR):
    temp = probS[:,i]
    temp2 = probH[i]
    box1 = ax.boxplot(temp,positions=[temp2],widths=.0006,notch=True,patch_artist=True,showfliers=False)
    plt.setp(box1['boxes'],color=dwtcolors[i])
    plt.setp(box1['means'],color=dwtcolors[i])
    plt.setp(box1['fliers'],color=dwtcolors[i])
    plt.setp(box1['whiskers'],color=dwtcolors[i])
    plt.setp(box1['caps'],color=dwtcolors[i])
    plt.setp(box1['medians'],color=dwtcolors[i],linewidth=0)

    #box1['boxes'].set(facecolor=dwtcolors[i])
    #plt.set(box1['fliers'],markeredgecolor=dwtcolors[i])
ax.plot([0,0.05],[0,0.05],'k.--', zorder=10)
plt.xlim([0,0.05])
plt.ylim([0,0.05])
plt.xticks([0,0.01,0.02,0.03, 0.04,0.05], ['0','0.01','0.02','0.03','0.04','0.05'])
plt.xlabel('Historical Probability')
plt.ylabel('Simulated Probability')
plt.title('Validation of ALR DWT Simulations')






timeAsArray = np.array(bmus_dates)
futureTimeAsArray = np.array(dates_sim)
plt.figure()
ax1 = plt.subplot2grid((2,1),(0,0))
for qq in range(len(np.unique(bmus))):
    getBMUS = np.where((bmus == qq))
    temp = bmus[getBMUS]
    tempTime = timeAsArray[getBMUS]
    ax1.plot(np.array(bmus_dates)[getBMUS[0]],qq*np.ones((len(temp),)),'.',color=dwtcolors[iceOrder[qq]])#[iceDWTs['kma_order'][qq]])

simIce = 0
ax2 = plt.subplot2grid((2,1),(1,0))
for qq in range(len(np.unique(bmus))):
    getBMUS = np.where((evbmus_sim[:,simIce] == qq))
    temp = evbmus_sim[getBMUS,simIce]
    tempTime = futureTimeAsArray[getBMUS]
    ax2.plot(np.array(futureTimeAsArray)[getBMUS[0]],qq*np.ones((len(temp[0]),)),'.',color=dwtcolors[iceOrder[qq]])#iceDWTs['kma_order'][qq]])









#dates_sim
#evbmus_sim
icecolors = cmocean.cm.ice(np.linspace(0, 1, 10))

testHist = evbmus_sim[:,0]-num_clusters+1
zeroConc = np.where(testHist < 2)
testHist[zeroConc] = 1
# testHist = testHist/10

resetBmus = bmus-num_clusters+1
zeroConc = np.where(resetBmus < 2)
resetBmus[zeroConc] = 1

bmus_dates_months = np.array([d.month for d in dates_sim])
bmus_dates_days = np.array([d.day for d in dates_sim])
#
# # generate perpetual year list
# list_pyear = GenOneYearDaily(month_ini=6)
# m_plot = np.zeros((num_clusters, len(list_pyear))) * np.nan
# num_sim=1
# # sort data
# for i, dpy in enumerate(list_pyear):
#    _, s = np.where([(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)])
#    # b = evbmus_sim[s,:]
#    b = testHist[s]
#    b = b.flatten()
#
#    for j in range(num_clusters):
#       _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!
#       # _, bb = np.where([(j == b)])  # j+1 starts at 1 bmus value!
#
#       m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)
#
# # plt.style.use('dark_background')
# fig = plt.figure(figsize=(10,4))
# ax = plt.subplot2grid((1,1),(0,0))
# # plot stacked bars
# bottom_val = np.zeros(m_plot[1, :].shape)
# for r in range(num_clusters):
#    row_val = m_plot[r, :]
#    ax.bar(
#       list_pyear, row_val, bottom=bottom_val,
#       width=1, color=np.array([icecolors[r]]))
#
#    # store bottom
#    bottom_val += row_val
#
# import matplotlib.dates as mdates
#
# # customize  axis
# months = mdates.MonthLocator()
# monthsFmt = mdates.DateFormatter('%b')
#
# ax.set_xlim(list_pyear[0], list_pyear[-1])
# ax.xaxis.set_major_locator(months)
# ax.xaxis.set_major_formatter(monthsFmt)
# ax.set_ylim(0, 1)
# ax.set_ylabel('Probability')



wavesYearly = np.nan*np.ones((43,365))
c = 61
fig = plt.figure()
ax1 = plt.subplot2grid((2,1),(0,0))

for hh in range(43):
    temp2 = resetBmus[c:c+365]
    # badhs = np.where((temp2 == 0))
    # temp2[badhs[0]] = np.nan*np.ones((len(badhs[0],)))
    # #temp2 = sg.medfilt(temp,5)
    wavesYearly[hh,:]=temp2/10
    c = c + 365

p1 = ax1.pcolor(wavesYearly,cmap=cmocean.cm.ice,vmin=0,vmax=1)
ax1.yaxis.set_ticks([1,11,21,31,41])
ax1.yaxis.set_ticklabels(['1980', '1990', '2000','2010','2020'])

ax1.xaxis.set_ticks([1,30,61,92,122,153,
                    183,214,244,275,305,336])
ax1.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax1.set_title('Historical')
# cax = fig.add_axes([0.1,0.1,0.3,0.05])
# fig.colorbar(p1,cax=cax,orientation='horizontal')
cax = ax1.inset_axes([7.5, 4, 35, 4], transform=ax1.transData)
cb = fig.colorbar(p1, cax=cax, orientation='horizontal')
cb.ax.set_title('Ice Concentration',fontsize=8)
# # single year
# wavesYearly = np.nan*np.ones((42,365))
# c = 214
# ax2 = plt.subplot2grid((2,1),(1,0))
# for hh in range(42):
#     temp2 = testHist[c:c+365]
#     wavesYearly[hh,:]=temp2/10
#     c = c + 365
# ax2.pcolor(wavesYearly,cmap=cmocean.cm.ice,vmin=0,vmax=1)

testHist = evbmus_sim[:,0]-num_clusters+1

# multi-year
wavesYearly = np.nan*np.ones((95,365))
c = 214
ax2 = plt.subplot2grid((2,1),(1,0))
for hh in range(95):
    temp2 = evbmus_sim[c:c+365,:]-8
    zeroConc = np.where(temp2 < 2)
    temp2[zeroConc] = 1
    temp3 = np.mean(temp2,axis=1)
    wavesYearly[hh,:]=temp3/10
    c = c + 365
p2 = ax2.pcolor(wavesYearly,cmap=cmocean.cm.ice,vmin=0,vmax=1)
# plt.colorbar(p2,ax=ax2)



iceConcHist = resetBmus/10

iceConcSims = evbmus_sim-num_clusters+1
zeroConc = np.where(iceConcSims < 2)
iceConcSims[zeroConc] = 1

iceConcHistSims = evbmus_simHist-num_clusters+1
zeroConc = np.where(iceConcHistSims < 2)
iceConcHistSims[zeroConc] = 1
# badIndW = np.where(np.isnan(wavesYearly))
# wavesYearly[badIndW] = 0
# ax2.plot(dayTime[0:365],np.nanmean(wavesYearly,axis=0),color='white',linewidth=2,label='Average Fetch')
# ax2.set_ylabel('Hs (m)')

ax2.xaxis.set_ticks([1,30,61,92,122,153,
                    183,214,244,275,305,336])
ax2.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax2.yaxis.set_ticks([1,11,21,31,41,51,61,71,81,91])
ax2.yaxis.set_ticklabels(['1980', '1990', '2000','2010','2020','2030','2040','2050','2060','2070'])
ax2.set_title('100 Simulations')
# ax2.xaxis.set_ticks([datetime(1979,1,1),datetime(1979,2,1),datetime(1979,3,1),datetime(1979,4,1),datetime(1979,5,1),datetime(1979,6,1),
#                     datetime(1979,7,1),datetime(1979,8,1),datetime(1979,9,1),datetime(1979,10,1),datetime(1979,11,1),datetime(1979,12,1)])
# ax2.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
# ax.set_xlabel('Fetch (km)')
# ax2.set_title('Average Fetch Length to Sea Ice')


#
#
# wavesYearly = np.nan*np.ones((42,365))
# c = 0
# ax2 = plt.subplot2grid((2,1),(1,0))
#
# for hh in range(42):
#     temp2 = slpWaves['wh_all'][c:c+365]
#     badhs = np.where((temp2 == 0))
#     temp2[badhs[0]] = np.nan*np.ones((len(badhs[0],)))
#     #temp2 = sg.medfilt(temp,5)
#     wavesYearly[hh,:]=temp2
#     c = c + 365
#     ax2.plot(dayTime[0:365],temp2,alpha=0.5)
#
#
# # badIndW = np.where(np.isnan(wavesYearly))
# # wavesYearly[badIndW] = 0
# ax2.plot(dayTime[0:365],np.nanmean(wavesYearly,axis=0),color='white',linewidth=2,label='Average Fetch')
# ax2.set_ylabel('Hs (m)')
#
# ax2.xaxis.set_ticks([datetime(1979,1,1),datetime(1979,2,1),datetime(1979,3,1),datetime(1979,4,1),datetime(1979,5,1),datetime(1979,6,1),
#                     datetime(1979,7,1),datetime(1979,8,1),datetime(1979,9,1),datetime(1979,10,1),datetime(1979,11,1),datetime(1979,12,1)])
# ax2.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
# # ax.set_xlabel('Fetch (km)')
# # ax2.set_title('Average Fetch Length to Sea Ice')

plt.figure()
# multi-year
c = 61
ax2 = plt.subplot2grid((1,1),(0,0))
for hh in range(44):
    temp2 = areaBelow[c:c+365]
    ax2.plot(np.arange(0,len(temp2)),temp2)
    c = c + 365
# plt.colorbar(p2,ax=ax2)

#
# from scipy.signal import medfilt
# import itertools
# import operator
# iceOnOff = []
# years = np.arange(1979,2022)
# # lets start in January and assume that we are always beginning with an iced ocean
# for hh in range(1):
#     simOfIce = evbmus_simHist[:,hh]
#     yearsStitched = []
#     timeStitched = []
#
#     concentration = []
#     area = []
#
#     for tt in range(len(years)):
#         yearInd = np.where((np.array(dates_simHist) >= datetime(years[tt],1,1)) & (np.array(dates_simHist) < datetime(years[tt]+1,1,1)))
#         timeSubset = np.array(dates_simHist)[yearInd]
#         tempOfSim = simOfIce[yearInd]
#         dayOfSim = np.arange(0,len(tempOfSim))
#         groupTemp = [[e[0] for e in d[1]] for d in
#                             itertools.groupby(enumerate(tempOfSim), key=operator.itemgetter(1))]
#         timeGroupTemp = [timeSubset[d] for d in groupTemp]
#         bmuGroupIce = (np.asarray([tempOfSim[i[0]] for i in groupTemp]))
#         dayGroupIce = [dayOfSim[d] for d in groupTemp]
#         # testHist = tempOfSim - num_clusters + 1
#         # zeroConc = np.where(testHist < 2)
#         # testHist[zeroConc] = 1
#
#         tempControl = 0
#
#         wavesStitched = []
#         for qq in range(len(groupTemp)):
#             tempBmu = bmuGroupIce[qq]
#             finder = np.where(bmus==tempBmu)
#             tempArea = areaBelow[finder]
#             tempRand = np.random.randint(len(tempArea), size=len(groupTemp[qq])) #np.random.uniform(size=len(groupTemp[qq]))
#             newArea = tempArea[tempRand]
#
#
#             if dayGroupIce[qq][0] > 50 and dayGroupIce[qq][-1] < 250:
#                 # lets say this 200 stretch increases in area
#                 sortedRand = np.sort(newArea)            # so we have a bunch of probabilities between 0 and 1
#             elif dayGroupIce[qq][0] < 250 and dayGroupIce[qq][-1] > 250:
#                 midFinder = np.where(np.array(groupTemp[qq]) == 250)
#                 beginFinder = 0
#                 endFinder = len(groupTemp[qq])
#                 overMid = midFinder[0]*2
#                 sortedRand = np.sort(newArea)[::-1]          # so we have a bunch of probabilities between 0 and 1
#                 endOfSort = sortedRand[overMid[0]:]
#                 beginOfSort = np.sort(sortedRand[0:overMid[0]:2])
#                 midOfSort = sortedRand[1:overMid[0]:2]
#                 sortedRand = np.concatenate((beginOfSort,midOfSort,endOfSort))
#             #    sortedRand = np.sort(newArea)            # so we have a bunch of probabilities between 0 and 1
#             else:# dayGroupIce[qq][0] < 51 | dayGroupIce[qq][0]>249:
#                 # and this stretch of time decreases in area
#                 sortedRand = np.sort(newArea)[::-1]          # so we have a bunch of probabilities between 0 and 1
#                 print('we should be decreasing')
#
#             ## set a condition for unrealistic jumps?
#
#             wavesStitched.append(sortedRand)
#         yearsStitched.append(wavesStitched)
#     iceOnOff.append(medfilt(np.concatenate([x for xs in yearsStitched for x in xs]).ravel(),5))
#



from scipy.signal import medfilt
import itertools
import operator
iceOnOff = []
areaBelow2 = areaBelow[6:-14]
# lets start in January and assume that we are always beginning with an iced ocean
for hh in range(100):
    simOfIce = evbmus_simHist[:,hh]
    yearsStitched = []
    timeStitched = []
    concentration = []
    area = []

    timeSubset = np.array(dates_simHist)
    dayOfSim = dayOfYear[6:-14]
    groupTemp = [[e[0] for e in d[1]] for d in
                 itertools.groupby(enumerate(simOfIce), key=operator.itemgetter(1))]
    timeGroupTemp = [timeSubset[d] for d in groupTemp]
    bmuGroupIce = (np.asarray([simOfIce[i[0]] for i in groupTemp]))
    dayGroupIce = [dayOfSim[d] for d in groupTemp]

    areaStitched = []
    for qq in range(len(groupTemp)):
        tempBmu = bmuGroupIce[qq]
        finder = np.where(bmus==tempBmu)
        tempArea = areaBelow2[finder]
        tempRand = np.random.randint(len(tempArea), size=len(groupTemp[qq])) #np.random.uniform(size=len(groupTemp[qq]))
        newArea = tempArea[tempRand]


        # So from April to late July we assume the wave area is growing
        if dayGroupIce[qq][0] > 100 and dayGroupIce[qq][0] < 241:
            # print('we''re in ice melt')
            # lets say this 200 stretch increases in area
            sortedRand = np.sort(newArea)            # so we have a bunch of probabilities between 0 and 1
        # If we are in August to mid-September then we split evenly into humps
        elif dayGroupIce[qq][0] > 241 and dayGroupIce[qq][0] < 300:
            if len(np.array(dayGroupIce[qq]))>2:

                print('in a parabola')
                midPoint = int(np.round(len(np.array(dayGroupIce[qq])) / 2)+dayGroupIce[qq][0])
                midFinder = np.where(np.array(dayGroupIce[qq]) == midPoint)
                beginFinder = 0
                endFinder = len(dayGroupIce[qq])
                overMid = midFinder[0]*2
                sortedRand = np.sort(newArea)[::-1]          # so we have a bunch of probabilities between 0 and 1
                endOfSort = sortedRand[overMid[0]:]
                beginOfSort = np.sort(sortedRand[0:overMid[0]:2])
                midOfSort = sortedRand[1:overMid[0]:2]
                sortedRand = np.concatenate((beginOfSort,midOfSort,endOfSort))
                print('lower ones at end: {} vs. {}'.format(len(sortedRand),np.array(len(dayGroupIce[qq]))))
            else:
                sortedRand=newArea
        else:
            sortedRand = np.sort(newArea)[::-1]          # so we have a bunch of probabilities between 0 and 1
                #print('we should be decreasing')

                ## set a condition for unrealistic jumps?
        #
        # # So from April to late July we assume the wave area is growing
        # if dayGroupIce[qq][0] > 100 and dayGroupIce[qq][-1] < 248:
        #     # print('we''re in ice melt')
        #     # lets say this 200 stretch increases in area
        #     sortedRand = np.sort(newArea)            # so we have a bunch of probabilities between 0 and 1
        # # If we are in August to mid-September then we split evenly into humps
        # elif dayGroupIce[qq][0] > 225 and dayGroupIce[qq][-1] < 300:
        #     if len(np.array(dayGroupIce[qq]))>2:
        #
        #         print('in a parabola')
        #         midPoint = int(np.round(len(np.array(dayGroupIce[qq])) / 2)+dayGroupIce[qq][0])
        #         midFinder = np.where(np.array(dayGroupIce[qq]) == midPoint)
        #         beginFinder = 0
        #         endFinder = len(dayGroupIce[qq])
        #         overMid = midFinder[0]*2
        #         if overMid[0] < endFinder/2:
        #             sortedRand = np.sort(newArea)[::-1]          # so we have a bunch of probabilities between 0 and 1
        #             endOfSort = sortedRand[overMid[0]:]
        #             beginOfSort = np.sort(sortedRand[0:overMid[0]:2])
        #             midOfSort = sortedRand[1:overMid[0]:2]
        #             sortedRand = np.concatenate((beginOfSort,midOfSort,endOfSort))
        #             print('lower ones at end: {} vs. {}'.format(len(sortedRand),np.array(len(dayGroupIce[qq]))))
        #         else:
        #             # print('should put lower ones at beginning')
        #             sortedRand = np.sort(newArea)[::-1]          # so we have a bunch of probabilities between 0 and 1
        #             endOfSort = sortedRand[overMid[0]:][::-1]
        #             beginOfSort = np.sort(sortedRand[0:overMid[0]:2])
        #             midOfSort = sortedRand[1:overMid[0]:2]
        #             sortedRand = np.concatenate((endOfSort,beginOfSort,midOfSort))
        #             print('lower ones at beginning: {} vs. {}'.format(len(sortedRand),np.array(len(dayGroupIce[qq]))))
        #
        #     # If we are in September into mid October
        #     elif dayGroupIce[qq][0] > 280 and dayGroupIce[qq][-1] < 320:
        #         if len(np.array(dayGroupIce[qq])) > 2:
        #
        #             print('in a parabola')
        #             midFinder = np.where(np.array(dayGroupIce[qq]) == midPoint)
        #             midPoint = int(np.round(len(np.array(dayGroupIce[qq])) / 2) + dayGroupIce[qq][0])
        #             midFinder = np.where(np.array(dayGroupIce[qq]) == midPoint)
        #             beginFinder = 0
        #             endFinder = len(dayGroupIce[qq])
        #             overMid = midFinder[0] * 2
        #             if overMid[0] < endFinder / 2:
        #                 sortedRand = np.sort(newArea)[::-1]  # so we have a bunch of probabilities between 0 and 1
        #                 endOfSort = sortedRand[overMid[0]:]
        #                 beginOfSort = np.sort(sortedRand[0:overMid[0]:2])
        #                 midOfSort = sortedRand[1:overMid[0]:2]
        #                 sortedRand = np.concatenate((beginOfSort, midOfSort, endOfSort))
        #             else:
        #                 print('should put lower ones at beginning')
        #                 sortedRand = np.sort(newArea)[::-1]  # so we have a bunch of probabilities between 0 and 1
        #                 endOfSort = sortedRand[overMid[0]:][::-1]
        #                 beginOfSort = np.sort(sortedRand[0:overMid[0]:2])
        #                 midOfSort = sortedRand[1:overMid[0]:2]
        #                 sortedRand = np.concatenate((endOfSort, beginOfSort, midOfSort))
        #
        #     else:
        #         sortedRand = newArea
        # else:# dayGroupIce[qq][0] < 51 | dayGroupIce[qq][0]>249:
        #             # and this stretch of time decreases in area
        #     # print('we''re in ice formation')
        #
        #     sortedRand = np.sort(newArea)[::-1]          # so we have a bunch of probabilities between 0 and 1
        #         #print('we should be decreasing')
        #
        #         ## set a condition for unrealistic jumps?


        areaStitched.append(sortedRand)

    # iceOnOff.append(medfilt(np.concatenate([x for xs in yearsStitched for x in xs]).ravel(),5))
    areaTS = np.concatenate([x for x in areaStitched]).ravel()

    # first step is just a median filter to remove singular spikes
    areaTSmed = medfilt(areaTS, 3)
    areaTSmed5 = medfilt(areaTS, 5)

    #
    # diffArea = np.diff(areaTS)
    # bigJump = np.where(diffArea > 200)
    # counter = 0
    # while len(bigJump)>0:
    #     counter = counter + 1
    #     print('removing big jumps: {} iteration'.format(counter))
    #     areaTS[bigJump[0]+1] = areaTS[bigJump[0]] + 149
    #     diffArea = np.diff(areaTS)
    #     bigJump = np.where(diffArea > 200)


    # diffArea = np.diff(areaTS)
    # bigJump = np.where(diffArea < 200)
    # areaTS[bigJump[0]+1] = areaTS[bigJump[0]] - 149
    # diffArea = np.diff(areaTS)
    # bigJump = np.where(diffArea < 200)
    # areaTS[bigJump[0]+1] = areaTS[bigJump[0]] - 149


    # for ff in range(len(bigJump)):
    #     areaTS[bigJump[ff]:bigJump[ff]+2] = medfilt(areaTS[bigJump[ff]])
    # iceOnOff.append(areaTS)
    iceOnOff.append(np.concatenate((areaTS[0:5], moving_average(areaTS, 11), areaTS[-5:])))

    # iceOnOff.append(np.concatenate([x for x in areaStitched]).ravel())




#
# plt.figure()
# plt.plot(areaTS)
# # plt.plot(areaTSmed)
# plt.plot(areaTSmed5)
# # plt.plot(np.concatenate((areaTSmed5[0:2],moving_average(areaTSmed5,5),areaTSmed5[-2:])))
# plt.plot(np.concatenate((areaTSmed[0:5],moving_average(areaTSmed,11),areaTSmed[-5:])))

areaSmoothed = np.concatenate((areaTS[0:5],moving_average(areaTS,11),areaTS[-5:]))

years = np.arange(1979,2023)
stackedAreaSim = np.zeros((365,len(years)))
stackedAreaHist = np.zeros((365,len(years)))

for tt in range(len(years)):
    yearInd = np.where((np.array(dates_simHist) >= datetime(years[tt],1,1)) & (np.array(dates_simHist) < datetime(years[tt]+1,1,1)))
    stackedAreaSim[:,tt] = areaSmoothed[yearInd[0][0:365]]*.25

    stackedAreaHist[:, tt] = areaBelow2[yearInd[0][0:365]]*.25

plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0))
ax1.plot(dayTime[61:(61+365)],np.mean(stackedAreaHist,axis=1),label='Historical')
# ax1.fill_between(dayTime[61:(61+365)], np.mean(stackedAreaHist,axis=1) - np.std(stackedAreaHist,axis=1), np.mean(stackedAreaHist,axis=1) + np.std(stackedAreaHist,axis=1), color='b', alpha=0.2)
ax1.fill_between(dayTime[61:(61+365)], np.percentile(stackedAreaHist,14,axis=1), np.percentile(stackedAreaHist,86,axis=1), color='b', alpha=0.2)

ax1.plot(dayTime[61:(61+365)],np.mean(stackedAreaSim,axis=1),color='r',label='Single Simulation')
# ax1.fill_between(dayTime[61:(61+365)], np.mean(stackedAreaSim,axis=1) - np.std(stackedAreaSim,axis=1), np.mean(stackedAreaSim,axis=1) + np.std(stackedAreaSim,axis=1), color='r', alpha=0.2)
ax1.fill_between(dayTime[61:(61+365)], np.percentile(stackedAreaSim,14,axis=1), np.percentile(stackedAreaSim,86,axis=1), color='r', alpha=0.2)
ax1.set_ylabel('Wave Basin ($10^3 km^2$)')
ax1.xaxis.set_ticks([dayTime[61],dayTime[61+61],dayTime[61+122],
                    dayTime[61+183],dayTime[61+244],dayTime[61+305],dayTime[61+365]])
ax1.xaxis.set_ticklabels(['Jan','Mar','May','Jul','Sep','Nov','Jan'])
ax1.legend()
#
# plt.figure()
# # multi-year
# c = 61
# ax2 = plt.subplot2grid((1,1),(0,0))
# for hh in range(44):
#     # temp2 = iceOnOff[0][c:c+365]
#     temp2 = areaTS[c:c+365]
#
#     ax2.plot(np.arange(0,len(temp2)),temp2)
#     c = c + 365





wavesYearly = np.nan*np.ones((43,365))
c = 61
fig = plt.figure()
ax1 = plt.subplot2grid((2,1),(0,0))

for hh in range(43):
    temp2 = areaBelow2[c:c+365]*25/100
    wavesYearly[hh,:]=temp2
    c = c + 365

p1 = ax1.pcolor(wavesYearly,cmap=cmocean.cm.ice_r,vmin=0,vmax=1000*.25)
ax1.yaxis.set_ticks([1,11,21,31,41])
ax1.yaxis.set_ticklabels(['1980', '1990', '2000','2010','2020'])

ax1.xaxis.set_ticks([1,30,61,92,122,153,
                    183,214,244,275,305,336])
ax1.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax1.set_title('Historical')
cax = ax1.inset_axes([7.5, 4, 35, 4], transform=ax1.transData)
cb = fig.colorbar(p1, cax=cax, orientation='horizontal')
cb.ax.set_title(r'Wave Basin ($10^{3} km^{2}$)',fontsize=8)
# # single year
# wavesYearly = np.nan*np.ones((42,365))
# c = 214
# ax2 = plt.subplot2grid((2,1),(1,0))
# for hh in range(42):
#     temp2 = testHist[c:c+365]
#     wavesYearly[hh,:]=temp2/10
#     c = c + 365
# ax2.pcolor(wavesYearly,cmap=cmocean.cm.ice,vmin=0,vmax=1)


# multi-year
wavesYearly = np.nan*np.ones((44,365))
c = 61
ax2 = plt.subplot2grid((2,1),(1,0))
for hh in range(44):
    temp2 = np.zeros((365,10))
    for qq in range(10):
        temp2[:,qq] = iceOnOff[qq][c:c+365]*25/100
    temp3 = np.mean(temp2,axis=1)
    wavesYearly[hh,:]=temp3
    c = c + 365
p2 = ax2.pcolor(wavesYearly,cmap=cmocean.cm.ice_r,vmin=0,vmax=1200*.25)
# plt.colorbar(p2,ax=ax2)



# badIndW = np.where(np.isnan(wavesYearly))
# wavesYearly[badIndW] = 0
# ax2.plot(dayTime[0:365],np.nanmean(wavesYearly,axis=0),color='white',linewidth=2,label='Average Fetch')
# ax2.set_ylabel('Hs (m)')

ax2.xaxis.set_ticks([1,30,61,92,122,153,
                    183,214,244,275,305,336])
ax2.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
# ax2.yaxis.set_ticks([1,11,21,31,41,51,61,71,81,91])
# ax2.yaxis.set_ticklabels(['1980', '1990', '2000','2010','2020','2030','2040','2050','2060','2070'])
ax2.yaxis.set_ticks([1,11,21,31,41,])
ax2.yaxis.set_ticklabels(['1980', '1990', '2000','2010','2020'])
ax2.set_title('100 Simulations')





dayOfYearFuture = np.array([hh.timetuple().tm_yday for hh in futureArcticTime[212:-520]])  # returns 1 for January 1st


iceOnOffSims = []
areaBelow2 = areaBelow[6:-14]
# lets start in January and assume that we are always beginning with an iced ocean
for hh in range(1000):
    if hh == 217:
        simOfIce = evbmus_sim[:,hh-1]
    else:
        simOfIce = evbmus_sim[:,hh]

    yearsStitched = []
    timeStitched = []
    concentration = []
    area = []

    timeSubset = np.array(dates_sim)
    dayOfSim = dayOfYearFuture
    groupTemp = [[e[0] for e in d[1]] for d in
                 itertools.groupby(enumerate(simOfIce), key=operator.itemgetter(1))]
    timeGroupTemp = [timeSubset[d] for d in groupTemp]
    bmuGroupIce = (np.asarray([simOfIce[i[0]] for i in groupTemp]))
    dayGroupIce = [dayOfSim[d] for d in groupTemp]

    areaStitched = []
    for qq in range(len(groupTemp)):
        tempBmu = bmuGroupIce[qq]
        finder = np.where(bmus==tempBmu)
        tempArea = areaBelow2[finder]
        tempRand = np.random.randint(len(tempArea), size=len(groupTemp[qq])) #np.random.uniform(size=len(groupTemp[qq]))
        newArea = tempArea[tempRand]


        # So from April to late July we assume the wave area is growing
        if dayGroupIce[qq][0] > 100 and dayGroupIce[qq][0] < 241:
            # print('we''re in ice melt')
            # lets say this 200 stretch increases in area
            sortedRand = np.sort(newArea)            # so we have a bunch of probabilities between 0 and 1
        # If we are in August to mid-September then we split evenly into humps
        elif dayGroupIce[qq][0] > 241 and dayGroupIce[qq][0] < 300:
            if len(np.array(dayGroupIce[qq]))>2:

                print('in a parabola')
                midPoint = int(np.round(len(np.array(dayGroupIce[qq])) / 2)+dayGroupIce[qq][0])
                midFinder = np.where(np.array(dayGroupIce[qq]) == midPoint)
                beginFinder = 0
                endFinder = len(dayGroupIce[qq])
                overMid = midFinder[0]*2
                sortedRand = np.sort(newArea)[::-1]          # so we have a bunch of probabilities between 0 and 1
                endOfSort = sortedRand[overMid[0]:]
                beginOfSort = np.sort(sortedRand[0:overMid[0]:2])
                midOfSort = sortedRand[1:overMid[0]:2]
                sortedRand = np.concatenate((beginOfSort,midOfSort,endOfSort))
                print('lower ones at end: {} vs. {}'.format(len(sortedRand),np.array(len(dayGroupIce[qq]))))
            else:
                sortedRand=newArea
        else:
            sortedRand = np.sort(newArea)[::-1]          # so we have a bunch of probabilities between 0 and 1

        areaStitched.append(sortedRand)

    # iceOnOff.append(medfilt(np.concatenate([x for xs in yearsStitched for x in xs]).ravel(),5))
    areaTS = np.concatenate([x for x in areaStitched]).ravel()

    # first step is just a median filter to remove singular spikes
    areaTSmed = medfilt(areaTS, 3)
    areaTSmed5 = medfilt(areaTS, 5)

    iceOnOffSims.append(np.concatenate((areaTS[0:5], moving_average(areaTS, 11), areaTS[-5:])))

# samplesPickle = 'ice18FutureSimulations1000PointLay.pickle'
# samplesPickle = 'ice18FutureSimulations1000Wainwright.pickle'
# samplesPickle = 'ice18FutureSimulations1000PointHope.pickle'
# samplesPickle = 'ice18FutureSimulations1000PointLay.pickle'

# samplesPickle = 'ice18FutureSimulations1000WevoknoCC.pickle'
# samplesPickle = 'ice18FutureSimulations1000KivalinanoCC.pickle'
# samplesPickle = 'ice18FutureSimulations1000WalesnoCC.pickle'
# samplesPickle = 'ice18FutureSimulations1000Deering.pickle'
samplesPickle = 'ice18FutureSimulations1000Utqiagvik.pickle'

outputSamples = {}
outputSamples['evbmus_sim'] = evbmus_sim
outputSamples['evbmus_simHist'] = evbmus_simHist

outputSamples['sim_years'] = sim_num
outputSamples['dates_sim'] = dates_sim
outputSamples['dates_simHist'] = dates_simHist

outputSamples['futureTemp'] = futureArcticTemp
outputSamples['futureArcticTime'] = futureArcticTime
outputSamples['iceOnOff'] = iceOnOff
outputSamples['iceOnOffSims'] = iceOnOffSims

outputSamples['iceConcHist'] = iceConcHist
outputSamples['iceConcSims'] = iceConcSims
outputSamples['iceConcHistSims'] = iceConcHistSims
outputSamples['areaBelow'] = areaBelow
outputSamples['dayTime'] = dayTime
outputSamples['bmus'] = OGbmus

with open(samplesPickle,'wb') as f:
    pickle.dump(outputSamples, f)
#