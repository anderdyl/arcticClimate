
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime as dt
from dateutil.relativedelta import relativedelta
import cmocean
import mat73
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

import sys
import subprocess

if 'darwin' in sys.platform:
    print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
    subprocess.Popen('caffeinate')

iceSubset = []
year = []
month = []
day = []
yStart = 135
yEnd = 448-135
xStart = 40
xEnd = 304-50

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
            # iceSubset.append(np.flipud(iceMasked))
            iceSubset.append(np.flipud(iceMasked[yStart:yEnd,xStart:xEnd]))


# allIce = np.ones((len(iceSubset),185*225))
# iceSubset.append(iceMasked[165:250, 70:175])

# allIce = np.ones((len(iceSubset),448*304))

allIce = np.ones((len(iceSubset),(yEnd-yStart)*(xEnd-xStart)))
# iceSubset.append(iceMasked[165:250, 70:175])

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
y = np.arange(-5350000, 5850000, dy)
# xAll = x[125:235]
# yAll = y[25:155]
xAll = x[xStart:xEnd]
yAll = y[yStart:yEnd]
# iceSubset.append(iceMasked[165:250,70:175])
# xAll = x[165:350]
# yAll = y[50:275]

xMesh,yMesh = np.meshgrid(xAll,yAll)
xFlat = xMesh.flatten()
yFlat = yMesh.flatten()

xPoints = np.delete(xFlat,badInds,axis=0)
yPoints = np.delete(yFlat,badInds,axis=0)
pointsAll = np.arange(0,len(xFlat))
points = np.delete(pointsAll,badInds,axis=0)




plt.style.use('default')
import cartopy.crs as ccrs
fig=plt.figure(figsize=(6, 6))

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ax = plt.subplot2grid((3,3),(0,0),rowspan=2,colspan=3)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
spatialField = onlyIceLessLand[-10]  # / 100 - np.nanmean(SLP_C, axis=0) / 100
linearField = np.ones((np.shape(xFlat))) * np.nan
linearField[points] = spatialField
# rectField = linearField.reshape(85, 105)
rectField = linearField.reshape((yEnd-yStart),(xEnd-xStart))
# cs = ax.coastlines(resolution='110m', linewidth=0.5)

# iceInd = np.where(spatialField < 1)

ax.set_extent([-180,180,50,90],crs=ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True)
extent=[-9.97,168.35,30.98,34.35]
dx = dy = 25000
x = np.arange(-3850000, +3750000, +dx)
y = np.arange(-5350000, +5850000, dy)
# xAll = x
# yAll = y
# ax.pcolormesh(x[55:140], y[150:225], rectField, cmap=cmocean.cm.ice)
# ax.pcolormesh(x, y, np.flipud(rectField), cmap=cmocean.cm.ice)
ax.pcolormesh(x[xStart:xEnd], y[yStart:yEnd], rectField, cmap=cmocean.cm.ice)
# ax.pcolormesh(x, y, rectField, cmap=plt.cm.Blues)

ax.set_xlim([-2822300,2420000])
ax.set_ylim([-2100000,2673000])
gl.xlabels_top = False
gl.ylabels_left = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
# ax.set_title('{}/{}/{}'.format(int(timeFile[0:4]),int(timeFile[4:6]),int(timeFile[6:8])))





# TODO: LETS PUT AN ICE PATTERN ON EVERY DAY AND THEN ADD WAVES ON TO IT

iceTime = [dt.datetime(np.array(year)[i],np.array(month)[i],np.array(day)[i]) for i in range(len(year))]

st = dt.datetime(1978, 10, 26)
# end = dt.datetime(2021,12,31)
end = dt.datetime(2023,11,1)
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



# define some constants
epoch = dt.datetime(1970, 1, 1)
matlab_to_epoch_days = 719529  # days from 1-1-0000 to 1-1-1970
matlab_to_epoch_seconds = matlab_to_epoch_days * 24 * 60 * 60

def matlab_to_datetime(matlab_date_num_seconds):
    # get number of seconds from epoch
    from_epoch = matlab_date_num_seconds - matlab_to_epoch_seconds

    # convert to python datetime
    return epoch + dt.timedelta(seconds=from_epoch)


#
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
# ds2 = xr.open_dataset('era5wavesJanToMay2022.nc')
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

st = dt.datetime(1979, 1, 1)
# end = dt.datetime(2021,12,31)
end = dt.datetime(2023,6,1)
step = relativedelta(days=1)
dayTime2 = []
while st < end:
    dayTime2.append(st)#.strftime('%Y-%m-%d'))
    st += step


# data = np.vstack((np.asarray(waveArrayTime),waveData['wh_all']))
# data = np.vstack((data,waveData['tp_all']))
# df = pd.DataFrame(data.T,columns=['time','wh_all','tp_all'])

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
# myDates = [dt.datetime.strftime(dayTime2[hh], "%Y-%m-%d") for hh in range(len(dayTime2))]
#
# slpWaves = waves[waves.index.isin(myDates)]
#
# slpWaves = slpWaves.fillna(0)
#
#











#
#
#
#
# # plt.figure()
# # plt.plot(dayTime,dayOfYearSine)
# # plt.plot(dayTime,dayOfYearCosine)
# plt.style.use('default')
# import cartopy.crs as ccrs
# fig=plt.figure(figsize=(6, 6))
#
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#
# # ax = plt.subplot2grid((3,3),(0,0),rowspan=2,colspan=3)
# ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
# spatialField = gapFilledIce[300]  # / 100 - np.nanmean(SLP_C, axis=0) / 100
# linearField = np.ones((np.shape(xFlat))) * np.nan
# linearField[points] = spatialField
# rectField = linearField.reshape(85, 105)
#
# # iceInd = np.where(spatialField < 1)
#
# ax.set_extent([-180,180,50,90],crs=ccrs.PlateCarree())
# gl = ax.gridlines(draw_labels=True)
# extent=[-9.97,168.35,30.98,34.35]
# dx = dy = 25000
# x = np.arange(-3850000, +3750000, +dx)
# y = np.arange(+5850000, -5350000, -dy)
# # xAll = x
# # yAll = y
# # ax.pcolormesh(x[55:140], y[150:225], rectField, cmap=cmocean.cm.ice)
# ax.pcolormesh(x[70:175], y[165:250], rectField, cmap=cmocean.cm.ice)
# ax.set_xlim([-3223000,1200000])
# ax.set_ylim([-400000,2730000])
# gl.xlabels_top = False
# gl.ylabels_left = False
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# # ax.set_title('{}/{}/{}'.format(int(timeFile[0:4]),int(timeFile[4:6]),int(timeFile[6:8])))
#
#
# xSmall = x[70:175]#[55:140]
# ySmall = y[160:245]#[150:225]
# xEdge = [xSmall[4],xSmall[4],xSmall[19],xSmall[19],xSmall[75],xSmall[76],xSmall[73],xSmall[69],xSmall[62],xSmall[56],xSmall[48]]
# yEdge = [ySmall[28],ySmall[14],ySmall[14],ySmall[20],ySmall[20],ySmall[30],ySmall[40],ySmall[50],ySmall[60],ySmall[70],ySmall[80]]
# plt.plot(xEdge,yEdge,'.',color='red')
#
# tupVerts = [(xSmall[4],ySmall[28]),(xSmall[4],ySmall[14]),(xSmall[19],ySmall[14]),(xSmall[19],ySmall[20]),(xSmall[74],ySmall[20]),
#             (xSmall[75],ySmall[30]),(xSmall[72],ySmall[40]),(xSmall[68],ySmall[50]),(xSmall[61],ySmall[60]),(xSmall[55],ySmall[70]),(xSmall[47],ySmall[80])]
#
# xMeshSmall,yMeshSmall = np.meshgrid(xSmall,ySmall)
# xMeshSmall,yMeshSmall = xMeshSmall.flatten(),yMeshSmall.flatten()
# points2 = np.vstack((xMeshSmall,yMeshSmall)).T
# from matplotlib.path import Path
# p = Path(tupVerts)
# grid = p.contains_points(points2)
# mask = grid.reshape(len(ySmall),len(xSmall))
#
# fig=plt.figure(figsize=(6, 6))
# ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
# spatialField = gapFilledIce[300]  # / 100 - np.nanmean(SLP_C, axis=0) / 100
# linearField = np.ones((np.shape(xFlat))) * np.nan
# linearField[points] = spatialField
# rectField = linearField.reshape(85, 105)
#
# ax.set_extent([-180,180,50,90],crs=ccrs.PlateCarree())
# gl = ax.gridlines(draw_labels=True)
# extent=[-9.97,168.35,30.98,34.35]
# dx = dy = 25000
# x = np.arange(-3850000, +3750000, +dx)
# y = np.arange(+5850000, -5350000, -dy)
# # xAll = x
# # yAll = y
# # ax.pcolormesh(x[55:140], y[150:225], rectField, cmap=cmocean.cm.ice)
# # ax.pcolormesh(x[70:175], y[165:250], rectField, cmap=cmocean.cm.ice)
# xMeshSmallx,yMeshSmallx = np.meshgrid(xSmall,ySmall)
# intensity = np.ma.masked_where(~mask,rectField)
#
# ax.pcolormesh(xMeshSmallx, yMeshSmallx, intensity)#, cmap=cmocean.cm.ice)
#
# ax.set_xlim([-3223000,1200000])
# ax.set_ylim([-400000,2730000])
#
# asdfg
#
#
# from shapely.geometry import Polygon
# pgon = Polygon(zip(xEdge, yEdge)) # Assuming the OP's x,y coordinates
#
# print(pgon.area)
# kiloArea = pgon.area/1000000
# print(kiloArea)
#
# fetchTotalConcentration = []
# c = 0
# for hh in range(len(gapFilledIce)):
#
#     if np.remainder(hh,365) == 0:
#         print('worked through {} years'.format(hh/365))
#
#     # First step is to make a gridded ice field
#     spatialField = gapFilledIce[hh]
#     linearField = np.ones((np.shape(xFlat))) * np.nan
#     linearField[points] = spatialField
#     rectField = linearField.reshape(85, 105)
#     intensity = np.ma.masked_where(~mask, rectField)
#     if np.nansum(intensity) == 0:
#         fetchTotalConcentration.append(fetchTotalConcentration[hh-1])
#     else:
#         fetchTotalConcentration.append(1-(np.nansum(intensity)/2658))
#
#
#
# plt.figure()
# plt.plot(dayTime,fetchTotalConcentration)
#
#
# from datetime import datetime
#
# plt.style.use('dark_background')
# plt.figure()
# iceYearly = np.nan*np.ones((42,365))
# c = 0
# ax = plt.subplot2grid((2,1),(0,0))
#
# for hh in range(42):
#     temp = np.asarray(fetchTotalConcentration)[c:c+365]
#     #temp2 = sg.medfilt(temp,5)
#     iceYearly[hh,:]=temp
#     c = c + 365
#     ax.plot(dayTime[0:365],temp*kiloArea,alpha=0.5)
#
# badInd = np.where(np.isnan(iceYearly))
# iceYearly[badInd] = 0
# ax.plot(dayTime[0:365],np.nanmean(iceYearly*kiloArea,axis=0),color='white',linewidth=2,label='Average Fetch')
# ax.set_ylabel('Wave Gen Area (km^2)')
#
# ax.xaxis.set_ticks([datetime(1979,1,1),datetime(1979,2,1),datetime(1979,3,1),datetime(1979,4,1),datetime(1979,5,1),datetime(1979,6,1),
#                     datetime(1979,7,1),datetime(1979,8,1),datetime(1979,9,1),datetime(1979,10,1),datetime(1979,11,1),datetime(1979,12,1)])
# ax.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
# # ax.set_xlabel('Fetch (km)')
# # ax.set_title('Average Fetch Length to Sea Ice')
#
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
#
#
#
#
# import pickle
#
# dwtPickle = 'fetchAreaAttempt1.pickle'
# outputDWTs = {}
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
#
#
# with open(dwtPickle,'wb') as f:
#     pickle.dump(outputDWTs, f)
#
#
#
# #
# # plt.figure()
# # spatialField = gapFilledIce[14130]  # / 100 - np.nanmean(SLP_C, axis=0) / 100
# # linearField = np.ones((np.shape(xFlat))) * np.nan
# # linearField[points] = spatialField
# # rectField = linearField.reshape(75, 110)
# # plt.plot(xSmall[9:]-np.min(xFetch),np.nanmean(rectField[25:31,9:],axis=0))
# # plt.plot(xSmall[9:]-np.min(xFetch),rectField[30,9:])
# # plt.plot(xSmall[9:]-np.min(xFetch),rectField[26,9:])
# # plt.plot(xSmall[9:]-np.min(xFetch),y2)
# # # plt.plot(xFetch[idx],y2[idx],'ro')
# # plt.plot(xx[idx] - np.min(xFetch),y1_interp[idx],'ro')
# #
# #
#
# plt.figure()
# plt.plot(dayTime,fetchFiltered)
#
# plt.figure()
# plt.plot(slpWaves['wh_all'][0:len(dayTime)],fetchFiltered,'.')
# #
# # def line_intersection(line1, line2):
# #     xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
# #     ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
# #
# #     def det(a, b):
# #         return a[0] * b[1] - a[1] * b[0]
# #
# #     div = det(xdiff, ydiff)
# #     if div == 0:
# #        raise Exception('lines do not intersect')
# #
# #     d = (det(*line1), det(*line2))
# #     x = det(d, xdiff) / div
# #     y = det(d, ydiff) / div
# #     return x, y
# #
# # line_intersection(icey, 0.5*np.ones((np.size(icey))))
# #
#
#


# iceMean = np.mean(onlyIceLessLand,axis=0)
# iceStd = np.std(onlyIceLessLand,axis=0)
# iceNorm = (onlyIceLessLand[:,:] - iceMean) / iceStd
# iceNorm[np.isnan(iceNorm)] = 0

lowConc = np.where(gapFilledIce < 0.4)
gapFilledIce[lowConc] = gapFilledIce[lowConc]*0

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
    rectField = linearField.reshape((yEnd-yStart),(xEnd-xStart))

    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True)
    extent = [-9.97, 168.35, 30.98, 34.35]
    # ax.pcolormesh(x[25:155], y[125:235], rectField)#, cmap=cmocean.cm.ice)
    ax.pcolormesh(x[xStart:xEnd], y[yStart:yEnd], rectField)#, cmap=cmocean.cm.ice)
    ax.set_xlim([-2822300, 2420000])
    ax.set_ylim([-2100000, 2673000])
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




# KMA Regression Guided
num_clusters = 25
repres = 0.80
alpha = 0.4
min_size = None  # any int will activate group_min_size iteration
min_group_size=100


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


medianTime = []
for hh in range(num_clusters):
    tempInd = np.where(bmus==hh)
    tempTime = dayOfYear[tempInd[0]]
    medianTime.append(np.median(tempTime))




medianTime = np.asarray(medianTime)
kma_orderTime = np.argsort(medianTime)


from matplotlib import gridspec
plt.style.use('default')
# plotting the EOF patterns
fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(int(np.sqrt(num_clusters)), int(np.sqrt(num_clusters)))
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
medianTime = []
for hh in range(num_clusters):
    ax = plt.subplot(gs1[hh])
    num = kma_orderTime[hh]
    tempInd = np.where(bmus==num)
    tempTime = dayOfYear[tempInd[0]]
    ax.hist(tempTime,np.linspace(0,365,20))
    ax.set_xlim([0, 365])
    medianTime.append(np.median(tempTime))
    # print(group_size[num])
    print(np.median(tempTime))
    ax.set_yticklabels('')
    if c1 == 4:
        ax.set_xticks([0,90,180,270])
        ax.set_xticklabels(['Jan','Apr','Jul','Oct'])
    else:
        ax.set_xticklabels('')
    c2 += 1
    if c2 == 5:
        c1 += 1
        c2 = 0


from matplotlib import gridspec
plt.style.use('default')
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
    num = kma_orderTime[hh]
    # num = kma_orderOG[hh]
    tempInd = np.where(bmus==num)

    # spatialField = kmOG[(hh - 1), :]# / 100 - np.nanmean(SLP_C, axis=0) / 100
    #spatialField = kmSorted[(hh), :]# / 100 - np.nanmean(SLP_C, axis=0) / 100
    spatialField = np.nanmean(gapFilledIce[tempInd[0]], axis=0)

    linearField = np.ones((np.shape(xFlat))) * np.nan
    linearField[points] = spatialField
    rectField = linearField.reshape((yEnd-yStart),(xEnd-xStart))

    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    # gl = ax.gridlines(draw_labels=True)
    extent = [-9.97, 168.35, 30.98, 34.35]
    # ax.pcolormesh(x[25:155], y[125:235], rectField, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[40:135], y[140:230], rectField, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[45:135], y[140:230], rectField, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[65:150], y[160:235], rectField, cmap=cmocean.cm.ice)
    ax.pcolormesh(x[xStart:xEnd], y[yStart:yEnd], rectField, cmap=cmocean.cm.ice)
    ax.set_xlim([-2822300, 2420000])
    ax.set_ylim([-2100000, 2673000])
    # ax.set_xlim([-2220000, -275000])
    # ax.set_ylim([50000, 1850000])
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




import pickle

dwtPickle = 'iceData25ClustersAllArcticRgSeasonality100min.pickle'
outputDWTs = {}
outputDWTs['APEV'] = APEV
outputDWTs['EOFs'] = EOFs
outputDWTs['EOFsub'] = EOFsub
# outputDWTs['bmus2'] = bmus2
# outputDWTs['bmus_final'] = bmus_final
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
outputDWTs['kma_orderTime'] = kma_orderTime

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











import sys
sys.path.append('/users/dylananderson/Documents/projects/gencadeClimate/')
import pickle
with open(r"/Users/dylananderson/Documents/projects/duneLifeCycles/ptLayWavesWinds164by70.pickle", "rb") as input_file:
    ptLayWavesWinds = pickle.load(input_file)

ptLayWaves = ptLayWavesWinds['ptLayWaves']
endTime = ptLayWavesWinds['endTime']
startTime = ptLayWavesWinds['startTime']





tWave = ptLayWaves.timeWave
time = dayTime
timeArray = np.asarray(time)


waveHieghts = []
for hh in range(num_clusters):
    dwtInd = np.where((bmus==hh))
    waveBin = []
    for qq in range(len(dwtInd[0])):
        dayInd = np.where((tWave >= dt.datetime(time[dwtInd[0][qq]].year, time[dwtInd[0][qq]].month,time[dwtInd[0][qq]].day,0,0,0)) &
                          (tWave <= dt.datetime(timeArray[dwtInd[0][qq]].year, timeArray[dwtInd[0][qq]].month,
                                               timeArray[dwtInd[0][qq]].day,23,0,0)))
        waveBin.append(ptLayWaves.Hs[dayInd[0]])
    waveHieghts.append(np.concatenate(waveBin,axis=0))








