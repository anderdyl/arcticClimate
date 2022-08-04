
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

endYear = 2023
grabYears = np.arange(1979,endYear)
counter = 0
for ff in range(len(grabYears)):

    icedir = '/users/dylananderson/Documents/data/ice/nsidc0051/daily/{}'.format(grabYears[ff])
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
        iceSubset.append(iceMasked[165:250,70:175])




allIce = np.ones((len(iceSubset),85*105))

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
xAll = x[165:250]
yAll = y[70:175]
xMesh,yMesh = np.meshgrid(xAll,yAll)
xFlat = xMesh.flatten()
yFlat = yMesh.flatten()

xPoints = np.delete(xFlat,badInds,axis=0)
yPoints = np.delete(yFlat,badInds,axis=0)
pointsAll = np.arange(0,len(xFlat))
points = np.delete(pointsAll,badInds,axis=0)


# TODO: LETS PUT AN ICE PATTERN ON EVERY DAY AND THEN ADD WAVES ON TO IT

iceTime = [dt.datetime(np.array(year)[i],np.array(month)[i],np.array(day)[i]) for i in range(len(year))]

st = dt.datetime(1979, 1, 1)
# end = dt.datetime(2021,12,31)
end = dt.datetime(2022,6,1)
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

waveData = mat73.loadmat('era5_waves_pthope.mat')


waveArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in waveData['time_all']]

# wlData = mat73.loadmat('updated_env_for_dylan.mat')
#
# wlArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in wlData['time']]


ds2 = xr.open_dataset('era5wavesJanToMay2022.nc')
df2 = ds2.to_dataframe()

wh_all = np.hstack((waveData['wh_all'],np.nan*np.zeros(3624,)))
tp_all = np.hstack((waveData['tp_all'],np.nan*np.zeros(3624,)))

st = dt.datetime(2022, 1, 1)
# end = dt.datetime(2021,12,31)
end = dt.datetime(2022,6,1)
step = relativedelta(hours=1)
hourTime = []
while st < end:
    hourTime.append(st)#.strftime('%Y-%m-%d'))
    st += step

time_all = np.hstack((waveArrayTime,hourTime))
# swh = ds2['swh'][0,0,:,:]


st = dt.datetime(1979, 1, 1)
# end = dt.datetime(2021,12,31)
end = dt.datetime(2022,6,1)
step = relativedelta(days=1)
dayTime2 = []
while st < end:
    dayTime2.append(st)#.strftime('%Y-%m-%d'))
    st += step


# data = np.vstack((np.asarray(waveArrayTime),waveData['wh_all']))
# data = np.vstack((data,waveData['tp_all']))
# df = pd.DataFrame(data.T,columns=['time','wh_all','tp_all'])



# data = np.vstack((waveData['wh_all'],waveData['tp_all']))
# df = pd.DataFrame(data=data.T,index=np.asarray(waveArrayTime),columns=['wh_all','tp_all'])
data = np.vstack((wh_all,tp_all))
df = pd.DataFrame(data=data.T,index=np.asarray(time_all),columns=['wh_all','tp_all'])

means = df.groupby(pd.Grouper(freq='1D')).mean()


waves = means.loc['1979-1-1':'2022-5-31']

# myDates = [dt.datetime.strftime(dt.datetime(year[hh],month[hh],day[hh]), "%Y-%m-%d") for hh in range(len(year))]
# iceTime = [dt.datetime(np.array(year)[i],np.array(month)[i],np.array(day)[i]) for i in range(len(year))]
myDates = [dt.datetime.strftime(dayTime2[hh], "%Y-%m-%d") for hh in range(len(dayTime2))]

slpWaves = waves[waves.index.isin(myDates)]

slpWaves = slpWaves.fillna(0)













# plt.figure()
# plt.plot(dayTime,dayOfYearSine)
# plt.plot(dayTime,dayOfYearCosine)
plt.style.use('default')
import cartopy.crs as ccrs
fig=plt.figure(figsize=(6, 6))

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ax = plt.subplot2grid((3,3),(0,0),rowspan=2,colspan=3)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
spatialField = gapFilledIce[300]  # / 100 - np.nanmean(SLP_C, axis=0) / 100
linearField = np.ones((np.shape(xFlat))) * np.nan
linearField[points] = spatialField
rectField = linearField.reshape(85, 105)

# iceInd = np.where(spatialField < 1)

ax.set_extent([-180,180,50,90],crs=ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True)
extent=[-9.97,168.35,30.98,34.35]
dx = dy = 25000
x = np.arange(-3850000, +3750000, +dx)
y = np.arange(+5850000, -5350000, -dy)
# xAll = x
# yAll = y
# ax.pcolormesh(x[55:140], y[150:225], rectField, cmap=cmocean.cm.ice)
ax.pcolormesh(x[70:175], y[165:250], rectField, cmap=cmocean.cm.ice)
ax.set_xlim([-3223000,1200000])
ax.set_ylim([-400000,2730000])
gl.xlabels_top = False
gl.ylabels_left = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
# ax.set_title('{}/{}/{}'.format(int(timeFile[0:4]),int(timeFile[4:6]),int(timeFile[6:8])))


xSmall = x[70:175]#[55:140]
ySmall = y[160:245]#[150:225]
xEdge = [xSmall[4],xSmall[4],xSmall[19],xSmall[19],xSmall[75],xSmall[76],xSmall[73],xSmall[69],xSmall[62],xSmall[56],xSmall[48]]
yEdge = [ySmall[28],ySmall[14],ySmall[14],ySmall[20],ySmall[20],ySmall[30],ySmall[40],ySmall[50],ySmall[60],ySmall[70],ySmall[80]]
plt.plot(xEdge,yEdge,'.',color='red')

tupVerts = [(xSmall[4],ySmall[28]),(xSmall[4],ySmall[14]),(xSmall[19],ySmall[14]),(xSmall[19],ySmall[20]),(xSmall[74],ySmall[20]),
            (xSmall[75],ySmall[30]),(xSmall[72],ySmall[40]),(xSmall[68],ySmall[50]),(xSmall[61],ySmall[60]),(xSmall[55],ySmall[70]),(xSmall[47],ySmall[80])]

xMeshSmall,yMeshSmall = np.meshgrid(xSmall,ySmall)
xMeshSmall,yMeshSmall = xMeshSmall.flatten(),yMeshSmall.flatten()
points2 = np.vstack((xMeshSmall,yMeshSmall)).T
from matplotlib.path import Path
p = Path(tupVerts)
grid = p.contains_points(points2)
mask = grid.reshape(len(ySmall),len(xSmall))

fig=plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
spatialField = gapFilledIce[300]  # / 100 - np.nanmean(SLP_C, axis=0) / 100
linearField = np.ones((np.shape(xFlat))) * np.nan
linearField[points] = spatialField
rectField = linearField.reshape(85, 105)

ax.set_extent([-180,180,50,90],crs=ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True)
extent=[-9.97,168.35,30.98,34.35]
dx = dy = 25000
x = np.arange(-3850000, +3750000, +dx)
y = np.arange(+5850000, -5350000, -dy)
# xAll = x
# yAll = y
# ax.pcolormesh(x[55:140], y[150:225], rectField, cmap=cmocean.cm.ice)
# ax.pcolormesh(x[70:175], y[165:250], rectField, cmap=cmocean.cm.ice)
xMeshSmallx,yMeshSmallx = np.meshgrid(xSmall,ySmall)
intensity = np.ma.masked_where(~mask,rectField)

ax.pcolormesh(xMeshSmallx, yMeshSmallx, intensity)#, cmap=cmocean.cm.ice)

ax.set_xlim([-3223000,1200000])
ax.set_ylim([-400000,2730000])

asdfg


from shapely.geometry import Polygon
pgon = Polygon(zip(xEdge, yEdge)) # Assuming the OP's x,y coordinates

print(pgon.area)
kiloArea = pgon.area/1000000
print(kiloArea)

fetchTotalConcentration = []
c = 0
for hh in range(len(gapFilledIce)):

    if np.remainder(hh,365) == 0:
        print('worked through {} years'.format(hh/365))

    # First step is to make a gridded ice field
    spatialField = gapFilledIce[hh]
    linearField = np.ones((np.shape(xFlat))) * np.nan
    linearField[points] = spatialField
    rectField = linearField.reshape(85, 105)
    intensity = np.ma.masked_where(~mask, rectField)
    if np.nansum(intensity) == 0:
        fetchTotalConcentration.append(fetchTotalConcentration[hh-1])
    else:
        fetchTotalConcentration.append(1-(np.nansum(intensity)/2658))



plt.figure()
plt.plot(dayTime,fetchTotalConcentration)


from datetime import datetime

plt.style.use('dark_background')
plt.figure()
iceYearly = np.nan*np.ones((42,365))
c = 0
ax = plt.subplot2grid((2,1),(0,0))

for hh in range(42):
    temp = np.asarray(fetchTotalConcentration)[c:c+365]
    #temp2 = sg.medfilt(temp,5)
    iceYearly[hh,:]=temp
    c = c + 365
    ax.plot(dayTime[0:365],temp*kiloArea,alpha=0.5)

badInd = np.where(np.isnan(iceYearly))
iceYearly[badInd] = 0
ax.plot(dayTime[0:365],np.nanmean(iceYearly*kiloArea,axis=0),color='white',linewidth=2,label='Average Fetch')
ax.set_ylabel('Wave Gen Area (km^2)')

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
    badhs = np.where((temp2 == 0))
    temp2[badhs[0]] = np.nan*np.ones((len(badhs[0],)))
    #temp2 = sg.medfilt(temp,5)
    wavesYearly[hh,:]=temp2
    c = c + 365
    ax2.plot(dayTime[0:365],temp2,alpha=0.5)


# badIndW = np.where(np.isnan(wavesYearly))
# wavesYearly[badIndW] = 0
ax2.plot(dayTime[0:365],np.nanmean(wavesYearly,axis=0),color='white',linewidth=2,label='Average Fetch')
ax2.set_ylabel('Hs (m)')

ax2.xaxis.set_ticks([datetime(1979,1,1),datetime(1979,2,1),datetime(1979,3,1),datetime(1979,4,1),datetime(1979,5,1),datetime(1979,6,1),
                    datetime(1979,7,1),datetime(1979,8,1),datetime(1979,9,1),datetime(1979,10,1),datetime(1979,11,1),datetime(1979,12,1)])
ax2.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
# ax.set_xlabel('Fetch (km)')
# ax2.set_title('Average Fetch Length to Sea Ice')




import pickle

dwtPickle = 'fetchAreaAttempt1.pickle'
outputDWTs = {}
outputDWTs['fetchTotalConcentration'] = fetchTotalConcentration
outputDWTs['slpWaves'] = slpWaves
outputDWTs['dayTime'] = dayTime
outputDWTs['gapFilledIce'] = gapFilledIce
outputDWTs['mask'] = mask
outputDWTs['xEdge'] = xEdge
outputDWTs['yEdge'] = yEdge
outputDWTs['kiloArea'] = kiloArea
outputDWTs['wavesYearly'] = wavesYearly
outputDWTs['iceYearly'] = iceYearly
outputDWTs['xMeshSmallx'] = xMeshSmallx
outputDWTs['yMeshSmallx'] = yMeshSmallx
outputDWTs['tupVerts'] = tupVerts
outputDWTs['points2'] = points2
outputDWTs['x'] = x
outputDWTs['y'] = y
outputDWTs['xSmall'] = xSmall
outputDWTs['ySmall'] = ySmall
outputDWTs['myDates'] = myDates
outputDWTs['waves'] = waves


with open(dwtPickle,'wb') as f:
    pickle.dump(outputDWTs, f)



#
# plt.figure()
# spatialField = gapFilledIce[14130]  # / 100 - np.nanmean(SLP_C, axis=0) / 100
# linearField = np.ones((np.shape(xFlat))) * np.nan
# linearField[points] = spatialField
# rectField = linearField.reshape(75, 110)
# plt.plot(xSmall[9:]-np.min(xFetch),np.nanmean(rectField[25:31,9:],axis=0))
# plt.plot(xSmall[9:]-np.min(xFetch),rectField[30,9:])
# plt.plot(xSmall[9:]-np.min(xFetch),rectField[26,9:])
# plt.plot(xSmall[9:]-np.min(xFetch),y2)
# # plt.plot(xFetch[idx],y2[idx],'ro')
# plt.plot(xx[idx] - np.min(xFetch),y1_interp[idx],'ro')
#
#

plt.figure()
plt.plot(dayTime,fetchFiltered)

plt.figure()
plt.plot(slpWaves['wh_all'][0:len(dayTime)],fetchFiltered,'.')
#
# def line_intersection(line1, line2):
#     xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
#     ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
#
#     def det(a, b):
#         return a[0] * b[1] - a[1] * b[0]
#
#     div = det(xdiff, ydiff)
#     if div == 0:
#        raise Exception('lines do not intersect')
#
#     d = (det(*line1), det(*line2))
#     x = det(d, xdiff) / div
#     y = det(d, ydiff) / div
#     return x, y
#
# line_intersection(icey, 0.5*np.ones((np.size(icey))))
#


