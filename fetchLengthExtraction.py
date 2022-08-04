
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
        iceSubset.append(iceMasked[160:245,65:175])




allIce = np.ones((len(iceSubset),85*110))

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
xAll = x[160:245]
yAll = y[65:175]
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
rectField = linearField.reshape(85, 110)
ax.set_extent([-180,180,50,90],crs=ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True)
extent=[-9.97,168.35,30.98,34.35]
dx = dy = 25000
x = np.arange(-3850000, +3750000, +dx)
y = np.arange(+5850000, -5350000, -dy)
# xAll = x
# yAll = y
# ax.pcolormesh(x[55:140], y[150:225], rectField, cmap=cmocean.cm.ice)
ax.pcolormesh(x[65:175], y[160:245], rectField, cmap=cmocean.cm.ice)
ax.set_xlim([-3223000,1200000])
ax.set_ylim([-400000,2730000])
gl.xlabels_top = False
gl.ylabels_left = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
ax.set_title('{}/{}/{}'.format(int(timeFile[0:4]),int(timeFile[4:6]),int(timeFile[6:8])))

xSmall = x[65:175]#[55:140]
ySmall = y[160:245]#[150:225]
plt.plot(xSmall[9],ySmall[28],'ro')
plt.plot((xSmall[9],xSmall[105]),(ySmall[28],ySmall[28]),'--',color='orange')
xAngle1 = np.array([xSmall[9],xSmall[107]])
yAngle1 = np.array([ySmall[28],ySmall[44]])
plt.plot(xAngle1,yAngle1,'--',color='orange')

xAngle1 = np.array([xSmall[9],xSmall[104]])
yAngle1 = np.array([ySmall[28],ySmall[15]])
plt.plot(xAngle1,yAngle1,'--',color='orange')

xAngle2 = np.array([xSmall[9],xSmall[105]])
yAngle2 = np.array([ySmall[28],ySmall[58]])
plt.plot(xAngle2,yAngle2,'--',color='orange')

xAngle1 = np.array([xSmall[9],xSmall[100]])
yAngle1 = np.array([ySmall[28],ySmall[71]])
plt.plot(xAngle1,yAngle1,'--',color='orange')

xAngle2 = np.array([xSmall[9],xSmall[94]])
yAngle2 = np.array([ySmall[28],ySmall[82]])
plt.plot(xAngle2,yAngle2,'--',color='orange')

xAngle2 = np.array([xSmall[9],xSmall[97]])
yAngle2 = np.array([ySmall[28],ySmall[76]])
plt.plot(xAngle2,yAngle2,'--',color='red')

xAngle2 = np.array([xSmall[9],xSmall[102]])
yAngle2 = np.array([ySmall[28],ySmall[65]])
plt.plot(xAngle2,yAngle2,'--',color='red')

xAngle2 = np.array([xSmall[9],xSmall[107]])
yAngle2 = np.array([ySmall[28],ySmall[51]])
plt.plot(xAngle2,yAngle2,'--',color='red')

xAngle2 = np.array([xSmall[9],xSmall[106]])
yAngle2 = np.array([ySmall[28],ySmall[37]])
plt.plot(xAngle2,yAngle2,'--',color='red')

xAngle1 = np.array([xSmall[9],xSmall[105]])
yAngle1 = np.array([ySmall[28],ySmall[22]])
plt.plot(xAngle1,yAngle1,'--',color='red')

#We need to define our 6 rays

xRay1 = np.linspace(xSmall[9],xSmall[105], 5000)
yRay1 = np.linspace(ySmall[28],ySmall[44], 5000)

xRay2 = np.linspace(xSmall[9],xSmall[105], 5000)
yRay2 = np.linspace(ySmall[28],ySmall[28], 5000)

xRay3 = np.linspace(xSmall[9],xSmall[105], 5000)
yRay3 = np.linspace(ySmall[28],ySmall[15], 5000)

xRay4 = np.linspace(xSmall[9],xSmall[100], 5000)
yRay4 = np.linspace(ySmall[28],ySmall[71], 5000)

xRay5 = np.linspace(xSmall[9],xSmall[94], 5000)
yRay5 = np.linspace(ySmall[28],ySmall[80], 5000)

xRay6 = np.linspace(xSmall[9],xSmall[105], 5000)
yRay6 = np.linspace(ySmall[28],ySmall[58], 5000)

xRay7 = np.linspace(xSmall[9],xSmall[105], 5000)
yRay7 = np.linspace(ySmall[28],ySmall[22], 5000)

xRay8 = np.linspace(xSmall[9],xSmall[106], 5000)
yRay8 = np.linspace(ySmall[28],ySmall[37], 5000)

xRay9 = np.linspace(xSmall[9],xSmall[107], 5000)
yRay9 = np.linspace(ySmall[28],ySmall[51], 5000)

xRay10 = np.linspace(xSmall[9],xSmall[102], 5000)
yRay10 = np.linspace(ySmall[28],ySmall[65], 5000)

xRay11 = np.linspace(xSmall[9],xSmall[97], 5000)
yRay11 = np.linspace(ySmall[28],ySmall[76], 5000)

grid_x,grid_y = np.meshgrid(x[65:175], y[160:245])
pointsA = np.array((grid_x.flatten(),grid_y.flatten())).T

# distances can be calculated for each ray
def get_distance(x, y):
    dist = np.nan*np.ones((len(x),))
    for hh in range(len(x)):
        dist[hh] = ((x[hh] - x[0])**2 + (y[hh] - y[0])**2)**0.5
    return dist
dist1 = get_distance(xRay1,yRay1)
dist2 = get_distance(xRay2,yRay2)
dist3 = get_distance(xRay3,yRay3)
dist4 = get_distance(xRay4,yRay4)
dist5 = get_distance(xRay5,yRay5)
dist6 = get_distance(xRay6,yRay6)
dist7 = get_distance(xRay7,yRay7)
dist8 = get_distance(xRay8,yRay8)
dist9 = get_distance(xRay9,yRay9)
dist10 = get_distance(xRay10,yRay10)
dist11 = get_distance(xRay11,yRay11)

#
# redoRange = np.arange(13600,13950,1)
fetch = []
rayDict1 = []
rayDict2 = []
rayDict3 = []
rayDict4 = []
rayDict5 = []
rayDict6 = []
rayDict7 = []
rayDict8 = []
rayDict9 = []
rayDict10 = []
rayDict11 = []

avgRayDict = []
c = 0
for hh in range(len(gapFilledIce)):

    if np.remainder(hh,365) == 0:
        print('worked through {} years'.format(hh/365))

    # Lets not even bother if waves say they are off...
    if slpWaves['wh_all'][hh] == 0:
            fetch.append(np.array([np.nan]))
            rayDict1.append(np.array([np.nan]))
            rayDict2.append(np.array([np.nan]))
            rayDict3.append(np.array([np.nan]))
            rayDict4.append(np.array([np.nan]))
            rayDict5.append(np.array([np.nan]))
            rayDict6.append(np.array([np.nan]))
            rayDict7.append(np.array([np.nan]))
            rayDict8.append(np.array([np.nan]))
            rayDict9.append(np.array([np.nan]))
            rayDict10.append(np.array([np.nan]))
            rayDict11.append(np.array([np.nan]))

            avgRayDict.append(np.array([np.nan]))

    else:
        # First step is to make a gridded ice field
        spatialField = gapFilledIce[hh]
        linearField = np.ones((np.shape(xFlat))) * np.nan
        linearField[points] = spatialField
        rectField = linearField.reshape(85, 110)
        badData = np.where(np.isnan(rectField))
        rectField[badData] = 0
        # We now need to interpolate our rays through that gridded data
        valuesA = rectField.flatten()
        ray1 = griddata(points=pointsA,values=valuesA, xi=(xRay1,yRay1), method="linear")
        ray2 = griddata(points=pointsA,values=valuesA, xi=(xRay2,yRay2), method="linear")
        ray3 = griddata(points=pointsA,values=valuesA, xi=(xRay3,yRay3), method="linear")
        ray4 = griddata(points=pointsA,values=valuesA, xi=(xRay4,yRay4), method="linear")
        ray5 = griddata(points=pointsA,values=valuesA, xi=(xRay5,yRay5), method="linear")
        ray6 = griddata(points=pointsA,values=valuesA, xi=(xRay6,yRay6), method="linear")
        ray7 = griddata(points=pointsA,values=valuesA, xi=(xRay7,yRay7), method="linear")
        ray8 = griddata(points=pointsA,values=valuesA, xi=(xRay8,yRay8), method="linear")
        ray9 = griddata(points=pointsA,values=valuesA, xi=(xRay9,yRay9), method="linear")
        ray10 = griddata(points=pointsA,values=valuesA, xi=(xRay10,yRay10), method="linear")
        ray11 = griddata(points=pointsA,values=valuesA, xi=(xRay11,yRay11), method="linear")

        # We want to collapse this down to a single distance vector
        f1 = interp1d(dist1, ray1, kind = 'linear')
        f2 = interp1d(dist2, ray2, kind = 'linear')
        f3 = interp1d(dist3, ray3, kind = 'linear')
        f4 = interp1d(dist4, ray4, kind = 'linear')
        f5 = interp1d(dist5, ray5, kind = 'linear')
        f6 = interp1d(dist6, ray6, kind = 'linear')
        f7 = interp1d(dist7, ray7, kind = 'linear')
        f8 = interp1d(dist8, ray8, kind = 'linear')
        f9 = interp1d(dist9, ray9, kind = 'linear')
        f10 = interp1d(dist10, ray10, kind = 'linear')
        f11 = interp1d(dist11, ray11, kind = 'linear')

        xx = np.linspace(0, 2400000, 4000)
        rInterp1 = f1(xx)
        rInterp2 = f2(xx)
        rInterp3 = f3(xx)
        rInterp4 = f4(xx)
        rInterp5 = f5(xx)
        rInterp6 = f6(xx)
        rInterp7 = f7(xx)
        rInterp8 = f8(xx)
        rInterp9 = f9(xx)
        rInterp10 = f10(xx)
        rInterp11 = f11(xx)

        rayDict1.append(rInterp1)
        rayDict2.append(rInterp2)
        rayDict3.append(rInterp3)
        rayDict4.append(rInterp4)
        rayDict5.append(rInterp5)
        rayDict6.append(rInterp6)
        rayDict7.append(rInterp7)
        rayDict8.append(rInterp8)
        rayDict9.append(rInterp9)
        rayDict10.append(rInterp10)
        rayDict11.append(rInterp11)

        rInterp4[2820:] = rInterp4[2820]*np.ones((len(rInterp4[2820:]),))
        rInterp5[2750:] = rInterp5[2750]*np.ones((len(rInterp5[2750:]),))
        rInterp6[3045:] = rInterp6[3045]*np.ones((len(rInterp6[3045:]),))
        rInterp10[2750:] = rInterp10[2750]*np.ones((len(rInterp10[2750:]),))
        rInterp11[2750:] = rInterp11[2750]*np.ones((len(rInterp11[2750:]),))

        avgRay = np.mean((rInterp1,rInterp2,rInterp3,rInterp4,rInterp5,rInterp6,
                          rInterp7,rInterp8,rInterp9,rInterp10,rInterp11),axis=0)
        avgRayDict.append(avgRay)

        # Now need to calculate the intersection
        y2 = 0.5 * np.ones((np.size(xx)))
        f2 = interp1d(xx, y2, kind='linear')
        y2_interp = f2(xx)
        idx = np.argwhere(np.diff(np.sign(avgRay - y2_interp))).flatten()
        if slpWaves['wh_all'][hh] > 0:
            if not np.any(np.isreal(idx)):
                print('so we have waves but no fetch {} times on {}'.format(c, dayTime[hh]))
                print(xx[idx])
                c = c + 1

                y2 = 0.75 * np.ones((np.size(xx)))
                f2 = interp1d(xx, y2, kind='linear')
                y2_interp = f2(xx)
                idx = np.argwhere(np.diff(np.sign(avgRay - y2_interp))).flatten()
                print('tried at 0.75 and got a fetch of {}'.format(xx[idx]))
        fetch.append(xx[idx])
        # fetch[hh] = xx[idx]
# plt.figure()
# plt.plot(dist1,ray1)
# plt.plot(dist2,ray2)
# plt.plot(dist3,ray3)
# plt.plot(dist4,ray4)
# plt.plot(dist5,ray5)
# plt.plot(dist6,ray6)
#
# plt.figure()
# plt.plot(xx,rInterp1)
# plt.plot(xx,rInterp2)
# plt.plot(xx,rInterp3)
# plt.plot(xx,rInterp4)
# plt.plot(xx,rInterp5)
# plt.plot(xx,rInterp6)
# plt.plot(xx,rInterp7)
# plt.plot(xx,rInterp8)
# plt.plot(xx,rInterp9)
# plt.plot(xx,rInterp10)
# plt.plot(xx,rInterp11)
#
# plt.plot(xx,avgRay,'k--',linewidth=2)

#
# plt.figure()
# plt.plot(dayTime[13600:13950],fetchFiltered)







#
# # plt.subplots_adjust(None,0.3,None,None,None)
# xFetch = xSmall[9:]
# y2 = 0.3*np.ones((np.size(xFetch)))
# from scipy.interpolate import interp1d
# f2 = interp1d(xFetch, y2, kind = 'linear')
# xx = np.linspace(max(xFetch[0], xFetch[0]), min(xFetch[-1], xFetch[-1]), 1000)
# y2_interp = f2(xx)
#
# fetch = []
# for hh in range(len(gapFilledIce)):
#     spatialField = gapFilledIce[hh]  # / 100 - np.nanmean(SLP_C, axis=0) / 100
#     linearField = np.ones((np.shape(xFlat))) * np.nan
#     linearField[points] = spatialField
#     rectField = linearField.reshape(75, 110)
#     y1 = np.nanmean(rectField[25:31,9:],axis=0)
#     f1 = interp1d(xFetch, y1, kind = 'linear')
#     y1_interp = f1(xx)
#     idx = np.argwhere(np.diff(np.sign(y1_interp - y2_interp))).flatten()
#     fetch.append(xx[idx] - np.min(xFetch))

import random

clickFetch = np.nan * np.zeros((len(fetch),))




#
# indexForFetch = np.arange(0,len(fetch))
# for qq in indexForFetch:#range(len(fetch)):
#     if np.all(np.isnan(fetch[qq])):
#         if dayTime[qq].month == 8 or dayTime[qq].month == 9:
#             clickFetch[qq] = 2250000 + random.random() * 100000 - 50000
#         else:
#             clickFetch[qq] = np.nan
#     else:
#         if slpWaves['wh_all'][qq] > 0:
#             plt.figure(figsize=(12,8))
#             plt.plot(xx,rayDict1[qq],color='orange',alpha=0.5)
#             plt.plot(xx,rayDict2[qq],color='orange',alpha=0.5)
#             plt.plot(xx,rayDict3[qq],color='orange',alpha=0.5)
#             plt.plot(xx,rayDict4[qq],color='orange',alpha=0.5)
#             plt.plot(xx,rayDict5[qq],color='orange',alpha=0.5)
#             plt.plot(xx,rayDict6[qq],color='orange',alpha=0.5)
#             plt.plot(xx,rayDict7[qq],color='orange',alpha=0.5)
#             plt.plot(xx,rayDict8[qq],color='orange',alpha=0.5)
#             plt.plot(xx,rayDict9[qq],color='orange',alpha=0.5)
#             plt.plot(xx,rayDict10[qq],color='orange',alpha=0.5)
#             plt.plot(xx,rayDict11[qq],color='orange',alpha=0.5)
#             plt.plot(xx,avgRayDict[qq],'w--',linewidth=2)
#             plt.plot([xx[0],xx[-1]],[0.75,0.75])
#             plt.title('{}/{}/{}'.format(dayTime[qq].year,dayTime[qq].month,dayTime[qq].day))
#             if len(fetch[qq]) >= 1:
#                 for yy in fetch[qq]:
#                     plt.plot([yy,yy],[0,1],'r--')
#
#             picked = plt.ginput(1)
#             plt.close()
#             clickFetch[qq] = picked[0][0]
#         else:
#             clickFetch[qq] = np.nan
# # newFetch = fetch
# # for qq in range(len(fetch)):
# #

fetchFiltered = np.nan * np.zeros((len(fetch),))
for qq in range(len(fetch)):
    # if len(fetch[qq]) == 0:
    #     fetchFiltered[qq] = np.nan
    if np.all(np.isnan(fetch[qq])):
        if dayTime[qq].month == 8 or dayTime[qq].month == 9:
            fetchFiltered[qq] = 2250000 + random.random() * 100000 - 50000
        else:
            fetchFiltered[qq] = np.nan
    else:
        # print('{}'.format(qq))
        if len(fetch[qq]) >= 1:

            if len(fetch[qq]) == 2:
                # if fetch[qq][0] == 0:
                #     fetchFiltered[qq] = fetch[qq][1]
                # else:
                if fetch[qq][1] < 1000000:
                    fetchFiltered[qq] = fetch[qq][1]
                else:
                    fetchFiltered[qq] = fetch[qq][0]
            elif len(fetch[qq]) == 3:
                if fetch[qq][1] < 150000:
                    fetchFiltered[qq] = fetch[qq][2]
                elif fetch[qq][0] < 20000:
                    fetchFiltered[qq] = fetch[qq][1]
                else:
                    fetchFiltered[qq] = fetch[qq][0]
            elif len(fetch[qq]) == 4:
                if fetch[qq][0] < 100000:
                    fetchFiltered[qq] = fetch[qq][1]
                else:
                    fetchFiltered[qq] = fetch[qq][0]
            elif len(fetch[qq]) == 5:
                if fetch[qq][1] < 100000:
                    fetchFiltered[qq] = fetch[qq][2]
                else:
                    fetchFiltered[qq] = fetch[qq][0]
            else:
                # print('we should be here')
                fetchFiltered[qq] = fetch[qq][0]

            if slpWaves['wh_all'][qq] == 0:
                fetchFiltered[qq] = np.nan

            if fetch[qq][0] > 600000:
                fetchFiltered[qq] = fetch[qq][0]
        elif len(fetch[qq]) == 0:
            # print('hey hi yo')
            if dayTime[qq].month == 8 or dayTime[qq].month == 9:
                fetchFiltered[qq] = 2000000 + random.random()*10000-5000
            else:
                fetchFiltered[qq] = np.nan


indexForFetch = []
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
                if fetch[qq][1] < 100000:
                    fetchFiltered2[qq] = fetch[qq][1]
                elif fetch[qq][0] < 250000:
                    fetchFiltered2[qq] = fetch[qq][1]
                else:
                    fetchFiltered2[qq] = fetch[qq][0]
            elif len(fetch[qq]) == 3:
                if fetch[qq][1] < 150000:
                    fetchFiltered2[qq] = fetch[qq][2]
                elif fetch[qq][0] < 20000:
                    fetchFiltered2[qq] = fetch[qq][1]
                else:
                    fetchFiltered2[qq] = fetch[qq][0]
            elif len(fetch[qq]) == 4:
                if fetch[qq][0] < 150000:
                #    fetchFiltered2[qq] = fetch[qq][2]
                #elif fetch[qq][2] < 100000:
                    fetchFiltered2[qq] = fetch[qq][3]
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
            print('so we have waves but no fetch {} times on {}'.format(c,dayTime[qq]))
            #print(fetch[qq])
            indexForFetch.append(qq)
            c = c +1









plt.figure()
plt.plot(dayTime,fetchFiltered2)

temp = 300
plt.style.use('dark_background')
plt.figure()
plt.plot(xx/1000,rayDict1[temp],color='orange',alpha=0.5)
plt.plot(xx/1000,rayDict2[temp],color='orange',alpha=0.5)
plt.plot(xx/1000,rayDict3[temp],color='orange',alpha=0.5)
plt.plot(xx/1000,rayDict4[temp],color='orange',alpha=0.5)
plt.plot(xx/1000,rayDict5[temp],color='orange',alpha=0.5)
plt.plot(xx/1000,rayDict6[temp],color='orange',alpha=0.5)
plt.plot(xx/1000,rayDict7[temp],color='orange',alpha=0.5)
plt.plot(xx/1000,rayDict8[temp],color='orange',alpha=0.5)
plt.plot(xx/1000,rayDict9[temp],color='orange',alpha=0.5)
plt.plot(xx/1000,rayDict10[temp],color='orange',alpha=0.5)
plt.plot(xx/1000,rayDict11[temp],color='orange',alpha=0.5)

plt.plot(xx/1000,avgRayDict[temp],'w--',linewidth=3)
plt.ylabel('Sea Ice Concentration')
plt.xlabel('distance (km)')



asdf
import pickle

dwtPickle = 'fetchLengthAttempt3.pickle'
outputDWTs = {}
outputDWTs['fetch'] = fetch
outputDWTs['fetchFiltered'] = fetchFiltered
outputDWTs['slpWaves'] = slpWaves
outputDWTs['dayTime'] = dayTime
outputDWTs['gapFilledIce'] = gapFilledIce
outputDWTs['xRay1'] = xRay1
outputDWTs['xRay2'] = xRay2
outputDWTs['xRay3'] = xRay3
outputDWTs['xRay4'] = xRay4
outputDWTs['xRay5'] = xRay5
outputDWTs['xRay6'] = xRay6
outputDWTs['xRay7'] = xRay7
outputDWTs['xRay8'] = xRay8
outputDWTs['xRay9'] = xRay9
outputDWTs['xRay10'] = xRay10
outputDWTs['xRay11'] = xRay11

outputDWTs['yRay1'] = yRay1
outputDWTs['yRay2'] = yRay2
outputDWTs['yRay3'] = yRay3
outputDWTs['yRay4'] = yRay4
outputDWTs['yRay5'] = yRay5
outputDWTs['yRay6'] = yRay6
outputDWTs['yRay7'] = yRay7
outputDWTs['yRay8'] = yRay8
outputDWTs['yRay9'] = yRay9
outputDWTs['yRay10'] = yRay10
outputDWTs['yRay11'] = yRay11

outputDWTs['dist1'] = dist1
outputDWTs['dist2'] = dist2
outputDWTs['dist3'] = dist3
outputDWTs['dist4'] = dist4
outputDWTs['dist5'] = dist5
outputDWTs['dist6'] = dist6
outputDWTs['dist7'] = dist7
outputDWTs['dist8'] = dist8
outputDWTs['dist9'] = dist9
outputDWTs['dist10'] = dist10
outputDWTs['dist11'] = dist11

outputDWTs['x'] = x
outputDWTs['y'] = y
outputDWTs['xSmall'] = xSmall
outputDWTs['ySmall'] = ySmall
outputDWTs['myDates'] = myDates
outputDWTs['waves'] = waves
outputDWTs['rayDict1'] = rayDict1
outputDWTs['rayDict2'] = rayDict2
outputDWTs['rayDict3'] = rayDict3
outputDWTs['rayDict4'] = rayDict4
outputDWTs['rayDict5'] = rayDict5
outputDWTs['rayDict6'] = rayDict6
outputDWTs['rayDict7'] = rayDict7
outputDWTs['rayDict8'] = rayDict8
outputDWTs['rayDict9'] = rayDict9
outputDWTs['rayDict10'] = rayDict10
outputDWTs['rayDict11'] = rayDict11

outputDWTs['avgRayDict'] = avgRayDict


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


