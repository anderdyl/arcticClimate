
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
grabYears = np.arange(2017,endYear)
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
        # iceSubset.append(iceMasked[165:250,70:175])
        # iceSubset.append(iceMasked[125:350,70:175])
        # iceSubset.append(np.flipud(iceMasked)
        iceSubset.append(iceMasked)




# allIce = np.ones((len(iceSubset),225*105))
allIce = np.ones((len(iceSubset),304*448))

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
# actually reversed...
# x = np.arange(-3850000, +3750000, +dx)
# y = np.arange(+5850000, -5350000, -dy)
y = np.arange(-3850000, +3750000, +dx)
x = np.arange(+5850000, -5350000, -dy)
# xAll = x[125:235]
# yAll = y[25:155]
xAll = x#[125:350]
yAll = y#[70:175]
xMesh,yMesh = np.meshgrid(xAll,yAll)
xFlat = xMesh.flatten()
yFlat = yMesh.flatten()

xPoints = np.delete(xFlat,badInds,axis=0)
yPoints = np.delete(yFlat,badInds,axis=0)
pointsAll = np.arange(0,len(xFlat))
points = np.delete(pointsAll,badInds,axis=0)


# TODO: LETS PUT AN ICE PATTERN ON EVERY DAY AND THEN ADD WAVES ON TO IT

iceTime = [dt.datetime(np.array(year)[i],np.array(month)[i],np.array(day)[i]) for i in range(len(year))]

st = dt.datetime(2017, 1, 1)
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


#
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
#
# st = dt.datetime(1979, 1, 1)
# # end = dt.datetime(2021,12,31)
# end = dt.datetime(2022,6,1)
# step = relativedelta(days=1)
# dayTime2 = []
# while st < end:
#     dayTime2.append(st)#.strftime('%Y-%m-%d'))
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
# myDates = [dt.datetime.strftime(dayTime2[hh], "%Y-%m-%d") for hh in range(len(dayTime2))]
#
# slpWaves = waves[waves.index.isin(myDates)]
#
# slpWaves = slpWaves.fillna(0)
#
#


plt.style.use('default')

for hh in range(len(dayTime)):

    # plt.figure()
    # plt.plot(dayTime,dayOfYearSine)
    # plt.plot(dayTime,dayOfYearCosine)
    import cartopy.crs as ccrs
    fig=plt.figure(figsize=(8, 8))
    import cartopy
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    # ax = plt.subplot2grid((3,3),(0,0),rowspan=2,colspan=3)
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
    spatialField = gapFilledIce[hh]  # / 100 - np.nanmean(SLP_C, axis=0) / 100
    linearField = np.ones((np.shape(xFlat))) * np.nan
    linearField[points] = spatialField
    # rectField = linearField.reshape(225, 105)
    rectField = linearField.reshape(448,304)#304, 448)

    # iceInd = np.where(spatialField < 1)
    ax.add_feature(cartopy.feature.LAND, zorder=0)#, edgecolor='black')

    ax.set_extent([-180,180,50,90],crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True)
    extent=[-9.97,168.35,30.98,34.35]
    dx = dy = 25000
    y = np.arange(-3850000, +3750000, +dx)
    x = np.arange(+5850000, -5350000, -dy)
    ax.pcolormesh(y, x, rectField, cmap=cmocean.cm.ice)

    ax.set_xlim([-3850000,3850000])
    # ax.set_ylim([-400000,2730000])
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # ax.set_title('{}/{}/{}'.format(int(timeFile[0:4]),int(timeFile[4:6]),int(timeFile[6:8])))
    ax.text(1000000,3000000,'{}/{}/{}'.format(dayTime[hh].year,dayTime[hh].month,dayTime[hh].day))
    if counter < 10:
        plt.savefig('/users/dylananderson/Documents/projects/plots/icePlots/frame0000'+str(counter)+'.png')
    elif counter < 100:
        plt.savefig('/users/dylananderson/Documents/projects/plots/icePlots/frame000'+str(counter)+'.png')
    elif counter < 1000:
        plt.savefig('/users/dylananderson/Documents/projects/plots/icePlots/frame00'+str(counter)+'.png')
    elif counter < 10000:
        plt.savefig('/users/dylananderson/Documents/projects/plots/icePlots/frame0'+str(counter)+'.png')
    else:
        plt.savefig('/users/dylananderson/Documents/projects/plots/icePLots/frame'+str(counter)+'.png')
    plt.close()
    counter = counter+1




geomorphdir = '/users/dylananderson/Documents/projects/plots/icePLots/'

files2 = os.listdir(geomorphdir)

files2.sort()

files_path2 = [os.path.join(geomorphdir,x) for x in os.listdir(geomorphdir)]

files_path2.sort()

import cv2

frame = cv2.imread(files_path2[0])
height, width, layers = frame.shape
forcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('iceVisualized2.avi', forcc, 36, (width, height))
for image in files_path2:
    video.write(cv2.imread(image))
cv2.destroyAllWindows()
video.release()