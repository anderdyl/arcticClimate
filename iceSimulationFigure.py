import numpy as np
import matplotlib.pyplot as plt
import cmocean
import datetime as dt
import pickle

# with open(r"ice18FutureSimulations1000Utqiagvik.pickle", "rb") as input_file:
# with open(r"ice18FutureSimulations100Wainwright.pickle", "rb") as input_file:
with open(r"ice18FutureSimulations1000PointHope.pickle", "rb") as input_file:
# with open(r"ice18FutureSimulations100Shishmaref.pickle", "rb") as input_file:
    inputIce = pickle.load(input_file)

evbmus_sim = inputIce['evbmus_sim']
dates_sim = inputIce['dates_sim']
dates_simHist = inputIce['dates_simHist']

iceOnOff = inputIce['iceOnOff']
iceOnOffSims = inputIce['iceOnOffSims']
areaBelow = inputIce['areaBelow']
dayTime = inputIce['dayTime']
iceConcSims = inputIce['iceConcSims']
bmus = inputIce['bmus']
iceConcHist = inputIce['iceConcHist']
iceConcHistSims = inputIce['iceConcHistSims']

num_clusters = 9
resetBmus = bmus-num_clusters+1
zeroConc = np.where(resetBmus < 2)
resetBmus[zeroConc] = 1
# outputSamples['evbmus_simHist'] = evbmus_simHist
# outputSamples['sim_years'] = sim_num
# outputSamples['futureTemp'] = futureArcticTemp
# outputSamples['futureArcticTime'] = futureArcticTime




wavesYearly = np.nan*np.ones((43,365))
c = 61
fig = plt.figure()
ax1 = plt.subplot2grid((1,2),(0,0))

for hh in range(43):
    pastDates = np.where((np.array(dates_simHist)>=dt.datetime(1980+hh,1,1)) & (np.array(dates_simHist)<dt.datetime(1981+hh,1,1)))
    # temp2 = resetBmus[c:c+365]]
    temp2 = resetBmus[pastDates[0][0:365]]
    wavesYearly[hh,:]=temp2/10
    c = c + 365

p1 = ax1.pcolor(wavesYearly,cmap=cmocean.cm.ice,vmin=0,vmax=1)
# ax1.yaxis.set_ticks([1,11,21,31,41])
# ax1.yaxis.set_ticklabels(['1980', '1990', '2000','2010','2020'])
ax1.yaxis.set_ticks([1,11,21,31,41])
ax1.yaxis.set_ticklabels(['1980', '1990', '2000','2010','2020'])
ax1.xaxis.set_ticks([1,30,61,92,122,153,
                    183,214,244,275,305,336])
ax1.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax1.set_title('Historical Ice Concentration')
cax = ax1.inset_axes([20, 8, 55, 4], transform=ax1.transData)
cb = fig.colorbar(p1, cax=cax, orientation='horizontal')
cb.ax.set_title('Ice Concentration',fontsize=8)

areaSmoothed = iceOnOff[0]
areaBelow2 = areaBelow[6:-14]

years = np.arange(1979,2023)
stackedAreaHist = np.zeros((365,len(years)))

for tt in range(len(years)):
    yearInd = np.where((np.array(dates_simHist) >= dt.datetime(years[tt],1,1)) & (np.array(dates_simHist) < dt.datetime(years[tt]+1,1,1)))
    stackedAreaHist[:, tt] = areaBelow2[yearInd[0][0:365]]*.25
ax2 = plt.subplot2grid((1,2),(0,1))

p2 = ax2.pcolor(stackedAreaHist.T,cmap=cmocean.cm.ice_r,vmin=0,vmax=1300*.25)

# ax2.set_ylabel('Wave Basin ($10^3 km^2$)')
ax2.xaxis.set_ticklabels(['Jan','Mar','May','Jul','Sep','Nov','Jan'])
cax2 = ax2.inset_axes([20, 8, 55, 4], transform=ax2.transData)
cb2 = fig.colorbar(p2, cax=cax2, orientation='horizontal')
cb2.ax.set_title(r'Wave Basin ($10^{3} km^{2}$)',fontsize=8)
ax2.set_title('Historical Wave Basin Size ($10^{3} km^{2}$)')










wavesYearly = np.nan*np.ones((43,365))
c = 61
fig = plt.figure()
ax1 = plt.subplot2grid((1,2),(0,0))

for hh in range(43):
    pastDates = np.where((np.array(dates_simHist)>=dt.datetime(1980+hh,1,1)) & (np.array(dates_simHist)<dt.datetime(1981+hh,1,1)))
    # temp2 = resetBmus[c:c+365]]
    temp2 = resetBmus[pastDates[0][0:365]]
    wavesYearly[hh,:]=temp2/10
    c = c + 365

p1 = ax1.pcolor(wavesYearly,cmap=cmocean.cm.ice,vmin=0,vmax=1)
# ax1.yaxis.set_ticks([1,11,21,31,41])
# ax1.yaxis.set_ticklabels(['1980', '1990', '2000','2010','2020'])
ax1.yaxis.set_ticks([1,11,21,31,41,51,61,71,81,91])
ax1.yaxis.set_ticklabels(['1980', '1990', '2000','2010','2020','2030','2040','2050','2060','2070'])
ax1.xaxis.set_ticks([1,30,61,92,122,153,
                    183,214,244,275,305,336])
ax1.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax1.set_title('Historical')




testHist = evbmus_sim[:,0]-num_clusters+1

# multi-year
wavesYearly = np.nan*np.ones((95,365))
c = 214
ax2 = plt.subplot2grid((1,2),(0,1))
for hh in range(95):
    pastDates = np.where((np.array(dates_sim)>=dt.datetime(1980+hh,1,1)) & (np.array(dates_sim)<dt.datetime(1981+hh,1,1)))
    temp2 = evbmus_sim[pastDates[0][0:365],:]-8
    zeroConc = np.where(temp2 < 2)
    temp2[zeroConc] = 1
    temp3 = np.mean(temp2,axis=1)
    wavesYearly[hh,:]=temp3/10
    c = c + 365
p2 = ax2.pcolor(wavesYearly,cmap=cmocean.cm.ice,vmin=0,vmax=1)


ax2.xaxis.set_ticks([1,30,61,92,122,153,
                    183,214,244,275,305,336])
ax2.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax2.yaxis.set_ticks([1,11,21,31,41,51,61,71,81,91])
ax2.yaxis.set_ticklabels(['1980', '1990', '2000','2010','2020','2030','2040','2050','2060','2070'])
ax2.set_title('Average of 100 Simulations')
cax = ax2.inset_axes([20, 8, 55, 4], transform=ax2.transData)
cb = fig.colorbar(p2, cax=cax, orientation='horizontal')
cb.ax.set_title('Ice Concentration',fontsize=8)







areaSmoothed = iceOnOff[0]
areaBelow2 = areaBelow[6:-14]

years = np.arange(1979,2023)
stackedAreaSim = np.zeros((365,len(years)))
stackedAreaHist = np.zeros((365,len(years)))

for tt in range(len(years)):
    yearInd = np.where((np.array(dates_simHist) >= dt.datetime(years[tt],1,1)) & (np.array(dates_simHist) < dt.datetime(years[tt]+1,1,1)))
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





wavesYearly = np.nan*np.ones((43,365))
c = 61
fig = plt.figure()
ax1 = plt.subplot2grid((2,1),(0,0))

for hh in range(43):
    temp2 = areaBelow2[c:c+365]*25/100
    wavesYearly[hh,:]=temp2
    c = c + 365

p1 = ax1.pcolor(wavesYearly,cmap=cmocean.cm.ice_r,vmin=0,vmax=1500*.25)
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
wavesYearly = np.nan*np.ones((95,365))
c = 61
plt.figure()
ax2 = plt.subplot2grid((1,2),(0,1))
for hh in range(95):
    temp2 = np.zeros((365,10))
    for qq in range(10):
        pastDates = np.where((np.array(dates_sim) >= dt.datetime(1980 + hh, 1, 1)) & (
                    np.array(dates_sim) < dt.datetime(1981 + hh, 1, 1)))

        temp2[:,qq] = iceOnOffSims[qq][pastDates[0][0:365]]*25/100
    temp3 = np.mean(temp2,axis=1)
    wavesYearly[hh,:]=temp3
    c = c + 365
p2 = ax2.pcolor(wavesYearly,cmap=cmocean.cm.ice_r,vmin=0,vmax=1300*.25)
ax2.xaxis.set_ticks([1,30,61,92,122,153,
                    183,214,244,275,305,336])
ax2.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax2.yaxis.set_ticks([1,11,21,31,41,51,61,71,81,91])
ax2.yaxis.set_ticklabels(['1980', '1990', '2000','2010','2020','2030','2040','2050','2060','2070'])
ax2.set_title('Average of 1000 Open Water Simulations')
cax2 = ax2.inset_axes([20, 8, 55, 4], transform=ax2.transData)
cb = fig.colorbar(p2, cax=cax2, orientation='horizontal')
cb.ax.set_title(r'Wave Basin ($10^{3} km^{2}$)',fontsize=8)




testHist = evbmus_sim[:,0]-num_clusters+1

# multi-year
wavesYearly = np.nan*np.ones((95,365))
c = 214
ax2b = plt.subplot2grid((1,2),(0,0))
for hh in range(95):
    pastDates = np.where((np.array(dates_sim)>=dt.datetime(1980+hh,1,1)) & (np.array(dates_sim)<dt.datetime(1981+hh,1,1)))
    temp2 = evbmus_sim[pastDates[0][0:365],:]-8
    zeroConc = np.where(temp2 < 2)
    temp2[zeroConc] = 1
    temp3 = np.mean(temp2,axis=1)
    wavesYearly[hh,:]=temp3/10
    c = c + 365
p2b = ax2b.pcolor(wavesYearly,cmap=cmocean.cm.ice,vmin=0,vmax=1)


ax2b.xaxis.set_ticks([1,30,61,92,122,153,
                    183,214,244,275,305,336])
ax2b.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax2b.yaxis.set_ticks([1,11,21,31,41,51,61,71,81,91])
ax2b.yaxis.set_ticklabels(['1980', '1990', '2000','2010','2020','2030','2040','2050','2060','2070'])
ax2b.set_title('Average of 1000 Sea Ice Concentrations')
caxb = ax2b.inset_axes([20, 8, 55, 4], transform=ax2b.transData)
cb = fig.colorbar(p2b, cax=caxb, orientation='horizontal')
cb.ax.set_title('Ice Concentration',fontsize=8)




asdfg



wavesYearly = np.nan*np.ones((43,365))
c = 61
fig = plt.figure()
ax1 = plt.subplot2grid((1,2),(0,0))

for hh in range(43):
    pastDates = np.where((np.array(dates_simHist)>=dt.datetime(1980+hh,1,1)) & (np.array(dates_simHist)<dt.datetime(1981+hh,1,1)))
    # temp2 = resetBmus[c:c+365]]
    temp2 = resetBmus[pastDates[0][0:365]]
    wavesYearly[hh,:]=temp2/10
    c = c + 365

p1 = ax1.pcolor(wavesYearly,cmap=cmocean.cm.ice,vmin=0,vmax=1)
ax1.yaxis.set_ticks([1,11,21,31,41])
ax1.yaxis.set_ticklabels(['1980', '1990', '2000','2010','2020'])
# ax1.yaxis.set_ticks([1,11,21,31,41,51,61,71,81,91])
# ax1.yaxis.set_ticklabels(['1980', '1990', '2000','2010','2020','2030','2040','2050','2060','2070'])
ax1.xaxis.set_ticks([1,30,61,92,122,153,
                    183,214,244,275,305,336])
ax1.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax1.set_title('Historical Sea Ice Concentration')
cax = ax1.inset_axes([20, 8, 55, 4], transform=ax1.transData)
cb = fig.colorbar(p1, cax=cax, orientation='horizontal')
cb.ax.set_title('Ice Concentration',fontsize=8)


wavesYearly = np.nan*np.ones((43,365))
ax1b = plt.subplot2grid((1,2),(0,1))
c = 61
for hh in range(43):
    pastDates = np.where((np.array(dates_simHist)>=dt.datetime(1980+hh,1,1)) & (np.array(dates_simHist)<dt.datetime(1981+hh,1,1)))
    temp2 = areaBelow2[pastDates[0][0:365]]*25/100

    # temp2 = areaBelow2[c:c+365]*25/100
    wavesYearly[hh,:]=temp2
    c = c + 365

p1b = ax1b.pcolor(wavesYearly,cmap=cmocean.cm.ice_r,vmin=0,vmax=1500*.25)
ax1b.yaxis.set_ticks([1,11,21,31,41])
ax1b.yaxis.set_ticklabels(['1980', '1990', '2000','2010','2020'])

ax1b.xaxis.set_ticks([1,30,61,92,122,153,
                    183,214,244,275,305,336])
ax1b.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax1b.set_title('Historical Open Water Area')
caxb = ax1b.inset_axes([20, 8, 55, 4], transform=ax1b.transData)
cb = fig.colorbar(p1b, cax=caxb, orientation='horizontal')
cb.ax.set_title(r'Wave Basin ($10^{3} km^{2}$)',fontsize=8)








dateIndex = np.where((np.array(dayTime)>dt.datetime(2000,1,1)) & (np.array(dayTime)<dt.datetime(2003,1,1)))
bmuSubset = bmus[dateIndex[0]]
matrix = np.nan * np.ones((18,len(dateIndex[0])))
for hh in range(18):
    tempBmu = bmuOrder[hh]
    subsetIndex = np.where(bmuSubset==tempBmu)
    matrix[hh,subsetIndex[0]] = hh*np.ones((len(subsetIndex[0],)))


plt.figure()
ax5 = plt.subplot2grid((1,1),(0,0))
ax5.pcolor(np.array(dayTime)[dateIndex[0]],np.arange(0,18),matrix)
ax5.yaxis.set_ticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
ax5.yaxis.set_ticklabels(['9','8','7','6','5','4','3','2','1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'])







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
yStart = 182
yEnd = 448-263
xStart = 70
xEnd = 304-231
yStartConc = 168
yEndConc = 448-236
xStartConc = 65
xEndConc = 304-195

# # Point Lay
# yStart = 185
# yEnd = 448-257
# xStart = 75
# xEnd = 304-223
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
x = np.arange(-3880000, +3720000, +dx)
# y = np.arange(5850000, -5350000,-dy)
y = np.arange(5890000, -5390000,-dy)

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
ax.pcolormesh(x[xStart:xEnd], y[yStart:yEnd], rectField3, cmap=cmocean.cm.ice,vmin=0,vmax=1)


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

st = dt.datetime(1978, 10, 26)
# end = dt.datetime(2021,12,31)
end = dt.datetime(2023,11,14)
step = relativedelta(days=1)
dayTime = []
while st < end:
    dayTime.append(st)#.strftime('%Y-%m-%d'))
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




bmuOrder = np.array([2,0,8,6,5,1,7,3,4,9,10,11,12,13,14,15,16,17])

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# plotting the EOF patterns
plt.figure()
c1 = 0
c2 = 0
for hh in range(18):
    ax = plt.subplot2grid((2,9),(c1,c2),projection=ccrs.NorthPolarStereo(central_longitude=-45))

    tempIndex = np.where(bmus==bmuOrder[hh])

    # spatialField = np.multiply(EOFs[hh,0:len(xPointsWR)],np.sqrt(variance[hh]))

    spatialField = np.nanmean(gapFilledIceP[tempIndex[0]],axis=0)  # / 100 - np.nanmean(SLP_C, axis=0) / 100

    linearField = np.ones((np.shape(xFlatWR))) * np.nan
    linearField[pointsWR] = spatialField
    # rectField = linearField.reshape(110,130)
    # rectField = linearField.reshape(75,85)
    rectField = np.flipud(linearField.reshape((yEndWR-yStartWR),(xEndWR-xStartWR)))

    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=False)
    extent = [-9.97, 168.35, 30.98, 34.35]
    # ax.pcolormesh(x[25:155], y[125:235], rectField)#, cmap=cmocean.cm.ice)
    # ax.pcolormesh(x[55:140], y[150:225], rectField)#, cmap=cmocean.cm.ice)
    ax.pcolormesh(x[xStartWR:xEndWR], y[yStartWR:yEndWR], rectField, cmap=cmocean.cm.ice)
    # m.drawcoastlines()
    ax.coastlines(resolution='50m')
    ax.set_xlim([-2400000, -500000])
    ax.set_ylim([-100000, 2030000])
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # ax.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
    if hh == 0:
        ax.set_title('C9')
    elif hh == 1:
        ax.set_title('C8')
    elif hh == 2:
        ax.set_title('C7')
    elif hh == 3:
        ax.set_title('C6')
    elif hh == 4:
        ax.set_title('C5')
    elif hh == 5:
        ax.set_title('C4')
    elif hh == 6:
        ax.set_title('C3')
    elif hh == 7:
        ax.set_title('C2')
    elif hh == 8:
        ax.set_title('C1')
    elif hh == 9:
        ax.set_title('0.2')
    elif hh == 10:
        ax.set_title('0.3')
    elif hh == 11:
        ax.set_title('0.4')
    elif hh == 12:
        ax.set_title('0.5')
    elif hh == 13:
        ax.set_title('0.6')
    elif hh == 14:
        ax.set_title('0.7')
    elif hh == 15:
        ax.set_title('0.8')
    elif hh == 16:
        ax.set_title('0.9')
    elif hh == 17:
        ax.set_title('1.0')

    c2 += 1
    if c2 == 9:
        c1 += 1
        c2 = 0







