import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import gridspec
from mpl_toolkits.basemap import Basemap

import cmocean


with open(r"dwts49ClustersArctic2023.pickle", "rb") as input_file:
   slpDWTs = pickle.load(input_file)

timeDWTs = slpDWTs['SLPtime']
bmus = slpDWTs['bmus_corrected']
num_clusters = slpDWTs['num_clusters']
kma_order = slpDWTs['kma_order']
lon = slpDWTs['lon']
lat = slpDWTs['lat']
SLP = slpDWTs['SLP']
GRD = slpDWTs['GRD']
kma = slpDWTs['kma']
Xsea = slpDWTs['Xsea']
centroids = slpDWTs['centroids']
sea_nodes = slpDWTs['sea_nodes']
X_in = slpDWTs['X_in']
group_size = slpDWTs['group_size']



SlpGrd = np.hstack((SLP,GRD))
SlpGrdMean = np.mean(SlpGrd,axis=0)
SlpGrdStd = np.std(SlpGrd,axis=0)
SlpGrdNorm = (SlpGrd[:,:] - SlpGrdMean) / SlpGrdStd
SlpGrdNorm[np.isnan(SlpGrdNorm)] = 0
sorted_cenEOFs = kma.cluster_centers_[kma_order, :]
sorted_centroids = centroids[kma_order, :]
repmatDesviacion = np.tile(SlpGrdStd, (num_clusters,1))
repmatMedia = np.tile(SlpGrdMean, (num_clusters,1))
Km_ = np.multiply(sorted_centroids,repmatDesviacion) + repmatMedia
Km_slp = Km_[:,0:int(len(Xsea))]
Km_grd = Km_[:,int(len(Xsea)):]
#
# outputDWTs['APEV'] = APEV
# outputDWTs['EOFs'] = EOFs
# outputDWTs['EOFsub'] = EOFsub
# outputDWTs['PCA'] = PCA
# outputDWTs['PCs'] = PCs
# outputDWTs['PCsub'] = PCsub
# outputDWTs['SLPtime'] = SLPtime
#
# outputDWTs['Ysea'] = Ysea
# outputDWTs['Y_in'] = Y_in
# outputDWTs['bmus_corrected'] = bmus_corrected
# outputDWTs['d_groups'] = d_groups
# outputDWTs['group_size'] = group_size
# outputDWTs['ipca'] = ipca
# outputDWTs['km'] = km
# outputDWTs['nPercent'] = nPercent
# outputDWTs['nterm'] = nterm
# outputDWTs['sorted_cenEOFs'] = sorted_cenEOFs
# outputDWTs['sorted_centroids'] = sorted_centroids
# outputDWTs['variance'] = variance



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
    spatialField = Km_slp[(num), :] / 100 - np.nanmean(SLP, axis=0) / 100

    rectField = np.ones((np.shape(X_in))) * np.nan
    for tt in range(len(sea_nodes)):
        rectField[sea_nodes[tt]] = spatialField[tt]

    m.fillcontinents(color=dwtcolors[hh], alpha=0.6)

    clevels = np.arange(-35,35,1)

    #ax.pcolormesh(cx, cy, rectField)#, cmap=cmocean.cm.ice)
    CS = m.contourf(cx, cy, rectField, clevels, vmin=-24, vmax=24, cmap=cm.RdBu_r, shading='gouraud')

    # ax.set_xlim([np.min(cx)-10000, np.max(cx)-10000])
    # ax.set_ylim([np.min(cy)-10000, np.max(cy)-10000])
    #tx, ty = m(320, -30)
    ax.text(np.min(cx)+(np.max(cx)-np.min(cx))/3.2*2+410000, np.min(cy)+(np.max(cy)-np.min(cy))/9, '{}'.format(group_size[num]))
    ax.set_xlim([np.min(cx)+2000000, np.max(cx)-1100000])
    ax.set_ylim([np.min(cy)+500000, np.max(cy)-1600000])
    #ax.set_title('{}'.format(group_size[num]))

    c2 += 1
    if c2 == 6:
        c1 += 1
        c2 = 0

    if plotIndx < 8:
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    counter = counter + 1
    if plotIndy < 8:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1




from datetime import datetime, timedelta, date
import numpy as np
from time_operations import xds2datetime as x2d
from time_operations import xds_reindex_daily as xr_daily
from time_operations import xds_common_dates_daily as xcd_daily
import pickle
from dateutil.relativedelta import relativedelta



#### AWT FROM ENSO SSTs
with open(r"AWT1880to2023.pickle", "rb") as input_file:
   historicalAWTs = pickle.load(input_file)
awtClusters = historicalAWTs['clusters']
awtPredictor = historicalAWTs['predictor']

awtBmus = awtClusters.bmus.values
pc1Annual = awtClusters.PCs[:,0]
pc2Annual = awtClusters.PCs[:,1]
pc3Annual = awtClusters.PCs[:,2]

awtVariance = awtPredictor['variance'].values
nPercent = awtVariance / np.sum(awtVariance)

dt = datetime(1880, 6, 1)
end = datetime(2023, 6, 1)
#step = datetime.timedelta(months=1)
step = relativedelta(years=1)
sstTime = []
while dt < end:
    sstTime.append(dt)
    dt += step

years = np.arange(1979,2023)
awtYears = np.arange(1880,2023)


def dateDay2datetimeDate(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [date(int(d[0]), int(d[1]), int(d[2])) for d in d_vec]

bmus_dates = dateDay2datetimeDate(timeDWTs)
bmus_dates_months = np.array([d.month for d in bmus_dates])
bmus_dates_days = np.array([d.day for d in bmus_dates])
# bmus_dates = dateDay2datetime(dayTime)




bmus = bmus[120:]+1
timeDWTs = timeDWTs[120:]
bmus_dates = bmus_dates[120:]

with open(r"arcticMJO.pickle", "rb") as input_file:
   mjoData = pickle.load(input_file)


timeMJO = mjoData['mjoTime']
# monthDWTS = historicalDWTs['month']
# yearsDWTS = historicalDWTs['month']
# daysDWTS = historicalDWTs['month']

bmusMJO = mjoData['bmus']
bmus_datesMJO = timeMJO#dateDay2datetimeDate(timeMJO)
bmus_dates_yearsMJO = np.array([d.year for d in bmus_datesMJO])
bmus_dates_monthsMJO = np.array([d.month for d in bmus_datesMJO])
bmus_dates_daysMJO = np.array([d.day for d in bmus_datesMJO])

awtDailyBmus = np.nan * np.ones(np.shape(bmus))
PC1 = np.nan * np.ones(np.shape(bmus))
PC2 = np.nan * np.ones(np.shape(bmus))
PC3 = np.nan * np.ones(np.shape(bmus))

for hh in years:
   indexDWT = np.where((np.asarray(bmus_dates) >= date(hh,6,1)) & (np.asarray(bmus_dates) <= date(hh+1,5,31)))
   indexAWT = np.where((awtYears == hh))
   awtDailyBmus[indexDWT] = awtBmus[indexAWT]*np.ones(len(indexDWT[0]))
   PC1[indexDWT] = pc1Annual[indexAWT]*np.ones(len(indexDWT[0]))
   PC2[indexDWT] = pc2Annual[indexAWT]*np.ones(len(indexDWT[0]))
   PC3[indexDWT] = pc3Annual[indexAWT]*np.ones(len(indexDWT[0]))


numOf1 = len(np.where(awtBmus[99:]==0)[0])
numOf2 = len(np.where(awtBmus[99:]==1)[0])
numOf3 = len(np.where(awtBmus[99:]==2)[0])
numOf4 = len(np.where(awtBmus[99:]==3)[0])
numOf5 = len(np.where(awtBmus[99:]==4)[0])
numOf6 = len(np.where(awtBmus[99:]==5)[0])

subsetYears1 = awtYears[99:][np.where(awtBmus[99:]==0)]
subsetYears2 = awtYears[99:][np.where(awtBmus[99:]==1)]
subsetYears3 = awtYears[99:][np.where(awtBmus[99:]==2)]
subsetYears4 = awtYears[99:][np.where(awtBmus[99:]==3)]
subsetYears5 = awtYears[99:][np.where(awtBmus[99:]==4)]
subsetYears6 = awtYears[99:][np.where(awtBmus[99:]==5)]


awt1Ind = np.where(awtDailyBmus==0)
awt2Ind = np.where(awtDailyBmus==1)
awt3Ind = np.where(awtDailyBmus==2)
awt4Ind = np.where(awtDailyBmus==3)
awt5Ind = np.where(awtDailyBmus==4)
awt6Ind = np.where(awtDailyBmus==5)

dwt1Ind = bmus[awt1Ind]
dwt2Ind = bmus[awt2Ind]
dwt3Ind = bmus[awt3Ind]
dwt4Ind = bmus[awt4Ind]
dwt5Ind = bmus[awt5Ind]
dwt6Ind = bmus[awt6Ind]

dwt1 = np.nan * np.ones((49,))
dwt2 = np.nan * np.ones((49,))
dwt3 = np.nan * np.ones((49,))
dwt4 = np.nan * np.ones((49,))
dwt5 = np.nan * np.ones((49,))
dwt6 = np.nan * np.ones((49,))
dwtAll = np.nan * np.ones((49,))

for qq in range(49):
    newInd = np.where(dwt1Ind==qq+1)
    dwt1[qq] = len(newInd[0])/(numOf1*365)
    newInd6 = np.where(dwt6Ind==qq+1)
    dwt6[qq] = len(newInd6[0])/(numOf6*365)
    newInd2 = np.where(dwt2Ind==qq+1)
    dwt2[qq] = len(newInd2[0])/(numOf2*365)
    newInd3 = np.where(dwt3Ind==qq+1)
    dwt3[qq] = len(newInd3[0])/(numOf3*365)
    newInd4 = np.where(dwt4Ind==qq+1)
    dwt4[qq] = len(newInd4[0])/(numOf4*365)
    newInd5 = np.where(dwt5Ind==qq+1)
    dwt5[qq] = len(newInd5[0])/(numOf5*365)
    newIndAll = np.where(bmus == qq + 1)
    dwtAll[qq] = len(newIndAll[0]) / (np.sum(numOf1+numOf6+numOf2+numOf3+numOf4+numOf5) * 365)

plt.figure()
ax1 = plt.subplot2grid((1,6),(0,0))
ax1.pcolor(np.flipud(np.reshape(dwt1,(7,7))),vmin=0,vmax=0.045,cmap='Reds')
ax2 = plt.subplot2grid((1,6),(0,1))
ax2.pcolor(np.flipud(np.reshape(dwt2,(7,7))),vmin=0,vmax=0.045,cmap='Reds')
ax3 = plt.subplot2grid((1,6),(0,2))
ax3.pcolor(np.flipud(np.reshape(dwt3,(7,7))),vmin=0,vmax=0.045,cmap='Reds')
ax4 = plt.subplot2grid((1,6),(0,3))
ax4.pcolor(np.flipud(np.reshape(dwt4,(7,7))),vmin=0,vmax=0.045,cmap='Reds')
ax5 = plt.subplot2grid((1,6),(0,4))
ax5.pcolor(np.flipud(np.reshape(dwt5,(7,7))),vmin=0,vmax=0.045,cmap='Reds')
ax6 = plt.subplot2grid((1,6),(0,5))
ax6.pcolor(np.flipud(np.reshape(dwt6,(7,7))),vmin=0,vmax=0.045,cmap='Reds')

plt.set_cmap('bwr')
plt.figure()
ax1 = plt.subplot2grid((1,6),(0,0))
ax1.pcolor(np.flipud(np.reshape(dwt1,(7,7)))-np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.02,vmax=0.02)
ax2 = plt.subplot2grid((1,6),(0,1))
ax2.pcolor(np.flipud(np.reshape(dwt2,(7,7)))-np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.02,vmax=0.02)
ax3 = plt.subplot2grid((1,6),(0,2))
ax3.pcolor(np.flipud(np.reshape(dwt3,(7,7)))-np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.02,vmax=0.02)
ax4 = plt.subplot2grid((1,6),(0,3))
ax4.pcolor(np.flipud(np.reshape(dwt4,(7,7)))-np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.02,vmax=0.02)
ax5 = plt.subplot2grid((1,6),(0,4))
ax5.pcolor(np.flipud(np.reshape(dwt5,(7,7)))-np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.02,vmax=0.02)
ax6 = plt.subplot2grid((1,6),(0,5))
ax6.pcolor(np.flipud(np.reshape(dwt6,(7,7)))-np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.02,vmax=0.02)


plt.set_cmap('bwr')
plt.figure()
ax1 = plt.subplot2grid((1,6),(0,0))
ax1.pcolor((np.flipud(np.reshape(dwt1,(7,7)))-np.flipud(np.reshape(dwtAll,(7,7))))/np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.5,vmax=0.5)
ax2 = plt.subplot2grid((1,6),(0,1))
ax2.pcolor((np.flipud(np.reshape(dwt2,(7,7)))-np.flipud(np.reshape(dwtAll,(7,7))))/np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.5,vmax=0.5)
ax3 = plt.subplot2grid((1,6),(0,2))
ax3.pcolor((np.flipud(np.reshape(dwt3,(7,7)))-np.flipud(np.reshape(dwtAll,(7,7))))/np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.5,vmax=0.5)
ax4 = plt.subplot2grid((1,6),(0,3))
ax4.pcolor((np.flipud(np.reshape(dwt4,(7,7)))-np.flipud(np.reshape(dwtAll,(7,7))))/np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.5,vmax=0.5)
ax5 = plt.subplot2grid((1,6),(0,4))
ax5.pcolor((np.flipud(np.reshape(dwt5,(7,7)))-np.flipud(np.reshape(dwtAll,(7,7))))/np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.5,vmax=0.5)
ax6 = plt.subplot2grid((1,6),(0,5))
ax6.pcolor((np.flipud(np.reshape(dwt6,(7,7)))-np.flipud(np.reshape(dwtAll,(7,7))))/np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.5,vmax=0.5)

numOfMJO = np.nan * np.ones((25))
mjoInd = []
dwtMJO = []
probDWTinMJO = []
for hh in range(16):
    numOfMJO[hh] = len(np.where(bmusMJO==hh+1 | bmusMJO==hh+9 | bmusMJO==hh+17)[0])
    mjoInd.append(np.where(bmusMJO==hh+1 | bmusMJO==hh+9 | bmusMJO==hh+17))
    dwtMJO.append(bmus[mjoInd[hh][0]])

    dwtInd = []
    for qq in range(49):
        newInd = np.where(dwtMJO[hh] == qq + 1)
        dwtInd.append(len(newInd[0]) / numOfMJO[hh])

    probDWTinMJO.append(np.flipud(np.reshape(np.array(dwtInd),(7,7))))







plt.set_cmap('bwr')
plt.figure()
ax1 = plt.subplot2grid((1,8),(0,0))
ax1.pcolor(probDWTinMJO[0]-np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.03,vmax=0.03)
ax2 = plt.subplot2grid((1,8),(0,1))
ax2.pcolor(probDWTinMJO[1]-np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.03,vmax=0.023)
ax3 = plt.subplot2grid((1,8),(0,2))
ax3.pcolor(probDWTinMJO[2]-np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.03,vmax=0.03)
ax4 = plt.subplot2grid((1,8),(0,3))
ax4.pcolor(probDWTinMJO[3]-np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.03,vmax=0.03)
ax5 = plt.subplot2grid((1,8),(0,4))
ax5.pcolor(probDWTinMJO[4]-np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.03,vmax=0.03)
ax6 = plt.subplot2grid((1,8),(0,5))
ax6.pcolor(probDWTinMJO[5]-np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.03,vmax=0.03)
ax7 = plt.subplot2grid((1,8),(0,6))
ax7.pcolor(probDWTinMJO[6]-np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.03,vmax=0.03)
ax8 = plt.subplot2grid((1,8),(0,7))
ax8.pcolor(probDWTinMJO[7]-np.flipud(np.reshape(dwtAll,(7,7))),vmin=-0.03,vmax=0.03)


