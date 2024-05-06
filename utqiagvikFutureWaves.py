import pickle

import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

hs = []
tp = []
dm = []
ss = []
numOfSims = 500
getSims = np.arange(0,500)
for hh in range(len(getSims)):
   numSim = getSims[hh]

   file = ('/volumes/macDrive/arcticSims/utqiagvik/futureSimulation{}.pickle'.format(numSim))

   with open(file, "rb") as input_file:
      simsInput = pickle.load(input_file)
   simulationData = simsInput['simulationData']
   df = simsInput['df']
   hourlyTime = simsInput['time']

   df.loc[df['hs'] == 0, 'dm'] = np.nan

   df.loc[df['hs'] == 0, 'hs'] = np.nan
   df.loc[df['tp'] == 0, 'tp'] = np.nan

   # angles = df['dm'].values
   # where360 = np.where((angles > 360))
   # angles[where360] = angles[where360] - 360
   # whereNeg = np.where((angles < 0))
   # angles[whereNeg] = angles[whereNeg] + 360

   # whereNan = np.where(np.isnan(angles))
   # angles[whereNan] = 0
   hsTemp = df['hs'].values
   dmTemp = df['dm'].values
   # whereNan = np.where(np.isnan(hsTemp))
   # hsTemp[whereNan] = 0
   tpTemp = df['tp'].values
   # whereNan = np.where(np.isnan(tpTemp))
   # tpTemp[whereNan] = 0
   ssTemp = df['ntr'].values
   # whereNan = np.where(np.isnan(ssTemp))
   # ssTemp[whereNan] = 0
   # hs.append(hsTemp)
   # tp.append(tpTemp)
   # dm.append(angles)
   # ss.append(ssTemp)
   hs.append(hsTemp)
   tp.append(tpTemp)
   dm.append(dmTemp)
   ss.append(ssTemp)
allHs = np.stack(hs, axis=0)
allTp = np.stack(tp, axis=0)
allDm = np.stack(dm, axis=0)
allSs = np.stack(ss, axis=0)



with open(r"realWavesUtqiagvik.pickle", "rb") as input_file:
   wavesInput = pickle.load(input_file)

tWave = wavesInput['tWave']#[5:]
tC = tWave#[0:-(2929+(365*24))]
# tC = wavesInput['tC'][5:]

hsCombined = wavesInput['hsCombined']
nanInd = np.where((hsCombined==0))
hsCombined[nanInd] = np.nan * np.ones((len(nanInd)))
#hsCombined = moving_average(hsCombined,3)
#hsCombined = hsCombined[3:]
tpCombined = wavesInput['tpCombined']#[5:]
nanInd = np.where((tpCombined==0))
tpCombined[nanInd] = np.nan * np.ones((len(nanInd)))

dmCombined = wavesInput['dmCombined']#[5:]
nanInd = np.where((dmCombined==0))
dmCombined[nanInd] = np.nan * np.ones((len(nanInd)))

waveNorm = wavesInput['waveNorm']

data = np.array([hsCombined,tpCombined,dmCombined])
ogdf = pd.DataFrame(data=data.T, index=tC, columns=["hs", "tp", "dm"])
year = np.array([tt.year for tt in tC])
ogdf['year'] = year
month = np.array([tt.month for tt in tC])
ogdf['month'] = month

dailyMaxHs = ogdf.resample("d")['hs'].max()
weeklyMaxHs = ogdf.resample('W-Mon')['hs'].mean()
monthlyMaxHs = ogdf.resample("M")['hs'].mean()

wavesYearly = np.nan*np.ones((43,365))
wavesYearly2 = np.nan*np.ones((43,52))
wavesYearly3 = np.nan*np.ones((43,12))

c = 61
fig = plt.figure()
ax1 = plt.subplot2grid((1,2),(0,0))
# tV = dailyMaxHs.index.values
tV = np.asarray([datetime.utcfromtimestamp((qq - np.datetime64(0, 's')) / np.timedelta64(1, 's'))for qq in dailyMaxHs.index.values])
tV2 = np.asarray([datetime.utcfromtimestamp((qq - np.datetime64(0, 's')) / np.timedelta64(1, 's'))for qq in weeklyMaxHs.index.values])
tV3 = np.asarray([datetime.utcfromtimestamp((qq - np.datetime64(0, 's')) / np.timedelta64(1, 's'))for qq in monthlyMaxHs.index.values])

for hh in range(43):
    pastDates = np.where((np.array(tV)>=datetime(1980+hh,1,1)) & (np.array(tV)<datetime(1981+hh,1,1)))
    pastDates2 = np.where((np.array(tV2)>=datetime(1980+hh,1,1)) & (np.array(tV2)<datetime(1981+hh,1,1)))
    pastDates3 = np.where((np.array(tV3)>=datetime(1980+hh,1,1)) & (np.array(tV3)<datetime(1981+hh,1,1)))

    temp2 = dailyMaxHs.values[pastDates[0][0:365]]
    temp3 = weeklyMaxHs.values[pastDates2[0][0:52]]
    temp4 = monthlyMaxHs.values[pastDates3[0][0:12]]

    wavesYearly[hh,:]=temp2
    wavesYearly2[hh,:]=temp3
    wavesYearly3[hh,:]=temp4

    c = c + 365


seasonalMean = ogdf.groupby('month').mean()
seasonalStd = ogdf.groupby('month').std()
yearlyMax = ogdf.groupby('year').max()

g2 = ogdf.groupby(pd.Grouper(freq="M")).mean()

c = 0
threeDayMax = []
while c < len(hsCombined):
    threeDayMax.append(np.nanmax(hsCombined[c:c+72]))
    c = c + 72
threeDayMaxHs = np.asarray(threeDayMax)

c = 0
fourDayMax = []
while c < len(hsCombined):
   fourDayMax.append(np.nanmax(hsCombined[c:c + 96]))
   c = c + 96
fourDayMaxHs = np.asarray(fourDayMax)




dt = datetime(2022, 1, 1)
end = datetime(2023, 1, 1)
step = relativedelta(months=1)
plotTime = []
while dt < end:
    plotTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step

plt.style.use('default')
# plt.style.use('dark_background')

plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
ax1.plot(plotTime,seasonalMean['hs'],label='ERA5 record (42 years)')
ax1.fill_between(plotTime, seasonalMean['hs'] - seasonalStd['hs'], seasonalMean['hs'] + seasonalStd['hs'], color='pink', alpha=0.4)
# ax1.plot(plotTime,df.groupby('month').mean()['hs'],label='Synthetic record (100 years)')
# ax1.fill_between(plotTime, df.groupby('month').mean()['hs'] - df.groupby('month').std()['hs'], df.groupby('month').mean()['hs'] + df.groupby('month').std()['hs'], color='orange', alpha=0.2)
# ax1.fill_between(plotTime, simSeasonalMean['hs'] - simSeasonalStd['hs'], simSeasonalMean['hs'] + simSeasonalStd['hs'], color='orange', alpha=0.2)
ax1.set_xticks([plotTime[0],plotTime[1],plotTime[2],plotTime[3],plotTime[4],plotTime[5],plotTime[6],plotTime[7],plotTime[8],plotTime[9],plotTime[10],plotTime[11]])
ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax1.legend()
plt.title('Monthly Average Wave Height')
ax1.set_ylabel('Hs (m)')






dt = datetime(2022, 1, 1)
end = datetime(2022, 12, 26)
step = relativedelta(days=7)
plotTime2 = []
while dt < end:
    plotTime2.append(dt)#.strftime('%Y-%m-%d'))
    dt += step

colormap = plt.cm.gist_ncar
plt.style.use('default')
fig = plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0))
ax1.set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, 43))))
labels = []
for i in range(43):
    data = wavesYearly2[i,:]

    ax1.plot(plotTime2, data, label=i)

ax1.set_ylabel('Hs (m)')
plt.title('Weekly Average Wave Heights')
# ax1.legend(loc='upper right')
# ax[0].set_xlim([50, 800])
# ax[0].set_ylim([-8, 4])
# ax[0].set_title('Profile variability in northern edge of property')


hsWeekly = np.nan*np.ones((51,53))
tpWeekly = np.nan*np.ones((51,53))
dmWeekly = np.nan*np.ones((51,53))
lwpWeekly = np.nan*np.ones((51,53))

years = np.arange(2024,2075)
for hh in range(len(years)):
    pastDates = np.where((np.array(hourlyTime)>=datetime(2024+hh,1,1)) & (np.array(hourlyTime)<datetime(2025+hh,1,1)))

    tempHs =allHs[:,pastDates[0]]
    tempTp =allTp[:,pastDates[0]]
    tempDm =allDm[:,pastDates[0]]
    tempLwp = 1025 * np.square(tempHs) * tempTp * (9.81 / (64 * np.pi)) * np.cos(tempDm * (np.pi / 180)) * np.sin(
        tempDm * (np.pi / 180))

    hsAnnual = np.nanmean(tempHs,axis=0)[0:8760]
    tpAnnual = np.nanmean(tempTp,axis=0)[0:8760]
    dmAnnual = np.nanmean(tempDm,axis=0)[0:8760]
    lwpAnnual = np.nanmean(np.abs(tempLwp),axis=0)[0:8760]

    data = np.array([hsAnnual,tpAnnual,dmAnnual,lwpAnnual])
    ndf = pd.DataFrame(data=data.T, index=list(np.asarray(hourlyTime)[pastDates[0][0:8760]]), columns=["hs", "tp", "dm","lwp"])
    # dailyMaxHs = ogdf.resample("d")['hs'].max()
    weeklyMeanHs = ndf.resample('W-Mon')['hs'].mean()
    weeklyMeanTp = ndf.resample('W-Mon')['tp'].mean()
    weeklyMeanDm = ndf.resample('W-Mon')['dm'].mean()
    weeklyMeanLwp = ndf.resample('W-Mon')['lwp'].mean()

    # monthlyMaxHs = ogdf.resample("M")['hs'].mean()
    hsWeekly[hh,:] = weeklyMeanHs.values
    tpWeekly[hh,:] = weeklyMeanTp.values
    dmWeekly[hh,:] = weeklyMeanDm.values
    lwpWeekly[hh,:] = weeklyMeanLwp.values

dt = datetime(2022, 1, 1)
end = datetime(2023, 1, 1)
step = relativedelta(days=7)
plotTime3 = []
while dt < end:
    plotTime3.append(dt)#.strftime('%Y-%m-%d'))
    dt += step

colormap = plt.cm.gist_ncar
plt.style.use('default')
fig = plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0))
ax1.set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, 51))))
labels = []
for i in range(50):
    data = hsWeekly[i,:]
    # ax1.plot( data[0:53], label=i)

    ax1.plot(plotTime3[0:53], data[0:53], label=i)

ax1.set_ylabel('Hs (m)')
plt.title('Daily Average Wave Heights')
# ax1.xaxis.set_ticks([1,30,61,92,122,153,
#                     183,214,244,275,305,336])
# ax1.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
from matplotlib import dates
ax1.xaxis.set_major_formatter(dates.DateFormatter('%b'))
# cb = plt.colorbar()






colormap = plt.cm.gist_ncar
plt.style.use('default')
fig = plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0))
ax1.set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, 51))))
labels = []
for i in range(50):
    data = lwpWeekly[i,:]
    # ax1.plot( data[0:53], label=i)

    ax1.plot(plotTime3[0:53], data[0:53], label=i)

ax1.set_ylabel('Longshore Wave Power (W/m)')
# plt.title('Daily Average Wave Heights')
# ax1.xaxis.set_ticks([1,30,61,92,122,153,
#                     183,214,244,275,305,336])
# ax1.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
from matplotlib import dates
ax1.xaxis.set_major_formatter(dates.DateFormatter('%b'))
# cb = plt.colorbar()





