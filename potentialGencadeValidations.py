import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
#Getting main packages
from scipy.stats import norm
import seaborn as sns; sns.set(style = 'whitegrid')
from scipy.stats import genpareto
import math as mt
import scipy.special as sm
#Getting main packages from R in order to apply the maximum likelihood function
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

POT = importr('POT') #importing POT package


def return_value(sample_real, threshold, alpha, block_size, return_period,
                 fit_method):  # return value plot and return value estimative
   sample = np.sort(sample_real)
   sample_excess = []
   sample_over_thresh = []
   for data in sample:
      if data > threshold + 0.00001:
         sample_excess.append(data - threshold)
         sample_over_thresh.append(data)

   rdata = FloatVector(sample)
   fit = POT.fitgpd(rdata, threshold, est=fit_method)  # fit data
   shape = fit[0][1]
   scale = fit[0][0]

   # Computing the return value for a given return period with the confidence interval estimated by the Delta Method
   m = return_period
   Eu = len(sample_over_thresh) / len(sample)
   x_m = threshold + (scale / shape) * (((m * Eu) ** shape) - 1)

   # Solving the delta method
   d = Eu * (1 - Eu) / len(sample)
   e = fit[3][0]
   f = fit[3][1]
   g = fit[3][2]
   h = fit[3][3]
   a = (scale * (m ** shape)) * (Eu ** (shape - 1))
   b = (shape ** -1) * (((m * Eu) ** shape) - 1)
   c = (-scale * (shape ** -2)) * ((m * Eu) ** shape - 1) + (scale * (shape ** -1)) * ((m * Eu) ** shape) * mt.log(
      m * Eu)
   CI = (norm.ppf(1 - (alpha / 2)) * ((((a ** 2) * d) + (b * ((c * g) + (e * b))) + (c * ((b * f) + (c * h)))) ** 0.5))

   print('The return value for the given return period is {} \u00B1 {}'.format(x_m, CI))

   ny = block_size  # defining how much observations will be a block (usually anual)
   N_year = return_period / block_size  # N_year represents the number of years based on the given return_period

   for i in range(0, len(sample)):
      if sample[i] > threshold + 0.0001:
         i_initial = i
         break

   p = np.arange(i_initial, len(sample)) / (len(sample))  # Getting Plotting Position points
   N = 1 / (ny * (1 - p))  # transforming plotting position points to years

   year_array = np.arange(min(N), N_year + 0.1, 0.1)  # defining a year array

   # Algorithm to compute the return value and the confidence intervals for plotting
   z_N = []
   CI_z_N_high_year = []
   CI_z_N_low_year = []
   for year in year_array:
      z_N.append(threshold + (scale / shape) * (((year * ny * Eu) ** shape) - 1))
      a = (scale * ((year * ny) ** shape)) * (Eu ** (shape - 1))
      b = (shape ** -1) * ((((year * ny) * Eu) ** shape) - 1)
      c = (-scale * (shape ** -2)) * (((year * ny) * Eu) ** shape - 1) + (scale * (shape ** -1)) * (
                 ((year * ny) * Eu) ** shape) * mt.log((year * ny) * Eu)
      CIyear = (norm.ppf(1 - (alpha / 2)) * (
                 (((a ** 2) * d) + (b * ((c * g) + (e * b))) + (c * ((b * f) + (c * h)))) ** 0.5))
      CI_z_N_high_year.append(threshold + (scale / shape) * (((year * ny * Eu) ** shape) - 1) + CIyear)
      CI_z_N_low_year.append(threshold + (scale / shape) * (((year * ny * Eu) ** shape) - 1) - CIyear)

   # Plotting Return Value
   # plt.figure(8)
   # plt.plot(year_array, CI_z_N_high_year, linestyle='--', color='red', alpha=0.8, lw=0.9, label='Confidence Bands')
   # plt.plot(year_array, CI_z_N_low_year, linestyle='--', color='red', alpha=0.8, lw=0.9)
   # plt.plot(year_array, z_N, color='black', label='Theoretical Return Level')
   # plt.scatter(N, sample_over_thresh, label='Empirical Return Level')
   # plt.xscale('log')
   # plt.xlabel('Return Period')
   # plt.ylabel('Return Level')
   # plt.title('Return Level Plot')
   # plt.legend()
   #
   # plt.show()

   output = dict()
   output['year_array'] = year_array
   output['N'] = N
   output['sample_over_thresh'] = sample_over_thresh
   output['CI_z_N_high_year'] = CI_z_N_high_year
   output['CI_z_N_low_year'] = CI_z_N_low_year
   output['z_N'] = z_N
   output['CI'] = CI
   return output


def moving_average(a, n=3):
   ret = np.cumsum(a, dtype=float)
   ret[n:] = ret[n:] - ret[:-n]
   return ret[n - 1:] / n


with open(r"realWavesPointHope.pickle", "rb") as input_file:
# with open(r"realWavesShishmaref.pickle", "rb") as input_file:
# with open(r"realWavesWainwright.pickle", "rb") as input_file:
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

# waveNorm = wavesInput['waveNorm']
# wlFRF = wavesInput['wlFRF']
# tFRF = wavesInput['tWl']
# resFRF = wavesInput['res']

data = np.array([hsCombined,tpCombined,dmCombined])
ogdf = pd.DataFrame(data=data.T, index=tC, columns=["hs", "tp", "dm"])
year = np.array([tt.year for tt in tC])
ogdf['year'] = year
month = np.array([tt.month for tt in tC])
ogdf['month'] = month

dailyMaxHs = ogdf.resample("d")['hs'].max()

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

simSeasonalMean = np.nan * np.ones((100,12))
simSeasonalStd = np.nan * np.ones((100,12))
simYearlyMax = np.nan * np.ones((100,45))
yearArray = []
zNArray = []
ciArray = []
for hh in range(30):
   # file = r"/home/dylananderson/projects/atlanticClimate/Sims/simulation{}.pickle".format(hh)
   # file = r"/media/dylananderson/Elements/historicSims/simulationOnlyWaves{}.pickle".format(hh)
   # file = r"/users/dylananderson/Documents/projects/historicalSims/simulationOnlyWaves{}.pickle".format(hh)
   # file = r"/volumes/macDrive/historicalSims2/simulationOnlyWaves{}.pickle".format(hh)
   # file = r"/Users/dylananderson/Documents/data/wainwright/simulation{}.pickle".format(hh)
   # file = r"/Users/dylananderson/Documents/data/shishmaref/simulation{}.pickle".format(hh)
   file = r"/Users/dylananderson/Documents/data/pointHope/simulation{}.pickle".format(hh)


   with open(file, "rb") as input_file:
      simsInput = pickle.load(input_file)
   # simulationData = simsInput['futureSimulationData']
   simulationData = simsInput['simulationData']

   df = simsInput['df']

   df.loc[df['hs'] == 0, 'hs'] = np.nan
   df.loc[df['tp'] == 0, 'tp'] = np.nan
   df.loc[df['dm'] == 0, 'dm'] = np.nan

   time = simsInput['time']
   year = np.array([tt.year for tt in time])
   df['year'] = year
   month = np.array([tt.month for tt in time])
   df['month'] = month

   g1 = df.groupby(pd.Grouper(freq="M")).mean()

   simSeasonalMean[hh,:] = df.groupby('month').mean()['hs']
   simSeasonalStd[hh,:] = df.groupby('month').std()['hs']
   simYearlyMax[hh,:] = df.groupby('year').max()['hs']
   dailyMaxHsSim = df.resample("d")['hs'].max()
   c = 0
   threeDayMaxSim = []
   while c < len(simulationData):
      threeDayMaxSim.append(np.nanmax(simulationData[c:c + 72, 0]))
      c = c + 72
   threeDayMaxHsSim = np.asarray(threeDayMaxSim)

   c = 0
   fourDayMaxSim = []
   while c < len(simulationData):
      fourDayMaxSim.append(np.nanmax(simulationData[c:c + 96, 0]))
      c = c + 96
   fourDayMaxHsSim = np.asarray(fourDayMaxSim)

   # sim = return_value(np.asarray(fourDayMaxHsSim)[0:365*40], 3, 0.05, 365/4, 36525/4, 'mle')
   sim = return_value(np.asarray(fourDayMaxHsSim)[0:365*44], 3, 0.05, 365/4, int((365*45)/4), 'mle')

   yearArray.append(sim['year_array'])
   zNArray.append(sim['z_N'])
   ciArray.append(sim['CI'])


#
# import pandas
# file = r"/home/dylananderson/projects/atlanticClimate/Sims/allSimulations.pickle"
#
# with open(file, "rb") as input_file:
#    simsInput = pickle.load(input_file)
# simulationHs = simsInput['simulationsHs']
# simulationTp = simsInput['simulationsTp']
# simulationTimes = simsInput['simulationsTime']
#
# simYearlyMaxNotInterped = np.nan * np.ones((50,101))
#
# for hh in range(50):
#    simData = np.array([simulationHs[hh],simulationTp[hh]])
#
#    simdf = pandas.DataFrame(data=simData.T, index=simulationTimes[hh], columns=["hs","tp"])
#    year = np.array([tt.year for tt in simulationTimes[hh]])
#    simdf['year'] = year
#    month = np.array([tt.month for tt in simulationTimes[hh]])
#    simdf['month'] = month
#
#    #g1 = df.groupby(pd.Grouper(freq="M")).mean()
#    #simSeasonalMean[hh,:] = df.groupby('month').mean()['hs']
#    #simSeasonalStd[hh,:] = df.groupby('month').std()['hs']
#    simYearlyMaxNotInterped[hh,:] = simdf.groupby('year').max()['hs']
#

import datetime as dt
st = dt.datetime(1980, 1, 1)
end = dt.datetime(2023,1,1)
from dateutil.relativedelta import relativedelta
step = relativedelta(days=1)
dailyTimeEval = []
while st < end:
    dailyTimeEval.append(st)#.strftime('%Y-%m-%d'))
    st += step

years = np.arange(1980,2023)
numDays = 14
st = dt.datetime(2022, 1, 1)
end = dt.datetime(2023, 1, 1)
step = relativedelta(days=numDays)
plotTimeW = []
while st < end:
    plotTimeW.append(st)#.strftime('%Y-%m-%d'))
    st += step


ogHsClimatologyMean = np.nan * np.ones((len(years),int(366/numDays)))
ogHsClimatologyUpper = np.nan * np.ones((len(years),int(366/numDays)))
ogHsClimatologyLower = np.nan * np.ones((len(years),int(366/numDays)))
yCounter = 0
for hh in years:
   ogTimeIndex = np.where((tC >= dt.datetime(hh,1,1)) & (tC < dt.datetime(hh+1,1,1)))
   ogYearlyDf = ogdf.hs[ogTimeIndex[0]]
   step = numDays*24
   c = 0
   for qq in range(int(366/numDays)):
      tempOG = np.where((ogYearlyDf[c:(c+numDays*24)]>0))

      if len(tempOG[0]) > (numDays*18):
         ogHsClimatologyMean[yCounter,qq] = np.nanmean(ogYearlyDf[c:(c+numDays*24)])
         ogHsClimatologyUpper[yCounter,qq] = np.nanpercentile(ogYearlyDf[c:(c+numDays*24)],90)
         ogHsClimatologyLower[yCounter,qq] = np.nanpercentile(ogYearlyDf[c:(c+numDays*24)],10)

      else:
         ogHsClimatologyMean[yCounter,qq] = np.nan
         ogHsClimatologyUpper[yCounter,qq] = np.nan
         ogHsClimatologyLower[yCounter,qq] = np.nan

      c = c + (numDays*24)
   yCounter = yCounter+1
simHsClimatologyMean = []
simHsClimatologyUpper = []
simHsClimatologyLower = []
for hh in range(10):
   # file = r"/media/dylananderson/Elements/historicSims/simulationOnlyWaves{}.pickle".format(hh)
   # file = r"/Users/dylananderson/Documents/data/wainwright/simulation{}.pickle".format(hh)
   # file = r"/Users/dylananderson/Documents/data/shishmaref/simulation{}.pickle".format(hh)
   file = r"/Users/dylananderson/Documents/data/pointHope/simulation{}.pickle".format(hh)

   with open(file, "rb") as input_file:
      simsInput = pickle.load(input_file)
      # simulationData = simsInput['futureSimulationData']
   simulationData = simsInput['simulationData']#[0:156000,:]

   df = simsInput['df']

   df.loc[df['hs'] == 0, 'hs'] = np.nan
   df.loc[df['tp'] == 0, 'tp'] = np.nan
   df.loc[df['dm'] == 0, 'dm'] = np.nan

   time = simsInput['time']
   year = np.array([tt.year for tt in time])
   df['year'] = year
   month = np.array([tt.month for tt in time])
   df['month'] = month
   timeArray = np.asarray(time)

   simHsClimatologyMeanTemp = np.nan * np.ones((len(years),int(366/numDays)))
   simHsClimatologyUpperTemp = np.nan * np.ones((len(years),int(366/numDays)))
   simHsClimatologyLowerTemp = np.nan * np.ones((len(years),int(366/numDays)))

   yCounter = 0
   for hh in years:
      timeIndex = np.where((timeArray >= dt.datetime(hh,1,1)) & (timeArray < dt.datetime(hh+1,1,1)))
      simYearlyDf = df.hs[timeIndex[0]]

      step = numDays*24
      c = 0
      for qq in range(int(366/numDays)):
         tempSim = np.where((simYearlyDf[c:(c+numDays*24)]>0))
         tempOG = np.where((ogYearlyDf[c:(c+numDays*24)]>0))
         if len(tempSim[0]) > (numDays*18):
            simHsClimatologyMeanTemp[yCounter,qq] = np.nanmean(simYearlyDf[c:(c+numDays*24)])
            simHsClimatologyUpperTemp[yCounter,qq] = np.nanpercentile(simYearlyDf[c:(c+numDays*24)],90)
            simHsClimatologyLowerTemp[yCounter,qq] = np.nanpercentile(simYearlyDf[c:(c+numDays*24)],10)

         else:
            simHsClimatologyMeanTemp[yCounter,qq] = np.nan
            simHsClimatologyUpperTemp[yCounter,qq] = np.nan
            simHsClimatologyLowerTemp[yCounter,qq] = np.nan

         c = c + (numDays*24)
      yCounter = yCounter+1

   simHsClimatologyMean.append(simHsClimatologyMeanTemp)
   simHsClimatologyUpper.append(simHsClimatologyUpperTemp)
   simHsClimatologyLower.append(simHsClimatologyLowerTemp)

simHsClimatologyMean = np.concatenate(simHsClimatologyMean, axis=0 )
simHsClimatologyUpper = np.concatenate(simHsClimatologyUpper, axis=0 )
simHsClimatologyLower = np.concatenate(simHsClimatologyLower, axis=0 )

plt.figure()
ax1 = plt.subplot2grid((1,2),(0,0))
for hh in range(len(years)):
   ax1.plot(plotTimeW[0:-1],ogHsClimatologyMean[hh,:],color=[0.5,0.5,0.5],alpha=0.2)
   ax1.plot(plotTimeW[0:-1],ogHsClimatologyUpper[hh,:],color=[1,0.25,0.25],alpha=0.2)
   ax1.plot(plotTimeW[0:-1],ogHsClimatologyLower[hh,:],color=[0.25,0.25,1],alpha=0.2)

ax2 = plt.subplot2grid((1, 2), (0, 1))
for hh in range(len(simHsClimatologyMean)):
   ax2.plot(plotTimeW[0:-1],simHsClimatologyUpper[hh,:],color=[1,0.25,0.25],alpha=0.2)
   ax2.plot(plotTimeW[0:-1],simHsClimatologyLower[hh,:],color=[0.25,0.25,1],alpha=0.2)
   ax2.plot(plotTimeW[0:-1],simHsClimatologyMean[hh,:],color=[0.5,0.5,0.5],alpha=0.2)


si = 12
ei = -4
ax1.plot(plotTimeW[si:ei-1], np.nanmean(ogHsClimatologyMean,axis=0)[si:ei], color=[0.05, 0.05, 0.05],linewidth=2)
ax1.plot(plotTimeW[si:ei-1], np.nanmean(ogHsClimatologyUpper,axis=0)[si:ei], color=[1, 0.05, 0.05],linewidth=2)
ax1.plot(plotTimeW[si:ei-1], np.nanmean(ogHsClimatologyLower,axis=0)[si:ei], color=[0.05, 0.205, 1],linewidth=2)
ax2.plot(plotTimeW[si:ei-1], np.nanmean(simHsClimatologyMean,axis=0)[si:ei], color=[0.05, 0.05, 0.05],linewidth=2)
ax2.plot(plotTimeW[si:ei-1], np.nanmean(simHsClimatologyUpper,axis=0)[si:ei], color=[1, 0.05, 0.05],linewidth=2)
ax2.plot(plotTimeW[si:ei-1], np.nanmean(simHsClimatologyLower,axis=0)[si:ei], color=[0.05, 0.05, 1],linewidth=2)
ax1.set_xlim([plotTimeW[9],plotTimeW[-3]])
ax2.set_xlim([plotTimeW[9],plotTimeW[-3]])
ax1.set_ylim([0,3.5])
ax2.set_ylim([0,3.5])


#
# plt.figure()
# plt.plot(time[0:-4],df['hs'])

st = datetime(2022, 1, 1)
end = datetime(2023, 1, 1)
step = relativedelta(months=1)
plotTime = []
while st < end:
    plotTime.append(st)#.strftime('%Y-%m-%d'))
    st += step

plt.style.use('default')
# plt.style.use('dark_background')

plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
ax1.plot(plotTime,seasonalMean['hs'],label='ERA5 record (42 years)')
ax1.fill_between(plotTime, seasonalMean['hs'] - seasonalStd['hs'], seasonalMean['hs'] + seasonalStd['hs'], color='pink', alpha=0.2)
ax1.plot(plotTime,df.groupby('month').mean()['hs'],label='Synthetic record (100 years)')
ax1.fill_between(plotTime, df.groupby('month').mean()['hs'] - df.groupby('month').std()['hs'], df.groupby('month').mean()['hs'] + df.groupby('month').std()['hs'], color='orange', alpha=0.2)
# ax1.fill_between(plotTime, simSeasonalMean['hs'] - simSeasonalStd['hs'], simSeasonalMean['hs'] + simSeasonalStd['hs'], color='orange', alpha=0.2)
ax1.set_xticks([plotTime[0],plotTime[1],plotTime[2],plotTime[3],plotTime[4],plotTime[5],plotTime[6],plotTime[7],plotTime[8],plotTime[9],plotTime[10],plotTime[11]])
ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax1.legend()


newDf = df.groupby(pd.Grouper(freq='1D')).mean()['hs']
result = newDf.groupby([newDf.index.month, newDf.index.day]).mean()
# result.plot()




import datetime as dt
st = dt.datetime(1979, 2, 1)
end = dt.datetime(2023,6,1)
from dateutil.relativedelta import relativedelta
step = relativedelta(days=1)
dailyTime = []
while st < end:
    dailyTime.append(st)#.strftime('%Y-%m-%d'))
    st += step


import datetime as dt
st = dt.datetime(1979, 6, 1)
end = dt.datetime(2023,6,1)
from dateutil.relativedelta import relativedelta
step = relativedelta(days=1)
dailyTimeSim = []
while st < end:
    dailyTimeSim.append(st)#.strftime('%Y-%m-%d'))
    st += step


dailyMeanHsSim = df.resample("d")['hs'].mean()
dailyMeanHs = ogdf.resample("d")['hs'].mean()

plt.figure()
plt.plot(dailyMeanHs)
plt.plot(dailyMeanHsSim)

years = np.arange(1980,2022)
for n in range(len(years)):
   ind = np.where((np.asarray(dailyTime) == dt.datetime(years[n],1,1))) # & (dailyTime < dt.datetime(year[n],12,31)))
   if n == 0:
      yearlyHsMat = dailyMeanHs[ind[0][0]:ind[0][0]+365]
   else:
      yearlyHsMat = np.vstack((yearlyHsMat,dailyMeanHs[ind[0][0]:ind[0][0]+365]))




years = np.arange(1980,2022)
for n in range(len(years)):
   ind2 = np.where((np.asarray(dailyTimeSim) == dt.datetime(years[n],1,1))) # & (dailyTime < dt.datetime(year[n],12,31)))
   if n == 0:
      yearlyHsSimMat = dailyMeanHsSim[ind2[0][0]:ind2[0][0]+365]
   else:
      yearlyHsSimMat = np.vstack((yearlyHsSimMat,dailyMeanHsSim[ind2[0][0]:ind2[0][0]+365]))



avgYearlyHs = np.nanmean(yearlyHsMat,axis=0)
avgYearlyHsSim = np.nanmean(yearlyHsSimMat,axis=0)

# plt.figure()
# plt.plot(dailyTime[ind[0][0]:ind[0][0]+365],avgYearlyHs)
# plt.plot(dailyTimeSim[ind2[0][0]:ind2[0][0]+365],avgYearlyHsSim)
#

# # dailyMeanHs = []
# # for n in range(len(dailyTime)):
# #    ind = np.where((tWave >= dailyTime[n]) & (tWave < dailyTime[n+1]))
# #    dailyMeanHs.append(np.nanmean(hsCombined))
# plt.figure()
# plt.plot(time[5136:5136+366*24],simulationData[5136:5136+366*24,0])
# for n in range(38):
#    plt.plot(time[5136:5136+365*24],simulationData[5136+365*24*(n+1):5136+365*24*(n+2),0])


#
# plt.figure()
# ax1 = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
#
#
#
# ax1.plot(result.values)
# # ax1.plot(plotTime,seasonalMean['hs'],label='ERA5 record (42 years)')
# # ax1.fill_between(plotTime, seasonalMean['hs'] - seasonalStd['hs'], seasonalMean['hs'] + seasonalStd['hs'], color='pink', alpha=0.2)
# ax1.plot(plotTime,df.groupby(pd.Grouper(freq='1D')).mean()['hs'],label='Synthetic record (100 years)')
# ax1.fill_between(plotTime, df.groupby('D').mean()['hs'] - df.groupby('D').std()['hs'], df.groupby('D').mean()['hs'] + df.groupby('D').std()['hs'], color='orange', alpha=0.2)
# # ax1.fill_between(plotTime, simSeasonalMean['hs'] - simSeasonalStd['hs'], simSeasonalMean['hs'] + simSeasonalStd['hs'], color='orange', alpha=0.2)
# ax1.set_xticks([plotTime[0],plotTime[1],plotTime[2],plotTime[3],plotTime[4],plotTime[5],plotTime[6],plotTime[7],plotTime[8],plotTime[9],plotTime[10],plotTime[11]])
# ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
# ax1.legend()


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
# ax.set_xlabel('Fetch (km)')
# ax2.set_title('Average Fetch Length to Sea Ice')






simMaxHs = np.nan*np.ones((100,44))
# simMaxHsNotInterped = np.nan*np.ones((50,100))
for hh in range(100):
   simMaxHs[hh,:] = np.sort(simYearlyMax[hh,0:44])
   # simMaxHsNotInterped[hh,:] = np.sort(simYearlyMaxNotInterped[hh,0:-1])

plt.style.use('default')

plt.figure()
ax2 = plt.subplot2grid((1,1),(0,0))
maxHs = np.sort(yearlyMax['hs'])
# maxHsNotInterped = np.sort(yearlyMax['hs'])

returnPeriod = np.flipud((len(maxHs)+1)/np.arange(1,len(maxHs)+1))
# simMaxHs = np.sort(simYearlyMax['hs'][0:-1])
# simReturnPeriod = np.flipud(100/np.arange(1,101))
simReturnPeriod = np.flipud(44/np.arange(1,45))

ax2.fill_between(simReturnPeriod, np.min(simMaxHs,axis=0), np.max(simMaxHs,axis=0), color='orange', alpha=0.2)
ax2.plot(returnPeriod[0:-1],maxHs[1:],'o',label='ERA5')
ax2.plot(simReturnPeriod,np.mean(simMaxHs,axis=0),'.-',label='Mean of Simulations')

# ax2.fill_between(simReturnPeriod, np.min(simMaxHsNotInterped,axis=0), np.max(simMaxHsNotInterped,axis=0), color='green', alpha=0.2)
# ax2.plot(simReturnPeriod,np.mean(simMaxHsNotInterped,axis=0),'.-')
# ax2.plot(simReturnPeriod,simMaxHs,'.')
ax2.set_xscale('log')
ax2.set_xlabel('return period (yrs)')
ax2.set_ylabel('hs (m)')
ax2.legend()
ax2.set_ylim([3,8.5])
# ax1.set_xticks([plotTime[0],plotTime[2],plotTime[4],plotTime[6],plotTime[8],plotTime[10]])
# ax1.set_xticklabels(['Jan','Mar','May','Jul','Sep','Nov'])


# from rpy2.robjects.packages import importr
# import rpy2.robjects.packages as rpackages
#
# base = importr('base')
# utils = importr('utils')
# utils.chooseCRANmirror(ind=1)
# utils.install_packages('POT') #installing POT package
# from thresholdmodeling import thresh_modeling #importing package
# import pandas as pd #importing pandas
#
# #url = 'https://raw.githubusercontent.com/iagolemos1/thresholdmodeling/master/dataset/rain.csv' #saving url
# #df =  pd.read_csv(url, error_bad_lines=False) #getting data
# data = df['hs'].values.ravel() #turning data into an array
#
data = np.asarray(dailyMaxHs)
# #data =
#
# # thresh_modeling.MRL(data, 0.05)
# # thresh_modeling.Parameter_Stability_plot(data, 0.05)
# # thresh_modeling.return_value(data, 30, 0.05, 365, 36500, 'mle')
#
# dataDecluster, data2 = thresh_modeling.decluster(data,3,24*2)
# thresh_modeling.return_value(data, 3, 0.05, 365.25*24, 36525*24, 'mle')
#


import matplotlib.cm as cm
import matplotlib.colors as mcolors

# historical = return_value(np.asarray(fourDayMax), 3, 0.05, 365/4, 36525/4, 'mle')
historical = return_value(np.asarray(fourDayMax), 3, 0.05, 365/4, (365*45)/4, 'mle')

plt.style.use('dark_background')


# to do order this by uncertainty
plt.figure(8)
colorparam = np.zeros((len(zNArray),))
for qq in range(len(zNArray)):
   normalize = mcolors.Normalize(vmin=0, vmax=5)
   colorparam[qq] = ciArray[qq]
   colormap = cm.Greys
   color = colormap(normalize(colorparam[qq]))
   plt.plot(yearArray[qq],zNArray[qq],color=color,alpha=0.75)#color=[0.5,0.5,0.5],alpha=0.5)

# returnPeriod = np.flipud((len(maxHs)+1)/np.arange(1,len(maxHs)+1))
# # simMaxHs = np.sort(simYearlyMax['hs'][0:-1])
# # simReturnPeriod = np.flipud(100/np.arange(1,101))
# simReturnPeriod = np.flipud(45/np.arange(1,46))

plt.plot(historical['year_array'], historical['CI_z_N_high_year'], linestyle='--', color='red', alpha=0.8, lw=0.9, label='Confidence Bands')
plt.plot(historical['year_array'], historical['CI_z_N_low_year'], linestyle='--', color='red', alpha=0.8, lw=0.9)
plt.plot(historical['year_array'], historical['z_N'], color='orange', label='Theoretical Return Level')
# plt.scatter(historical['N'], historical['sample_over_thresh'], color='orange',label='Empirical Return Level',zorder=10)
plt.xscale('log')
plt.xlabel('Return Period')
plt.ylabel('Return Level')
plt.title('Return Level Plot')
plt.legend()

plt.show()

