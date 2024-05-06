import os
import numpy as np
import datetime
from netCDF4 import Dataset
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec
import pickle
from scipy.io.matlab.mio5_params import mat_struct
from datetime import datetime, date, timedelta
import random
import itertools
import operator
import scipy.io as sio
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from scipy.stats import norm, genpareto, t
from scipy.special import ndtri  # norm inv
import matplotlib.dates as mdates
from scipy.stats import  genextreme, gumbel_l, spearmanr, norm, weibull_min
from scipy.spatial import distance
import pickle
import calendar
import xarray as xr
import pandas


def toTimestamp(d):
    return calendar.timegm(d.timetuple())


with open(r"simulations100Chopped49_2Dist.pickle", "rb") as input_file:
   simsChoppedInput = pickle.load(input_file)
simBmuLengthChopped = simsChoppedInput['simBmuLengthChopped']
simBmuGroupsChopped = simsChoppedInput['simBmuGroupsChopped']
simBmuChopped = simsChoppedInput['simBmuChopped']
simIceGroupsChopped = simsChoppedInput['simIceGroupsChopped']
simIceSlpGroupsChopped = simsChoppedInput['simIceSlpGroupsChopped']
simTimeGroupsChopped = simsChoppedInput['simTimeGroupsChopped']

simIceAreaGroupsChopped = simsChoppedInput['simIceAreaGroupsChopped']


with open(r"gevCopulaSims100000ptLay.pickle", "rb") as input_file:
   gevCopulaSimsInput = pickle.load(input_file)
gevCopulaSims = gevCopulaSimsInput['gevCopulaSims']

with open(r"normalizedWaveHydrographsHope2Dist49ptLay.pickle", "rb") as input_file:
   normalizedWaveHydrographs = pickle.load(input_file)
normalizedHydros = normalizedWaveHydrographs['normalizedHydros']
bmuDataMin = normalizedWaveHydrographs['bmuDataMin']
bmuDataMax = normalizedWaveHydrographs['bmuDataMax']
bmuDataStd = normalizedWaveHydrographs['bmuDataStd']
bmuDataNormalized = normalizedWaveHydrographs['bmuDataNormalized']

with open(r"hydrographCopulaDataHope2Dist49ptLay.pickle", "rb") as input_file:
   hydrographCopulaData = pickle.load(input_file)
copulaData = hydrographCopulaData['copulaData']
copulaDataNoNaNs = hydrographCopulaData['copulaDataNoNaNs']

with open(r"iceFutureSimulations1000.pickle", "rb") as input_file:

# with open(r"iceWithTempElNinosSimulations100.pickle", "rb") as input_file:
   iceSimulations = pickle.load(input_file)
iceSims = iceSimulations['evbmus_sim']
iceDates = iceSimulations['dates_sim']

with open(r"processedWaveWaterLevels49DWTs25IWTs2022v2.pickle", "rb") as input_file:
   onOffData = pickle.load(input_file)
percentWaves = onOffData['percentWaves']
percentIce = onOffData['percentIce']

# with open(r"iceData25ClustersMediumishMin150withDayNumRG.pickle", "rb") as input_file:
with open(r"iceData25ClustersDayNumRGAdjusted.pickle", "rb") as input_file:
   iceDWTs = pickle.load(input_file)




# with open(r"iceTempWith2PCsNAO36DWTsSimulations100.pickle", "rb") as input_file:
# # with open(r"iceWithTempElNinosSimulations100.pickle", "rb") as input_file:
#    iceSimulations = pickle.load(input_file)
# iceSims = iceSimulations['evbmus_sim']
# iceDates = iceSimulations['dates_sim']
#
#
# with open(r"processedWaveWaterLevels25DWTs36IWTs.pickle", "rb") as input_file:
#    onOffData = pickle.load(input_file)
# percentWaves = onOffData['percentWaves']
# percentIce = onOffData['percentIce']
#
# with open(r"iceData36ClustersMin60.pickle", "rb") as input_file:
#    iceDWTs = pickle.load(input_file)
#

#
# iceOnOff = []
# years = np.arange(1979,2022)
# # lets start in January and assume that we are always beginning with an iced ocean
# for hh in range(100):
#     simOfIce = iceSims[:,hh]
#     yearsStitched = []
#     timeStitched = []
#
#     for tt in range(len(years)):
#         yearInd = np.where((np.array(iceDates) >= datetime(years[tt],1,1)) & (np.array(iceDates) < datetime(years[tt]+1,1,1)))
#         timeSubset = np.array(iceDates)[yearInd]
#         tempOfSim = simOfIce[yearInd]
#         groupTemp = [[e[0] for e in d[1]] for d in
#                             itertools.groupby(enumerate(tempOfSim), key=operator.itemgetter(1))]
#         timeGroupTemp = [timeSubset[d] for d in groupTemp]
#         bmuGroupIce = (np.asarray([tempOfSim[i[0]] for i in groupTemp]))
#         tempControl = 0
#
#         # lets a
#         wavesStitched = []
#         for qq in range(len(groupTemp)):
#             tempRand = np.random.uniform(size=len(groupTemp[qq]))
#             if tempControl == 0:
#                 # lets enter the next group assuming that waves were previously off
#                 sortedRand = np.sort(tempRand)            # so we have a bunch of probabilities between 0 and 1
#                 finder = np.where((sortedRand > percentIce[int(bmuGroupIce[qq]-1)]))     # if greater than X then waves are ON
#                 # arbitrary, but lets say waves have to be turned on for 3 days before we say, sure the ocean broke up
#                 waveDays = np.zeros((len(sortedRand),))
#                 if timeGroupTemp[qq][-1].month > 3:
#                     if len(finder[0]) > 2:
#                         waveDays[finder[0]] = np.ones((len(finder[0])))       # flip those days that are waves to ON
#                         tempControl = 1
#                 wavesStitched.append(waveDays)
#             else:
#                 # now we're entering with waves ON and want to turn them OFF
#                 sortedRand = np.sort(tempRand)#[::-1]        # again a bunch of probabilities between 0 and 1
#                 finder = np.where((sortedRand > percentWaves[int(bmuGroupIce[qq]-1)]))  # if greater than X then waves are OFF
#                 # arbitrary, but again, waves need to be off for 3 days to really freeze up the ocean
#                 waveDays = np.ones((len(sortedRand),))
#                 if timeGroupTemp[qq][0].month < 8 or timeGroupTemp[qq][-1].month > 9:
#
#                     if len(finder[0]) > 2:
#                         waveDays[finder[0]] = np.zeros((len(finder[0])))
#                         tempControl = 0
#                 wavesStitched.append(waveDays)
#         yearsStitched.append(wavesStitched)
#     iceOnOff.append(np.concatenate([x for xs in yearsStitched for x in xs]).ravel())
#
#
#
#
# def GenOneYearDaily(yy=1981, month_ini=1):
#    'returns one generic year in a list of datetimes. Daily resolution'
#
#    dp1 = datetime(yy, month_ini, 1)
#    dp2 = dp1 + timedelta(days=365)
#
#    return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]
#
# list_pyear = GenOneYearDaily(month_ini=1)
#
# histDates = onOffData['iceDateTimes']
# bmus_dates_months = np.array([d.month for d in histDates])
# bmus_dates_days = np.array([d.day for d in histDates])
# yearsOfHist= np.arange(1980,2022)
# stacked = np.zeros((len(yearsOfHist),365))
# for qq in range(len(yearsOfHist)):
#     # index = np.where((np.array(histDates) >= datetime(yearsOfHist[qq],6,1)) & (np.array(histDates) < datetime(yearsOfHist[qq]+1,5,31)))
#     index = np.where((np.array(histDates) >= datetime(yearsOfHist[qq],1,1)) & (np.array(histDates) < datetime(yearsOfHist[qq],12,31)))
#     stacked[qq,0:len(index[0])] = onOffData['wavesOnOff'][index]
#
# import cmocean
# fig = plt.figure(figsize=(10,4))
# ax = plt.subplot2grid((1,1),(0,0))
# X,Y = np.meshgrid(list_pyear,np.arange(1980,2022))
# ax.pcolormesh(X,Y,stacked,cmap=cmocean.cm.ice_r,vmin=0,vmax=2)
# import matplotlib.dates as mdates
# # customize  axis
# months = mdates.MonthLocator()
# monthsFmt = mdates.DateFormatter('%b')
# ax.set_xlim(list_pyear[0], list_pyear[-1])
# ax.xaxis.set_major_locator(months)
# ax.xaxis.set_major_formatter(monthsFmt)
#
#
#
#
# simulat = 0
#
# histDates = iceDates#[214:]#onOffData['iceDateTimes']
# bmus_dates_months = np.array([d.month for d in histDates])
# bmus_dates_days = np.array([d.day for d in histDates])
# yearsOfHist= np.arange(1980,2021)
# stacked = np.zeros((len(yearsOfHist),365))
# for qq in range(len(yearsOfHist)):
#     # index = np.where((np.array(histDates) >= datetime(yearsOfHist[qq],6,1)) & (np.array(histDates) < datetime(yearsOfHist[qq]+1,5,31)))
#     index = np.where((np.array(histDates) >= datetime(yearsOfHist[qq],1,1)) & (np.array(histDates) < datetime(yearsOfHist[qq],12,31)))
#     stacked[qq,0:len(index[0])] = iceOnOff[simulat][index]
#
# import cmocean
# fig = plt.figure(figsize=(10,4))
# ax = plt.subplot2grid((1,1),(0,0))
# X,Y = np.meshgrid(list_pyear,np.arange(1980,2022))
# ax.pcolormesh(X,Y,stacked,cmap=cmocean.cm.ice_r,vmin=0,vmax=2)
# import matplotlib.dates as mdates
# # customize  axis
# months = mdates.MonthLocator()
# monthsFmt = mdates.DateFormatter('%b')
# ax.set_xlim(list_pyear[0], list_pyear[-1])
# ax.xaxis.set_major_locator(months)
# ax.xaxis.set_major_formatter(monthsFmt)
#
#
# num_simulations = 100
# histDates = iceDates#[214:]#onOffData['iceDateTimes']
# bmus_dates_months = np.array([d.month for d in histDates])
# bmus_dates_days = np.array([d.day for d in histDates])
# yearsOfHist= np.arange(1980,2021)
# stacked = np.zeros((len(yearsOfHist),365))
# for qq in range(len(yearsOfHist)):
#     index = np.where((np.array(histDates) >= datetime(yearsOfHist[qq], 1, 1)) & (
#                 np.array(histDates) < datetime(yearsOfHist[qq], 12, 31)))
#     yearlyStacked = np.zeros((num_simulations,len(index[0])))
#     for tt in range(num_simulations):
#         # index = np.where((np.array(histDates) >= datetime(yearsOfHist[qq],6,1)) & (np.array(histDates) < datetime(yearsOfHist[qq]+1,5,31)))
#         yearlyStacked[tt,0:len(index[0])] = iceOnOff[tt][index]
#     stacked[qq,0:len(index[0])] = np.nanmean(yearlyStacked,axis=0)
#
#
# fig = plt.figure(figsize=(10,4))
# ax = plt.subplot2grid((1,1),(0,0))
# X,Y = np.meshgrid(list_pyear,np.arange(1980,2022))
# ax.pcolormesh(X,Y,stacked,cmap=cmocean.cm.ice_r,vmin=0,vmax=2)
# # customize  axis
# months = mdates.MonthLocator()
# monthsFmt = mdates.DateFormatter('%b')
# ax.set_xlim(list_pyear[0], list_pyear[-1])
# ax.xaxis.set_major_locator(months)
# ax.xaxis.set_major_formatter(monthsFmt)
#

dt = datetime(1979, 6, 1, 0, 0, 0)
end = datetime(2021, 5, 31, 23, 0, 0)
# dt = datetime(1980, 1, 1, 0, 0, 0)
# end = datetime(2020, 12, 31, 23, 0, 0)
# dt = datetime(2022, 6, 1, 0, 0, 0)
# end = datetime(2122, 5, 31, 23, 0, 0)
step = timedelta(hours=1)
hourlyTime = []
while dt < end:
    hourlyTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step

deltaT = [(tt - hourlyTime[0]).total_seconds() / (3600*24) for tt in hourlyTime]
# # Create datetime objects for each time (a and b)
# dateTimeA = datetime.combine(datetime.date.today(), a)
# dateTimeB = datetime.combine(datetime.date.today(), b)
# # Get the difference between datetimes (as timedelta)
# dateTimeDifference = dateTimeA - dateTimeB
# # Divide difference in seconds by number of seconds in hour (3600)
# dateTimeDifferenceInHours = dateTimeDifference.total_seconds() / 3600


def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index], closest_index


simulationsHs = list()
simulationsTp = list()
simulationsDm = list()
simulationsSs = list()
simulationsArea = list()
simulationsTime = list()

for simNum in range(100):

    simHs = []
    simTp = []
    simDm = []
    simSs = []
    simArea = []
    simTime = []
    print('filling in simulation #{}'.format(simNum))

    for i in range(len(simBmuChopped[simNum])):
        if np.remainder(i,1000) == 0:
            print('done with {} hydrographs'.format(i))

        tempIceSlpBmus = simIceSlpGroupsChopped[simNum][i][0]
        tempTimeMonth = simTimeGroupsChopped[simNum][i][0]
        tempArea = simIceAreaGroupsChopped[simNum][i][0]

        if tempIceSlpBmus == 1:
            tempBmu = int(simBmuChopped[simNum][i]-1)+49
        else:
            tempBmu = int(simBmuChopped[simNum][i]-1)


        if tempBmu == 91:
            tempBmu = 42
        elif tempBmu ==43:
            tempBmu = 42
        elif tempBmu ==36:
            tempBmu = 37
        elif tempBmu ==85:
            tempBmu = 37
        elif tempBmu ==77:
            tempBmu = 35
        elif tempBmu ==86:
            tempBmu = 37

        #
        # if tempBmu == 62:
        #     tempBmu = 122

        # so i think at this stage we need to know what the correspond ICE dwt is... then whether its a 0 or 1, and then choose from the gevCopulaSims based on that
        randStorm = [random.randint(0, 49999) for n in range(100)]

        stormDetails = gevCopulaSims[tempBmu][randStorm]

        possibleAreas = stormDetails[:,5]


        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            #return array[idx]
            return idx
        chosen = find_nearest(possibleAreas,tempArea)
        stormDetails = stormDetails[chosen]
        if stormDetails[0] > 10:
            print('oh boy, we''ve picked a {}m storm wave in BMU #{}'.format(stormDetails[0],tempBmu))
        durSim = simBmuLengthChopped[simNum][i]

        simHsNorm = (stormDetails[0] - np.asarray(bmuDataMin)[tempBmu,0]) / (np.asarray(bmuDataMax)[tempBmu,0]-np.asarray(bmuDataMin)[tempBmu,0])
        simTpNorm = (stormDetails[1] - np.asarray(bmuDataMin)[tempBmu,1]) / (np.asarray(bmuDataMax)[tempBmu,1]-np.asarray(bmuDataMin)[tempBmu,1])
        # simDmNorm = (stormDetails[4] - np.asarray(bmuDataMin)[tempBmu,0]) / (np.asarray(bmuDataMax)[tempBmu,0]-np.asarray(bmuDataMin)[tempBmu,0])
        # simDurNorm = (durSim - np.asarray(bmuDataMin)[tempBmu,1]) / (np.asarray(bmuDataMax)[tempBmu,1]-np.asarray(bmuDataMin)[tempBmu,1])
        # closeIndex, test = min(enumerate(np.asarray(bmuDataNormalized)[tempBmu][:,0]), key=lambda x: abs(x[1] - simDmNorm))

        #test, closeIndex = closest_node([simDmNorm],np.asarray(bmuDataNormalized)[tempBmu][:,0])

        # simDmNorm = (stormDetails[4] - np.asarray(bmuDataMin)[tempBmu,0]) / (np.asarray(bmuDataMax)[tempBmu,0]-np.asarray(bmuDataMin)[tempBmu,0])
        # simSsNorm = (stormDetails[5] - np.asarray(bmuDataMin)[tempBmu,1]) / (np.asarray(bmuDataMax)[tempBmu,1]-np.asarray(bmuDataMin)[tempBmu,1])
        # test, closeIndex = closest_node([simDmNorm,simSsNorm],np.asarray(bmuDataNormalized)[tempBmu])


        test, closeIndex = closest_node([simHsNorm,simTpNorm],np.asarray(bmuDataNormalized)[tempBmu])

        actualIndex = closeIndex#int(np.asarray(copulaDataNoNaNs[tempBmu])[closeIndex,6])
        # if tempBmu==0 or tempBmu==10 or tempBmu==21 or tempBmu==27 or tempBmu==28 or tempBmu==30 or tempBmu==34 or tempBmu==35 or tempBmu==36 or tempBmu==37 or tempBmu==42\
        #         or tempBmu==43 or tempBmu==47 or tempBmu==48 or tempBmu==98 or tempBmu==99 or tempBmu==100 or tempBmu==101 or tempBmu==102 or tempBmu==108 or tempBmu==112\
        #         or tempBmu==116 or tempBmu==117 or tempBmu==118 or tempBmu==119 or tempBmu==121 or tempBmu==122 or tempBmu==124 or tempBmu==125 or tempBmu==126\
        #         or tempBmu==127 or tempBmu==128 or tempBmu==132 or tempBmu==133 or tempBmu==134 or tempBmu==135 or tempBmu==140 or tempBmu==141 or tempBmu==142\
        #         or tempBmu==143 or tempBmu==145 or tempBmu==146:
        # if tempBmu==2:
        #     tempHs = np.array([0.25,0.25,0.25,0.25])
        #     tempTp = np.array([3,3,3,3])
        #     tempDm = np.array([0,0,0,0])
        #     simTime.append(np.array([0.25,0.25,0.25,0.25])*durSim)
        #
        # elif tempBmu == 83 or tempBmu == 84 or tempBmu == 125 or tempBmu == 142 or tempBmu == 158:
        #     tempHs = np.array([1.5,1.5,1.5,1.5])
        #     tempTp = np.array([8,8,8,8])
        #     tempDm = np.array([0,0,0,0])
        #     simTime.append(np.array([0.25,0.25,0.25,0.25])*durSim)

        # elif tempBmu == 188 or tempBmu == 189 or tempBmu == 191:
        #     tempHs = np.array([1.5,1.5,1.5])
        #     tempTp = np.array([8,8,8])
        #     tempDm = np.array([0,0,0])
        #     simTime.append(np.array([0,0.5,0.99999]))
        # else:
        tempHs = ((normalizedHydros[tempBmu][actualIndex]['hsNorm']) * (stormDetails[0]-stormDetails[1]) + stormDetails[1])#.filled()
        tempTp = ((normalizedHydros[tempBmu][actualIndex]['tpNorm']) * (stormDetails[2]-stormDetails[3]) + stormDetails[3])#.filled()
        tempDm = ((normalizedHydros[tempBmu][actualIndex]['dmNorm']) + stormDetails[4])
            # tempSs = ((normalizedHydros[tempBmu][actualIndex]['ssNorm']) + stormDetails[5])
        if len(normalizedHydros[tempBmu][actualIndex]['hsNorm']) < len(normalizedHydros[tempBmu][actualIndex]['timeNorm']):
            print('Time is shorter than Hs in bmu {}, index {}'.format(tempBmu,actualIndex))
        if stormDetails[1] < 0:
            print('woah, we''re less than 0 over here')
            asdfg
        # if len(tempSs) < len(normalizedHydros[tempBmu][actualIndex]['timeNorm']):
        #     print('Ss is shorter than Time in bmu {}, index {}'.format(tempBmu,actualIndex))
        #     tempLength = len(normalizedHydros[tempBmu][actualIndex]['timeNorm'])
        #     tempSs = np.zeros((len(normalizedHydros[tempBmu][actualIndex]['timeNorm']),))
        #     tempSs[0:len((normalizedHydros[tempBmu][actualIndex]['ssNorm']) + stormDetails[5])] = ((normalizedHydros[tempBmu][actualIndex]['ssNorm']) + stormDetails[5])
        # if len(tempSs) > len(normalizedHydros[tempBmu][actualIndex]['timeNorm']):
        #     print('Now Ss is longer than Time in bmu {}, index {}'.format(tempBmu,actualIndex))
        #     print('{} vs. {}'.format(len(tempSs),len(normalizedHydros[tempBmu][actualIndex]['timeNorm'])))
        #     tempSs = tempSs[0:-1]
        simTime.append(np.hstack((np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim), np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)[-1])))

        if simIceGroupsChopped[simNum][i][0] > 0:
            simHs.append(tempHs)
            simTp.append(tempTp)
            simDm.append(tempDm)
            # simSs.append(tempSs)
        else:
            simHs.append(tempHs*0)
            simTp.append(tempTp*0)
            simDm.append(tempDm*0)
            # simSs.append(tempSs)
        #simTime.append(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
        #dt = np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
        # simTime.append(np.hstack((np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim), np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)[-1])))


    #
    # asdfg
    #
    #
    #
    # simulationsHs.append(np.hstack(simHs))
    # simulationsTp.append(np.hstack(simTp))
    # simulationsDm.append(np.hstack(simDm))
    # simulationsSs.append(np.hstack(simSs))
    cumulativeHours = np.cumsum(np.hstack(simTime))
    newDailyTime = [datetime(2022, 6, 1) + timedelta(days=ii) for ii in cumulativeHours]
    simDeltaT = [(tt - newDailyTime[0]).total_seconds() / (3600 * 24) for tt in newDailyTime]

    # simulationsTime.append(newDailyTime)
    # rng = newDailyTime
    #
    simData = np.array(np.vstack((np.hstack(simHs).T,np.hstack(simTp).T,np.hstack(simDm))))#.T,np.hstack(simSs).T)))
    # simData = np.array((np.ma.asarray(np.hstack(simHs)),np.ma.asarray(np.hstack(simTp)),np.ma.asarray(np.hstack(simDm)),np.ma.asarray(np.hstack(simSs))))
    # simData = np.array([np.hstack(simHs).filled(),np.hstack(simTp).filled(),np.hstack(simDm).filled(),np.hstack(simSs)])

    ogdf = pandas.DataFrame(data=simData.T,index=newDailyTime,columns=["hs","tp","dm"])#,"ss"])

    print('interpolating')
    interpHs = np.interp(deltaT,simDeltaT,np.hstack(simHs))
    interpTp = np.interp(deltaT,simDeltaT,np.hstack(simTp))
    interpDm = np.interp(deltaT,simDeltaT,np.hstack(simDm))
    # interpSs = np.interp(deltaT,simDeltaT,np.hstack(simSs))

    simDataInterp = np.array([interpHs,interpTp,interpDm])#,interpSs])

    df = pandas.DataFrame(data=simDataInterp.T,index=hourlyTime,columns=["hs","tp","dm"])#,"ss"])
    # resampled = df.resample('H')
    # interped = resampled.interpolate()
    # simulationData = interped.values
    # testTime = interped.index  # to_pydatetime()
    # testTime2 = testTime.to_pydatetime()

    # simsPickle = ('/home/dylananderson/projects/atlanticClimate/Sims/simulation{}.pickle'.format(simNum))
    simsPickle = ('/volumes/macDrive/historicalSims2/simulationOnlyWaves{}.pickle'.format(simNum))

    outputSims= {}
    outputSims['simulationData'] = simDataInterp.T
    outputSims['df'] = df
    outputSims['simHs'] = np.hstack(simHs)
    outputSims['simTp'] = np.hstack(simTp)
    outputSims['simDm'] = np.hstack(simDm)
    # outputSims['simSs'] = np.hstack(simSs)
    outputSims['time'] = hourlyTime

    with open(simsPickle, 'wb') as f:
        pickle.dump(outputSims, f)

    #
    # # ts = pandas.Series(np.hstack(simHs), index=newDailyTime)
    # # resampled = ts.resample('H')
    # # interp = resampled.interpolate()
    #
    # testTime = interped.index  # to_pydatetime()
    # testTime2 = testTime.to_pydatetime()
    # testData = interped.values



# simsPickle = '/home/dylananderson/projects/atlanticClimate/Sims/allSimulations.pickle'
# outputSims= {}
# outputSims['simulationsHs'] = simulationsHs
# outputSims['simulationsTime'] = simulationsTime
# outputSims['simulationsTp'] = simulationsTp
# outputSims['simulationsDm'] = simulationsDm
# outputSims['simulationsSs'] = simulationsSs
#
# with open(simsPickle, 'wb') as f:
#     pickle.dump(outputSims, f)


plt.figure()
ax1 = plt.subplot2grid((3,1),(0,0),rowspan=1,colspan=1)
hs = simDataInterp[0,:]
where0 = np.where((hs == 0))
hs[where0] = np.nan
ax1.plot(hourlyTime,hs)
ax2 = plt.subplot2grid((3,1),(1,0),rowspan=1,colspan=1)
tp = simDataInterp[1,:]
where0 = np.where((tp < 0.5))
tp[where0] = np.nan
ax2.plot(hourlyTime,tp)
ax3 = plt.subplot2grid((3,1),(2,0),rowspan=1,colspan=1)
dm = simDataInterp[2,:]
where0 = np.where((dm == 0))
dm[where0] = np.nan
where360 = np.where((dm > 360))
dm[where360] = dm[where360]-360
whereNeg = np.where((dm < 0))
dm[whereNeg] = dm[whereNeg]+360
ax3.plot(hourlyTime,dm)

### TODO: Need to assess the statistics of these hypothetical scenarios... Yearly max Hs? Wave Energy?

### TODO: Which requires interpolating the time series to hourly values...

# for qq in len(simulationsTime):

