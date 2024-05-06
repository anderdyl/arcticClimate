import numpy as np
from datetime import datetime, date, timedelta
import random
import itertools
import operator
import pickle
import matplotlib.pyplot as plt

with open(r"normalizedWaveHydrographsHope2Dist49.pickle", "rb") as input_file:
   normalizedWaveHydrographs = pickle.load(input_file)
normalizedHydros = normalizedWaveHydrographs['normalizedHydros']
bmuDataMin = normalizedWaveHydrographs['bmuDataMin']
bmuDataMax = normalizedWaveHydrographs['bmuDataMax']
bmuDataStd = normalizedWaveHydrographs['bmuDataStd']
bmuDataNormalized = normalizedWaveHydrographs['bmuDataNormalized']


with open(r"hydrographCopulaDataHope2Dist49.pickle", "rb") as input_file:
   hydrographCopulaData = pickle.load(input_file)
copulaData = hydrographCopulaData['copulaData']

with open(r"historicalDataHope2Dist49.pickle", "rb") as input_file:
   historicalData = pickle.load(input_file)
grouped = historicalData['grouped']
groupLength = historicalData['groupLength']
bmuGroup = historicalData['bmuGroup']
timeGroup = historicalData['timeGroup']

with open(r"gevCopulaSims1000002Dist49.pickle", "rb") as input_file:
   gevCopulaSimsInput = pickle.load(input_file)
gevCopulaSims = gevCopulaSimsInput['gevCopulaSims']

# with open(r"dwt81RGHistoricalSimulations100withTemp.pickle", "rb") as input_file:
# with open(r"dwt64HistoricalSimulations100withTemp.pickle", "rb") as input_file:
with open(r"dwt49RGHistoricalSimulations100withTemp.pickle", "rb") as input_file:
# with open(r"dwtFutureSimulations1000.pickle", "rb") as input_file:
   dwtFutureSimulations = pickle.load(input_file)
evbmus_sim = dwtFutureSimulations['evbmus_sim']
# sim_years = dwtFutureSimulations['sim_years']
dates_sim = dwtFutureSimulations['dates_sim']

# with open(r"iceTempWithNAO36DWTsSimulations100.pickle", "rb") as input_file:
with open(r"iceFutureSimulations1000.pickle", "rb") as input_file:
    iceSimulations = pickle.load(input_file)
iceSims = iceSimulations['evbmus_sim']
iceDates = iceSimulations['dates_sim']


with open(r"processedWaveWaterLevels49DWTs25IWTs2022v2.pickle", "rb") as input_file:
   onOffData = pickle.load(input_file)
percentWaves = onOffData['percentWaves']
percentIce = onOffData['percentIce']


# with open(r"iceData36ClustersMin60.pickle", "rb") as input_file:
#    iceDWTs = pickle.load(input_file)
# with open(r"iceData25ClustersMediumishMin150withDayNumRG.pickle", "rb") as input_file:

with open(r"iceData25ClustersDayNumRGAdjusted.pickle", "rb") as input_file:
   iceDWTs = pickle.load(input_file)

with open(r"iceFetchSimsv2.pickle", "rb") as input_file:
   fwtchSimsPickle = pickle.load(input_file)
fetchSims = fwtchSimsPickle['fetchSims']



#
# iceOnOff = []
# years = np.arange(1979,2022)
# # lets start in January and assume that we are always beginning with an iced ocean
# for hh in range(100):
#     simOfIce = iceSims[:,hh]
#     yearsStitched = []
#     timeStitched = []
#     tempControl = 0
#
#     for tt in range(len(years)):
#         yearInd = np.where((np.array(iceDates) >= datetime(years[tt],1,1)) & (np.array(iceDates) < datetime(years[tt]+1,1,1)))
#         timeSubset = np.array(iceDates)[yearInd]
#         tempOfSim = simOfIce[yearInd]
#         groupTemp = [[e[0] for e in d[1]] for d in
#                             itertools.groupby(enumerate(tempOfSim), key=operator.itemgetter(1))]
#         timeGroupTemp = [timeSubset[d] for d in groupTemp]
#         bmuGroupIce = (np.asarray([tempOfSim[i[0]] for i in groupTemp]))
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
#                 if timeGroupTemp[qq][0].month > 4:
#                     if len(finder[0]) > 3:
#                         waveDays[finder[0]] = np.ones((len(finder[0])))       # flip those days that are waves to ON
#                         tempControl = 1
#                 if timeGroupTemp[qq][0].month > 7 and timeGroupTemp[qq][-1].month < 9:
#                     waveDays = np.ones((len(sortedRand),))
#                     tempControl = 1
#                 wavesStitched.append(waveDays)
#             else:
#                 # now we're entering with waves ON and want to turn them OFF
#                 sortedRand = np.sort(tempRand)#[::-1]        # again a bunch of probabilities between 0 and 1
#                 finder = np.where((sortedRand > percentWaves[int(bmuGroupIce[qq]-1)]))  # if greater than X then waves are OFF
#                 # arbitrary, but again, waves need to be off for 3 days to really freeze up the ocean
#                 waveDays = np.ones((len(sortedRand),))
#                 if timeGroupTemp[qq][-1].month < 8 or timeGroupTemp[qq][0].month > 9:
#
#                     if len(finder[0]) > 3:
#                         waveDays[finder[0]] = np.zeros((len(finder[0])))
#                         tempControl = 0
#                 if timeGroupTemp[qq][0].month > 0 and timeGroupTemp[qq][-1].month < 5:
#                     waveDays = np.zeros((len(sortedRand),))
#                     tempControl = 0
#                 wavesStitched.append(waveDays)
#         yearsStitched.append(wavesStitched)
#     iceOnOff.append(np.concatenate([x for xs in yearsStitched for x in xs]).ravel())
#


iceOnOff = []
iceArea = []
years = np.arange(1979,2022)
# lets start in January and assume that we are always beginning with an iced ocean
for hh in range(100):
    simOfIce = iceSims[:,hh]
    yearsStitched = []
    yearsIceStitched = []
    timeStitched = []
    tempControl = 0

    for tt in range(len(years)):
        yearInd = np.where((np.array(iceDates) >= datetime(years[tt],1,1)) & (np.array(iceDates) < datetime(years[tt]+1,1,1)))
        timeSubset = np.array(iceDates)[yearInd]
        tempOfSim = simOfIce[yearInd]
        groupTemp = [[e[0] for e in d[1]] for d in
                            itertools.groupby(enumerate(tempOfSim), key=operator.itemgetter(1))]
        timeGroupTemp = [timeSubset[d] for d in groupTemp]
        bmuGroupIce = (np.asarray([tempOfSim[i[0]] for i in groupTemp]))
        withinYear = 0

        # lets a
        wavesStitched = []
        iceStitched = []
        for qq in range(len(groupTemp)):
            tempRand = np.random.uniform(size=len(groupTemp[qq]))
            randIceSize = [random.randint(0, 9999) for xx in range(len(tempRand))]
            iceDetails = fetchSims[int(bmuGroupIce[qq]-1)][randIceSize]
            if tempControl == 0:
                # lets enter the next group assuming that waves were previously off
                sortedRand = np.sort(tempRand)            # so we have a bunch of probabilities between 0 and 1
                finder = np.where((sortedRand > percentIce[int(bmuGroupIce[qq]-1)]))     # if greater than X then waves are ON
                # arbitrary, but lets say waves have to be turned on for 3 days before we say, sure the ocean broke up
                waveDays = np.zeros((len(sortedRand),))
                if timeGroupTemp[qq][0].month > 3:
                    if len(finder[0]) > 3:
                        if timeGroupTemp[qq][0].month > 10:
                            if (len(finder[0])/len(sortedRand))>0.7:
                                waveDays[finder[0]] = np.ones((len(finder[0])))  # flip those days that are waves to ON
                                tempControl = 1
                            else:
                                print('prevented wave from turning on Sim {}'.format(hh))
                        else:
                            waveDays[finder[0]] = np.ones((len(finder[0])))       # flip those days that are waves to ON
                            tempControl = 1
                    if timeGroupTemp[qq][0].month > 7 and timeGroupTemp[qq][-1].month < 9:
                        waveDays = np.ones((len(sortedRand),))
                        tempControl = 1
                # if len(wavesStitched) > 0:
                    # arrayOfWavesStitched = np.abs(np.concatenate(wavesStitched,axis=0)-1)
                    # whereDidTheyTurnOff = np.where(np.diff(arrayOfWavesStitched) == 1)
                    # if len(whereDidTheyTurnOff[0]) > 0:
                    #     timeSince = len(arrayOfWavesStitched)-whereDidTheyTurnOff[0][-1]
                    # if timeGroupTemp[qq][0].month > 10 and timeSince > 7:
                    #         waveDays = np.zeros((len(sortedRand),))
                    #         tempControl = 0

                wavesStitched.append(waveDays)
                # add iceDetails here
                if timeGroupTemp[qq][0].month < 9:
                    iceStitched.append(np.sort(iceDetails))
                else:
                    iceStitched.append(np.sort(iceDetails)[::-1])

            else:
                # now we're entering with waves ON and want to turn them OFF
                sortedRand = np.sort(tempRand)#[::-1]        # again a bunch of probabilities between 0 and 1
                finder = np.where((sortedRand > percentWaves[int(bmuGroupIce[qq]-1)]))  # if greater than X then waves are OFF
                # arbitrary, but again, waves need to be off for 3 days to really freeze up the ocean
                waveDays = np.ones((len(sortedRand),))
                if timeGroupTemp[qq][-1].month < 8 or timeGroupTemp[qq][0].month > 9:
                    if len(finder[0]) > 1:
                        waveDays[finder[0]] = np.zeros((len(finder[0])))
                        tempControl = 0
                if timeGroupTemp[qq][0].month > 1 and timeGroupTemp[qq][-1].month < 5:
                    waveDays = np.zeros((len(sortedRand),))
                    tempControl = 0

                if len(wavesStitched) > 0:
                    countOfStitched = np.sum(np.concatenate(wavesStitched,axis=0))
                    if timeGroupTemp[qq][0].month > 5 and countOfStitched > 7:
                        waveDays = np.ones((len(sortedRand),))

                wavesStitched.append(waveDays)
                if timeGroupTemp[qq][0].month < 9:
                    iceStitched.append(np.sort(iceDetails))
                else:
                    iceStitched.append(np.sort(iceDetails)[::-1])

        yearsStitched.append(wavesStitched)
        yearsIceStitched.append(iceStitched)
    iceOnOff.append(np.concatenate([x for xs in yearsStitched for x in xs]).ravel())
    iceArea.append(np.concatenate([x for xs in yearsIceStitched for x in xs]).ravel())


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# tempIceArea = iceArea[0]
# smoothedIce = running_mean(tempIceArea,30)
# smoothedIce2 = running_mean(tempIceArea,60)
#
# plt.figure()
#
# plt.plot(dates_sim,tempIceArea[0:len(dates_sim)])
# plt.plot(dates_sim[15:len(dates_sim)-15],smoothedIce[0:len(dates_sim)-30])
# plt.plot(dates_sim[30:len(dates_sim)-30],smoothedIce2[0:len(dates_sim)-60])

startIce = np.array([8,9,10,12,14,16,18,20,22,24,26,29,32,35,39,43,47,51,55,60,65,70,75,80,85,90,95,100,106,112])
endIce = np.array([10,12,13,14,15,16,17,16,15,14,13,14,15,16,16,17,18,19,20,16,17,15,14,20,22,16,18,22,26])

# contIce = np.hstack((startIce,smoothedIce2,endIce))

iceAreaSmooth = []
for n in range(100):
    print('smoothing ice area simulation {}'.format(n))
    tempIceArea = iceArea[n]
    smoothedIce2 = running_mean(tempIceArea, 60)
    contIce = np.hstack((startIce,smoothedIce2,endIce))
    iceAreaSmooth.append(contIce)
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
# #
# #
# # num_simulations = 100
# # histDates = iceDates#[214:]#onOffData['iceDateTimes']
# # bmus_dates_months = np.array([d.month for d in histDates])
# # bmus_dates_days = np.array([d.day for d in histDates])
# # yearsOfHist= np.arange(1980,2021)
# # stacked = np.zeros((len(yearsOfHist),365))
# # for qq in range(len(yearsOfHist)):
# #     index = np.where((np.array(histDates) >= datetime(yearsOfHist[qq], 1, 1)) & (
# #                 np.array(histDates) < datetime(yearsOfHist[qq], 12, 31)))
# #     yearlyStacked = np.zeros((num_simulations,len(index[0])))
# #     for tt in range(num_simulations):
# #         # index = np.where((np.array(histDates) >= datetime(yearsOfHist[qq],6,1)) & (np.array(histDates) < datetime(yearsOfHist[qq]+1,5,31)))
# #         yearlyStacked[tt,0:len(index[0])] = iceOnOff[tt][index]
# #     stacked[qq,0:len(index[0])] = np.nanmean(yearlyStacked,axis=0)
# #
# #
# # fig = plt.figure(figsize=(10,4))
# # ax = plt.subplot2grid((1,1),(0,0))
# # X,Y = np.meshgrid(list_pyear,np.arange(1980,2022))
# # ax.pcolormesh(X,Y,stacked,cmap=cmocean.cm.ice_r,vmin=0,vmax=2)
# # # customize  axis
# # months = mdates.MonthLocator()
# # monthsFmt = mdates.DateFormatter('%b')
# # ax.set_xlim(list_pyear[0], list_pyear[-1])
# # ax.xaxis.set_major_locator(months)
# # ax.xaxis.set_major_formatter(monthsFmt)
# #
#



# newOrderIce = np.array([0,0,0,1,1,1,0,1,0,0,0,1,0,1,1,0,1,1,1,1,1,1,1,1,1])
# newOrderIce = np.array([0,0,0,0,0,
#                         0,0,0,0,0,
#                         0,0,0,1,1,
#                         0,1,1,1,1,
#                         1,1,1,1,1])
newOrderIce = np.array([0,0,0,1,0,
                        0,0,0,0,0,
                        0,0,0,0,1,
                        0,1,1,1,1,
                        1,1,1,1,1])
# newOrderIce = np.array([0,0,0,1,1,
#                         1,0,1,0,0,
#                         0,1,0,1,2,
#                         1,2,2,2,2,
#                         2,2,2,2,2])


dt = datetime(1979,6, 1)
end = datetime(2022, 6, 1)
# dt = datetime(2021,6, 1)
# end = datetime(2121, 6, 1)
step = timedelta(days=1)
midnightTime = []
while dt < end:
    midnightTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step


bmus_dates_months = np.array([d.month for d in midnightTime])

groupedList = list()
groupLengthList = list()
bmuGroupList = list()
iceOnOffGroupList = list()
iceAreaGroupList = list()
iceSlpGroupList = list()
timeGroupList = list()
for hh in range(100):
    print('breaking up hydrogrpahs for simulation {}'.format(hh))
    bmus = evbmus_sim[:,hh]
    iceBmus = iceOnOff[hh]
    actualIceBmus = iceSims[:,hh]
    iceAreaBmus = iceAreaSmooth[hh]

    tempBmusGroup = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(bmus), key=operator.itemgetter(1))]

    bmusICESLP = np.zeros((len(actualIceBmus),), ) * np.nan
    for i in range(2):
        posc = np.where(newOrderIce == i)
        for hh in range(len(posc[0])):
            posc2 = np.where(actualIceBmus == posc[0][hh])
            bmusICESLP[posc2] = i

    groupedList.append(tempBmusGroup)
    groupLengthList.append(np.asarray([len(i) for i in tempBmusGroup]))
    bmuGroupList.append(np.asarray([bmus[i[0]] for i in tempBmusGroup]))
    iceOnOffGroupList.append([iceBmus[i] for i in tempBmusGroup])
    iceAreaGroupList.append([iceAreaBmus[i] for i in tempBmusGroup])
    iceSlpGroupList.append([bmusICESLP[i] for i in tempBmusGroup])
    timeGroupList.append([bmus_dates_months[i] for i in tempBmusGroup])



### TODO: use the historical to develop a new chronology

numRealizations = 100


simBmuChopped = []
simBmuLengthChopped = []
simBmuGroupsChopped = []
simIceGroupsChopped = []
simIceSlpGroupsChopped = []
simIceAreaGroupsChopped = []
simTimeGroupsChopped = []
for pp in range(numRealizations):

    print('working on realization #{}'.format(pp))
    bmuGroup = bmuGroupList[pp]
    groupLength = groupLengthList[pp]
    grouped = groupedList[pp]
    iceGroup = iceOnOffGroupList[pp]
    iceSlpGroup = iceSlpGroupList[pp]
    iceAreaGroup = iceAreaGroupList[pp]
    timeGroup = timeGroupList[pp]

    simGroupLength = []
    simGrouped = []
    simBmu = []
    simIceGroupLength = []
    simIceGrouped = []
    simIceBmu = []
    simIceSlpGrouped = []
    simIceAreaGrouped = []
    simTimeGrouped = []
    for i in range(len(groupLength)):
        # if np.remainder(i,10000) == 0:
        #     print('done with {} hydrographs'.format(i))
        tempGrouped = grouped[i]
        tempBmu = int(bmuGroup[i])
        tempIceGrouped = iceGroup[i]
        tempIceSlpGrouped = iceSlpGroup[i]
        tempTimeGrouped = timeGroup[i]
        tempIceAreaGrouped = iceAreaGroup[i]
        remainingDays = groupLength[i]
        counter = 0
        # if groupLength[i] < 2:
        #     simGroupLength.append(int(groupLength[i]))
        #     simGrouped.append(grouped[i])
        #     simIceGrouped.append(iceGroup[i])
        #     simIceSlpGrouped.append(iceSlpGroup[i])
        #     simTimeGrouped.append(timeGroup[i])
        #     simBmu.append(tempBmu)
        # else:
        #     counter = 0
        while (len(grouped[i]) - counter) > 1:
                # print('we are in the loop with remainingDays = {}'.format(remainingDays))
                # random days between 3 and 5
            randLength = 1#random.randint(1, 2)
                # add this to the record
            simGroupLength.append(int(randLength))
                # simGrouped.append(tempGrouped[0:randLength])
                # print('should be adding {}'.format(grouped[i][counter:counter+randLength]))
            simGrouped.append(grouped[i][counter:counter+randLength])
            simIceGrouped.append(iceGroup[i][counter:counter+randLength])
            simIceSlpGrouped.append(iceSlpGroup[i][counter:counter+randLength])
            simTimeGrouped.append(timeGroup[i][counter:counter+randLength])
            simIceAreaGrouped.append(iceAreaGroup[i][counter:counter+randLength])
            simBmu.append(tempBmu)
                # remove those from the next step
                # tempGrouped = np.delete(tempGrouped,np.arange(0,randLength))
                # do we continue forward
            remainingDays = remainingDays - randLength
            counter = counter + randLength

            # if (len(grouped[i]) - counter) > 0:
        simGroupLength.append(int((len(grouped[i]) - counter)))
                # simGrouped.append(tempGrouped[0:])
        simGrouped.append(grouped[i][counter:])
        simIceGrouped.append(iceGroup[i][counter:])
        simIceSlpGrouped.append(iceSlpGroup[i][counter:])
        simTimeGrouped.append(timeGroup[i][counter:])
        simIceAreaGrouped.append(iceAreaGroup[i][counter:])

        simBmu.append(tempBmu)

    simBmuLengthChopped.append(np.asarray(simGroupLength))
    simBmuGroupsChopped.append(simGrouped)
    simIceGroupsChopped.append(simIceGrouped)
    simBmuChopped.append(np.asarray(simBmu))
    simIceSlpGroupsChopped.append(simIceSlpGrouped)
    simIceAreaGroupsChopped.append(simIceAreaGrouped)
    simTimeGroupsChopped.append(simTimeGrouped)





simsChoppedPickle = 'simulations100Chopped49_2Dist.pickle'
outputSimsChopped = {}
outputSimsChopped['simBmuLengthChopped'] = simBmuLengthChopped
outputSimsChopped['simBmuGroupsChopped'] = simBmuGroupsChopped
outputSimsChopped['simBmuChopped'] = simBmuChopped
outputSimsChopped['simIceGroupsChopped'] = simIceGroupsChopped
outputSimsChopped['simIceSlpGroupsChopped'] = simIceSlpGroupsChopped
outputSimsChopped['simTimeGroupsChopped'] = simTimeGroupsChopped
outputSimsChopped['simIceAreaGroupsChopped'] = simIceAreaGroupsChopped

with open(simsChoppedPickle,'wb') as f:
    pickle.dump(outputSimsChopped, f)


