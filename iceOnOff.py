
import numpy as np
import pickle
with open(r"iceData25ClustersAllArcticRgSeasonality100min.pickle", "rb") as input_file:
   historicalDWTs = pickle.load(input_file)

order = historicalDWTs['kma_order']
gapFilledIce = historicalDWTs['gapFilledIce']
iceBmus = historicalDWTs['bmus_corrected']
iceTime = historicalDWTs['dayTime']
timeArray = np.array(iceTime)
kma_order = historicalDWTs['kma_order']
kmSorted = historicalDWTs['kmSorted']
xFlat = historicalDWTs['xFlat']
points = historicalDWTs['points']
group_size = historicalDWTs['group_size']
x = historicalDWTs['x']
y = historicalDWTs['y']
num_clusters = historicalDWTs['num_clusters']




import sys
sys.path.append('/users/dylananderson/Documents/projects/gencadeClimate/')
import pickle
with open(r"/Users/dylananderson/Documents/projects/duneLifeCycles/ptLayWavesWinds164by70.pickle", "rb") as input_file:
    ptLayWavesWinds = pickle.load(input_file)

ptLayWaves = ptLayWavesWinds['ptLayWaves']
endTime = ptLayWavesWinds['endTime']
startTime = ptLayWavesWinds['startTime']

import datetime as dt
from dateutil.relativedelta import relativedelta
st = dt.datetime(startTime[0], startTime[1], startTime[2])
end = dt.datetime(endTime[0],endTime[1]+1,1)
step = relativedelta(hours=1)
hourTime = []
while st < end:
    hourTime.append(st)
    st += step


from datetime import datetime, date, timedelta

# tWave = ptLayWaves.timeWave
tWave = np.array([datetime.utcfromtimestamp((qq - np.datetime64(0, 's')) / np.timedelta64(1, 's'))for qq in ptLayWaves.timeWave])
tC = tWave


waveHieghts = []
for hh in range(num_clusters):
    dwtInd = np.where((iceBmus==hh))
    waveBin = []
    for qq in range(len(dwtInd[0])):
        dayInd = np.where((tC >= datetime(iceTime[dwtInd[0][qq]].year, iceTime[dwtInd[0][qq]].month,iceTime[dwtInd[0][qq]].day,0,0,0)) &
                          (tC <= datetime(timeArray[dwtInd[0][qq]].year, timeArray[dwtInd[0][qq]].month,
                                               timeArray[dwtInd[0][qq]].day,23,0,0)))
        waveBin.append(ptLayWaves.Hs[dayInd[0]])
    waveHieghts.append(np.concatenate(waveBin,axis=0))





import itertools
import operator


dt = iceTime[0]
end = iceTime[-1]
step = timedelta(days=1)
midnightTime = []
while dt < end:
    midnightTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step
midnightTime.append(dt)

# bmus = newBmus[0:len(midnightTime)]

# newBmus
# bmus = bmus[0:len(midnightTime)]

grouped = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(iceBmus), key=operator.itemgetter(1))]

groupLength = np.asarray([len(i) for i in grouped])
bmuGroup = np.asarray([iceBmus[i[0]] for i in grouped])
timeGroup = [np.asarray(midnightTime)[i] for i in grouped]

# fetchGroup = np.asarray([fetchFiltered[i[0]] for i in grouped])

startTimes = [i[0] for i in timeGroup]
endTimes = [i[-1] for i in timeGroup]

hydrosInds = np.unique(bmuGroup)

hsCombined = ptLayWaves.Hs
tpCombined = ptLayWaves.Tp
dmCombined = ptLayWaves.Dm

hydros = list()
c = 0
for p in range(len(np.unique(bmuGroup))):

    tempInd = p#hydrosInds[p]
    print('working on bmu = {}'.format(tempInd))
    index = np.where((bmuGroup==tempInd))[0][:]
    if len(index) == 0:
        print('no waves here')
        tempList = list()
    else:

        print('should have at least {} days in it'.format(len(index)))
        subLength = groupLength[index]
        m = np.ceil(np.sqrt(len(index)))
        tempList = list()


        for i in range(len(index)):
            st = startTimes[index[i]]
            et = endTimes[index[i]] + timedelta(days=1)

            # tempFetch = fetchGroup[index[i]]#np.array([fetchFiltered[index[i]]])

            if et > datetime(2023,8,31):
                print('done up to 2022/5/31')
            else:
                waveInd = np.where((tWave < et) & (tWave >= st))
                # waterInd = np.where((tWl < et) & (tWl >= st))
                if len(waveInd[0]) > 0:
                    newTime = startTimes[index[i]]

                    tempHydroLength = subLength[i]
                    counter = 0
                    while tempHydroLength > 1:
                        #randLength = random.randint(1,2)
                        etNew = newTime + timedelta(days=1)
                        waveInd = np.where((tWave < etNew) & (tWave >= newTime))
                        # fetchInd = np.where((np.asarray(dayTime) == newTime))

                        # waterInd = np.where((tWl < etNew) & (tWl >= st))
                        deltaDays = et-etNew
                        #print('After first cut off = {}'.format(deltaDays.days))
                        c = c + 1
                        counter = counter + 1
                        if counter > 65:
                            print('we have split 15 times')

                        tempDict = dict()
                        tempDict['time'] = tWave[waveInd[0]]
                        tempDict['numDays'] = subLength[i]
                        tempDict['hs'] = hsCombined[waveInd[0]]
                        tempDict['tp'] = tpCombined[waveInd[0]]
                        tempDict['dm'] = dmCombined[waveInd[0]]
                        # tempDict['fetch'] = fetchTotalConcentration[fetchInd[0][0]]#tempFetch[0]
                        # tempDict['res'] = res[waterInd[0]]
                        # tempDict['wl'] = wlHycom[waterInd[0]]
                        tempDict['cop'] = np.asarray([np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                                                      np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                                                      np.nanmean(dmCombined[waveInd[0]])])#,fetchTotalConcentration[fetchInd[0][0]]])#, np.nanmean(res[waterInd[0]])])
                        tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                        tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                        tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                        tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                        tempDict['dmMean'] = np.nanmean(dmCombined[waveInd[0]])
                        # tempDict['ssMean'] = np.nanmean(res[waterInd[0]])
                        tempList.append(tempDict)
                        tempHydroLength = tempHydroLength-1
                        newTime = etNew
                    else:
                        waveInd = np.where((tWave < et) & (tWave >= newTime))
                        # fetchInd = np.where((np.asarray(dayTime) == newTime))

                        # waterInd = np.where((tWl < et) & (tWl >= etNew7))
                        c = c + 1
                        tempDict = dict()
                        tempDict['time'] = tWave[waveInd[0]]
                        tempDict['numDays'] = subLength[i]
                        tempDict['hs'] = hsCombined[waveInd[0]]
                        tempDict['tp'] = tpCombined[waveInd[0]]
                        tempDict['dm'] = dmCombined[waveInd[0]]
                        # tempDict['fetch'] = fetchTotalConcentration[fetchInd[0][0]]#tempFetch[0]

                        # tempDict['res'] = res[waterInd[0]]
                        # tempDict['wl'] = wlHycom[waterInd[0]]
                        tempDict['cop'] = np.asarray(
                            [np.nanmin(hsCombined[waveInd[0]]),
                             np.nanmax(hsCombined[waveInd[0]]),
                             np.nanmin(tpCombined[waveInd[0]]),
                             np.nanmax(tpCombined[waveInd[0]]),
                             np.nanmean(
                                 dmCombined[waveInd[0]])])#,fetchTotalConcentration[fetchInd[0][0]]])  # ,
                        # np.nanmean(res[waterInd[0]])])
                        tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                        tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                        tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                        tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                        tempDict['dmMean'] = np.nanmean(dmCombined[waveInd[0]])
                        # tempDict['ssMean'] = np.nanmean(res[waterInd[0]])
                        tempList.append(tempDict)
    hydros.append(tempList)





