import scipy.io
import h5py
import mat73
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from datetime import timedelta
import itertools
import operator

# define some constants
epoch = dt.datetime(1970, 1, 1)
matlab_to_epoch_days = 719529  # days from 1-1-0000 to 1-1-1970
matlab_to_epoch_seconds = matlab_to_epoch_days * 24 * 60 * 60

def matlab_to_datetime(matlab_date_num_seconds):
    # get number of seconds from epoch
    from_epoch = matlab_date_num_seconds - matlab_to_epoch_seconds

    # convert to python datetime
    return epoch + dt.timedelta(seconds=from_epoch)

def dateDay2datetimeDate(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [dt.date(d[0], d[1], d[2]) for d in d_vec]

def dateDay2datetime(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [dt.datetime(int(d[0]), int(d[1]), int(d[2])) for d in d_vec]


def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return dt.datetime.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           + timedelta(hours=int(hours)) \
           + timedelta(minutes=int(minutes)) \
           + timedelta(seconds=round(seconds)) \
           - timedelta(days=366)


def datenum_to_date(datenum):
    """
    Convert Matlab datenum into Python date.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return dt.datetime.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           - timedelta(days=366)
           #+ timedelta(hours=int(hours)) \
           #+ timedelta(minutes=int(minutes)) \
           #+ timedelta(seconds=round(seconds)) \


def dt2cal(dt):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.

    Parameters
    ----------
    dt : datetime64 array (...)
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 7)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """

    # allocate output
    out = np.empty(dt.shape + (7,), dtype="u4")
    # decompose calendar floors
    Y, M, D, h, m, s = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = Y + 1970 # Gregorian Year
    out[..., 1] = (M - Y) + 1 # month
    out[..., 2] = (D - M) + 1 # dat
    out[..., 3] = (dt - D).astype("m8[h]") # hour
    out[..., 4] = (dt - h).astype("m8[m]") # minute
    out[..., 5] = (dt - m).astype("m8[s]") # second
    out[..., 6] = (dt - s).astype("m8[us]") # microsecond
    return out


waveData = mat73.loadmat('era5_waves_pthope.mat')
wlData = mat73.loadmat('pthope_tides_for_dylan.mat')
environData = mat73.loadmat('updated_env_for_dylan.mat')
environDates = [datenum_to_date(t) for t in environData['time']]
environTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in environData['time']]




wlArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in wlData['time_all']]
waveArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in waveData['time_all']]


waveDates = [datenum_to_date(t) for t in waveData['time_all']]

plt.style.use('dark_background')
plt.figure()
ax1 = plt.subplot2grid((4,1),(0,0))
ax2 = plt.subplot2grid((4,1),(1,0))
ax3 = plt.subplot2grid((4,1),(2,0))
ax4 = plt.subplot2grid((4,1),(3,0))

allWLs = environData['wl']
wlsInd = np.where(allWLs < -2.5)
allWLs[wlsInd] = np.zeros((len(wlsInd),))

# ax1.plot(wlArrayTime,wlData['wl_all'])
ax1.plot(environTime,allWLs)

ax1.set_xlim([dt.datetime(1979,1,1), dt.datetime(2022,6,1)])
ax1.set_ylabel('wl (m)')
ax2.plot(waveArrayTime,waveData['wh_all'])
# ax2.plot(environTime,environData['hs'])
ax2.set_xlim([dt.datetime(1979,1,1), dt.datetime(2022,6,1)])
ax2.set_ylabel('hs (m)')

ax3.plot(waveArrayTime,waveData['tp_all'])
# ax3.plot(environTime,environData['tp'])
ax3.set_xlim([dt.datetime(1979,1,1), dt.datetime(2022,6,1)])
ax3.set_ylabel('tp (s)')

ax4.plot(waveArrayTime,waveData['mwd_all'])
# ax4.plot(environTime,environData['wd'])
ax4.set_xlim([dt.datetime(1979,1,1), dt.datetime(2022,6,1)])
ax4.set_ylabel('mwd (deg)')


asfg
# Can we identify which days have waves and which don't...

import pickle
with open(r"dwts49ClustersArctic2y2022.pickle", "rb") as input_file:
   slpDWTs = pickle.load(input_file)

# with open(r"iceData25Clusters.pickle", "rb") as input_file:
#    iceDWTs = pickle.load(input_file)
with open(r"iceData25ClustersMediumishMin150withDayNumRG.pickle", "rb") as input_file:
   iceDWTs = pickle.load(input_file)


dayTime = slpDWTs['SLPtime']
iceTime = iceDWTs['dayTime']#[np.array((iceDWTs['year'][i],iceDWTs['month'][i],iceDWTs['day'][i])) for i in range(len(iceDWTs['year']))]

slpDates = dateDay2datetime(dayTime)
slpDatetimes = dateDay2datetime(dayTime)
iceDateTimes = iceTime#dateDay2datetime(iceTime)
iceDateTimes = iceDateTimes[31:]
iceBmus = iceDWTs['bmus_corrected'][31:]
iceDWT = iceBmus



#
# slpDaysWithNoIce = [x for x in slpDatetimes if x not in iceDateTimes]
# ind_dict = dict((k,i) for i,k in enumerate(slpDatetimes))
# inter = set(slpDaysWithNoIce).intersection(slpDatetimes)
# indices = [ ind_dict[x] for x in inter ]
# indices.sort()
#
# slpDaysWithIce = [x for x in slpDatetimes if x in iceDateTimes]
# ind_dict = dict((k,i) for i,k in enumerate(slpDatetimes))
# inter2 = set(slpDaysWithIce).intersection(slpDatetimes)
# indices2 = [ ind_dict[x] for x in inter2]
# indices2.sort()
#
# gapFilledIce = np.nan * np.ones((len(slpDatetimes),))
# gapFilledIce[indices] = iceBmus[0:len(indices)]
# gapFilledIce[indices2] = iceBmus
#
# iceDWT = gapFilledIce
slpDWT = slpDWTs['bmus_corrected']


## lets get a binary of waves on/off

onOff = np.nan * np.ones((len(iceDWT)))

for hh in range(len(onOff)):
    #ind = np.where((np.array(waveArrayTime) >= dt.datetime(dayTime[hh,0],dayTime[hh,1],dayTime[hh,2],0,0,0)) &
     #              (np.array(waveArrayTime) <= dt.datetime(dayTime[hh,0],dayTime[hh,1],dayTime[hh,2],23,0,0)))
    ind = np.where((np.array(waveDates) == slpDates[hh]))
    onOff[hh] = np.nanmean(waveData['wh_all'][ind])
     #              (np.array(waveArrayTime) <= dt.datetime(dayTime[hh,0],dayTime[hh,1],dayTime[hh,2],23,0,0)))
    # if np.isnan(np.nanmean(waveData['wh_all'][ind])):
    #     onOff[hh] = 0
    # else:
    #     onOff[hh] = 1

plt.plot(slpDates[0:len(onOff)],onOff)
# plt.plot(iceDateTimes,onOff)

wavesOnOff = np.ones((len(onOff),))
nanInd = np.isnan((onOff))
wavesOnOff[nanInd] = 0

percentWaves = np.nan*np.ones((len(np.unique(iceDWT)),))
percentIce = np.nan*np.ones((len(np.unique(iceDWT)),))

for qq in range(len(np.unique(iceDWT))):
    getBMUS = np.where((iceDWT == qq))
    temp = wavesOnOff[getBMUS]
    getWaves = np.where((temp == 1))
    getIce = np.where((temp == 0))
    percentWaves[qq] = len(getWaves[0])/len(getBMUS[0])
    percentIce[qq] = len(getIce[0])/len(getBMUS[0])

from datetime import datetime

import matplotlib.cm as cm
dwtcolors = cm.rainbow(np.linspace(0, 1,36))

years = np.arange(1979,2022)
iceGone = []
iceBack = []
iceGoneDatetime = []
iceBackDatetime = []
wavesOn = []
yearlyIceDWT = np.nan * np.ones((len(np.unique(iceDWT)),len(years)))
for tt in range(len(years)):
    yearInd = np.where((np.array(waveArrayTime) >= datetime(years[tt],1,1)) &
                       (np.array(waveArrayTime) <= datetime(years[tt]+1,1,1)))
    subsetWaves = waveData['wh_all'][yearInd]
    icePortion = np.where(np.isnan(subsetWaves))
    wavePortion = np.where((subsetWaves > 0))
    iceGoneDatetime.append(waveArrayTime[wavePortion[0][0]])
    iceBackDatetime.append(waveArrayTime[wavePortion[0][-1]])
    iceGone.append(wavePortion[0][0])
    iceBack.append(wavePortion[0][-1])
    wavesOn.append((wavePortion[0][-1]-wavePortion[0][0])/24)
    iceYearInd = np.where((np.array(slpDatetimes) >= datetime(years[tt],1,1)) &
                       (np.array(slpDatetimes) <= datetime(years[tt]+1,1,1)))
    tempIce = iceDWT[iceYearInd]
    for qq in range(len(np.unique(iceDWT))):
        dwtInd = np.where((tempIce == qq))
        yearlyIceDWT[qq,tt] = len(dwtInd[0])

plt.figure()
ax1 = plt.subplot2grid((3,1),(0,0))
ax1.plot(years,wavesOn)
ax1.set_ylabel('Days with Waves')
ax2 = plt.subplot2grid((3,1),(1,0))
ax2.plot(years,iceGoneDatetime)
ax2.set_ylabel('Waves Turn On')
ax3 = plt.subplot2grid((3,1),(2,0))
ax3.plot(years,iceBackDatetime)
ax3.set_ylabel('Waves Turn Off')

timeAsArray = np.array(slpDatetimes[0:len(onOff)])
plt.figure()
ax4 = plt.subplot2grid((1,1),(0,0))
for qq in range(len(np.unique(iceDWT))):
    getBMUS = np.where((iceDWT == qq))
    temp = iceDWT[getBMUS]
    tempTime = timeAsArray[getBMUS]
    ax4.plot(np.array(slpDatetimes)[getBMUS[0]],qq*np.ones((len(temp),)),'.',color=dwtcolors[iceDWTs['kma_order'][qq]])



# so we should figure out the relative percentage of each ice DWT with waves on/off
dt = datetime(1979, 1, 31)
end = datetime(2022, 1, 1)
step = timedelta(days=1)
midnightTime = []
while dt < end:
    midnightTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step

grouped = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(iceDWT), key=operator.itemgetter(1))]

groupLength = np.asarray([len(i) for i in grouped])
bmuGroup = np.asarray([iceDWT[i[0]] for i in grouped])
# timeGroup = [np.asarray(midnightTime)[i] for i in grouped]
timeGroup = [np.array(iceDateTimes)[i] for i in grouped]

startTimes = [i[0] for i in timeGroup]
endTimes = [i[-1] for i in timeGroup]


import pickle

dwtPickle = 'processedWaveWaterLevels49DWTs25IWTs2022.pickle'
outputDWTs = {}
outputDWTs['wavesOnOff'] = wavesOnOff
outputDWTs['onOff'] = onOff

outputDWTs['slpDates'] = slpDates
outputDWTs['iceDWT'] = iceDWT
outputDWTs['slpDWT'] = slpDWT
outputDWTs['slpDatetimes'] = slpDatetimes
outputDWTs['iceDateTimes'] = iceDateTimes
outputDWTs['iceBmus'] = iceBmus
outputDWTs['wlArrayTime'] = wlArrayTime
outputDWTs['waveArrayTime'] = waveArrayTime
outputDWTs['waveDates'] = waveDates
outputDWTs['allWLs'] = allWLs
outputDWTs['environTime'] = environTime
outputDWTs['percentWaves'] = percentWaves
outputDWTs['percentIce'] = percentIce
outputDWTs['iceGone'] = iceGone
outputDWTs['iceBack'] = iceBack
outputDWTs['iceGoneDatetime'] = iceGoneDatetime
outputDWTs['iceBackDatetime'] = iceBackDatetime
outputDWTs['wavesOn'] = wavesOn
outputDWTs['yearlyIceDWT'] = yearlyIceDWT



with open(dwtPickle,'wb') as f:
    pickle.dump(outputDWTs, f)


