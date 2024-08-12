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
from scipy.stats import gumbel_l, genextreme
from scipy.spatial import distance
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
   return [dt.date(int(d[0]), int(d[1]), int(d[2])) for d in d_vec]

def dateDay2datetime(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [dt.datetime(int(d[0]), int(d[1]), int(d[2])) for d in d_vec]

import datetime as dt
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




import scipy.io as sio

def ReadMatfile(p_mfile):
    'Parse .mat file to nested python dictionaries'

    def RecursiveMatExplorer(mstruct_data):
        # Recursive function to extrat mat_struct nested contents

        if isinstance(mstruct_data, mat_struct):
            # mstruct_data is a matlab structure object, go deeper
            d_rc = {}
            for fn in mstruct_data._fieldnames:
                d_rc[fn] = RecursiveMatExplorer(getattr(mstruct_data, fn))
            return d_rc

        else:
            # mstruct_data is a numpy.ndarray, return value
            return mstruct_data

    # base matlab data will be in a dict
    mdata = sio.loadmat(p_mfile, squeeze_me=True, struct_as_record=False)
    mdata_keys = [x for x in mdata.keys() if x not in
                  ['__header__','__version__','__globals__']]

    # use recursive function
    dout = {}
    for k in mdata_keys:
        dout[k] = RecursiveMatExplorer(mdata[k])
    return dout

def dateDay2datetimeDate(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [date(int(d[0]), int(d[1]), int(d[2])) for d in d_vec]

def dateDay2datetime(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [datetime(int(d[0]), int(d[1]), int(d[2])) for d in d_vec]






with open(r"dwts49ClustersArctic2023.pickle", "rb") as input_file:
    historicalDWTs = pickle.load(input_file)

numClusters = 49
timeDWTs = historicalDWTs['SLPtime']
# outputDWTs['slpDates'] = slpDates
bmus = historicalDWTs['bmus_corrected']

bmus_dates = dateDay2datetimeDate(timeDWTs)
bmus_dates_months = np.array([d.month for d in bmus_dates])
bmus_dates_days = np.array([d.day for d in bmus_dates])


# with open(r"ice18FutureSimulations1000PointHope.pickle", "rb") as input_file:
# with open(r"ice18FutureSimulations1000Shishmaref.pickle", "rb") as input_file:
with open(r"ice18FutureSimulations1000Wainwright.pickle", "rb") as input_file:
# with open(r"ice18FutureSimulations1000Wales.pickle", "rb") as input_file:
# with open(r"ice18FutureSimulations1000Wevok.pickle", "rb") as input_file:
# with open(r"ice18FutureSimulations1000PointLay.pickle", "rb") as input_file:
# with open(r"ice18FutureSimulations1000Kivalina.pickle", "rb") as input_file:

    historicalICEs = pickle.load(input_file)

bmusIce = historicalICEs['bmus']
timeIce = historicalICEs['dayTime']
timeArrayIce = np.array(timeIce)
areaBelow = historicalICEs['areaBelow']
bmus_corrected = bmusIce


# with open(r"historicalNTRDataPointHope.pickle", "rb") as input_file:
# with open(r"historicalNTRDataShishmaref.pickle", "rb") as input_file:
with open(r"historicalNTRDataWainwright.pickle", "rb") as input_file:
# with open(r"historicalNTRDataWales.pickle", "rb") as input_file:
# with open(r"historicalNTRDataWevok.pickle", "rb") as input_file:
# with open(r"historicalNTRDataPointLay.pickle", "rb") as input_file:
# with open(r"historicalNTRDataKivalina.pickle", "rb") as input_file:
    historicalWLs = pickle.load(input_file)

# bmusIce = historicalICEs['bmus']
tNTR = historicalWLs['time']
ntr = historicalWLs['ntr1']
# ntr2 = historicalWLs['ntr2']
# timeArrayIce = np.array(timeIce)
# areaBelow = historicalICEs['areaBelow']
# bmus_corrected = bmusIce


# bmus_corrected = np.zeros((len(bmusIce),), ) * np.nan
# for i in range(2):
#     posc = np.where(newOrderIce == i)
#     for hh in range(len(posc[0])):
#         posc2 = np.where(bmusIce == posc[0][hh])
#         bmus_corrected[posc2] = i






import sys
sys.path.append('/users/dylananderson/Documents/projects/gencadeClimate/')
import pickle
with open(r"/Users/dylananderson/Documents/projects/arcticClimate/utqiagvikWavesWindsTemps157pt25by71pt5.pickle","rb") as input_file:
# with open(r"/Users/dylananderson/Documents/projects/arcticClimate/pointHopeWavesWindsTemps166pt75by68pt5.pickle","rb") as input_file:
# with open(r"/Users/dylananderson/Documents/projects/arcticClimate/shishmarefWavesWindsTemps166pt5by66pt5.pickle","rb") as input_file:
# with open(r"/Users/dylananderson/Documents/projects/arcticClimate/wainwrightWavesWindsTemps160pt5by70pt75.pickle","rb") as input_file:
# with open(r"/Users/dylananderson/Documents/projects/arcticClimate/walesNorthWavesWindsTemps168pt5by66.pickle","rb") as input_file:
# with open(r"/Users/dylananderson/Documents/projects/arcticClimate/wevokWavesWindsTemps166by69pt25.pickle","rb") as input_file:
# with open(r"/Users/dylananderson/Documents/projects/arcticClimate/pointLayWavesWindsTemps164by70.pickle", "rb") as input_file:
# with open(r"/Users/dylananderson/Documents/projects/arcticClimate/kivalina2WavesWindsTemps165by67pt5.pickle","rb") as input_file:
    wavesWinds = pickle.load(input_file)

# ptLayWaves = ptLayWavesWinds['metOcean']
endTime = wavesWinds['endTime']
startTime = wavesWinds['startTime']

import datetime as dt
from dateutil.relativedelta import relativedelta
st = dt.datetime(startTime[0], startTime[1], startTime[2])
end = dt.datetime(endTime[0],endTime[1]+1,1)
step = relativedelta(hours=1)
hourTime = []
while st < end:
    hourTime.append(st)#.strftime('%Y-%m-%d'))
    st += step

# import datetime as dt
# waveDates = [dt.datetime(t.year,t.month,t.day) for t in hourTime]

beginTime = np.where((np.asarray(hourTime) == datetime(bmus_dates[0].year,bmus_dates[0].month,bmus_dates[0].day,0,0)))
endingTime = np.where((np.asarray(hourTime) == datetime(bmus_dates[-1].year,bmus_dates[-1].month,bmus_dates[-1].day,0,0)))

wh = wavesWinds['metOcean'].Hs[beginTime[0][0]:endingTime[0][0]+24]
tp = wavesWinds['metOcean'].Tp[beginTime[0][0]:endingTime[0][0]+24]
dm = wavesWinds['metOcean'].Dm[beginTime[0][0]:endingTime[0][0]+24]
u10 = wavesWinds['metOcean'].u10[beginTime[0][0]:endingTime[0][0]+24]
v10 = wavesWinds['metOcean'].v10[beginTime[0][0]:endingTime[0][0]+24]
sst = wavesWinds['metOcean'].sst[beginTime[0][0]:endingTime[0][0]+24]
ssr = wavesWinds['metOcean'].ssr[beginTime[0][0]:endingTime[0][0]+24]
t2m = wavesWinds['metOcean'].t2m[beginTime[0][0]:endingTime[0][0]+24]

# # Point Hope
# waveNorm = dm - 334
# # Shishmaref
# waveNorm = dm - 326
# # Wainwright
# waveNorm = dm - 308
# # Wales
# waveNorm = dm - 327
# # Wevok
# waveNorm = dm - 11
# # Point Lay
# waveNorm = dm - 280
# # Kivalina
# waveNorm = dm - 225
# Utqiagvik
waveNorm = dm - 300
neg = np.where((waveNorm > 180))
waveNorm[neg[0]] = waveNorm[neg[0]]-360
neg2 = np.where((waveNorm < -180))
waveNorm[neg2[0]] = waveNorm[neg2[0]]+360
dmOG = dm
dm = waveNorm

time_all = np.asarray([datetime.utcfromtimestamp((qq - np.datetime64(0, 's')) / np.timedelta64(1, 's'))for qq in wavesWinds['metOcean'].timeWave])[beginTime[0][0]:endingTime[0][0]+24]

# wlData = mat73.loadmat('updated_env_for_dylan.mat')
# wlArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in wlData['time']]





def hourlyVec2datetime(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [datetime(d[0], d[1], d[2], d[3]) for d in d_vec]

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)



# realWavesPickle = 'realWavesPointHope.pickle'
# realWavesPickle = 'realWavesShishmaref.pickle'
# realWavesPickle = 'realWavesWainwright.pickle'
# realWavesPickle = 'realWavesWales.pickle'
# realWavesPickle = 'realWavesWevok.pickle'
# realWavesPickle = 'realWavesPointLay.pickle'
# realWavesPickle = 'realWavesKivalina.pickle'
realWavesPickle = 'realWavesUtqiagvik.pickle'

outputReal = {}
outputReal['tWave'] = time_all
outputReal['hsCombined'] = wh
outputReal['tpCombined'] = tp
outputReal['dmCombined'] = dmOG
outputReal['waveNorm'] = waveNorm
outputReal['ntr'] = ntr
outputReal['tNTR'] = tNTR
# outputReal['res'] = res

with open(realWavesPickle, 'wb') as f:
    pickle.dump(outputReal, f)





dt = datetime(bmus_dates[0].year, bmus_dates[0].month, bmus_dates[0].day)
end = datetime(bmus_dates[-1].year,bmus_dates[-1].month+1,1)
step = timedelta(days=1)
midnightTime = []
while dt < end:
    midnightTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step

asdfg

grouped = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(bmus), key=operator.itemgetter(1))]

groupLength = np.asarray([len(i) for i in grouped])
bmuGroup = np.asarray([bmus[i[0]] for i in grouped])
timeGroup = [np.asarray(midnightTime)[i] for i in grouped]

startTimes = [i[0] for i in timeGroup]
endTimes = [i[-1] for i in timeGroup]

hydrosInds = np.unique(bmuGroup)

hydros = list()
c = 0
for p in range(len(np.unique(bmuGroup))):

    tempInd = p
    print('working on bmu = {}'.format(tempInd))
    index = np.where((bmuGroup==tempInd))[0][:]

    print('should have at least {} days in it'.format(len(index)))
    subLength = groupLength[index]
    m = np.ceil(np.sqrt(len(index)))
    tempList = list()
    counter = 0
    for i in range(len(index)):
        st = startTimes[index[i]]
        et = endTimes[index[i]] + timedelta(days=1)

        waveInd = np.where((time_all < et) & (time_all >= st))
        ntrInd = np.where((tNTR < et) & (tNTR >= st))

        if len(waveInd[0]) > 0:
            newTime = startTimes[index[i]]

            tempHydroLength = subLength[i]

            while tempHydroLength > 1:
                # randLength = random.randint(1, 2)
                etNew = newTime + timedelta(days=1)
                # if etNew >= et:
                #     etNew = newTime + timedelta(days=1)
                waveInd = np.where((time_all < etNew) & (time_all >= newTime))
                fetchInd = np.where((np.asarray(timeArrayIce) == newTime))
                ntrInd = np.where((tNTR < etNew) & (tNTR >= newTime))

                deltaDays = et-etNew
                c = c + 1
                counter = counter + 1
                # if counter > 15:
                #     print('we have split 15 times')

                tempDict = dict()
                tempDict['time'] = time_all[waveInd[0]]
                tempDict['numDays'] = subLength[i]
                tempDict['hs'] = wh[waveInd[0]]
                tempDict['tp'] = tp[waveInd[0]]
                tempDict['dm'] = dm[waveInd[0]]
                tempDict['v10'] = v10[waveInd[0]]
                tempDict['u10'] = u10[waveInd[0]]
                tempDict['sst'] = sst[waveInd[0]]
                tempDict['ssr'] = ssr[waveInd[0]]
                tempDict['t2m'] = t2m[waveInd[0]]
                tempDict['ntr'] = ntr[ntrInd[0]]

                tempDict['fetch'] = areaBelow[fetchInd[0][0]]
                tempDict['cop'] = np.asarray([np.nanmin(wh[waveInd[0]]), np.nanmax(wh[waveInd[0]]),
                                              np.nanmin(tp[waveInd[0]]), np.nanmax(tp[waveInd[0]]),
                                              np.nanmean(dm[waveInd[0]]),np.nanmean(u10[waveInd[0]]),
                                              np.nanmean(v10[waveInd[0]]),np.nanmean(sst[waveInd[0]]),
                                              np.nanmean(ssr[waveInd[0]]),np.nanmean(t2m[waveInd[0]]),
                                              areaBelow[fetchInd[0][0]], np.nanmean(ntr[ntrInd[0]])])

                tempDict['hsMin'] = np.nanmin(wh[waveInd[0]])
                tempDict['hsMax'] = np.nanmax(wh[waveInd[0]])
                tempDict['tpMin'] = np.nanmin(tp[waveInd[0]])
                tempDict['tpMax'] = np.nanmax(tp[waveInd[0]])
                tempDict['dmMean'] = np.nanmean(dm[waveInd[0]])
                tempDict['u10Mean'] = np.nanmean(u10[waveInd[0]])
                tempDict['u10Max'] = np.nanmax(u10[waveInd[0]])
                tempDict['u10Min'] = np.nanmin(u10[waveInd[0]])
                tempDict['v10Max'] = np.nanmax(v10[waveInd[0]])
                tempDict['v10Mean'] = np.nanmean(v10[waveInd[0]])
                tempDict['v10Min'] = np.nanmin(v10[waveInd[0]])
                tempDict['sstMean'] = np.nanmean(sst[waveInd[0]])
                tempDict['ssrMean'] = np.nanmean(ssr[waveInd[0]])
                tempDict['ssrMin'] = np.nanmin(ssr[waveInd[0]])
                tempDict['ssrMax'] = np.nanmax(ssr[waveInd[0]])
                tempDict['t2mMean'] = np.nanmean(t2m[waveInd[0]])
                tempDict['t2mMin'] = np.nanmin(t2m[waveInd[0]])
                tempDict['t2mMax'] = np.nanmax(t2m[waveInd[0]])
                tempDict['ntrMean'] = np.nanmean(ntr[ntrInd[0]])
                tempDict['ntrMin'] = np.nanmin(ntr[ntrInd[0]])
                tempDict['ntrMax'] = np.nanmax(ntr[ntrInd[0]])
                tempList.append(tempDict)
                tempHydroLength = tempHydroLength-1
                newTime = etNew
            else:
                waveInd = np.where((time_all < et) & (time_all >= newTime))
                fetchInd = np.where((np.asarray(timeArrayIce) == newTime))
                ntrInd = np.where((tNTR < et) & (tNTR >= newTime))

                c = c + 1
                counter = counter+1
                tempDict = dict()
                tempDict['time'] = time_all[waveInd[0]]
                tempDict['numDays'] = subLength[i]
                tempDict['hs'] = wh[waveInd[0]]
                tempDict['tp'] = tp[waveInd[0]]
                tempDict['dm'] = dm[waveInd[0]]
                tempDict['v10'] = v10[waveInd[0]]
                tempDict['u10'] = u10[waveInd[0]]
                tempDict['sst'] = sst[waveInd[0]]
                tempDict['ssr'] = ssr[waveInd[0]]
                tempDict['t2m'] = t2m[waveInd[0]]
                tempDict['ntr'] = ntr[ntrInd[0]]

                tempDict['fetch'] = areaBelow[fetchInd[0][0]]
                tempDict['cop'] = np.asarray([np.nanmin(wh[waveInd[0]]), np.nanmax(wh[waveInd[0]]),
                                              np.nanmin(tp[waveInd[0]]), np.nanmax(tp[waveInd[0]]),
                                              np.nanmean(dm[waveInd[0]]),np.nanmean(u10[waveInd[0]]),
                                              np.nanmean(v10[waveInd[0]]),np.nanmean(sst[waveInd[0]]),
                                              np.nanmean(ssr[waveInd[0]]),np.nanmean(t2m[waveInd[0]]),
                                              areaBelow[fetchInd[0][0]], np.nanmean(ntr[ntrInd[0]])])
                tempDict['hsMin'] = np.nanmin(wh[waveInd[0]])
                tempDict['hsMax'] = np.nanmax(wh[waveInd[0]])
                tempDict['tpMin'] = np.nanmin(tp[waveInd[0]])
                tempDict['tpMax'] = np.nanmax(tp[waveInd[0]])
                tempDict['dmMean'] = np.nanmean(dm[waveInd[0]])
                tempDict['u10Mean'] = np.nanmean(u10[waveInd[0]])
                tempDict['u10Max'] = np.nanmax(u10[waveInd[0]])
                tempDict['u10Min'] = np.nanmin(u10[waveInd[0]])
                tempDict['v10Max'] = np.nanmax(v10[waveInd[0]])
                tempDict['v10Mean'] = np.nanmean(v10[waveInd[0]])
                tempDict['v10Min'] = np.nanmin(v10[waveInd[0]])
                tempDict['sstMean'] = np.nanmean(sst[waveInd[0]])
                tempDict['ssrMean'] = np.nanmean(ssr[waveInd[0]])
                tempDict['ssrMin'] = np.nanmin(ssr[waveInd[0]])
                tempDict['ssrMax'] = np.nanmax(ssr[waveInd[0]])
                tempDict['t2mMean'] = np.nanmean(t2m[waveInd[0]])
                tempDict['t2mMin'] = np.nanmin(t2m[waveInd[0]])
                tempDict['t2mMax'] = np.nanmax(t2m[waveInd[0]])
                tempDict['ntrMean'] = np.nanmean(ntr[ntrInd[0]])
                tempDict['ntrMin'] = np.nanmin(ntr[ntrInd[0]])
                tempDict['ntrMax'] = np.nanmax(ntr[ntrInd[0]])
                tempList.append(tempDict)
    print('we have split {} times in bmu {}'.format(counter,p))
    hydros.append(tempList)




myFmt = mdates.DateFormatter('%d')
#
# bmuPlot = 2
#
# m = np.ceil(np.sqrt(len(hydros[bmuPlot])))
# plt.figure()
# gs = gridspec.GridSpec(int(m), int(m), wspace=0.1, hspace=0.15)
# for i in range(len(hydros[bmuPlot])):
#     ax = plt.subplot(gs[i])
#     ax.plot(hydros[bmuPlot][i]['time'], hydros[bmuPlot][i]['tp'], 'k-')
#     ax.set_ylim([4, 12])
#     ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
#     ax.xaxis.set_major_formatter(myFmt)
#
#
# ### TODO: look at the distributions within each copula
# plt.figure()
# gs = gridspec.GridSpec(10, 7, wspace=0.1, hspace=0.15)
# bins = np.linspace(0,4,25)
# for bmuPlot in range(len(np.unique(bmuGroup))):
#     ax = plt.subplot(gs[bmuPlot])
#     tempHsMin = list()
#     tempHsMax = list()
#     for i in range(len(hydros[bmuPlot])):
#         tempHsMin.append(hydros[bmuPlot][i]['hsMin'])
#         tempHsMax.append(hydros[bmuPlot][i]['hsMax'])
#
#     ax.hist(np.asarray(tempHsMax),bins,alpha=0.5,label='hs max',density=True)
#     ax.hist(np.asarray(tempHsMin),bins,alpha=0.5,label='hs min',density=True)
#     ax.set_ylim([0,2])
# ax.legend(loc='upper right')
#
# plt.figure()
# gs = gridspec.GridSpec(10, 7, wspace=0.1, hspace=0.15)
# bins = np.linspace(3,14,40)
# for bmuPlot in range(len(np.unique(bmuGroup))):
#     ax = plt.subplot(gs[bmuPlot])
#     tempHsMin = list()
#     tempHsMax = list()
#     for i in range(len(hydros[bmuPlot])):
#         tempHsMin.append(hydros[bmuPlot][i]['tpMin'])
#         tempHsMax.append(hydros[bmuPlot][i]['tpMax'])
#
#     ax.hist(np.asarray(tempHsMax),bins,alpha=0.5,label='tp max',density=True)
#     ax.hist(np.asarray(tempHsMin),bins,alpha=0.5,label='tp min',density=True)
#     ax.set_ylim([0,0.6])
# ax.legend(loc='upper right')
#
#
# plt.figure()
# gs = gridspec.GridSpec(10, 7, wspace=0.1, hspace=0.15)
# bins = np.linspace(-0.4,0.8,20)
# for bmuPlot in range(len(np.unique(bmuGroup))):
#     ax = plt.subplot(gs[bmuPlot])
#     tempSS= list()
#     for i in range(len(hydros[bmuPlot])):
#         tempSS.append(hydros[bmuPlot][i]['ssMean'])
#
#     ax.hist(np.asarray(tempSS),bins,alpha=0.5,label='tp min',density=True)
#     ax.set_ylim([0,3.25])
# ax.legend(loc='upper right')



### TODO: make a copula fit for each of the 70 DWTs
copulaData = list()
copulaDataOnlyWaves = list()
for i in range(len(np.unique(bmuGroup))):
    tempHydros = hydros[i]
    dataCop = []
    dataCopOnlyWaves = []
    for kk in range(len(tempHydros)):
        dataCop.append(list([tempHydros[kk]['hsMax'], tempHydros[kk]['hsMin'], tempHydros[kk]['tpMax'],
                             tempHydros[kk]['tpMin'], tempHydros[kk]['dmMean'], tempHydros[kk]['u10Max'],
                             tempHydros[kk]['u10Min'], tempHydros[kk]['v10Max'], tempHydros[kk]['v10Min'],
                              tempHydros[kk]['ssrMean'], tempHydros[kk]['t2mMean'],
                             tempHydros[kk]['fetch'], tempHydros[kk]['ntrMean'], tempHydros[kk]['sstMean'],len(tempHydros[kk]['time']), kk]))

        if np.isnan(tempHydros[kk]['hsMax'])==1:
            print('no waves here')

        else:
            dataCopOnlyWaves.append(list([tempHydros[kk]['hsMax'],tempHydros[kk]['hsMin'],tempHydros[kk]['tpMax'],
                             tempHydros[kk]['tpMin'], tempHydros[kk]['dmMean'], tempHydros[kk]['u10Max'],
                             tempHydros[kk]['u10Min'], tempHydros[kk]['v10Max'], tempHydros[kk]['v10Min'],
                             tempHydros[kk]['ssrMean'], tempHydros[kk]['t2mMean'],
                             tempHydros[kk]['fetch'], tempHydros[kk]['ntrMean'], tempHydros[kk]['sstMean'],len(tempHydros[kk]['time']),kk]))

    copulaData.append(dataCop)
    copulaDataOnlyWaves.append(dataCopOnlyWaves)

# bmuData = list()
# for i in range(len(np.unique(bmuGroup))):
#     tempHydros = hydros[i]
#     dataCop = []
#     for kk in range(len(tempHydros)):
#         if np.isnan(tempHydros[kk]['ssMean']):
#             print('skipping a Nan in storm surge')
#         else:
#             dataCop.append(list([tempHydros[kk]['hsMax'],tempHydros[kk]['hsMin'],tempHydros[kk]['tpMax'],
#                        tempHydros[kk]['tpMin'],tempHydros[kk]['dmMean'],tempHydros[kk]['ssMean'],kk]))
#     bmuData.append(dataCop)



# bmuData = list()
# for i in range(len(np.unique(bmuGroup))):
#     tempHydros = hydros[i]
#     dataCop = []
#     for kk in range(len(tempHydros)):
#         if np.isnan(tempHydros[kk]['ssMean']):
#             print('skipping a Nan in storm surge')
#         else:
#             dataCop.append(list([tempHydros[kk]['hsMax'],tempHydros[kk]['hsMin'],tempHydros[kk]['tpMax'],
#                        tempHydros[kk]['tpMin'],tempHydros[kk]['dmMean'],tempHydros[kk]['ssMean'],kk]))
#     bmuData.append(dataCop)


bmuDataNormalized = list()
bmuDataMin = list()
bmuDataMax = list()
bmuDataStd = list()
for i in range(len(np.unique(bmuGroup))):
    temporCopula = np.asarray(copulaDataOnlyWaves[i])
    if len(temporCopula) == 0:
        bmuDataNormalized.append(np.vstack((0, 0)).T)
        bmuDataMin.append([0, 0])
        bmuDataMax.append([0, 0])
        bmuDataStd.append([0, 0])
    else:
        dataHs = np.array([sub[0] for sub in copulaDataOnlyWaves[i]])
        data = temporCopula[~np.isnan(dataHs)]
        data2 = data[~np.isnan(data[:,0])]
        if len(data2) == 0:
            print('woah, no waves here bub')
        #     data2 = data
        #     data2[:,5] = 0
            bmuDataNormalized.append(np.vstack((0, 0)).T)
            bmuDataMin.append([0, 0])
            bmuDataMax.append([0, 0])
            bmuDataStd.append([0, 0])
        else:

            # maxHs = np.nanmax(data2[:,0])
            # minHs = np.nanmin(data2[:,0])
            # stdHs = np.nanstd(data2[:,0])
            # hsNorm = (data2[:,0] - minHs) / (maxHs-minHs)
            # maxTp = np.nanmax(data2[:,1])
            # minTp = np.nanmin(data2[:,1])
            # stdTp = np.nanstd(data2[:,1])
            # tpNorm = (data2[:,1] - minTp) / (maxTp-minTp)
            maxDm = np.nanmax(data2[:,4])
            minDm = np.nanmin(data2[:,4])
            stdDm = np.nanstd(data2[:,4])
            dmNorm = (data2[:,4] - minDm) / (maxDm-minDm)
            maxSs = np.nanmax(data2[:,12])
            minSs = np.nanmin(data2[:,12])
            stdSs = np.nanstd(data2[:,12])
            ssNorm = (data2[:,12] - minSs) / (maxSs-minSs)
        # maxDm = np.nanmax(np.asarray(copulaData[i])[:,4])
        # minDm = np.nanmin(np.asarray(copulaData[i])[:,4])
        # stdDm = np.nanstd(np.asarray(copulaData[i])[:,4])
        # dmNorm = (np.asarray(copulaData[i])[:,4] - minDm) / (maxDm-minDm)
        # maxSs = np.nanmax(np.asarray(copulaData[i])[:,5])
        # minSs = np.nanmin(np.asarray(copulaData[i])[:,5])
        # stdSs = np.nanstd(np.asarray(copulaData[i])[:,5])
        # ssNorm = (np.asarray(copulaData[i])[:,5] - minSs) / (maxSs-minSs)
        #     bmuDataNormalized.append(np.vstack((hsNorm,tpNorm)).T)
        #     bmuDataMin.append([minHs,minTp])
        #     bmuDataMax.append([maxHs,maxTp])
        #     bmuDataStd.append([stdHs,stdTp])
            bmuDataNormalized.append(np.vstack((dmNorm,ssNorm)).T)
            bmuDataMin.append([minDm,minSs])
            bmuDataMax.append([maxDm,maxSs])
            bmuDataStd.append([stdDm,stdSs])
#
#
# bmuDataNormalized2 = list()
# bmuDataMin2 = list()
# bmuDataMax2 = list()
# bmuDataStd2 = list()
# for i in range(len(np.unique(bmuGroup))):
#     maxDm = np.nanmax(np.asarray(copulaData[i])[:,4])
#     minDm = np.nanmin(np.asarray(copulaData[i])[:,4])
#     stdDm = np.nanstd(np.asarray(copulaData[i])[:,4])
#     dmNorm = (np.asarray(copulaData[i])[:,4] - minDm) / (maxDm-minDm)
#     maxSs = np.nanmax(np.asarray(copulaData[i])[:,5])
#     minSs = np.nanmin(np.asarray(copulaData[i])[:,5])
#     stdSs = np.nanstd(np.asarray(copulaData[i])[:,5])
#     ssNorm = (np.asarray(copulaData[i])[:,5] - minSs) / (maxSs-minSs)
#     bmuDataNormalized.append(np.vstack((dmNorm,ssNorm)).T)
#     bmuDataMin.append([minDm,minSs])
#     bmuDataMax.append([maxDm,maxSs])
#     bmuDataStd.append([stdDm,stdSs])

#
#
# def closest_node(node, nodes):
#     closest_index = distance.cdist([node], nodes).argmin()
#     return nodes[closest_index], closest_index
#

### TODO: Need to normalize all of the hydrographs with respect to time AND magnitude


normalizedHydros = list()
for i in range(len(np.unique(bmuGroup))):
    tempHydros = hydros[i]
    tempList = list()
    for mm in range(len(tempHydros)):
        if np.isnan(tempHydros[mm]['hsMin']):
            print('no waves')
        else:
            tempDict = dict()
            tempDict['hsNorm'] = (tempHydros[mm]['hs'] - tempHydros[mm]['hsMin']) / (tempHydros[mm]['hsMax']- tempHydros[mm]['hsMin'])
            tempDict['tpNorm'] = (tempHydros[mm]['tp'] - tempHydros[mm]['tpMin']) / (tempHydros[mm]['tpMax']- tempHydros[mm]['tpMin'])
            tempDict['timeNorm'] = np.arange(0,1,1/len(tempHydros[mm]['time']))[0:len(tempDict['hsNorm'])]
            tempDict['dmNorm'] = (tempHydros[mm]['dm']) - tempHydros[mm]['dmMean']
            tempDict['uNorm'] = (tempHydros[mm]['u10'] - tempHydros[mm]['u10Min']) / (
                        tempHydros[mm]['u10Max'] - tempHydros[mm]['u10Min'])
            tempDict['vNorm'] = (tempHydros[mm]['v10'] - tempHydros[mm]['v10Min']) / (
                        tempHydros[mm]['v10Max'] - tempHydros[mm]['v10Min'])
            # tempDict['ntrNorm'] = (tempHydros[mm]['ntr'] - tempHydros[mm]['ntrMin']) / (tempHydros[mm]['ntrMax']- tempHydros[mm]['ntrMin'])
            tempDict['ntrNorm'] = (tempHydros[mm]['ntr'] - tempHydros[mm]['ntrMean'])

            tempList.append(tempDict)
    normalizedHydros.append(tempList)




import pickle

# normHydrosPickle = 'normalizedWaveHydrographsPointHope.pickle'
# normHydrosPickle = 'normalizedWaveHydrographsShishmaref.pickle'
# normHydrosPickle = 'normalizedWaveHydrographsWainwright.pickle'
# normHydrosPickle = 'normalizedWaveHydrographsWales.pickle'
# normHydrosPickle = 'normalizedWaveHydrographsWevok.pickle'
# normHydrosPickle = 'normalizedWaveHydrographsPointLay.pickle'
# normHydrosPickle = 'normalizedWaveHydrographsKivalina.pickle'
normHydrosPickle = 'normalizedWaveHydrographsUtqiagvik.pickle'

outputHydrosNorm = {}
outputHydrosNorm['normalizedHydros'] = normalizedHydros
outputHydrosNorm['bmuDataMin'] = bmuDataMin
outputHydrosNorm['bmuDataMax'] = bmuDataMax
outputHydrosNorm['bmuDataStd'] = bmuDataStd
outputHydrosNorm['bmuDataNormalized'] = bmuDataNormalized

with open(normHydrosPickle,'wb') as f:
    pickle.dump(outputHydrosNorm, f)

# hydrosPickle = 'waveHydrographsPointHope.pickle'
# hydrosPickle = 'waveHydrographsShishmaref.pickle'
# hydrosPickle = 'waveHydrographsWainwright.pickle'
# hydrosPickle = 'waveHydrographsWales.pickle'
# hydrosPickle = 'waveHydrographsWevok.pickle'
# hydrosPickle = 'waveHydrographsPointLay.pickle'
# hydrosPickle = 'waveHydrographsKivalina.pickle'
hydrosPickle = 'waveHydrographsUtqiagvik.pickle'

outputHydros = {}
outputHydros['hydros'] = hydros
with open(hydrosPickle,'wb') as f:
    pickle.dump(outputHydros, f)

# copPickle = 'hydrographCopulaDataPointHope.pickle'
# copPickle = 'hydrographCopulaDataShishmaref.pickle'
# copPickle = 'hydrographCopulaDataWainwright.pickle'
# copPickle = 'hydrographCopulaDataWales.pickle'
# copPickle = 'hydrographCopulaDataWevok.pickle'
# copPickle = 'hydrographCopulaDataPointLay.pickle'
# copPickle = 'hydrographCopulaDataKivalina.pickle'
copPickle = 'hydrographCopulaDataUtqiagvik.pickle'

outputCopula = {}
outputCopula['copulaData'] = copulaData
outputCopula['copulaDataNoNaNs'] = copulaDataOnlyWaves
with open(copPickle,'wb') as f:
    pickle.dump(outputCopula, f)


# historicalPickle = 'historicalDataPointHope.pickle'
# historicalPickle = 'historicalDataShishmaref.pickle'
# historicalPickle = 'historicalDataWainwright.pickle'
# historicalPickle = 'historicalDataWales.pickle'
# historicalPickle = 'historicalDataWevok.pickle'
# historicalPickle = 'historicalDataPointLay.pickle'
# historicalPickle = 'historicalDataKivalina.pickle'
historicalPickle = 'historicalDataUtqiagvik.pickle'

outputHistorical = {}
outputHistorical['grouped'] = grouped
outputHistorical['groupLength'] = groupLength
outputHistorical['bmuGroup'] = bmuGroup
outputHistorical['timeGroup'] = timeGroup

with open(historicalPickle,'wb') as f:
    pickle.dump(outputHistorical, f)












