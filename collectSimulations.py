import pickle

import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta

import pandas as pd

hs = []
tp = []
dm = []
ss = []
numOfSims = 100
getSims = np.arange(0,100)
for hh in range(len(getSims)):
   numSim = getSims[hh]
   # file = r"/home/dylananderson/projects/atlanticClimate/Sims/simulation{}.pickle".format(hh)
   file = r"/media/dylananderson/Elements/SimsArctic/simulation{}.pickle".format(numSim)

   with open(file, "rb") as input_file:
      simsInput = pickle.load(input_file)
   simulationData = simsInput['futureSimulationData']
   df = simsInput['df']
   hourlyTime = simsInput['time']
   df.loc[df['hs'] == 0, 'hs'] = np.nan
   df.loc[df['tp'] == 0, 'tp'] = np.nan
   df.loc[df['dm'] == 0, 'dm'] = np.nan

   angles = df['dm'].values
   where360 = np.where((angles > 360))
   angles[where360] = angles[where360] - 360
   whereNeg = np.where((angles < 0))
   angles[whereNeg] = angles[whereNeg] + 360

   # whereNan = np.where(np.isnan(angles))
   # angles[whereNan] = 0
   hsTemp = df['hs'].values
   # whereNan = np.where(np.isnan(hsTemp))
   # hsTemp[whereNan] = 0
   tpTemp = df['tp'].values
   # whereNan = np.where(np.isnan(tpTemp))
   # tpTemp[whereNan] = 0
   ssTemp = df['ss'].values
   # whereNan = np.where(np.isnan(ssTemp))
   # ssTemp[whereNan] = 0
   # hs.append(hsTemp)
   # tp.append(tpTemp)
   # dm.append(angles)
   # ss.append(ssTemp)
   hs.append(hsTemp)
   tp.append(tpTemp)
   dm.append(angles)
   ss.append(ssTemp)
allHs = np.stack(hs, axis=0)
allTp = np.stack(tp, axis=0)
allDm = np.stack(dm, axis=0)
allSs = np.stack(ss, axis=0)
#simNum = np.arange(0,numOfSims)
# import xarray as xr
# ds = xr.Dataset(
#    {"Hs": (("sim","time"), allHs),
#     "Tp": (("sim","time"), allTp),
#     "Dm": (("sim","time"), allDm),
#     "Ss": (("sim","time"), allSs)},
#    coords={
#          "sim": simNum,
#          "time": hourlyTime,
#    },
# )
# ds.to_netcdf("saved_on_disk.nc")

from datetime import datetime as dt

def datenum(d):
    return 366 + d.toordinal() + (d - dt.fromordinal(d.toordinal())).total_seconds()/(24*60*60)

d = dt.strptime('2019-2-1 12:24','%Y-%m-%d %H:%M')
dn = datenum(d)

simTime = [datenum(hh) for hh in hourlyTime]








import datetime as dt

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

# define some constants
epoch = dt.datetime(1970, 1, 1)
matlab_to_epoch_days = 719529  # days from 1-1-0000 to 1-1-1970
matlab_to_epoch_seconds = matlab_to_epoch_days * 24 * 60 * 60

def matlab_to_datetime(matlab_date_num_seconds):
    # get number of seconds from epoch
    from_epoch = matlab_date_num_seconds - matlab_to_epoch_seconds

    # convert to python datetime
    return epoch + dt.timedelta(seconds=from_epoch)

import mat73
waveData = mat73.loadmat('era5_waves_pthope.mat')
# wlData = mat73.loadmat('pthope_tides_for_dylan.mat')
wlData = mat73.loadmat('updated_env_for_dylan.mat')

environData = mat73.loadmat('updated_env_for_dylan.mat')
environDates = [datenum_to_date(t) for t in environData['time']]
environTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in environData['time']]

wlArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in wlData['time']]
waveArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in waveData['time_all']]

allWLs = wlData['wl']
wlsInd = np.where(allWLs < -2.5)
allWLs[wlsInd] = np.zeros((len(wlsInd),))

from scipy.stats.kde import gaussian_kde
import matplotlib.cm as cm

res = allWLs#wlData['wl']
wlHycom = allWLs#wlData['wl']
tWl = np.array(wlArrayTime)
waveDates = [datenum_to_date(t) for t in waveData['time_all']]
tWave = np.array(waveArrayTime)
hsCombined = waveData['wh_all']
tpCombined = waveData['tp_all']
dmCombined = waveData['mwd_all']
waveNorm = dmCombined

# dist_space = np.arange(0,5,100)
# kde = gaussian_kde(hsCombined)
# # colorparam[counter] = np.nanmean(data)
# colormap = cm.Reds
# # color = colormap(normalize(colorparam[counter]))
# plt.figure()
# ax = plt.subplot2grid((1,1),(0,0))
# ax.plot(dist_space, kde(dist_space), linewidth=1)#, color=color)


plt.style.use('dark_background')
plt.figure()
ax1 = plt.subplot2grid((3,1),(0,0))
ax2 = plt.subplot2grid((3,1),(1,0))
ax3 = plt.subplot2grid((3,1),(2,0))
# ax4 = plt.subplot2grid((4,1),(3,0))

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




plt.style.use('dark_background')
plt.figure()
ax1 = plt.subplot2grid((3,1),(0,0))
ax2 = plt.subplot2grid((3,1),(1,0))
ax3 = plt.subplot2grid((3,1),(2,0))
# ax4 = plt.subplot2grid((4,1),(3,0))
versn = 3

# ax1.plot(wlArrayTime,wlData['wl_all'])
ax1.plot(hourlyTime,ss[versn])

# ax1.set_xlim([dt.datetime(1979,1,1), dt.datetime(2022,6,1)])
ax1.set_ylabel('wl (m)')
ax2.plot(hourlyTime,hs[versn])
# ax2.plot(environTime,environData['hs'])
# ax2.set_xlim([dt.datetime(1979,1,1), dt.datetime(2022,6,1)])
ax2.set_ylabel('hs (m)')

tempTP = tp[versn]
weirdVals = np.where((tempTP < 1))
tempTP[weirdVals] = np.nan
ax3.plot(hourlyTime,tempTP)
# ax3.plot(environTime,environData['tp'])
# ax3.set_xlim([dt.datetime(1979,1,1), dt.datetime(2022,6,1)])
ax3.set_ylabel('tp (s)')


#
# ax4.plot(waveArrayTime,waveData['mwd_all'])
# # ax4.plot(environTime,environData['wd'])
# ax4.set_xlim([dt.datetime(1979,1,1), dt.datetime(2022,6,1)])
# ax4.set_ylabel('mwd (deg)')


# def datetime2matlabdn(dt):
#    mdn = dt + timedelta(days = 366)
#    frac_seconds = (dt-datetime.datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
#    frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
#    return mdn.toordinal() + frac_seconds + frac_microseconds

#
# mdict = dict()
# mdict['Hs'] = allHs
# mdict['Tp'] = allTp
# mdict['Dm'] = allDm
# mdict['SWL'] = allSs
# mdict['time'] = simTime
#
#
# from scipy.io import savemat
#
# savemat('simulations1to100.mat',mdict)

# import matplotlib.pyplot as plt
# plt.plot(simTime,allDm[0,:])

#
# In [2]: ds.to_netcdf("saved_on_disk.nc")
