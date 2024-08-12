import pickle

import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta

import pandas as pd

sSims = np.arange(0,1000,100)
eSims = np.arange(100,1001,100)
# sSims = np.arange(0,400,100)
# eSims = np.arange(100,501,100)

with open(r"predictedArctic1978to2023.pickle", "rb") as input_file:
   arcticTemperatures = pickle.load(input_file)

timeTemp = arcticTemperatures['futureDate']
arcticTemps = np.array(arcticTemperatures['futureSims'])
alternateArcticTemps = np.array(arcticTemperatures['alternateFutureSims'])
arcticTempTrend = np.array(arcticTemperatures['futureTrend'])

from datetime import datetime
pastDatesIndex = np.where((np.asarray(timeTemp) < datetime(2023,5,31)))
futureDatesIndex = np.where((np.asarray(timeTemp) > datetime(2023,5,31)) & (np.asarray(timeTemp) < datetime(2075,6,1)))
trendLine = np.mean(arcticTempTrend[pastDatesIndex])
futureTrendLine = arcticTempTrend[futureDatesIndex]-trendLine
dailyInt = np.arange(0,24*len(futureTrendLine),24)
hourlyInt = np.arange(0,24*len(futureTrendLine))
interpTrend = np.interp(hourlyInt, dailyInt, futureTrendLine)

for beginSim in sSims:
    print('working on {} to {}'.format(beginSim,beginSim+100))
    hs = []
    tp = []
    dm = []
    ss = []
    t2m = []
    numOfSims = 100
    getSims = np.arange(beginSim,beginSim+100)
    for hh in range(len(getSims)):
       numSim = getSims[hh]

       file = ('/volumes/macDrive/arcticSims/utqiagvik/futureSimulation{}.pickle'.format(numSim))

       meanTemp = arcticTemps[int(np.floor(numSim/10))]
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
       t2mTemp = df['t2m'].values+interpTrend
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
       t2m.append(t2mTemp)




    allHs = np.stack(hs, axis=0)
    allTp = np.stack(tp, axis=0)
    allDm = np.stack(dm, axis=0)
    allSs = np.stack(ss, axis=0)
    allT2m = np.stack(t2m, axis=0)

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

    # d = dt.strptime('2019-2-1 12:24','%Y-%m-%d %H:%M')
    # dn = datenum(d)

    simTime = [datenum(hh) for hh in hourlyTime]


    import scipy.io as sio
    from scipy.io.matlab.mio5_params import mat_struct


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

    from scipy.io import loadmat
    tidePurdoe = loadmat('/users/dylananderson/documents/projects/arcticClimate/tide_emulation_Purdoe.mat')
    tideNome = loadmat('/users/dylananderson/documents/projects/arcticClimate/tide_emulation_Nome.mat')
    tideRed = loadmat('/users/dylananderson/documents/projects/arcticClimate/tide_emulation_Red.mat')

    syntideNome = tideNome['synTide']
    syntimeNome = tideNome['synTime']
    environTimeNome = tideNome['synTime']#[matlab_to_datetime(t * 24 * 60 * 60) for t in tideNome['synTime']]
    environIndNome = np.where((environTimeNome>=simTime[0]) & (environTimeNome<=simTime[-1]))
    tideNome = syntideNome[environIndNome]

    syntideRed = tideRed['synTide']
    syntimeRed = tideRed['synTime']
    environTimeRed = tideRed['synTime']#[matlab_to_datetime(t * 24 * 60 * 60) for t in tideRed['synTime']]
    environIndRed = np.where((environTimeRed>=simTime[0]) & (environTimeRed<=simTime[-1]))
    tideRed = syntideRed[environIndRed]


    syntidePurdoe = tidePurdoe['synTide']
    syntimePurdoe = tidePurdoe['synTime']
    environTimePurdoe = tidePurdoe['synTime']#[matlab_to_datetime(t * 24 * 60 * 60) for t in tideRed['synTime']]
    environIndPurdoe = np.where((environTimePurdoe>=simTime[0]) & (environTimePurdoe<=simTime[-1]))
    tidePurdoe = syntideRed[environIndPurdoe]

    # %distance weight (roughly) all tide locations
    shish_tide = tideNome *(170/370) + tideRed*(200/370)
    pthope_tide = tideNome *(160/920) + tideRed*(760/920)
    wainright_tide = tideNome *(175/620) + tideRed*(445/620)
    # shish_wl2 = gapFilledNomeNTR *(865/1065) + gapFilledPurdoeNTR*(200/1065)
    # pthope_wl2 = gapFilledPurdoeNTR *(430/1190) + gapFilledNomeNTR*(760/1190)
    # wainright_wl2 = gapFilledPurdoeNTR *(720/1165) + gapFilledNomeNTR*(445/1165)

    shish_tide = tideNome *(170/370) + tideRed*(200/370)
    pthope_tide = tideNome *(160/920) + tideRed*(760/920)
    wainright_tide = tideNome *(175/620) + tideRed*(445/620)

    wevok_tide = tidePurdoe*(166/856) + tideRed*(690/856)
    wales_tide = tideNome *(269/450) + tideRed*(181/450)
    kivalina_tide = tideRed
    ptlay_tide = tidePurdoe *(556/716) + tideRed*(160/716)
    utqiagvik_tide = tidePurdoe*(505/825) + tideRed*(320/825)



    data = ReadMatfile('/users/dylananderson/documents/projects/arcticClimate/arcticTides.mat')
    # tidePurdoe = ReadMatfile('/users/dylananderson/documents/projects/arcticClimate/tide_emulation_Purdoe.mat')
    # tideNome = ReadMatfile('/users/dylananderson/documents/projects/arcticClimate/tide_emulation_Nome.mat')
    # tideRed = ReadMatfile('/users/dylananderson/documents/projects/arcticClimate/tide_emulation_Red.mat')
    #
    # tideTimeEmPurdoe = np.array([matlab_to_datetime(tempTime * 24 * 60 * 60) for tempTime in tidePurdoe['time_emulator']])
    # tideTimeEmNome = np.array([matlab_to_datetime(tempTime * 24 * 60 * 60) for tempTime in tideNome['time_emulator']])
    # tideTimeEmRed = np.array([matlab_to_datetime(tempTime * 24 * 60 * 60) for tempTime in tideRed['time_emulator']])

    mslEmNome = data['nomeDailyData']['B']['msl'][1]*syntimeNome[environIndNome]+data['nomeDailyData']['B']['msl'][0]
    mslEmPurdoe = data['purdoeDailyData']['B']['msl'][1]*syntimePurdoe[environIndPurdoe]+data['purdoeDailyData']['B']['msl'][0]
    mslEmRed = data['redDailyData']['B']['msl'][1]*syntimeRed[environIndRed]+data['redDailyData']['B']['msl'][0]

    shish_msl = mslEmNome *(170/370) + mslEmRed*(200/370)
    pthope_msl = mslEmNome *(160/920) + mslEmRed*(760/920)
    wainright_msl = mslEmNome *(175/620) + mslEmRed*(445/620)

    wevok_msl = mslEmPurdoe*(166/856) + mslEmRed*(690/856)
    wales_msl = mslEmNome *(269/450) + mslEmRed*(181/450)
    kivalina_msl = mslEmRed
    ptlay_msl = mslEmPurdoe *(556/716) + mslEmRed*(160/716)


    utqiagvik_msl = mslEmPurdoe*(505/825) + mslEmRed*(320/825)

    #
    # import hdf5storage
    # hdf5storage.write('/users/dylananderson/documents/projects/arcticClimate/wainwrightSimulations500to600.mat', mdict, format=7.3, matlab_compatible=True, compress=False )
    # # import matplotlib.pyplot as plt
    # # plt.plot(simTime,allDm[0,:])

    #
    # In [2]: ds.to_netcdf("saved_on_disk.nc")




    #
    # waveData = mat73.loadmat('era5_waves_pthope.mat')
    # # wlData = mat73.loadmat('pthope_tides_for_dylan.mat')
    # wlData = mat73.loadmat('updated_env_for_dylan.mat')
    #
    # environData = mat73.loadmat('updated_env_for_dylan.mat')
    # environDates = [datenum_to_date(t) for t in environData['time']]
    # environTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in environData['time']]
    #
    # wlArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in wlData['time']]
    # waveArrayTime = [matlab_to_datetime(t * 24 * 60 * 60) for t in waveData['time_all']]
    #
    # allWLs = wlData['wl']
    # wlsInd = np.where(allWLs < -2.5)
    # allWLs[wlsInd] = np.zeros((len(wlsInd),))
    #
    # from scipy.stats.kde import gaussian_kde
    # import matplotlib.cm as cm
    #
    # res = allWLs#wlData['wl']
    # wlHycom = allWLs#wlData['wl']
    # tWl = np.array(wlArrayTime)
    # waveDates = [datenum_to_date(t) for t in waveData['time_all']]
    # tWave = np.array(waveArrayTime)
    # hsCombined = waveData['wh_all']
    # tpCombined = waveData['tp_all']
    # dmCombined = waveData['mwd_all']
    # waveNorm = dmCombined
    #
    # # dist_space = np.arange(0,5,100)
    # # kde = gaussian_kde(hsCombined)
    # # # colorparam[counter] = np.nanmean(data)
    # # colormap = cm.Reds
    # # # color = colormap(normalize(colorparam[counter]))
    # # plt.figure()
    # # ax = plt.subplot2grid((1,1),(0,0))
    # # ax.plot(dist_space, kde(dist_space), linewidth=1)#, color=color)


    ## WAS PLOTTING THIS AREA
    # # plt.style.use('dark_background')
    # plt.figure()
    # ax1 = plt.subplot2grid((3,1),(0,0))
    # ax2 = plt.subplot2grid((3,1),(1,0))
    # ax3 = plt.subplot2grid((3,1),(2,0))
    # # ax4 = plt.subplot2grid((4,1),(3,0))
    #
    # # allWLs = environData['wl']
    # # wlsInd = np.where(allWLs < -2.5)
    # # allWLs[wlsInd] = np.zeros((len(wlsInd),))
    #
    #
    # # ax1.plot(wlArrayTime,wlData['wl_all'])
    # ax1.plot(hourlyTime,pthope_msl+pthope_tide+allSs[0,:],label='still water level',color='k')
    # ax1.plot(hourlyTime,pthope_msl+pthope_tide,label='MSL + tides',color='orange')
    # ax1.legend()
    # ax1.set_xlim([hourlyTime[0],hourlyTime[-1]])
    # # ax1.set_xlim([dt.datetime(1979,1,1), dt.datetime(2022,6,1)])
    # ax1.set_ylabel('wl (m)')
    # ax2.plot(hourlyTime,allHs[0,:],color='k')
    # # ax2.plot(environTime,environData['hs'])
    # # ax2.set_xlim([dt.datetime(1979,1,1), dt.datetime(2022,6,1)])
    # ax2.set_xlim([hourlyTime[0],hourlyTime[-1]])
    #
    # ax2.set_ylabel('hs (m)')
    # tpSub = allTp[0,:]
    # badtp = np.where(tpSub<1)
    # tpSub[badtp[0]] = np.nan * tpSub[badtp[0]]
    # ax3.plot(hourlyTime,tpSub,color='k')
    # # ax3.plot(environTime,environData['tp'])
    # # ax3.set_xlim([dt.datetime(1979,1,1), dt.datetime(2022,6,1)])
    # ax3.set_ylabel('tp (s)')
    # ax3.set_xlim([hourlyTime[0],hourlyTime[-1]])
    # ax3.set_ylim([2,10])


    #
    #
    # plt.style.use('dark_background')
    # plt.figure()
    # ax1 = plt.subplot2grid((3,1),(0,0))
    # ax2 = plt.subplot2grid((3,1),(1,0))
    # ax3 = plt.subplot2grid((3,1),(2,0))
    # # ax4 = plt.subplot2grid((4,1),(3,0))
    # versn = 3
    #
    # # ax1.plot(wlArrayTime,wlData['wl_all'])
    # ax1.plot(hourlyTime,ss[versn])
    #
    # # ax1.set_xlim([dt.datetime(1979,1,1), dt.datetime(2022,6,1)])
    # ax1.set_ylabel('wl (m)')
    # ax2.plot(hourlyTime,hs[versn])
    # # ax2.plot(environTime,environData['hs'])
    # # ax2.set_xlim([dt.datetime(1979,1,1), dt.datetime(2022,6,1)])
    # ax2.set_ylabel('hs (m)')
    #
    # tempTP = tp[versn]
    # weirdVals = np.where((tempTP < 1))
    # tempTP[weirdVals] = np.nan
    # ax3.plot(hourlyTime,tempTP)
    # # ax3.plot(environTime,environData['tp'])
    # # ax3.set_xlim([dt.datetime(1979,1,1), dt.datetime(2022,6,1)])
    # ax3.set_ylabel('tp (s)')
    #
    #
    # #
    # # ax4.plot(waveArrayTime,waveData['mwd_all'])
    # # # ax4.plot(environTime,environData['wd'])
    # # ax4.set_xlim([dt.datetime(1979,1,1), dt.datetime(2022,6,1)])
    # # ax4.set_ylabel('mwd (deg)')

    #
    # def datetime2matlabdn(dt):
    #    mdn = dt + timedelta(days = 366)
    #    frac_seconds = (dt-datetime.datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
    #    frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    #    return mdn.toordinal() + frac_seconds + frac_microseconds

    mdict = dict()
    mdict['Hs'] = allHs
    mdict['Tp'] = allTp
    mdict['Dm'] = allDm
    mdict['Ss'] = allSs
    mdict['T2m'] = allT2m
    # mdict['tide'] = ptlay_tide
    # mdict['msl'] = ptlay_msl
    mdict['tide'] = utqiagvik_tide
    mdict['msl'] = utqiagvik_msl
    # mdict['tide'] = pthope_tide
    # mdict['msl'] = pthope_msl
    # mdict['tide'] = shish_tide
    # mdict['msl'] = shish_msl
    # mdict['tide'] = kivalina_tide
    # mdict['msl'] = kivalina_msl
    # mdict['tide'] = ptlay_tide
    # mdict['msl'] = ptlay_msl
    # mdict['tide'] = wales_tide
    # mdict['msl'] = wales_msl
    # mdict['tide'] = wevok_tide
    # mdict['msl'] = wevok_msl
    # mdict['tide'] = wainright_tide
    # mdict['msl'] = wainright_msl
    mdict['time'] = simTime
    #
    #
    from scipy.io import savemat
    # savemat('arcticFutureTidesMSL.mat',mdict)
    # savemat('utqiagvikSimulations901to1000.mat',mdict)
    # savemat('pointHopeSimulations901to1000.mat',mdict)
    # # savemat('shishmarefSimulations400to500.mat',mdict)
    savemat('/volumes/macDrive/arcticSims/utqiagvikSimulations{}to{}.mat'.format(beginSim+1,beginSim+100),mdict)




#
# plt.figure()
# p1 = plt.subplot2grid((1,1),(0,0))
# p1.plot(df.index,df.t2m)
# # simSeasonalMean = df.groupby('month').mean()['t2m']
# simSeasonalMean2 = df.resample('m').mean()['t2m']
#
# # p2 = plt.subplot2grid((2,1),(1,0))
# p1.plot(simSeasonalMean2.index,simSeasonalMean2)
#
#
# plt.figure()
# p1 = plt.subplot2grid((1,1),(0,0))
# p1.plot(wavesWinds['metOcean'].timeWave,wavesWinds['metOcean'].t2m)
# # # simSeasonalMean = df.groupby('month').mean()['t2m']
# # simSeasonalMean2 = df.resample('m').mean()['t2m']
# #
# # # p2 = plt.subplot2grid((2,1),(1,0))
# # p1.plot(simSeasonalMean2.index,simSeasonalMean2)
