import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# with open(r"realWavesPointHope.pickle", "rb") as input_file:
# with open(r"realWavesShishmaref.pickle", "rb") as input_file:
with open(r"realWavesWainwright.pickle", "rb") as input_file:
   wavesInput = pickle.load(input_file)

tWave = wavesInput['tWave']#[5:]
tC = tWave#[0:-(2929+(365*24))]
# tC = wavesInput['tC'][5:]

t2m = wavesInput['t2m']#[5:]


hsCombined = wavesInput['hsCombined']
nanInd = np.where((hsCombined==0))
hsCombined[nanInd] = np.nan * np.ones((len(nanInd)))
tpCombined = wavesInput['tpCombined']#[5:]
nanInd = np.where((tpCombined==0))
tpCombined[nanInd] = np.nan * np.ones((len(nanInd)))

dmCombined = wavesInput['dmCombined']#[5:]
nanInd = np.where((dmCombined==0))
dmCombined[nanInd] = np.nan * np.ones((len(nanInd)))

data = np.array([hsCombined,tpCombined,dmCombined,t2m-273.15])
ogdf = pd.DataFrame(data=data.T, index=tC, columns=["hs", "tp", "dm","t2m"])
year = np.array([tt.year for tt in tC])
ogdf['year'] = year
month = np.array([tt.month for tt in tC])
ogdf['month'] = month

seasonalMean = ogdf.groupby('month').mean()
seasonalStd = ogdf.groupby('month').std()
yearlyMax = ogdf.groupby('year').max()

g2 = ogdf.groupby(pd.Grouper(freq="M")).mean()

from datetime import datetime
from dateutil.relativedelta import relativedelta

st = datetime(2022, 1, 1)
end = datetime(2023, 1, 1)
step = relativedelta(months=1)
plotTime = []
while st < end:
    plotTime.append(st)#.strftime('%Y-%m-%d'))
    st += step

numSim = 1
# file = ('/volumes/macDrive/arcticSims/pointHope/futureSimulation{}.pickle'.format(numSim))
# file = ('/volumes/macDrive/arcticSims/shishmaref/futureSimulation{}.pickle'.format(numSim))
file = ('/volumes/macDrive/arcticSims/wainwright/futureSimulation{}.pickle'.format(numSim))

with open(file, "rb") as input_file:
    simsInput = pickle.load(input_file)
simulationData = simsInput['simulationData']
simdf = simsInput['df']
simTime = simdf.index
from datetime import datetime, timedelta
#
from scipy.io import loadmat
# sims100 = loadmat('/volumes/macDrive/arcticSims/pointHopeSimulations1to100.mat')
# sims100 = loadmat('/volumes/macDrive/arcticSims/shishmarefSimulations1to100.mat')
sims100 = loadmat('/volumes/macDrive/arcticSims/wainwrightSimulations1to100.mat')

simTime = sims100['time']
python_datetime = [datetime.fromordinal(int(tt)) + timedelta(days=tt%1) - timedelta(days=366) for tt in simTime[0]]
# data = np.array([sims100['Hs'][0,:],sims100['Tp'][0,:],sims100['Dm'][0,:],sims100['T2m'][0,:]])
# simdf = pd.DataFrame(data=data.T, index=python_datetime, columns=["hs", "tp", "dm","t2m"])
year = np.array([tt.year for tt in np.asarray(python_datetime)])
simdf['year'] = year
month = np.array([tt.month for tt in np.asarray(python_datetime)])
simdf['month'] = month
simdf['t2m'] = simdf['t2m']-273.15


seasonalMeanSim = simdf.groupby('month').mean()
seasonalStdSim = simdf.groupby('month').std()

plt.figure()
plt.plot(plotTime,seasonalMean.t2m,label='real temps')
plt.fill_between(plotTime, seasonalMean.t2m - seasonalStd.t2m, seasonalMean.t2m + seasonalStd.t2m, color='blue', alpha=0.2)

plt.plot(plotTime,seasonalMeanSim.t2m,label='simulation')
plt.fill_between(plotTime, seasonalMeanSim.t2m - seasonalStdSim.t2m, seasonalMeanSim.t2m + seasonalStdSim.t2m, color='orange', alpha=0.2)
plt.legend()


