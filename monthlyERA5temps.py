import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_in = "/media/dylananderson/Elements/pointHope/"
path_out = "./"

df = xr.open_dataset(path_in + 'ERA5monthlyTemps.nc')
print(df)
print(df.dims)

df2 = xr.open_dataset('ERA5monthlyMay2022.nc')
print(df2.dims)

temps = df['t2m']-273.15
temps2 = df2['t2m']-273.15
aoTemps = temps[:,:,0:90,:].mean(dim='expver',skipna=True)

aoTemps2 = temps2[:,0:90,:]
# aoTemps3 = aoTemps.drop('expver')
tempCon = xr.concat([aoTemps,aoTemps2],dim='time')

temp1 = tempCon[:,0:90,:].mean(dim=['longitude','latitude'])

# testTemp1 = aoTemps[:,0:90,:].mean(dim=['longitude','latitude'])


yearly = temp1.rolling(time=12).mean()

timeTemp = tempCon['time'].values




from datetime import datetime

pdt = pd.to_datetime(timeTemp)
ts = pdt.to_pydatetime()



c = 0
borealTemp = []
for tt in range(43):
    borealTemp.append(temp1[5+c:5+c+12].values.mean())
    c = c + 12

# plt.figure()
# # plt.plot(ts,temp1)
# plt.plot(ts,temp1)
# plt.plot(ts,yearly)
# plt.plot(ts[6+6::12],borealTemp)
#

pdf = temp1.to_dataframe()

pdfy = pdf.squeeze()

y_to_train = pdfy[:'2016-05-01'] # dataset to train
y_to_val = pdfy['2016-06-01':] # last X months for test
predict_date = len(pdfy) - len(pdfy[:'2016-05-01']) # the number of data points for the test set



import statsmodels.api as sm

# graphs to show seasonal_decompose
def seasonal_decompose (y):
    decomposition = sm.tsa.seasonal_decompose(y, model='additive',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()

# seasonal_decompose(pdf)

decomposition = sm.tsa.seasonal_decompose(pdf, model='additive', extrapolate_trend='freq')

sl = decomposition.seasonal
rl = decomposition.resid
tr = decomposition.trend






#
#
#
# ### plot for Rolling Statistic for testing Stationarity
# def test_stationarity(timeseries, title):
#     # Determing rolling statistics
#     rolmean = pd.Series(timeseries).rolling(window=12).mean()
#     rolstd = pd.Series(timeseries).rolling(window=12).std()
#
#     fig, ax = plt.subplots(figsize=(16, 4))
#     ax.plot(timeseries, label=title)
#     ax.plot(rolmean, label='rolling mean');
#     ax.plot(rolstd, label='rolling std (x10)');
#     ax.legend()
#
# # pd.options.display.float_format = '{:.8f}'.format
# test_stationarity(pdfy,'raw data')
#
# import itertools
#
#
# def sarima_grid_search(y, seasonal_period):
#     p = d = q = range(0, 2)
#     pdq = list(itertools.product(p, d, q))
#     seasonal_pdq = [(x[0], x[1], x[2], seasonal_period) for x in list(itertools.product(p, d, q))]
#
#     mini = float('+inf')
#
#     for param in pdq:
#         for param_seasonal in seasonal_pdq:
#             try:
#                 mod = sm.tsa.statespace.SARIMAX(y,
#                                                 order=param,
#                                                 seasonal_order=param_seasonal,
#                                                 enforce_stationarity=False,
#                                                 enforce_invertibility=False)
#
#                 results = mod.fit()
#
#                 if results.aic < mini:
#                     mini = results.aic
#                     param_mini = param
#                     param_seasonal_mini = param_seasonal
#
#             #                 print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
#             except:
#                 continue
#     print('The set of parameters with the minimum AIC is: SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini,
#                                                                                        mini))
#
# sarima_grid_search(pdfy,12)
#
#
# # Call this function after pick the right(p,d,q) for SARIMA based on AIC
# def sarima_eva(y, order, seasonal_order, seasonal_period, pred_date, y_to_test):
#     # fit the model
#     mod = sm.tsa.statespace.SARIMAX(y,
#                                     order=order,
#                                     seasonal_order=seasonal_order,
#                                     enforce_stationarity=False,
#                                     enforce_invertibility=False)
#
#     results = mod.fit()
#     print(results.summary().tables[1])
#
#
#
#     # results.plot_diagnostics(figsize=(16, 8))
#     # plt.show()
#
#     # The dynamic=False argument ensures that we produce one-step ahead forecasts,
#     # meaning that forecasts at each point are generated using the full history up to that point.
#     pred = results.get_prediction(start=pd.to_datetime(pred_date), dynamic=False)
#     pred_ci = pred.conf_int()
#     y_forecasted = pred.predicted_mean
#     mse = ((y_forecasted - y_to_test) ** 2).mean()
#     print('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = False {}'.format(seasonal_period,
#                                                                                                       round(
#                                                                                                           np.sqrt(mse),
#                                                                                                           2)))
#
#     ax = y.plot(label='observed')
#     y_forecasted.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
#     ax.fill_between(pred_ci.index,
#                     pred_ci.iloc[:, 0],
#                     pred_ci.iloc[:, 1], color='k', alpha=.2)
#
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Sessions')
#     plt.legend()
#     plt.show()
#
#     # A better representation of our true predictive power can be obtained using dynamic forecasts.
#     # In this case, we only use information from the time series up to a certain point,
#     # and after that, forecasts are generated using values from previous forecasted time points.
#     pred_dynamic = results.get_prediction(start=pd.to_datetime(pred_date), dynamic=True, full_results=True)
#     pred_dynamic_ci = pred_dynamic.conf_int()
#     y_forecasted_dynamic = pred_dynamic.predicted_mean
#     mse_dynamic = ((y_forecasted_dynamic - y_to_test) ** 2).mean()
#     print('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = True {}'.format(seasonal_period,
#                                                                                                      round(np.sqrt(
#                                                                                                          mse_dynamic),
#                                                                                                            2)))
#
#     ax = y.plot(label='observed')
#     y_forecasted_dynamic.plot(label='Dynamic Forecast', ax=ax, figsize=(14, 7))
#     ax.fill_between(pred_dynamic_ci.index,
#                     pred_dynamic_ci.iloc[:, 0],
#                     pred_dynamic_ci.iloc[:, 1], color='k', alpha=.2)
#
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Sessions')
#
#     plt.legend()
#     plt.show()
#
#     return (results)
#
# model = sarima_eva(pdfy,(1, 1, 1),(0, 1, 1, 12),12,'2016-06-01 00:00:00',y_to_val)
#




from scipy.optimize import curve_fit
import time

from dateutil.relativedelta import relativedelta
from datetime import date
dt = date(1979, 1, 1)
end = date(2022, 6, 1)
step = relativedelta(days=1)
atTime = []
while dt < end:
    atTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step


dtime=[]
for i in ts:
    dtime.append(time.mktime(i.timetuple()))

dailyTime = []
for i in atTime:
    dailyTime.append(time.mktime(i.timetuple()))

dt2 = date(1979, 6, 1)
end2 = date(2050, 6, 1)
step2 = relativedelta(days=1)
atTime2 = []
while dt2 < end2:
    atTime2.append(dt2)#.strftime('%Y-%m-%d'))
    dt2 += step2


dailyTime2 = []
for i in atTime2:
    dailyTime2.append(time.mktime(i.timetuple()))



DATE = [datetime.utcfromtimestamp(x) for x in dtime]
DAILYDATE = [datetime.utcfromtimestamp(x) for x in dailyTime]
DAILYDATE2 = [datetime.utcfromtimestamp(x) for x in dailyTime2]

def objective(x,a,b,c):
    return a * x**2 + b * x + c

popt, _ = curve_fit(objective, np.array(dtime), tr.values.squeeze())

a, b, c = popt
print('y = %.5f * x^2 + %.5f * x + %.5f' % (a, b, c))

# define a sequence of inputs between the smallest and largest known inputs
x_line = np.array(dailyTime)#np.arange(np.nanmin(d), np.nanmax(d), 1)
# calculate the output for the range
y_line = objective(x_line, a, b, c)
# create a line plot for the mapping function
plt.plot(DAILYDATE, y_line, '--', color='red')

# define a sequence of inputs between the smallest and largest known inputs
x_lineFuture = np.array(dailyTime2)#np.arange(np.nanmin(d), np.nanmax(d), 1)
# calculate the output for the range
y_lineFuture = objective(x_lineFuture, a, b, c)
# create a line plot for the mapping function
plt.plot(DAILYDATE2, y_lineFuture, '--', color='green')




def objective2(x,a,b,):
    return a * x*2 + b

popt2, _ = curve_fit(objective2, np.array(dtime), tr.values.squeeze())

a2, b2 = popt2
print('y = %.5f * x + %.5f' % (a, b))

x_lineMonthlyHistorical = np.array(dtime)
y_lineMonthlyHistorical = objective2(x_lineMonthlyHistorical, a2, b2)
y_lineFuture = objective2(x_lineFuture, a2, b2)


plt.figure()
ax1 = plt.subplot2grid((6,1),(0,0))
ax1.plot(ts,tr.values)
ax1.plot(DAILYDATE2,y_lineFuture)
ax1.plot(ts,y_lineMonthlyHistorical)

rd2 = tr.values.squeeze()-y_lineMonthlyHistorical
ax2 = plt.subplot2grid((6,1),(1,0))
ax2.plot(ts,rd2)

# Test function with coefficients as parameters
def testSine(x, a, b, c):
    return a * np.sin(b * x + c)
# curve_fit() function takes the test-function
# x-data and y-data as argument and returns
# the coefficients a and b in param and
# the estimated covariance of param in param_cov
dailyNumber = np.array(dtime)/(3600*24*365.25*10)
param, param_cov = curve_fit(testSine, dailyNumber, rd2)
ansSine = (param[0]*(np.sin(param[1]*dailyNumber+param[2])))
# ansSine = (0.5*(np.sin((2*np.pi/20)*dailyNumber)))
dailyNumber2 = np.array(dailyTime2)/(3600*24*365.25*10)
ansSineFuture = (param[0]*(np.sin(param[1]*dailyNumber2+param[2])))
ax2.plot(DAILYDATE2,ansSineFuture)
ax2.plot(ts,ansSine)


rd3 = rd2-ansSine
ax3 = plt.subplot2grid((6,1),(2,0))
ax3.plot(ts,rd3)

dailyNumber = np.array(dtime)/(3600*24*365.25*1)
param, param_cov = curve_fit(testSine, dailyNumber, rd3)
ansSine2 = (param[0]*(np.sin(param[1]*dailyNumber+param[2])))
# ansSine = (0.5*(np.sin((2*np.pi/20)*dailyNumber)))
dailyNumber2 = np.array(dailyTime2)/(3600*24*365.25*1)
ansSine2Future = (param[0]*(np.sin(param[1]*dailyNumber2+param[2])))

ax3.plot(DAILYDATE2,ansSine2Future)
ax3.plot(ts,ansSine2)


rd4 = rd3-ansSine2
ax4 = plt.subplot2grid((6,1),(3,0))
ax4.plot(ts,rd4)

dailyNumber = np.array(dtime)/(3600*24*365.25*1)
param, param_cov = curve_fit(testSine, dailyNumber, rd4)
ansSine3 = (param[0]*(np.sin(param[1]*dailyNumber+param[2])))
# ansSine = (0.5*(np.sin((2*np.pi/20)*dailyNumber)))
dailyNumber2 = np.array(dailyTime2)/(3600*24*365.25*1)
ansSine3Future = (param[0]*(np.sin(param[1]*dailyNumber2+param[2])))

ax4.plot(DAILYDATE2,ansSine3Future)
ax4.plot(ts,ansSine3)

rd5 = rd4-ansSine3

ax5 = plt.subplot2grid((6,1),(4,0))
ax5.plot(ts,rd5)

dailyNumber = np.array(dtime)/(3600*24*365.25*0.255)
param, param_cov = curve_fit(testSine, dailyNumber, rd5)
ansSine4 = (param[0]*(np.sin(param[1]*dailyNumber+param[2])))
# ansSine = (0.5*(np.sin((2*np.pi/20)*dailyNumber)))
dailyNumber2 = np.array(dailyTime2)/(3600*24*365.25*0.255)
ansSine4Future = (param[0]*(np.sin(param[1]*dailyNumber2+param[2])))

ax5.plot(DAILYDATE2,ansSine4Future)
ax5.plot(ts,ansSine4)

ax6 = plt.subplot2grid((6,1),(5,0))
ax6.plot(ts,tr.values)
ax6.plot(DAILYDATE2,ansSineFuture+ansSine2Future+ansSine3Future+ansSine4Future+y_lineFuture)
ax6.plot(ts,ansSine+ansSine2+ansSine3+ansSine4+y_lineMonthlyHistorical)


futureTrend = ansSineFuture+ansSine2Future+ansSine3Future+ansSine4Future+y_lineFuture


def testSineSeasonal(x, a, b, c, d):
    return a * np.sin(2*np.pi * x + b) + c * np.cos(2*np.pi * x + d)

sl = decomposition.seasonal.values.squeeze() - (np.max(decomposition.seasonal.values.squeeze())+np.min(decomposition.seasonal.values.squeeze()))/2
dailyNumber = np.array(dtime)/(3600*24*365.25)
dailyNumber2 = np.array(dailyTime2)/(3600*24*365.25)
param, param_cov = curve_fit(testSineSeasonal, dailyNumber, sl)
seasonalSine = (param[0]*(np.sin(2*np.pi*dailyNumber+param[1]))) + (param[2]*(np.cos(2*np.pi*dailyNumber+param[3])))
seasonalSineFuture = (param[0]*(np.sin(2*np.pi*dailyNumber2+param[1]))) + (param[2]*(np.cos(2*np.pi*dailyNumber2+param[3]))) + (np.max(decomposition.seasonal.values.squeeze())+np.min(decomposition.seasonal.values.squeeze()))/2

# plt.figure()
# plt.plot(ts,sl)
# plt.plot(DAILYDATE2,seasonalSineFuture)
# plt.plot(ts,seasonalSine)
#



plt.figure()
plt.plot(ts,pdf)
plt.plot(DAILYDATE2,seasonalSineFuture+futureTrend)

futureTemps = seasonalSineFuture+futureTrend


import datetime as dt
from dateutil.relativedelta import relativedelta
st = dt.datetime(1979, 1, 1)
# end = dt.datetime(2021,12,31)
end = dt.datetime(2022,6,1)
step = relativedelta(days=1)
dayTime = []
while st < end:
    dayTime.append(st)#.strftime('%Y-%m-%d'))
    st += step


st2 = dt.datetime(1979, 6, 1)
# end = dt.datetime(2021,12,31)
end2 = dt.datetime(2050,6,1)
step2 = relativedelta(days=1)
dayTime2 = []
while st2 < end2:
    dayTime2.append(st2)#.strftime('%Y-%m-%d'))
    st2 += step2



dt3 = date(1979, 1, 1)
end3 = date(2050, 7, 1)
step3 = relativedelta(months=1)
monthlyTime = []
while dt3 < end3:
    monthlyTime.append(dt3)#.strftime('%Y-%m-%d'))
    dt3 += step3

dtimeFuture=[]
for i in monthlyTime:
    dtimeFuture.append(time.mktime(i.timetuple()))



plt.figure()
# plt.hist(rl.values.squeeze())
plt.plot(ts,rl.values.squeeze())
possibleRes = rl.values.squeeze()
lengthHistory = len(possibleRes)
numSims = 100
sims = []
for hh in range(numSims):
    simInts = np.random.randint(521, size=337)
    newRes = possibleRes[simInts]
    combinedRes = np.hstack((possibleRes,newRes))

    interpTempFuture = np.interp(np.array(dailyTime2), np.array(dtimeFuture), combinedRes)+futureTemps


    sims.append(interpTempFuture)

# plt.figure()
# plt.plot(dayTime2,interpTempFuture)
# #
# # define a sequence of inputs between the smallest and largest known inputs
# x_line3 = np.array(dailyTime)#np.arange(np.nanmin(d), np.nanmax(d), 1)
# # calculate the output for the range
# y_line3 = objective2(x_line3, a2, b2)
# # create a line plot for the mapping function
# plt.plot(DAILYDATE, y_line3, '-.', color='red')
#
# # define a sequence of inputs between the smallest and largest known inputs
# x_lineFuture2 = np.array(dailyTime2)#np.arange(np.nanmin(d), np.nanmax(d), 1)
# # calculate the output for the range
# y_lineFuture2 = objective2(x_lineFuture2, a2, b2)
# # create a line plot for the mapping function
# plt.plot(DAILYDATE2, y_lineFuture2, '--', color='green')



interpTemp = np.interp(np.array(dailyTime),np.array(dtime),temp1.values)

# plt.figure()
# plt.plot(ts,temp1.values,'o')
# plt.plot(DAILYDATE,interpTemp)



import pickle

dwtPickle = 'predictedArcticTemp.pickle'
outputDWTs = {}
outputDWTs['dailyDate'] = dayTime
outputDWTs['arcticTemp'] = interpTemp
outputDWTs['futureDate'] = dayTime2
outputDWTs['futureTemp'] = futureTemps
outputDWTs['futureSims'] = sims


with open(dwtPickle,'wb') as f:
    pickle.dump(outputDWTs, f)





asdfg
#
# residuals = yearly[11:-2].values-y_line
#
#
#
#
#
# c = 0
# borealTemp = []
# for tt in range(43):
#     borealTemp.append(temp1[5+c:5+c+12].values.mean())
#     c = c + 12
#
# plt.figure()
# plt.plot(ts[11:-2],residuals)
# plt.plot(ts[8+6::12],borealTemp-y_line[1::12])
#
# residual2 = borealTemp-y_line[1::12]
#
#
#
#
#
# # import pickle
# #
# # with open(r"AWT1880to2020.pickle", "rb") as input_file:
# #    awt = pickle.load(input_file)
# #
# # predictor = awt['predictor']
# #
# # plt.figure()
# # plt.plot(ts[11:-2],residuals)
# # plt.plot(ts[8+6::12],borealTemp-y_line[1::12])
# # plt.plot(predictor.time,predictor.PCs.values[:,0]/np.max(predictor.PCs.values[:,0]))
# #
# #
# # PC1 = predictor.PCs.values[99:,0]
# # PC2 = predictor.PCs.values[99:,1]
# # PC3 = predictor.PCs.values[99:,2]
# #
# #
# # plt.figure()
# # plt.plot(PC1,residual2[0:-1],'.')
#
# file = 'AO.txt'
# data = pd.read_csv(file,header=None,delim_whitespace=True)
#
# aoTime = [datetime(data[0][hh],data[1][hh],1) for hh in range(len(data))]
# aoValues = data[2]
#
# def moving_average(a, n=3) :
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n
#
# plt.figure()
# plt.plot(aoTime[6:-5],moving_average(np.array(aoValues),12))
# plt.plot(ts[11:-2],residuals)
#
#


#
# import numpy, scipy.optimize
#
# def fit_sin(tt, yy):
#     '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
#     tt = numpy.array(tt)
#     yy = numpy.array(yy)
#     ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
#     Fyy = abs(numpy.fft.fft(yy))
#     guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
#     guess_amp = numpy.std(yy) * 2.**0.5
#     guess_offset = numpy.mean(yy)
#     guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, 0., guess_offset])
#
#     def sinfunc(t, A, w, p, c):  return A * numpy.sin(w*t + p) + c
#     popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
#     A, w, p, c = popt
#     f = w/(2.*numpy.pi)
#     fitfunc = lambda t: A * numpy.sin(w*t + p) + c
#     return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}
#
# #
# # # define the true objective function
# # def objective2(x, a, b, c):
# #     return a * np.sin(2*np.pi/b*x + c)
# # popt, _ = curve_fit(objective2, np.array(dtime[11:-2]), residuals)
# # a2, b2, c2 = popt
# # print('y = %.5f sin %.5f x %.5f ' % (a2, b2, c2))
# # # define a sequence of inputs between the smallest and largest known inputs
# # x_line = np.array(dtime[11:-2])#np.arange(np.nanmin(d), np.nanmax(d), 1)
# # # calculate the output for the range
# # y_line2 = objective2(x_line, a2, b2, c2)#, c, d)
# # # create a line plot for the mapping function
#
# res = fit_sin(x_line,residuals)
#
# plt.figure()
# plt.plot(ts[11:-2],residuals)
# plt.plot(ts[11:-2], res["fitfunc"](x_line), '--', color='green')
#
# # our_dictionary = {'time' : ts, 'temp': temp1, 'lat':df['latitude'][0:100], 'lon':df['longitude']}
# # # our_dictionary = {'time' : pd.to_datetime(time)}
# # pdf = pd.DataFrame(our_dictionary, columns=['time','temp','lat','lon'])
# #
