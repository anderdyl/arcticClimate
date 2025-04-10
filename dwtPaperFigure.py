from dateutil.relativedelta import relativedelta

import numpy as np
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import windrose
from matplotlib import gridspec
from mpl_toolkits.basemap import Basemap
import pickle
from matplotlib.colors import Normalize

# with open(r"dwtsAll6TCTracksALLDATA.pickle", "rb") as input_file:
with open(r"dwts49ClustersArctic2023.pickle", "rb") as input_file:
   historicalDWTs = pickle.load(input_file)

sorted_centroidsET = historicalDWTs['sorted_centroids']
X_inET = historicalDWTs['X_in']
Y_inET = historicalDWTs['Y_in']
kma_orderET = historicalDWTs['kma_order']
SLPET = historicalDWTs['SLP']
group_sizeET = historicalDWTs['group_size']

bmus = historicalDWTs['bmus_corrected']
lon = historicalDWTs['lon']
lat = historicalDWTs['lat']
sea_nodes = historicalDWTs['sea_nodes']

dwtcolors = cm.rainbow(np.linspace(0, 1, 50))





# plotting the EOF patterns
fig2 = plt.figure(figsize=(6.5,9))
gs1 = gridspec.GridSpec(7, 7)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for hh in range(49):
    #p1 = plt.subplot2grid((6,6),(c1,c2))
    ax = plt.subplot(gs1[hh])

    if hh <= 48:
        print('working on plotting DWT {}'.format(hh))
        num = kma_orderET[hh]
        finder = np.where((bmus == hh))
        SLPtemp = SLPET[finder[0],:]

        #spatialField = Km_slpET[(num - 1), :] / 100 - np.nanmean(SLPET, axis=0) / 100
        # spatialField = Km_slpET[(hh), :] / 100 - np.nanmean(SLPET, axis=0) / 100
        spatialField = np.nanmean(SLPtemp,axis=0) / 100 - np.nanmean(SLPET, axis=0) / 100

        m = Basemap(projection='npstere', boundinglat=50, lon_0=180, resolution='l')

        cx, cy = m(lon, lat)


        rectField = np.ones((np.shape(X_inET))) * np.nan
        for tt in range(len(sea_nodes)):
            rectField[sea_nodes[tt]] = spatialField[tt]

        clevels = np.arange(-27,27,1)
        m = Basemap(projection='npstere', boundinglat=50, lon_0=180, resolution='l')

        cx, cy = m(lon, lat)
        m.drawcoastlines()

        rectField = np.ones((np.shape(X_inET))) * np.nan
        for tt in range(len(sea_nodes)):
            rectField[sea_nodes[tt]] = spatialField[tt]

        # ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
        # gl = ax.gridlines(draw_labels=True)
        # extent = [-9.97, 168.35, 30.98, 34.35]

        clevels = np.arange(-70, 70, 1)

        # ax.pcolormesh(cx, cy, rectField)#, cmap=cmocean.cm.ice)
        CS = m.contourf(cx, cy, rectField, clevels, vmin=-27, vmax=27, cmap=cm.RdBu_r)#, shading='gouraud')
        # ax.set_xlim([-3223000, 0])
        # ax.set_ylim([0, 2730000])
        ax.set_xlim([np.min(cx) + 2400000, np.max(cx) - 2200000])
        ax.set_ylim([np.min(cy) + 900000, np.max(cy) - 1700000])
        m.fillcontinents(color=dwtcolors[hh])

        #p1.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
        # tx,ty = m(323,-27)
        # ax.text(tx,ty,'{}'.format(group_sizeET[num]))


    c2 += 1
    if c2 == 6:
        c1 += 1
        c2 = 0

    if plotIndx < 8:
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    counter = counter + 1
    if plotIndy < 8:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1













import pickle
import numpy as np
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def GenOneYearDaily(yy=1981, month_ini=1):
   'returns one generic year in a list of datetimes. Daily resolution'

   dp1 = datetime.datetime(yy, month_ini, 1)
   dp2 = dp1 + datetime.timedelta(days=365)

   return [dp1 + datetime.timedelta(days=i) for i in range((dp2 - dp1).days)]



def dateDay2datetimeDate(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [datetime.date(d[0], d[1], d[2]) for d in d_vec]

def mVec2datetimeDate(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [datetime.date(int(d_vec[hh,0]), int(d_vec[hh,1]), int(d_vec[hh,2])) for hh in range(len(d_vec[:,0]))]


timeDWTs = historicalDWTs['SLPtime']
# outputDWTs['slpDates'] = slpDates
dwtBmus = historicalDWTs['bmus_corrected']

timeDWTs = timeDWTs[120:]
bmus = dwtBmus[120:]
bmus_dates = mVec2datetimeDate(timeDWTs)
bmus_dates = bmus_dates#[120:]

bmus_dates_months = np.array([d.month for d in bmus_dates])
bmus_dates_days = np.array([d.day for d in bmus_dates])


# generate perpetual year list
list_pyear = GenOneYearDaily(month_ini=1)
m_plot = np.zeros((49, len(list_pyear))) * np.nan
num_clusters=49
num_sim=1
# sort data
for i, dpy in enumerate(list_pyear):
   _, s = np.where(
      [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)]
   )
   #b = bmus_values[s, :]
   b = bmus[s]
   b = b.flatten()

   for j in range(num_clusters):
      _, bb = np.where([(j == b)])  # j+1 starts at 1 bmus value!

      m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)




fig = plt.figure()
ax = plt.subplot2grid((1,1),(0,0))
# plot stacked bars
bottom_val = np.zeros(m_plot[1, :].shape)
for r in range(num_clusters):
   row_val = m_plot[r, :]
   ax.bar(
      list_pyear, row_val, bottom=bottom_val,
      width=1, color=np.array([dwtcolors[r]])
   )

   # store bottom
   bottom_val += row_val

# customize  axis
months = mdates.MonthLocator()
monthsFmt = mdates.DateFormatter('%b')

ax.set_xlim(list_pyear[0], list_pyear[-1])
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.set_ylim(0, 1)
ax.set_ylabel('Probability')



#### AWT FROM ENSO SSTs
with open(r"AWT1880to2023.pickle", "rb") as input_file:
   historicalAWTs = pickle.load(input_file)
awtClusters = historicalAWTs['clusters']
awtPredictor = historicalAWTs['predictor']

awtBmus = awtClusters.bmus.values
pc1Annual = awtClusters.PCs[:,0]
pc2Annual = awtClusters.PCs[:,1]
pc3Annual = awtClusters.PCs[:,2]

awtVariance = awtPredictor['variance'].values
nPercent = awtVariance / np.sum(awtVariance)
import datetime as DT
dt = DT.datetime(1880, 6, 1)
end = DT.datetime(2023, 6, 1)
#step = datetime.timedelta(months=1)
step = relativedelta(years=1)
sstTime = []
while dt < end:
    sstTime.append(dt)
    dt += step

years = np.arange(1979,2023)
awtYears = np.arange(1880,2023)

awtDailyBmus = np.nan * np.ones(np.shape(bmus))
PC1 = np.nan * np.ones(np.shape(bmus))
PC2 = np.nan * np.ones(np.shape(bmus))
PC3 = np.nan * np.ones(np.shape(bmus))

for hh in years:
   indexDWT = np.where((np.asarray(bmus_dates) >= DT.date(hh,6,1)) & (np.asarray(bmus_dates) <= DT.date(hh+1,5,31)))
   indexAWT = np.where((awtYears == hh))
   awtDailyBmus[indexDWT] = awtBmus[indexAWT]*np.ones(len(indexDWT[0]))
   PC1[indexDWT] = pc1Annual[indexAWT]*np.ones(len(indexDWT[0]))
   PC2[indexDWT] = pc2Annual[indexAWT]*np.ones(len(indexDWT[0]))
   PC3[indexDWT] = pc3Annual[indexAWT]*np.ones(len(indexDWT[0]))

dailyAWT = awtDailyBmus

from dateutil.relativedelta import relativedelta

###### LOADING A SIMULATED SWT AND ALIGNING WITH JUNE/MAY and daily values
# simulated seasonal
with open(r"awtSimulationsENSO.pickle", "rb") as input_file:
   simSWTs = pickle.load(input_file)
swtBMUS = simSWTs['evbmus_sim']
swtPC1 = simSWTs['pc1Sims']
swtPC2 = simSWTs['pc2Sims']
swtPC3 = simSWTs['pc3Sims']
#swtPC4 = simSWTs['pc4Sims']
swtDatesSim = simSWTs['dates_sim']
awtDatesSim = simSWTs['dates_sim'][0::12]


with open(r"dwt49FutureSimulations500.pickle", "rb") as input_file:
   simDWTs = pickle.load(input_file)

evbmus_sim = simDWTs['evbmus_sim']
sim_years = simDWTs['sim_years']
dates_sim = simDWTs['dates_sim']
awtBMUSsim = simDWTs['awtBMUsim']
awtPC1sim = simDWTs['awtPC1sim']
awtPC2sim = simDWTs['awtPC2sim']
awtPC3sim = simDWTs['awtPC3sim']
mjoRMM1Sim = simDWTs['mjoRMM1Sim']
mjoRMM2Sim = simDWTs['mjoRMM2Sim']

d1 = DT.datetime(2022, 6, 1)
dt = DT.datetime(2022, 6, 1)
end = DT.datetime(2122, 6, 2)
# step = datetime.timedelta(months=1)
step = relativedelta(days=1)
simDailyTime = []
while dt < end:
    simDailyTime.append(dt)
    dt += step
simDailyDatesMatrix = np.array([[r.year,r.month,r.day] for r in simDailyTime])

dailyAWT = np.ones(len(simDailyDatesMatrix),)
awtDaily = []
pc1Daily = []
for simIndex in range(100):
    awtBMUtemp = awtBMUSsim[simIndex][0:100]
    awtPC1temp = awtPC1sim[simIndex][0:100]  # [0:len(awt_bmus)]
    # awtPC2temp = awtPC2sim[simIndex][0:100]  # [0:len(awt_bmus)]
    # awtPC3temp = awtPC3sim[simIndex][0:100]  # [0:len(awt_bmus)]

    # trainingDates = mjoDatesSim#[datetime(r[0],r[1],r[2]) for r in dailyDates]
    trainingDates = [DT.datetime(r[0], r[1], r[2]) for r in simDailyDatesMatrix]
    dailyAWTsim = np.ones((len(trainingDates),))
    dailyPC1sim = np.ones((len(trainingDates),))
    # dailyPC2sim = np.ones((len(trainingDates),))
    # dailyPC3sim = np.ones((len(trainingDates),))

    dailyDatesSWTyear = np.array([r[0] for r in simDailyDatesMatrix])
    dailyDatesSWTmonth = np.array([r[1] for r in simDailyDatesMatrix])
    dailyDatesSWTday = np.array([r[2] for r in simDailyDatesMatrix])
    normPC1 = awtPC1temp
    # normPC2 = awtPC2temp
    # normPC3 = awtPC3temp

    for i in range(len(awtBMUtemp)):
        sSeason = np.where((simDailyDatesMatrix[:, 0] == awtDatesSim[i].year) & (
                simDailyDatesMatrix[:, 1] == awtDatesSim[i].month) & (simDailyDatesMatrix[:, 2] == 1))
        ssSeason = np.where((simDailyDatesMatrix[:, 0] == awtDatesSim[i].year + 1) & (
                simDailyDatesMatrix[:, 1] == awtDatesSim[i].month) & (simDailyDatesMatrix[:, 2] == 1))

        dailyAWTsim[sSeason[0][0]:ssSeason[0][0] + 1] = awtBMUtemp[i] * dailyAWT[sSeason[0][0]:ssSeason[0][0] + 1]
        dailyPC1sim[sSeason[0][0]:ssSeason[0][0] + 1] = normPC1[i] * np.ones(
            len(dailyAWT[sSeason[0][0]:ssSeason[0][0] + 1]), )
        # dailyPC2sim[sSeason[0][0]:ssSeason[0][0] + 1] = normPC2[i] * np.ones(
        #     len(dailyAWT[sSeason[0][0]:ssSeason[0][0] + 1]), )
        # dailyPC3sim[sSeason[0][0]:ssSeason[0][0] + 1] = normPC3[i] * np.ones(
        #     len(dailyAWT[sSeason[0][0]:ssSeason[0][0] + 1]), )
    awtDaily.append(dailyAWTsim)
    pc1Daily.append(dailyPC1sim)



numDWTsAWT1 = np.nan * np.ones((100,49))
numDWTsAWT2 = np.nan * np.ones((100,49))
numDWTsAWT3 = np.nan * np.ones((100,49))
numDWTsAWT4 = np.nan * np.ones((100,49))
numDWTsAWT5 = np.nan * np.ones((100,49))
numDWTsAWT6 = np.nan * np.ones((100,49))
numPC1s = np.nan * np.ones((100,6))
numDWTsTotal = np.nan * np.ones((100,49))


for ff in range(100):
    finder = np.where((awtDaily[ff][0:18992] == 0))
    finder2 = np.where((awtDaily[ff][0:18992] == 1))
    finder3 = np.where((awtDaily[ff][0:18992] == 2))
    finder4 = np.where((awtDaily[ff][0:18992] == 3))
    finder5 = np.where((awtDaily[ff][0:18992] == 4))
    finder6 = np.where((awtDaily[ff][0:18992] == 5))

    f1 = np.where((swtBMUS[ff][0:51] == 0))
    f2 = np.where((swtBMUS[ff][0:51] == 1))
    f3 = np.where((swtBMUS[ff][0:51] == 2))
    f4 = np.where((swtBMUS[ff][0:51] == 3))
    f5 = np.where((swtBMUS[ff][0:51] == 4))
    f6 = np.where((swtBMUS[ff][0:51] == 5))

    dwt1s = evbmus_sim[finder,ff].flatten()
    dwt2s = evbmus_sim[finder2,ff].flatten()
    dwt3s = evbmus_sim[finder3,ff].flatten()
    dwt4s = evbmus_sim[finder4,ff].flatten()
    dwt5s = evbmus_sim[finder5,ff].flatten()
    dwt6s = evbmus_sim[finder6,ff].flatten()

    pc1s = pc1Daily[ff][finder].flatten()
    pc2s = pc1Daily[ff][finder2].flatten()
    pc3s = pc1Daily[ff][finder3].flatten()
    pc4s = pc1Daily[ff][finder4].flatten()
    pc5s = pc1Daily[ff][finder5].flatten()
    pc6s = pc1Daily[ff][finder6].flatten()
    numPC1s[ff,0] = np.nanmean(pc1s)
    numPC1s[ff,1] = np.nanmean(pc2s)
    numPC1s[ff,2] = np.nanmean(pc3s)
    numPC1s[ff,3] = np.nanmean(pc4s)
    numPC1s[ff,4] = np.nanmean(pc5s)
    numPC1s[ff,5] = np.nanmean(pc6s)

    for qq in range(49):
        findDays1 = np.where((dwt1s == qq+1))
        findDays2 = np.where((dwt2s == qq+1))
        findDays3 = np.where((dwt3s == qq+1))
        findDays4 = np.where((dwt4s == qq+1))
        findDays5 = np.where((dwt5s == qq+1))
        findDays6 = np.where((dwt6s == qq+1))

        numDWTsAWT1[ff,qq] = len(findDays1[0])/len(f1[0])
        numDWTsAWT2[ff,qq] = len(findDays2[0])/len(f2[0])
        numDWTsAWT3[ff,qq] = len(findDays3[0])/len(f3[0])
        numDWTsAWT4[ff,qq] = len(findDays4[0])/len(f4[0])
        numDWTsAWT5[ff,qq] = len(findDays5[0])/len(f5[0])
        numDWTsAWT6[ff,qq] = len(findDays6[0])/len(f6[0])
        numDWTsTotal[ff,qq] = (len(findDays1[0])+len(findDays2[0])+len(findDays3[0])+len(findDays4[0])+len(findDays5[0])+len(findDays6[0]))/\
                              (len(f1[0])+len(f2[0])+len(f3[0])+len(f4[0])+len(f5[0])+len(f6[0]))


from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import numpy as np

def boxplot_2d(x,y, ax, whis=1.5, colorP=[0.5,0.5,0.5]):
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0],ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        color=colorP,
        ec = colorP,
        alpha = 0.5,
        zorder=0
    )
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
        color=colorP,
        zorder=1
    )
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
        color=colorP,
        zorder=1
    )
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]],[ylimits[1]], color=colorP, marker='o')

    # ##the x-whisker
    # ##defined as in matplotlib boxplot:
    # ##As a float, determines the reach of the whiskers to the beyond the
    # ##first and third quartiles. In other words, where IQR is the
    # ##interquartile range (Q3-Q1), the upper whisker will extend to
    # ##last datum less than Q3 + whis*IQR). Similarly, the lower whisker
    # ####will extend to the first datum greater than Q1 - whis*IQR. Beyond
    # ##the whiskers, data are considered outliers and are plotted as
    # ##individual points. Set this to an unreasonably high value to force
    # ##the whiskers to show the min and max values. Alternatively, set this
    # ##to an ascending sequence of percentile (e.g., [5, 95]) to set the
    # ##whiskers at specific percentiles of the data. Finally, whis can
    # ##be the string 'range' to force the whiskers to the min and max of
    # ##the data.
    # iqr = xlimits[2]-xlimits[0]
    #
    # ##left
    # left = np.min(x[x > xlimits[0]-whis*iqr])
    # whisker_line = Line2D(
    #     [left, xlimits[0]], [ylimits[1],ylimits[1]],
    #     color = colorP,
    #     zorder = 1
    # )
    # ax.add_line(whisker_line)
    # whisker_bar = Line2D(
    #     [left, left], [ylimits[0],ylimits[2]],
    #     color = colorP,
    #     zorder = 1
    # )
    # ax.add_line(whisker_bar)
    #
    # ##right
    # right = np.max(x[x < xlimits[2]+whis*iqr])
    # whisker_line = Line2D(
    #     [right, xlimits[2]], [ylimits[1],ylimits[1]],
    #     color = colorP,
    #     zorder = 1
    # )
    # ax.add_line(whisker_line)
    # whisker_bar = Line2D(
    #     [right, right], [ylimits[0],ylimits[2]],
    #     color = colorP,
    #     zorder = 1
    # )
    # ax.add_line(whisker_bar)
    #
    # ##the y-whisker
    # iqr = ylimits[2]-ylimits[0]
    #
    # ##bottom
    # bottom = np.min(y[y > ylimits[0]-whis*iqr])
    # whisker_line = Line2D(
    #     [xlimits[1],xlimits[1]], [bottom, ylimits[0]],
    #     color = colorP,
    #     zorder = 1
    # )
    # ax.add_line(whisker_line)
    # whisker_bar = Line2D(
    #     [xlimits[0],xlimits[2]], [bottom, bottom],
    #     color = colorP,
    #     zorder = 1
    # )
    # ax.add_line(whisker_bar)
    #
    # ##top
    # top = np.max(y[y < ylimits[2]+whis*iqr])
    # whisker_line = Line2D(
    #     [xlimits[1],xlimits[1]], [top, ylimits[2]],
    #     color = colorP,
    #     zorder = 1
    # )
    # ax.add_line(whisker_line)
    # whisker_bar = Line2D(
    #     [xlimits[0],xlimits[2]], [top, top],
    #     color = colorP,
    #     zorder = 1
    # )
    # ax.add_line(whisker_bar)

    # ##outliers
    # mask = (x<left)|(x>right)|(y<bottom)|(y>top)
    # ax.scatter(
    #     x[mask],y[mask],
    #     facecolors='none', edgecolors='k'
    # )








plt.figure()
p100 = plt.subplot2grid((1,1),(0,0))
for qq in range(49):
    #plt.scatter(numDWTsAWT4[:,qq],numDWTsAWT2[:,qq],color=dwtcolors[qq])
    boxplot_2d(numDWTsAWT4[:,qq], numDWTsAWT2[:,qq], ax=p100, whis=1,colorP=dwtcolors[qq])

p100.plot([0,30],[0,30],'--',color='k')


# plt.figure()
# p1 = plt.subplot2grid((5,5),(0,0))
# for qq in range(49):
#     p1.scatter(np.mean(numDWTsAWT1[:,qq]),np.mean(numDWTsAWT2[:,qq]),color=dwtcolors[qq])
# p2 = plt.subplot2grid((5,5),(1,0))
# for qq in range(49):
#     p2.scatter(np.mean(numDWTsAWT1[:,qq]),np.mean(numDWTsAWT3[:,qq]),color=dwtcolors[qq])
# p3 = plt.subplot2grid((5,5),(2,0))
# for qq in range(49):
#     p3.scatter(np.mean(numDWTsAWT1[:,qq]),np.mean(numDWTsAWT4[:,qq]),color=dwtcolors[qq])
# p4 = plt.subplot2grid((5,5),(3,0))
# for qq in range(49):
#     p4.scatter(np.mean(numDWTsAWT1[:,qq]),np.mean(numDWTsAWT5[:,qq]),color=dwtcolors[qq])
# p5 = plt.subplot2grid((5,5),(4,0))
# for qq in range(49):
#     p5.scatter(np.mean(numDWTsAWT1[:,qq]),np.mean(numDWTsAWT6[:,qq]),color=dwtcolors[qq])
#
# p6 = plt.subplot2grid((5,5),(0,1))
# for qq in range(49):
#     p6.scatter(np.mean(numDWTsAWT2[:,qq]),np.mean(numDWTsAWT3[:,qq]),color=dwtcolors[qq])
# p7 = plt.subplot2grid((5,5),(1,1))
# for qq in range(49):
#     p7.scatter(np.mean(numDWTsAWT2[:,qq]),np.mean(numDWTsAWT4[:,qq]),color=dwtcolors[qq])
# p8 = plt.subplot2grid((5,5),(2,1))
# for qq in range(49):
#     p8.scatter(np.mean(numDWTsAWT2[:,qq]),np.mean(numDWTsAWT5[:,qq]),color=dwtcolors[qq])
# p9 = plt.subplot2grid((5,5),(3,1))
# for qq in range(49):
#     p9.scatter(np.mean(numDWTsAWT2[:,qq]),np.mean(numDWTsAWT6[:,qq]),color=dwtcolors[qq])
#
# p10 = plt.subplot2grid((5,5),(0,2))
# for qq in range(49):
#     p10.scatter(np.mean(numDWTsAWT3[:,qq]),np.mean(numDWTsAWT4[:,qq]),color=dwtcolors[qq])
# p11 = plt.subplot2grid((5,5),(1,2))
# for qq in range(49):
#     p11.scatter(np.mean(numDWTsAWT3[:,qq]),np.mean(numDWTsAWT5[:,qq]),color=dwtcolors[qq])
# p12 = plt.subplot2grid((5,5),(2,2))
# for qq in range(49):
#     p12.scatter(np.mean(numDWTsAWT3[:,qq]),np.mean(numDWTsAWT6[:,qq]),color=dwtcolors[qq])
#
# p13 = plt.subplot2grid((5,5),(0,3))
# for qq in range(49):
#     p13.scatter(np.mean(numDWTsAWT4[:,qq]),np.mean(numDWTsAWT5[:,qq]),color=dwtcolors[qq])
# p14 = plt.subplot2grid((5,5),(1,3))
# for qq in range(49):
#     p14.scatter(np.mean(numDWTsAWT4[:,qq]),np.mean(numDWTsAWT6[:,qq]),color=dwtcolors[qq])
#
# p15 = plt.subplot2grid((5,5),(0,4))
# for qq in range(49):
#     p15.scatter(np.mean(numDWTsAWT5[:,qq]),np.mean(numDWTsAWT6[:,qq]),color=dwtcolors[qq])



bmus2 = bmus+1 #[120:]+1

# Lets make a plot comparing probabilities in sim vs. historical
probH = np.nan*np.ones((num_clusters,))
probS = np.nan * np.ones((500,num_clusters))

for h in np.unique(bmus2):
    findH = np.where((bmus2 == h))[0][:]
    probH[int(h-1)] = len(findH)/len(bmus2)

    for s in range(500):
        findS = np.where((evbmus_sim[:,s] == h))[0][:]
        probS[s,int(h-1)] = len(findS)/len(evbmus_sim[:,s])




diffAWT = 100*(numDWTsAWT4-numDWTsTotal)/numDWTsTotal
# diffAWT = (numDWTsAWT4-numDWTsTotal)

plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0))
for i in range(49):
    data =diffAWT[:,i].flatten()
    box1 = ax1.boxplot(data, positions=[int(i+1)], widths=.6, notch=True, patch_artist=True, showfliers=False)
    plt.setp(box1['boxes'],color=dwtcolors[i])
    plt.setp(box1['means'],color=dwtcolors[i])
    plt.setp(box1['fliers'],color=dwtcolors[i])
    plt.setp(box1['whiskers'],color=dwtcolors[i])
    plt.setp(box1['caps'],color=dwtcolors[i])
    plt.setp(box1['medians'],color=dwtcolors[i],linewidth=1)

ax1.plot([0,50],[0,0],'k--')
ax1.set_xlim([0,50])
ax1.set_ylabel('Percent Change During El Nino')
ax1.set_xticks([5,10,15,20,25,30,35,40,45])
ax1.set_xticklabels(['5','10','15','20','25','30','35','40','45'])




plt.figure()
# plt.plot(probH,np.mean(probS,axis=0),'.')
# plt.plot([0,0.03],[0,0.03],'.--')
ax = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
for i in range(num_clusters):
    temp = probS[:,i]
    temp2 = probH[i]
    box1 = ax.boxplot(temp,positions=[temp2],widths=.002,notch=True,patch_artist=True,showfliers=False)
    plt.setp(box1['boxes'],color=dwtcolors[i])
    plt.setp(box1['means'],color=dwtcolors[i])
    plt.setp(box1['fliers'],color=dwtcolors[i])
    plt.setp(box1['whiskers'],color=dwtcolors[i])
    plt.setp(box1['caps'],color=dwtcolors[i])
    plt.setp(box1['medians'],color=dwtcolors[i],linewidth=0)

    #box1['boxes'].set(facecolor=dwtcolors[i])
    #plt.set(box1['fliers'],markeredgecolor=dwtcolors[i])
ax.plot([0,0.06],[0,0.06],'k.--', zorder=10)
plt.xlim([0,0.06])
plt.ylim([0,0.06])
plt.xticks([0,0.02,0.04,0.06], ['0','0.02','0.04','0.06'])
plt.xlabel('Historical Probability')
plt.ylabel('Simulated Probability')
plt.title('Validation of ALR DWT Simulations')
#
#
#
#
from datetime import timedelta

def GenOneYearDaily(yy=1981, month_ini=1):
   'returns one generic year in a list of datetimes. Daily resolution'

   dp1 = DT.datetime(yy, month_ini, 2)
   dp2 = dp1 + timedelta(days=364)

   return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]


# generate perpetual year list
list_pyear = GenOneYearDaily(month_ini=1)
m_plot = np.zeros((num_clusters, len(list_pyear))) * np.nan
num_sim=1
# sort data
for i, dpy in enumerate(list_pyear):
   _, s = np.where(
      [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)]
   )
   b = evbmus_sim[s,:]
   # b = bmus[s]
   b = b.flatten()

   for j in range(num_clusters):
      _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!
      # _, bb = np.where([(j == b)])  # j+1 starts at 1 bmus value!

      m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)

fig = plt.figure(figsize=(10,4))
ax = plt.subplot2grid((1,1),(0,0))
# plot stacked bars
bottom_val = np.zeros(m_plot[1, :].shape)
for r in range(num_clusters):
   row_val = m_plot[r, :]
   ax.bar(list_pyear, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
   # store bottom
   bottom_val += row_val

import matplotlib.dates as mdates
# customize  axis
months = mdates.MonthLocator()
monthsFmt = mdates.DateFormatter('%b')
ax.set_xlim(list_pyear[0], list_pyear[-1])
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.set_ylim(0, 500)
ax.set_ylabel('Average Probability')
ax.set_yticks([0,100,200,300,400,500])
ax.set_yticklabels(['0','0.2','0.4','0.6','0.8','1'])



# generate perpetual year list
list_pyear = GenOneYearDaily(month_ini=1)
m_plot = np.zeros((num_clusters, len(list_pyear))) * np.nan
num_sim=1
# sort data
for i, dpy in enumerate(list_pyear):
   _, s = np.where(
      [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)]
   )
   b = evbmus_sim[s,2]
   # b = bmus[s]
   b = b.flatten()

   for j in range(num_clusters):
      _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!
      # _, bb = np.where([(j == b)])  # j+1 starts at 1 bmus value!

      m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)

import matplotlib.cm as cm
# etcolors = cm.viridis(np.linspace(0, 1, 70-20))
# tccolors = np.flipud(cm.autumn(np.linspace(0,1,21)))
# dwtcolors = np.vstack((etcolors,tccolors[1:,:]))

fig = plt.figure(figsize=(10,4))
ax = plt.subplot2grid((1,1),(0,0))
# plot stacked bars
bottom_val = np.zeros(m_plot[1, :].shape)
for r in range(num_clusters):
   row_val = m_plot[r, :]
   ax.bar(list_pyear, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
   # store bottom
   bottom_val += row_val

import matplotlib.dates as mdates
# customize  axis
months = mdates.MonthLocator()
monthsFmt = mdates.DateFormatter('%b')
ax.set_xlim(list_pyear[0], list_pyear[-1])
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
# ax.set_ylim(0, 100)
ax.set_ylim(0, 1)
ax.set_ylabel('Probability')



f2 = plt.figure()
# plt.pcolor(np.asarray(bmus_dates)[0:365],np.ones((len(bmus[0:365]),)),np.vstack((bmus[0:365],bmus[0:365])))
p10 = plt.subplot2grid((1,1),(0,0))
p10.pcolor(np.vstack((bmus[(-730-150):(-365-150)],bmus[(-730-150):(-365-150)])),cmap='rainbow')
p10.set_xticks(np.array([0,90,180,270,365]))
p10.set_xticklabels(['Jan 2022','Apr 2022','Jul 2022','Oct 2022','Jan 2023'])
