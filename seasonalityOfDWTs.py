


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


# with open(r"dwtsAll6TCTracksClusters.pickle", "rb") as input_file:
with open(r"dwts25ClustersArctic2.pickle", "rb") as input_file:
   historicalDWTs = pickle.load(input_file)

num_clusters = 25

timeDWTs = historicalDWTs['SLPtime']
bmus = historicalDWTs['bmus_corrected']

timeSLPs = dateDay2datetimeDate(timeDWTs)


bmus_dates = dateDay2datetimeDate(timeDWTs)
bmus_dates_months = np.array([d.month for d in bmus_dates])
bmus_dates_days = np.array([d.day for d in bmus_dates])


dwtcolors = cm.rainbow(np.linspace(0, 1,num_clusters))


# generate perpetual year list
list_pyear = GenOneYearDaily(month_ini=6)
m_plot = np.zeros((70, len(list_pyear))) * np.nan
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
ax.set_ylabel('')




# Lets do the same for ice...


# with open(r"dwtsAll6TCTracksClusters.pickle", "rb") as input_file:
with open(r"iceData25Clusters.pickle", "rb") as input_file:
   iceDWTs = pickle.load(input_file)

num_clusters = 25
iceTime = [np.array((iceDWTs['year'][i],iceDWTs['month'][i],iceDWTs['day'][i])) for i in range(len(iceDWTs['year']))]
#timeDWTs = iceDWTs['SLPtime']
bmus = iceDWTs['bmus_corrected']
#timeSLPs = dateDay2datetimeDate(timeDWTs)
bmus_dates = dateDay2datetimeDate(iceTime)
bmus_dates_months = np.array([d.month for d in bmus_dates])
bmus_dates_days = np.array([d.day for d in bmus_dates])
dwtcolors = cm.rainbow(np.linspace(0, 1,num_clusters))



# generate perpetual year list
list_pyear = GenOneYearDaily(month_ini=6)
m_plot = np.zeros((num_clusters, len(list_pyear))) * np.nan
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
ax.set_ylabel('')

