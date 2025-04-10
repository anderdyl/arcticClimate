
import pickle
import numpy as np
from datetime import date,datetime
def dateDay2datetimeDate(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [date(int(d[0]), int(d[1]), int(d[2])) for d in d_vec]

with open(r"dwts49ClustersArctic2023.pickle", "rb") as input_file:
    historicalDWTs = pickle.load(input_file)

timeDWTs = historicalDWTs['SLPtime']
bmus = historicalDWTs['bmus_corrected']

bmus_dates = dateDay2datetimeDate(timeDWTs)
# bmus_dates_months = np.array([d.month for d in bmus_dates])
# bmus_dates_days = np.array([d.day for d in bmus_dates])

with open(r"ice18FutureSimulations1000PointHope.pickle", "rb") as input_file:

    historicalICEs = pickle.load(input_file)

bmusIce = historicalICEs['bmus']
timeIce = historicalICEs['dayTime']
timeArrayIce = np.array(timeIce)
areaBelow = historicalICEs['areaBelow']
bmus_corrected = bmusIce


with open(r"realWavesPointHope.pickle", "rb") as input_file:

    historical = pickle.load(input_file)

time_all = historical['tWave']
wh = historical['hsCombined']
tp = historical['tpCombined']
dmOG = historical['dmCombined']
waveNorm = historical['waveNorm']
ntr = historical['ntr']
t2m = historical['t2m']
tNTR = historical['tNTR']

import matplotlib.cm as cm
dwtcolors = cm.rainbow(np.linspace(0, 1, 50))


import matplotlib.pyplot as plt

f1 = plt.figure()
p1 = plt.subplot2grid((15,1),(3,0),rowspan=2)
p2 = plt.subplot2grid((15,1),(5,0),rowspan=2)
p3 = plt.subplot2grid((15,1),(7,0),rowspan=2)
p4 = plt.subplot2grid((15,1),(9,0),rowspan=2)
p5 = plt.subplot2grid((15,1),(11,0),rowspan=2)
p6 = plt.subplot2grid((15,1),(13,0),rowspan=2)


# plt.pcolor(np.asarray(bmus_dates)[0:365],np.ones((len(bmus[0:365]),)),np.vstack((bmus[0:365],bmus[0:365])))
p10 = plt.subplot2grid((15,1),(0,0))
p10.pcolor(np.vstack((bmus[(-730-150):(-150)],bmus[(-730-150):(-150)])),cmap='rainbow')
# p10.set_xticks(np.array([0,180,365,(365+180),725]))
# p10.set_xticklabels(['Jan 2021','Jul 2021','Jan 2022','Jul 2022','Jan 2023'])
p10.set_xticklabels([''])
p10.set_yticklabels([''])
p10.set_ylabel('DWT',fontweight='bold')

import cmocean

p11 = plt.subplot2grid((15,1),(1,0))
# p11.pcolor(np.vstack((bmus_corrected[(-730-318):(-317)],bmus_corrected[(-730-318):(-317)])),cmap=cmocean.cm.ice)
# p11.set_xticks(np.array([0,180,365,(365+180),725]))
p11.pcolor(np.vstack((bmus_corrected[(-730-(730*4)-318):(-317-(730*4))],bmus_corrected[(-730-(730*4)-318):(-317-(730*4))])),cmap=cmocean.cm.ice)
p11.set_xticks(np.array([0,180,365,(365+180),725]))
p11.set_xticklabels(['Jan 2013','Jul 2013','Jan 2014','Jul 2014','Jan 2015'],fontweight='bold')
p11.set_yticklabels([''])
p11.set_ylabel('SIC',fontweight='bold')
p1.plot(timeArrayIce,areaBelow,color='k')
p2.plot(time_all,wh,color='k')
p3.plot(time_all,tp,color='k')
p4.plot(time_all,waveNorm,'.',color='k',markersize=4)
p5.plot(tNTR,ntr,color='k')
p6.plot(time_all,t2m-273.15,color='k')

monthsTime = np.asarray([dd.month for dd in time_all])
julyIndex = np.where((monthsTime==7))
octoberIndex = np.where((monthsTime==10))

# p1.set_xlim([time_all[0],time_all[-1]])
# p2.set_xlim([time_all[0],time_all[-1]])
# p3.set_xlim([time_all[0],time_all[-1]])
# p4.set_xlim([time_all[0],time_all[-1]])
# p5.set_xlim([time_all[0],time_all[-1]])
# p6.set_xlim([time_all[0],time_all[-1]])
p1.set_xlim([datetime(2013,1,1),datetime(2015,1,1)])
p2.set_xlim([datetime(2013,1,1),datetime(2015,1,1)])
p3.set_xlim([datetime(2013,1,1),datetime(2015,1,1)])
p4.set_xlim([datetime(2013,1,1),datetime(2015,1,1)])
p5.set_xlim([datetime(2013,1,1),datetime(2015,1,1)])
p6.set_xlim([datetime(2013,1,1),datetime(2015,1,1)])
p1.set_yticklabels(['0','0','$10^{3}$'],fontweight='bold')

p1.set_ylabel('Basin ($km^{2}$)',fontweight='bold')
p2.set_ylabel('Hs (m)',fontweight='bold')
p3.set_ylabel('Tp (s)',fontweight='bold')
p4.set_ylabel('Dm ($^{\circ}$N)',fontweight='bold')
p5.set_ylabel('SS (m)',fontweight='bold')
p6.set_ylabel('Air Temp (C)',fontweight='bold')

p2.set_yticks([0,5])
p2.set_yticklabels(['0','5'],fontweight='bold')
p3.set_yticks([2.5,5.0,7.5])
p3.set_yticklabels(['2.5','5.0','7.5'],fontweight='bold')
p4.set_yticks([-100,0,100])
p4.set_yticklabels(['-100','0','100'],fontweight='bold')
p5.set_yticks([-1,0,1])
p5.set_yticklabels(['-1','0','1'],fontweight='bold')
p6.set_yticks([-25,0])
p6.set_yticklabels(['-25','0'],fontweight='bold')

p1.set_xticklabels([''])
p2.set_xticklabels([''])
p3.set_xticklabels([''])
p4.set_xticklabels([''])
p5.set_xticklabels([''])
p6.set_xticks(np.array([datetime(2013,1,1),datetime(2013,4,1),datetime(2013,7,1),datetime(2013,10,1),
                        datetime(2014,1,1),datetime(2014,4,1),datetime(2014,7,1),datetime(2014,10,1),datetime(2015,1,1)]))
p6.set_xticklabels(['Jan 2013','','Jul 2013','','Jan 2014','','Jul 2014','','Jan 2015'],fontweight='bold')

p10.text(-0.12, 1.03, 'a.', transform=p10.transAxes, size=12,fontweight='bold')#, weight='bold')
p11.text(-0.12, 1.03, 'b.', transform=p11.transAxes, size=12,fontweight='bold')#, weight='bold')
p1.text(-0.14, 1.03, 'c.', transform=p1.transAxes, size=12,fontweight='bold')#, weight='bold')
p2.text(-0.14, 1.03, 'd.', transform=p2.transAxes, size=12,fontweight='bold')#, weight='bold')
p3.text(-0.14, 1.03, 'e.', transform=p3.transAxes, size=12,fontweight='bold')#, weight='bold')
p4.text(-0.14, 1.03, 'f.', transform=p4.transAxes, size=12,fontweight='bold')#, weight='bold')
p5.text(-0.14, 1.03, 'g.', transform=p5.transAxes, size=12,fontweight='bold')#, weight='bold')
p6.text(-0.14, 1.03, 'h.', transform=p6.transAxes, size=12,fontweight='bold')#, weight='bold')


plt.tight_layout()