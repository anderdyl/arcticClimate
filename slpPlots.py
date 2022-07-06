
import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import numpy as np
import pandas as pd
import os
import datetime
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import mda
from scipy.spatial import distance_matrix

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import datetime
from dateutil.relativedelta import relativedelta
import random
import xarray as xr
import geopy.distance
from dipy.segment.metric import Metric
from dipy.segment.metric import ResampleFeature
from dipy.segment.clustering import QuickBundles
from mpl_toolkits.basemap import Basemap





import mat73

SLPs = mat73.loadmat('/media/dylananderson/Elements/pointHope/slpsPolarGrid2v4y2022.mat')

X_in = SLPs['X_in']
Y_in = SLPs['Y_in']
Xsea = SLPs['Xsea']
Ysea = SLPs['Ysea']
SLP = SLPs['SLPsea']
GRD = SLPs['GRDsea']
SLPtime = SLPs['time']
inSea = SLPs['in']
onCoast = SLPs['on']


sea_nodes = []
for qq in range(len(Xsea)):
    sea_nodes.append(np.where((X_in == Xsea[qq]) & (Y_in == Ysea[qq])))



plt.style.use('dark_background')

lat = SLPs['lat']
lon = SLPs['lon']
# plotting the EOF patterns
c1 = 0
c2 = 0
counter = 1
for hh in range(2000):
    print('working on {}'.format(hh))
    #ax = plt.subplot2grid((3,3),(c1,c2),projection=ccrs.NorthPolarStereo(central_longitude=-45))
    #ax = plt.subplot2grid((3,3),(c1,c2))#,projection=ccrs.NorthPolarStereo(central_longitude=-45))
    # m = Basemap(projection='merc',llcrnrlat=-40,urcrnrlat=55,llcrnrlon=255,urcrnrlon=375,lat_ts=10,resolution='c')
    plt.figure(figsize=(8,8))
    ax = plt.subplot2grid((1,1),(0,0))
    m = Basemap(projection='npstere', boundinglat=50, lon_0=180, resolution='l')

    cx,cy =m(lon,lat)
    m.drawcoastlines()

    spatialField = SLP[hh,:] / 100 - np.nanmean(SLP, axis=0) / 100

    # linearField = np.ones((np.shape(X_in.flatten()))) * np.nan
    # linearField[indexIn[0]] = spatialField
    # rectField = linearField.reshape(96,91)

    rectField = np.ones((np.shape(X_in))) * np.nan
    for tt in range(len(sea_nodes)):
        rectField[sea_nodes[tt]] = spatialField[tt]

    #ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    #gl = ax.gridlines(draw_labels=True)
    #extent = [-9.97, 168.35, 30.98, 34.35]

    clevels = np.arange(-70,70,1)

    #ax.pcolormesh(cx, cy, rectField)#, cmap=cmocean.cm.ice)
    CS = m.contourf(cx, cy, rectField, clevels, vmin=-35, vmax=35, cmap=cm.RdBu_r, shading='gouraud')
    # ax.set_xlim([-3223000, 0])
    # ax.set_ylim([0, 2730000])
    ax.set_xlim([np.min(cx)+2200000, np.max(cx)-1600000])
    ax.set_ylim([np.min(cy)+900000, np.max(cy)-500000])
    m.fillcontinents()
    if counter < 10:
        plt.savefig('/media/dylananderson/Elements/slpPlots/frame0000'+str(counter)+'.png')
    elif counter < 100:
        plt.savefig('/media/dylananderson/Elements/slpPlots/frame000'+str(counter)+'.png')
    elif counter < 1000:
        plt.savefig('/media/dylananderson/Elements/slpPlots/frame00'+str(counter)+'.png')
    elif counter < 10000:
        plt.savefig('/media/dylananderson/Elements/slpPlots/frame0'+str(counter)+'.png')
    else:
        plt.savefig('/media/dylananderson/Elements/slpPlots/plots2/frame'+str(counter)+'.png')
    plt.close()
    counter = counter+1
    #
    #gl.xlabels_top = False
    #gl.ylabels_left = False
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    #
    # Xs = np.arange(np.min(X_in),np.max(X_in),2)
    # Ys = np.arange(np.min(Y_in),np.max(Y_in),2)
    # lenXB = len(X_in)
    # [XR,YR] = np.meshgrid(Xs,Ys)
    # sea_nodes = []
    # for qq in range(lenXB-1):
    #     sea_nodes.append(np.where((XR == X_in[qq]) & (YR == Y_in[qq])))
    #
    # rectField = np.ones((np.shape(XR))) * np.nan
    # for tt in range(len(sea_nodes)):
    #     rectField[sea_nodes[tt]] = spatialField[tt]
    #
    # clevels = np.arange(-2,2,.05)
    # m = Basemap(projection='merc',llcrnrlat=-40,urcrnrlat=55,llcrnrlon=255,urcrnrlon=375,lat_ts=10,resolution='c')
    # #m.fillcontinents(color=dwtcolors[qq])
    # cx,cy =m(XR,YR)
    # m.drawcoastlines()
    # CS = m.contourf(cx,cy,rectField,clevels,vmin=-1.2,vmax=1.2,cmap=cm.RdBu_r,shading='gouraud')
    # ax.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
#




geomorphdir = '/media/dylananderson/Elements/slpPlots/'

files2 = os.listdir(geomorphdir)

files2.sort()

files_path2 = [os.path.join(geomorphdir,x) for x in os.listdir(geomorphdir)]

files_path2.sort()

files_path2 = files_path2[550:900]
import cv2

frame = cv2.imread(files_path2[0])
height, width, layers = frame.shape
forcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('slpVisualized.avi', forcc, 5, (width, height))
for image in files_path2:
    video.write(cv2.imread(image))
cv2.destroyAllWindows()
video.release()


