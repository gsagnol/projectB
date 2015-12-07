north_pole = (90,0)
weight_limit = 1000#TMP 1000
sleigh_weight = 10


import pandas as pd
import numpy as np
from haversine import haversine
from mpl_toolkits.basemap import Basemap
import Kmeans as km
#---------------------#
#   make subsinstance #
#---------------------#


gifts = pd.read_csv('../input/gifts.csv')
ngifts = gifts.count()[0]
lamin = -35
lamax = 28
lomin = -15
lomax = 50

sgif = gifts[np.all([gifts.Latitude<lamax,gifts.Latitude>lamin,gifts.Longitude>lomin,gifts.Longitude<lomax],axis=0)]


m = Basemap(llcrnrlon=lomin,llcrnrlat=lamin,urcrnrlon=lomax,urcrnrlat=lamax,
            projection='lcc',lat_0=0.5*(lamin+lamax),lon_0=0.5*(lomin+lomax),
            resolution ='l',area_thresh=1000.)

# draw coastlines, state and country boundaries, edge of map.
m.drawcoastlines()
m.drawstates()
m.drawcountries()
x,y = m(sgif.Longitude.values,sgif.Latitude.values)
m.scatter(x,y)

X = sgif[['Latitude','Longitude']].values
k = int(1.05 * sgif.Weight.sum()/1000.)
init_centres = X[[int(len(sgif)*p) for p in np.random.sample(k)]]
centroids,Xto,dist = km.kmeans(X,init_centres)
x,y = m(centroids[:,1],centroids[:,0])
m.scatter(x,y,color='r')