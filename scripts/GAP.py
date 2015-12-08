north_pole = (90,0)
weight_limit = 1000#TMP 1000
sleigh_weight = 10


import pandas as pd
import numpy as np
from haversine import haversine
from mpl_toolkits.basemap import Basemap
import Kmeans as km
import build_clusters as bc

#---------------------#
#   make subsinstance #
#---------------------#

#(restricts attention to Africa)
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

#dummy Kmeans in Euclidean metric
X = sgif[['Latitude','Longitude']].values
k = int(1.05 * sgif.Weight.sum()/1000.)
init_centres = X[[int(len(sgif)*p) for p in np.random.sample(k)]]
centroids,Xto,dist = km.kmeans(X,init_centres,metric=haversine)
x,y = m(centroids[:,1],centroids[:,0])
m.scatter(x,y,color='r')

#TODO find a better (faster to compute) distance
"""
a = 100.
mydis = lambda x,y: 0.5*( a*haversine(x,(x[0],y[1])) + haversine(y,(x[0],y[1]))) + 0.5*(haversine(x,(y[0],x[1])) + a*haversine(y,(y[0],x[1])))
D = sdis.cdist(X,init_centres)
close = np.argsort(D,axis=1)[:,:10]
sdis.cdist(np.atleast_2d(X[0]),init_centres[close[0]],mydis)
[close[i][np.argmin(sdis.cdist(np.atleast_2d(X[i]),init_centres[close[i]],mydis))] for i in range(len(X))]
"""

colors = np.random.rand(k,3)
centroids,Xto,dist = km.kmeans(X,init_centres,metric='mydist:100',verbose=2)
x,y = m(sgif.Longitude.values,sgif.Latitude.values)
m.scatter(x,y,color=colors[Xto])

wgt_per_cluster = [gifts.Weight[Xto==i].sum() for i in range(k)]

wgt = sgif.Weight.values
a = 20.#100. TODO this param will probably help to make cluster of the desired width...
mydis = lambda x,y: 0.5*( a*haversine(x,(x[0],y[1])) + haversine(y,(x[0],y[1]))) + 0.5*(haversine(x,(y[0],x[1])) + a*haversine(y,(y[0],x[1])))
cb = bc.Cluster_Builder(X,centroids,mydis,wgt)
cb.compute_initial_allocation()

Xto2 = np.zeros(cb.N).tolist()
for k in cb.clusters:
    for i in cb.clusters[k]:
        Xto2[i] = int(k)

figure(3)

m = Basemap(llcrnrlon=lomin,llcrnrlat=lamin,urcrnrlon=lomax,urcrnrlat=lamax,
            projection='lcc',lat_0=0.5*(lamin+lamax),lon_0=0.5*(lomin+lomax),
            resolution ='l',area_thresh=1000.)

# draw coastlines, state and country boundaries, edge of map.
m.drawcoastlines()
m.drawstates()
m.drawcountries()

colors = np.random.rand(cb.K,3)
x,y = m(sgif.Longitude.values,sgif.Latitude.values)
m.scatter(x,y,color=colors[Xto2])



'''

#TEST to solve GAP with CPLEX... but when restricting to close cluster, CPLEX detects infeasibility
import picos as pic
import sklearn.neighbors

P = pic.Problem()
x = {}
d = {}
n = len(X)
K = len(centroids)
kdtree = sklearn.neighbors.KDTree(X)
obj = 0.
cons = {}
for i in range(n):
    if i%1000 == 0:
        print i
    close_centroids = np.unique(Xto[kdtree.query(X[i],500)[1]])
    for k in close_centroids:
        x[i,k] = P.add_variable('x[{0},{1}]'.format(i,k),1,'binary')
        d[i,k] = mydis(X[i],centroids[k])
        obj += x[i,k] * d[i,k]
        cons.setdefault(k,0.)
        cons[k] += wgt[i] * x[i,k]
    P.add_constraint(pic.sum([x[i,k] for k in close_centroids],'k')==1)
for k in range(K):
    P.add_constraint(cons[k]<=1000)
P.minimize(obj)
'''