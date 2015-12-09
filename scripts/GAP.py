north_pole = (90,0)
weight_limit = 1000#TMP 1000
sleigh_weight = 10


import pandas as pd
import numpy as np
from haversine import haversine,AVG_EARTH_RADIUS
from mpl_toolkits.basemap import Basemap
import Kmeans as km
import build_clusters as bc

#--------------------------------------#
#   Some preliminary tests with Africa #
#--------------------------------------#

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

a = 20.#100. TODO this param will probably help to make cluster of the desired width...
mydis = lambda x,y: haversine(y,(x[0],y[1])) + 0.5* a* ( haversine(x,(x[0],y[1])) + haversine(y,(y[0],x[1])))
mydis2 = lambda x,y: AVG_EARTH_RADIUS * np.pi/180 * (a/2. * abs((x[1]-y[1]+180)%360 -180) * (np.cos(x[0]*np.pi/180)+np.cos(y[0]*np.pi/180))  + abs(x[0]-y[0]))
mydis3 = lambda x,y: AVG_EARTH_RADIUS * np.pi/180 * (a * abs((x[1]-y[1]+180)%360 -180) * np.cos((x[0]+y[0])*np.pi/360)  + abs(x[0]-y[0]))


colors = np.random.rand(k,3)
#centroids,Xto,dist = km.kmeans(X,init_centres,metric='mydist:100',verbose=2)
centroids,Xto,dist = km.kmeans(X,init_centres,metric=mydis2,verbose=2)
x,y = m(sgif.Longitude.values,sgif.Latitude.values)
m.scatter(x,y,color=colors[Xto])

wgt_per_cluster = [gifts.Weight[Xto==i].sum() for i in range(k)]

wgt = sgif.Weight.values



centroids,Xto,dist = km.kmeans(X,init_centres,metric=mydis3,verbose=2)

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

#----------------------------------------#
#  try to build cluster with Thin_Kmeans #
#----------------------------------------#
gifts = pd.read_csv('../input/gifts.csv')
thinf = 20
fullf = 1.02
tkm = bc.Thin_Kmeans(gifts,thinf,fullf)
tkm.run_thinkmeans()

#draw clusters
figure(1)
clf()
m = Basemap(llcrnrlon=-170,llcrnrlat=-80,urcrnrlon=190,urcrnrlat=80,projection='mill')
m.drawcoastlines()
m.drawcountries()
colors = np.random.rand(tkm.K,3)
x,y = m(gifts.Longitude.values,gifts.Latitude.values)
m.scatter(x,y,color=colors[tkm.Xto])


f = open('../clusters/tkmeans_{0}_{1}'.format(thinf,tkm.K),'w')
f.write(str(tkm.Xto.tolist()))
f.close()

#----------------------------------------------------#
#  Allocate points to centroids with Cluster_Builder #
#----------------------------------------------------#
thinf2 = 20
cb = bc.Cluster_Builder(tkm.X,tkm.centroids,bc.Thin_Metric(thinf2),gifts.Weight.values)
cb.compute_initial_allocation()

Xto2 = np.zeros(cb.N).tolist()
for k in cb.clusters:
    for i in cb.clusters[k]:
        Xto2[i] = int(k)

clusters = {c:[i+1 for i in cb.clusters[c]] for c in cb.clusters}

f = open('../clusters/greedy_tkmeans_{0}_{1}_{2}'.format(thinf,thinf2,cb.K),'w')
f.write(str(clusters))
f.close()

#draw allocation to trip
figure(2)
clf()
m = Basemap(llcrnrlon=-170,llcrnrlat=-80,urcrnrlon=190,urcrnrlat=80,projection='mill')
m.drawcoastlines()
m.drawcountries()
colors = np.random.rand(tkm.K,3)
x,y = m(gifts.Longitude.values,gifts.Latitude.values)
m.scatter(x,y,color=colors[Xto2],s=1)
#savefig('tst.png',dpi=500,bbox_inches='tight')


cb = bc.Cluster_Builder(tkm.X,None,bc.Thin_Metric(thinf2),gifts.Weight.values,clusters,gifts)



#distrubution of trip weights: Too many 'light trips' -> TODO merge procedure
figure(3)
hist(cb.weight_per_cluster.values(),20)
#distribution of trip height and width
height = []
width = []
for k in range(tkm.K):
    lamax = gifts.loc[cb.clusters[k]].Latitude.max()
    lamin = gifts.loc[cb.clusters[k]].Latitude.min()
    lomean = gifts.loc[cb.clusters[k]].Longitude.mean()
    height.append(AVG_EARTH_RADIUS * np.pi/180 * (lamax-lamin))
    widths = []
    for la,lo in zip(gifts.loc[cb.clusters[k]].Latitude,gifts.loc[cb.clusters[k]].Longitude):
        dlo = abs((lo-lomean+180)%360 -180) * np.pi/180
        widths.append(AVG_EARTH_RADIUS * dlo * np.cos(la*np.pi/180.))
    width.append(mean(widths)*2)
        
figure(4)
hist(height,20)
figure(5)
hist(width,20)