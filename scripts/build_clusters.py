north_pole = (90,0)
weight_limit = 1000#TMP 1000
sleigh_weight = 10


import pandas as pd
import numpy as np
from haversine import haversine,AVG_EARTH_RADIUS
from mpl_toolkits.basemap import Basemap
import Kmeans as km
import networkx as nx
import scipy.spatial.distance as sdis
from progress_bar import ProgressBar
import sys


class Cluster_Builder:
    """
    Given centroids, this class assigns the gifts to cluster, while respecting the knapsack constraint,
    and must try to minimize the distance ``metric`` from point to centroids

    can also be initialized with predetermined cluster, to optimize them
    """
    def __init__(self,X,centroids,metric,weights,clusters=None,gifts = None):
        self.metric = metric
        self.weights = np.array(weights)
        self.N = len(X)
        self.X = X
        self.gifts = gifts

        if clusters is None:
            self.centroids = centroids
            self.K = len(centroids)
            self.clusters = {k:[] for k in range(self.K)}
            self.weight_per_cluster = {k:0. for k in range(self.K)}
        else:
            self.clusters = {c:[i-1 for i in clusters[c]] for c in clusters}
            self.K = len(clusters)
            self.to_assign = []
            self.update_stats()

    def compute_initial_allocation(self,disp_progress = True):

        self.to_assign = np.random.permutation(range(self.N)).tolist()

        if disp_progress:
            prog = ProgressBar(0, len(self.to_assign), 77, mode='fixed')
            oldprog = str(prog)

        
        while True:
            if disp_progress:
                #<--display progress
                prog.increment_amount()
                if oldprog != str(prog):
                        print prog, "\r",
                        sys.stdout.flush()
                        oldprog=str(prog)
                #-->

            i = self.to_assign.pop()
            dists = sdis.cdist(np.atleast_2d(self.X[i]),self.centroids,self.metric).ravel()
            sorted_centroids = np.argsort(dists)
            for c in sorted_centroids:
                if self.weight_per_cluster[c]+self.weights[i]<weight_limit:
                    self.clusters[c].append(i)
                    self.weight_per_cluster[c]+=self.weights[i]
                    break

            if not(self.to_assign):
                break
            
    def update_stats(self):
        assert(self.gifts is not None)
        wgt = {}
        height = []
        width = []
        centroids = []
        avdists = {}
        num_per_cluster = {}
        for k in range(self.K):
            lamax = self.gifts.loc[self.clusters[k]].Latitude.max()
            lamin = self.gifts.loc[self.clusters[k]].Latitude.min()
            lamean = self.gifts.loc[self.clusters[k]].Latitude.mean()
            #TODO might cause problem for clusters from both sides of the 'date change line'
            lomean = self.gifts.loc[self.clusters[k]].Longitude.mean()
            centroids.append(np.array([lamean,lomean]))
            height.append(AVG_EARTH_RADIUS * np.pi/180 * (lamax-lamin))
            widths = []
            sumd = 0.
            for la,lo in zip(self.gifts.loc[self.clusters[k]].Latitude,self.gifts.loc[self.clusters[k]].Longitude):
                dlo = abs((lo-lomean+180)%360 -180) * np.pi/180
                widths.append(AVG_EARTH_RADIUS * dlo * np.cos(la*np.pi/180.))
                sumd += self.metric(centroids[-1],(la,lo))

            width.append(np.mean(widths)*2)
            wgt[k] = sum([self.weights[i] for i in self.clusters[k]])
            num_per_cluster[k] = len(self.clusters[k])
            avdists[k] = sumd/float(num_per_cluster[k])


        self.height_distribution = height
        self.width_distribution = width
        self.weight_per_cluster = wgt
        self.centroids = np.array(centroids)
        self.average_dists_to_centroid = avdists
        self.num_per_cluster = num_per_cluster
        

class Thin_Metric:
    def __init__(self,thin_factor):
        self.thin_factor = thin_factor
        
    def __call__(self,x,y):
        return AVG_EARTH_RADIUS * np.pi/180 * (self.thin_factor  * abs((x[1]-y[1]+180)%360 -180) * np.cos((x[0]+y[0])*np.pi/360)  + abs(x[0]-y[0]))

class Thin_Kmeans:
    def __init__(self,gifts,thin_factor=100.,number_trip_factor=1.05):
        self.X = gifts[['Latitude','Longitude']].values
        self.K = int(number_trip_factor * gifts.Weight.sum()/1000.)
        self.N = len(self.X)
        self.init_centres = self.X[[int(self.N*p) for p in np.random.sample(self.K)]]
        self.gifts = gifts
        self.thin_factor = thin_factor
        self.metric = Thin_Metric(thin_factor)
        
    def run_thinkmeans(self):
        self.centroids,self.Xto,self.dist = km.kmeans(self.X,self.init_centres,metric=self.metric,verbose=2,restrict_comp_to_close=True)
        
