north_pole = (90,0)
weight_limit = 1000#TMP 1000
sleigh_weight = 10


import pandas as pd
import numpy as np
from haversine import haversine
from mpl_toolkits.basemap import Basemap
import Kmeans as km
import networkx as nx
import scipy.spatial.distance as sdis


class Cluster_Builder:
    def __init__(self,X,centroids,metric,weights):
        self.centroids = centroids
        self.metric = metric
        self.weights = np.array(weights)
        self.K = len(centroids)
        self.N = len(X)
        self.X = X
        self.clusters = {k:[] for k in range(self.K)}
        self.weight_per_cluster = {k:0. for k in range(self.K)}
        self.to_assign = np.random.permutation(range(self.N)).tolist()

    def compute_initial_allocation(self):
        while True:
            i = self.to_assign.pop()
            dists = sdis.cdist(np.atleast_2d(self.X[i]),self.centroids,self.metric).ravel()
            sorted_centroids = np.argsort(dists)
            for c in sorted_centroids:
                if self.weight_per_cluster[c]+self.weights[i]<weight_limit:
                    self.clusters[c].append(i)
                    self.weight_per_cluster[c]+=self.weights[i]
                    break

            if len(self.to_assign) % 100 == 0:
                print len(self.to_assign)

            if not(self.to_assign):
                break


