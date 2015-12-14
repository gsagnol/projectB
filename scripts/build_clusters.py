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
from scipy.spatial.distance import cdist

class Cluster_Builder:
    """
    Given centroids, this class assigns the gifts to cluster, while respecting the knapsack constraint,
    and must try to minimize the distance ``metric`` from point to centroids

    * can also be initialized with predetermined cluster, to optimize them (TODO)

    * or for the greedy_bound heuristic, that sequentially adds gifts that minimize the difference with the bound 1.02w_i d_i
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
            self.latitudes_in_cluster = {k:[] for k in range(self.K)} #gifts ordered by lattitude for the greedy bound optimization
            self.distances_to_pole = cdist(np.atleast_2d(north_pole),X,haversine).ravel()
            self.latencies_in_cluster = {k:[] for k in range(self.K)} #for gifts ordered by lattitude for the greedy bound optimization
            self.cost_per_cluster = {k:0. for k in range(self.K)} #for gifts ordered by lattitude for the greedy bound optimization
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

    def init_clusters_with_centroids(self):
        for k in range(self.K):
            igift = np.argmin(np.linalg.norm(self.X[self.to_assign]-self.centroids[k],axis=1))
            gift = self.to_assign[igift]
            if gift not in self.to_assign:
                raise Exception('this should not happen: one gift assigned to 2 centroids')
            self.latitudes_in_cluster[k] = [self.X[gift][0]]
            self.latencies_in_cluster[k] = [self.distances_to_pole[gift]]
            self.clusters[k].append(gift)
            self.weight_per_cluster[k] = self.weights[gift]
            self.cost_per_cluster[k] = (self.weights[gift]+2*sleigh_weight) * self.latencies_in_cluster[k]
            self.to_assign.remove(gift)


    def bound_increase_for_adding_gift_in_cluster(self,giftID,centroidID):
        i,k = giftID,centroidID
        n = len(self.clusters[k])
        if n==0:
            raise Exception('cluster was not initialized')
        lati,longi = self.X[i]
        dpole_i = self.distances_to_pole[i]
        j = n-np.searchsorted(self.latitudes_in_cluster[k],lati)
        if j==0:
            latency_i = dpole_i
        else:
            previous_gift = self.clusters[k][j-1]
            latency_i = self.latencies_in_cluster[k][j-1] + haversine(self.X[previous_gift],self.X[i])
        if j==n:
            #add gift in last position
            delta_latency = latency_i - self.latencies_in_cluster[k][j-1]
            delta_d = dpole_i - self.distances_to_pole[self.clusters[k][-1]]
            return self.weights[i]*(latency_i-1.02 * dpole_i) + sleigh_weight * (delta_latency + delta_d)
        else:
            next_gift = self.clusters[k][j]
            delta_latency = latency_i +  haversine(self.X[next_gift],self.X[i]) - self.latencies_in_cluster[k][j]
            weight_after_j = sum(self.weights[self.clusters[k][j:]])
            return self.weights[i]*(latency_i-1.02 * dpole_i) + delta_latency * (weight_after_j + sleigh_weight)

    def add_in_tour(self,giftID,centroidID):
        i,k = giftID,centroidID
        n = len(self.clusters[k])
        if n==0:
            raise Exception('cluster was not initialized')
        lati,longi = self.X[i]
        dpole_i = self.distances_to_pole[i]
        j = n-np.searchsorted(self.latitudes_in_cluster[k],lati)
        if j==0:
            latency_i = dpole_i
        else:
            previous_gift = self.clusters[k][j-1]
            latency_i = self.latencies_in_cluster[k][j-1] + haversine(self.X[previous_gift],self.X[i])
        if j==n:
            #add gift in last position
            delta_latency = latency_i - self.latencies_in_cluster[k][j-1]
            weight_after_j = 0.
            delta_d = dpole_i - self.distances_to_pole[self.clusters[k][-1]]
        else:
            next_gift = self.clusters[k][j]
            delta_latency = latency_i +  haversine(self.X[next_gift],self.X[i]) - self.latencies_in_cluster[k][j]
            weight_after_j = sum(self.weights[self.clusters[k][j:]])
            delta_d = 0.
        
        self.clusters[k].insert(j,i)
        self.latitudes_in_cluster[k].insert(j,lati)
        for jj in range(j,n):
            self.latencies_in_cluster[k][jj]+=delta_latency
        
        self.latencies_in_cluster[k].insert(j,latency_i)
        
        self.weight_per_cluster[k]+= self.weights[i]
        
        self.cost_per_cluster[k] +=  self.weights[i]*latency_i+ (weight_after_j + sleigh_weight) * delta_latency + sleigh_weight * delta_d
        
        


        

    def greedy_for_bound(self,disp_progress=True):
        print 'initialization...'
        self.to_assign = np.random.permutation(range(self.N)).tolist()
        if not(np.all([self.centroids[i][1]<=self.centroids[i+1][1] for i in range(len(self.centroids)-1)])):
            print 'warning: centroids were not sorted by longitude. I reorder them'
            self.centroids = np.array(sorted(self.centroids, key=lambda x: x[1]))
            
        self.init_clusters_with_centroids()
        print 'done.'
        
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
            km = np.searchsorted(self.centroids[:,1], self.X[i][1]-15)
            kp = np.searchsorted(self.centroids[:,1], self.X[i][1]+15)
            
            bounds_inc = [(self.bound_increase_for_adding_gift_in_cluster(i,k),k) for k in range(km,kp)]
            sorted_bounds_inc = sorted(bounds_inc)
            for inc,c in sorted_bounds_inc:
                if self.weight_per_cluster[c]+self.weights[i]<weight_limit:
                    self.clusters[c].append(i)
                    self.weight_per_cluster[c]+=self.weights[i]
                    break

            if not(self.to_assign):
                break
        

        for k in range(km,kp):
            tab.append((cb.bound_increase_for_adding_gift_in_cluster(i,k),i,k))


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
        

class Cluster:
    """
    A class for trips (group of gifts < 1000 kg)
    Gifts are with id from 0 to make it simpler, but ``save()`` writes the true IDs in the file
    """
    def __init__(self,cluster,gifts):
        self.gifts = gifts
        if isinstance(cluster,str):
            self.load(cluster)
        else:
            imin = min([min([vi for vi in v]) for v in cluster.values()])
            if imin == 0:
                self.cluster = cluster
            elif imin==1:
                self.cluster = {c: [i-1 for i in v] for c,v in cluster.iteritems()}
            else:
                raise Exception('bad cluster indices')
        assert(min([min([vi for vi in v]) for v in self.cluster.values()]) == 0 )

    def save(self,name):
        clusters_from_1 = {c: [i+1 for i in v] for c,v in self.cluster.iteritems()}
        f = open('../clusters/'+name,'w')
        f.write(str(clusters_from_1))
        f.close()

    def load(self,name):
        f = open('../clusters/'+name,'r')
        clusters_from_1 = eval(f.read())
        f.close()
        self.cluster =  {c: [i-1 for i in v] for c,v in clusters_from_1.iteritems()}

    def compute_wgt_per_cluster(self):
        self.wgts = self.gifts.Weight.values
        self.wgt_per_cluster = {c:sum([self.wgts[i] for i in v]) for c,v in self.cluster.iteritems()}

    def lower_bound_per_cluster(self):
        latitude = self.gifts.Latitude.values
        d_from_pole = AVG_EARTH_RADIUS * (90-latitude)*np.pi/180.
        self.bound_per_cluster = {c:sum([self.wgts[i]*d_from_pole[i] for i in v]) *
                                  (1.+2*sleigh_weight/float(self.wgt_per_cluster[c])) for c,v in self.cluster.iteritems()}




class Capactited_MST:
    def __init__(self,gifts,nb_neighbors=50,metric=None):
        """
        metric=None uses the chord distance
        """
        self.gifts = gifts
        self.X = gifts[['Latitude','Longitude']].values
        self.N = len(self.X)
        self.wgt = gifts.Weight.values
        #root of subtree -> list of nodes in this subtree
        self.subtrees = {i:[i] for i in range(self.N)}
        #node -> root of subtree
        self.Xto = range(self.N)
        #weight of subtrees
        self.subtree_weights = {i: self.wgt[i] for i in range(self.N)}
        #cartesian coordinates (ignoring earth radius)
        self.Z = np.apply_along_axis(self.to_cartesian,1,self.X)
        #distance from north pole to root points
        to_pole = cdist(np.atleast_2d(self.to_cartesian(north_pole)),self.Z)
        if metric is None:
            self.gates = to_pole[0].tolist()
        else:
            if isinstance(metric,Thin_Metric):
                self.gates = (AVG_EARTH_RADIUS * to_pole[0]).tolist()
            else:
                self.gates = cdist(np.atleast_2d(north_pole),self.X)[0].to_list()
        self.subtree_costs = {i:self.gates[i] for i in range(self.N)}
        self.total_cost = sum(self.subtree_costs.values())
        self.nb_neighbors = nb_neighbors
        import sklearn.neighbors
        self.kdtree = sklearn.neighbors.KDTree(self.Z)
        self.metric=metric


    def to_cartesian(self,x):
        phi = (90-x[0]) * np.pi/180.
        theta = x[1] * np.pi/180.
        sphi = np.sin(phi)
        return np.array([sphi*np.cos(theta),sphi*np.sin(theta),np.cos(phi)])

    def closest_point_in_other_subtree(self,i):
        close = self.kdtree.query(self.Z[i],self.nb_neighbors)
        rooti = self.Xto[i]
        wgti = self.subtree_weights[rooti]
        if self.metric is None:
            for d,ind in zip(close[0][0][1:],close[1][0][1:]):
                rootind = self.Xto[ind]
                if rooti!=rootind and wgti + self.subtree_weights[rootind] <= weight_limit:
                    return d,ind
        else:
            dd = []
            for ind in close[1][0][1:]:
                dd.append(self.metric(self.X[ind],self.X[i]))
            dd = np.array(dd)
            sd = np.argsort(dd)
            for d,ind in zip(dd[sd],close[1][0][sd]):
                rootind = self.Xto[ind]
                if rooti!=rootind and wgti + self.subtree_weights[rootind] <= weight_limit:
                    return d,ind
                
                
        return None,None

    def evaluate_tradeoffs(self):
        tradeoffs = []
        for i in range(self.N):
            dij,j = self.closest_point_in_other_subtree(i)
            if dij is None:
                tradeoffs.append((-np.inf,0,0,0,0,0))
            else:
                tradeoffs.append((self.gates[i]-dij,i,j,dij,self.Xto[i],self.Xto[j]))
        return tradeoffs

    def connect_subtrees(self,i,j,cij,tij):
        rooti = self.Xto[i]
        rootj = self.Xto[j]
        wij = self.subtree_weights[rooti] + self.subtree_weights[rootj]
        assert(wij <= weight_limit)
        self.subtrees[rootj].extend(self.subtrees[rooti])
        for k in self.subtrees[rooti]:
            self.Xto[k]=rootj
        del self.subtrees[rooti]

        self.subtree_weights[rootj] += self.subtree_weights[rooti]
        del self.subtree_weights[rooti]
        self.subtree_costs[rootj] += self.subtree_costs[rooti] + cij
        del self.subtree_costs[rooti]
        self.total_cost -= tij


    def essau_williams(self,max_shift_per_iter=1000):
        while True:
            print 'Evaluating tradeoffs...'
            to = self.evaluate_tradeoffs()
            print 'done.'
            sto = sorted(to)
            if sto[-1]<0:
                print 'no more changes available. Stop.'
                return
            nbshift = 0
            root_seen = []
            while sto and nbshift<max_shift_per_iter:
                tradeoff,i,j,dij,rooti,rootj = sto.pop()
                if tradeoff<=0:
                    break
                if rooti in root_seen or rootj in root_seen:
                    continue
                else:
                    self.connect_subtrees(i,j,dij,tradeoff)
                    root_seen.extend([rooti,rootj])
                    nbshift += 1
            print '{0} changes done. New value: {1}'.format(nbshift,self.total_cost)






