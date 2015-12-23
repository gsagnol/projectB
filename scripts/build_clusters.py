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
            #self.clusters = {c:[i-1 for i in clusters[c]] for c in clusters}
            if min(clusters)==1:
                self.clusters = {c-1:v for c,v in clusters.iteritems()}
            else:
                self.clusters = clusters
            self.K = len(clusters)
            #self.to_assign = []
            #self.update_stats()
            self.update_instance()
            
    def update_instance(self):
        all_gifts = set(range(self.N))
        centroids = []
        self.distances_to_pole = cdist(np.atleast_2d(north_pole),self.X,haversine).ravel()
        for c in sorted(self.clusters):
            all_gifts = all_gifts - set(self.clusters[c])
            lamean = sum(self.X[self.clusters[c]][:,0] * self.weights[self.clusters[c]])/sum(self.weights[self.clusters[c]])
            longs = self.X[self.clusters[c]][:,1]
            if min(longs)<-150 and max(longs)>150:
                import pdb;pdb.set_trace()
            else:
                lomean = sum(self.X[self.clusters[c]][:,1] * self.weights[self.clusters[c]])/sum(self.weights[self.clusters[c]])
            centroids.append(np.array([lamean,lomean]))
            
        self.to_assign = list(all_gifts)
        order_cluster = np.argsort(np.array(centroids)[:,1])
        self.centroids = np.array(centroids)[order_cluster]
    
        newcls = {}
        for i,k in enumerate(order_cluster):
            newcls[i] = self.clusters[k][:]
        
        self.clusters = newcls
        
        self.weight_per_cluster = {}
        self.latencies_in_cluster = {}
        self.cost_per_cluster = {}
        self.latitudes_in_cluster = {}
        for c in sorted(self.clusters):    
            self.weight_per_cluster[c] = sum(self.weights[self.clusters[c]])
            lats = self.X[self.clusters[c]][:,0]
            assert( all([lats[i]>lats[i+1] for i in range(len(lats)-1)]) ) 
            self.latitudes_in_cluster[c] = list(lats)
            lat = 0.
            latencies = []
            current = north_pole
            for x in self.X[self.clusters[c]]:
                lat += haversine(current,x)
                latencies.append(lat)
                current = np.array(x)
            
            self.latencies_in_cluster[c] = latencies
            
            self.cost_per_cluster[c] = sum(np.array(latencies)*self.weights[self.clusters[c]]) + (
                                        sleigh_weight * (latencies[-1] + self.distances_to_pole[self.clusters[c][-1]]))

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
            self.cost_per_cluster[k] = (self.weights[gift]+2*sleigh_weight) * self.distances_to_pole[gift]
            self.to_assign.remove(gift)


    def bound_increase_for_adding_gift_in_cluster(self,giftID,centroidID):
        i,k = giftID,centroidID
        n = len(self.clusters[k])
        if n==0:
            raise Exception('cluster was not initialized')
        lati,longi = self.X[i]
        dpole_i = self.distances_to_pole[i]
        j = n-np.searchsorted(self.latitudes_in_cluster[k][::-1],lati)
        if j==0:
            latency_i = dpole_i
        else:
            previous_gift = self.clusters[k][j-1]
            latency_i = self.latencies_in_cluster[k][j-1] + haversine(self.X[previous_gift],self.X[i])
        if j==n:
            #add gift in last position
            delta_latency = latency_i - self.latencies_in_cluster[k][j-1]
            delta_d = dpole_i - self.distances_to_pole[self.clusters[k][-1]]
            return self.weights[i]*(latency_i - dpole_i) + sleigh_weight * (delta_latency + delta_d)
        else:
            next_gift = self.clusters[k][j]
            delta_latency = latency_i +  haversine(self.X[next_gift],self.X[i]) - self.latencies_in_cluster[k][j]
            weight_after_j = sum(self.weights[self.clusters[k][j:]])
            return self.weights[i]*(latency_i - dpole_i) + delta_latency * (weight_after_j + sleigh_weight)

    def add_in_tour(self,giftID,centroidID):
        i,k = giftID,centroidID
        n = len(self.clusters[k])
        if n==0:
            raise Exception('cluster was not initialized')
        lati,longi = self.X[i]
        dpole_i = self.distances_to_pole[i]
        j = n-np.searchsorted(self.latitudes_in_cluster[k][::-1],lati)
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
        
        if min(self.centroids[k][1],self.X[i][1])<-150 and max(self.centroids[k][1],self.X[i][1])>150:
            lamean = (self.centroids[k][0] * self.weight_per_cluster[k] + self.X[i][0] *
                                 self.weights[i])/(self.weights[i]+self.weight_per_cluster[k])
            lo = np.array([self.centroids[k][1],self.X[i][1]])
            lo = np.where(lo<0,lo+360,lo)
            lomean = (lo[0] * self.weight_per_cluster[k] + lo[1]*self.weights[i])/(self.weights[i]+self.weight_per_cluster[k])
            if lomean>180:
                lomean = lomean-360.
            self.centroids[k] = np.array([lamean,lomean])
            #import pdb;pdb.set_trace()
        else:
            self.centroids[k] = (self.centroids[k] * self.weight_per_cluster[k] + self.X[i] *
                                 self.weights[i])/(self.weights[i]+self.weight_per_cluster[k])
        self.weight_per_cluster[k]+= self.weights[i]
        
        self.cost_per_cluster[k] +=  self.weights[i]*latency_i+ (weight_after_j + sleigh_weight) * delta_latency + sleigh_weight * delta_d

    def create_new_cluster(self,i):
        kk = np.searchsorted(self.centroids[:,1],self.X[i][1])
        for k in range(self.K,kk,-1):
            self.clusters[k] = self.clusters[k-1][:]
            self.weight_per_cluster[k] = self.weight_per_cluster[k-1]
            self.latitudes_in_cluster[k] = self.latitudes_in_cluster[k-1][:]
            self.latencies_in_cluster[k] = self.latencies_in_cluster[k-1][:]
            self.cost_per_cluster[k] = self.cost_per_cluster[k-1]

        self.K += 1
        self.centroids = np.insert(self.centroids, kk, self.X[i],axis=0)
        self.clusters[kk] = [i]
        self.latitudes_in_cluster[kk] = [self.X[i][0]]
        self.latencies_in_cluster[kk] = [self.distances_to_pole[i]]
        self.weight_per_cluster[kk] = self.weights[i]
        self.cost_per_cluster[kk] = (self.weights[i]+2*sleigh_weight) * self.distances_to_pole[i]

    def greedy_for_bound(self,best_in_next = 100, direction = 'west', disp_progress=True,width=40,wgpenalty=0,start=-180,init=True):
        if init:
            print 'initialization...'
            self.to_assign = np.random.permutation(range(self.N)).tolist()
            '''
            if not(np.all([self.centroids[i][1]<=self.centroids[i+1][1] for i in range(len(self.centroids)-1)])):
                print 'warning: centroids were not sorted by longitude. I reorder them'
                self.centroids = np.array(sorted(self.centroids, key=lambda x: x[1]))
                
            self.init_clusters_with_centroids()
            '''
            self.centroids = np.zeros((0,2))
            self.K = 0
        
        shiftedlong = np.where(self.X[self.to_assign][:,1]<start,self.X[self.to_assign][:,1]+360,self.X[self.to_assign][:,1])
        if direction=='west':
            print 'sorting gifts per longitude...'
            self.to_assign = np.array(self.to_assign)[np.argsort(shiftedlong)]
        elif direction=='east':
            print 'sorting gifts per longitude...'
            self.to_assign = np.array(self.to_assign)[np.argsort(shiftedlong)[::-1]]
        elif direction=='south':
            print 'sorting gifts per latitude...'
            self.to_assign = np.array(self.to_assign)[np.argsort(self.X[self.to_assign][:,0])[::-1]]
        elif direction=='north':
            print 'sorting gifts per latitude...'
            self.to_assign = np.array(self.to_assign)[np.argsort(self.X[self.to_assign][:,0])]
        else:
            raise ValueError('direction not implemented')
        self.to_assign = self.to_assign.tolist()
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
        
            
            if self.K == 0:
                self.create_new_cluster(self.to_assign[0])
                del self.to_assign[0]
                continue
            
            candidates = self.to_assign[:best_in_next]
                
            bounds_inc = []
            for i in candidates:
                km = np.searchsorted(self.centroids[:,1], self.X[i][1]-width)
                kp = np.searchsorted(self.centroids[:,1], self.X[i][1]+width)
                bounds_inc.extend([(self.bound_increase_for_adding_gift_in_cluster(i,k),i,k)
                                   for k in range(km,kp) if self.weight_per_cluster[k]+self.weights[i]<weight_limit-wgpenalty])
                
            if not bounds_inc:
                self.create_new_cluster(self.to_assign[0])
                del self.to_assign[0]
                continue
                
            sorted_bounds_inc = sorted(bounds_inc)
            assigned = False
            for inc,i,c in sorted_bounds_inc:
                if inc> 2*self.distances_to_pole[i]*sleigh_weight:
                    #import pdb;pdb.set_trace()
                    self.create_new_cluster(self.to_assign[0])
                    assigned = True
                    del self.to_assign[0]
                    #print 'one more clust '+str(self.K)
                    #if self.K>1500:
                    #    import pdb;pdb.set_trace()#TMP
                    break
                if self.weight_per_cluster[c]+self.weights[i]<weight_limit-wgpenalty:
                    self.add_in_tour(i,c)
                    assigned = True
                    self.to_assign.remove(i)
                    break

            if not assigned:
                raise Exception('not able to assign a trip in this window of longitudes.')

            if not(self.to_assign):
                break


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
