north_pole = (90,0)
weight_limit = 1000#TMP 1000
sleigh_weight = 10

import numpy as np
from haversine import haversine,AVG_EARTH_RADIUS
import pandas as pd
import sub_mlp as smlp
from mpl_toolkits.basemap import Basemap
import pylab

class Solution:
    """
    A class for a solution of the problem solution
    can be initialized with
     * a cluster file ('../clusters/xxx')
     * a csv-solution file ('../solutions/xxx')
     * a cluster dictionary

    the data is stored as a dict tripID ->MLT
    """
    def __init__(self,cluster_or_solution,gifts,compute_tours=False):
        self.gifts = gifts
        self.X = gifts[['Latitude','Longitude']].values
        if isinstance(cluster_or_solution,str):
            cluster = self.load(cluster_or_solution)
        else:
            imin = min([min([vi for vi in v]) for v in cluster_or_solution.values()])
            if imin == 0:
                cluster = cluster_or_solution
            elif imin==1:
                cluster = {c: [i-1 for i in v] for c,v in cluster_or_solution.iteritems()}
            else:
                raise Exception('bad cluster indices')
        assert(min([min([vi for vi in v]) for v in cluster.values()]) == 0 )
        self.cluster = cluster
        self.compute_wgt_per_cluster()
        self.lower_bound_per_cluster(true_weight=False)
        self.compute_latencies()
        self.compute_cluster_sizes()
        self.efficiencies = {k: self.wlatencies[k]/self.bound_per_cluster[k] for k in self.cluster}

        if compute_tours:
            self.mlts = {c: smlp.MLT(self.gifts,[i+1 for i in self.cluster.cluster[c]]) for c in self.cluster}
            self.tours = {c: smlp.Tour(self.mlts[c],range(1,self.mlts[c].n)) for c in self.cluster}

    def load(self,name):
        if name.startswith('../clusters'):
            f = open(name,'r')
            clusters_from_1 = eval(f.read())
            f.close()
            return {c: [i-1 for i in v] for c,v in clusters_from_1.iteritems()}
        elif name.startswith('../solutions'):
            sol = pd.read_csv(name)
            cluster = {}
            for i,c in sol.values:
                cluster.setdefault(c,[])
                cluster[c].append(i-1)
            return cluster
        else:
            raise ValueError('You must give a relative path starting with ../clusters or ../solutions')

    def save_cluster(self,name):
        f = open('../clusters/' + name,'w')
        clusters_from_1 = {c: [i+1 for i in v] for c,v in self.cluster.iteritems()}
        f.write(str(clusters_from_1))
        f.close()

    def write(self,name):
        if not name.startswith('../solutions'):
            name = '../solutions/' + name
        f = open(name,'w')
        f.write('GiftId,TripId\n')
        for c,v in self.cluster.cluster.iteritems():
            for i in v:
                f.write('{0},{1}\n'.format(i+1,c))
        
        f.close()

    def compute_wgt_per_cluster(self):
        self.wgts = self.gifts.Weight.values
        self.wgt_per_cluster = {c:sum([self.wgts[i] for i in v]) for c,v in self.cluster.iteritems()}

    def lower_bound_per_cluster(self,true_weight=False):
        latitude = self.gifts.Latitude.values
        d_from_pole = AVG_EARTH_RADIUS * (90-latitude)*np.pi/180.
        if true_weight:
            self.bound_per_cluster = {c:sum([self.wgts[i]*d_from_pole[i] for i in v]) *
                                  (1.+2*sleigh_weight/float(self.wgt_per_cluster[c])) for c,v in self.cluster.iteritems()}
        else:
            self.bound_per_cluster = {c:sum([self.wgts[i]*d_from_pole[i] for i in v]) * 1.02 for c,v in self.cluster.iteritems()}

    def compute_cluster_sizes(self):
        self.cluster_widths = {}
        self.cluster_angles = {}
        self.cluster_heights = {}
        for c in self.cluster:
            longs = np.array([self.X[i][1] for i in self.cluster[c]])
            lats = np.array([self.X[i][0] for i in self.cluster[c]])
            if min(longs)<-150 and max(longs)>150:
                lomean = ((np.mean(longs%360)+180)%360) -180
                ang = max(longs%360)-min(longs%360)
            else:
                lomean = np.mean(longs)
                ang = max(longs)-min(longs)
            wd = 0
            for la,lo in zip(lats,longs):
                wd += AVG_EARTH_RADIUS * np.cos(la*np.pi/180.) * np.abs(lo-lomean) * np.pi/180.
            self.cluster_widths[c] = 2*wd /len(lats)
            self.cluster_heights[c] = (max(lats)-min(lats))*np.pi/180 * AVG_EARTH_RADIUS
            self.cluster_angles[c] = ang

    def compute_latencies(self):
        self.latencies = {}
        self.wlatencies = {}
        self.max_pole_dist2 = {}

        for c in self.cluster:
            latency = 0.
            weighted_latency = 0.
            max_pole_dist2 = 0.
            current_node = north_pole

            for j in self.cluster[c]:
                latency += haversine(current_node,self.X[j])
                weighted_latency += latency * self.gifts.Weight[j]
                current_node = self.X[j]
                if self.X[j][1] > max_pole_dist2:
                    max_pole_dist2 = self.X[j][1]

            latency += haversine(current_node,north_pole)
            weighted_latency += latency * sleigh_weight
            max_pole_dist2 *= 2

            self.wlatencies[c] = weighted_latency
            self.latencies[c] = latency
            self.max_pole_dist2[c] = max_pole_dist2

class TripSplitter:
    def __init__(self,solution,trip1,trip2):
        self.items1 = solution.cluster[trip1]
        self.items2 = solution.cluster[trip2]
        self.all_its = np.array(self.items1+self.items2)
        self.X1 = solution.X[self.items1]
        self.X2 = solution.X[self.items2]
        self.wg1 = solution.gifts.Weight[self.items1].values
        self.wg2 = solution.gifts.Weight[self.items2].values
        self.ww = np.r_[self.wg1,self.wg2]
        self.SW = sum(self.wg1) + sum(self.wg2)
        self.trip1 = trip1
        self.trip2 = trip2
        self.distances = None #dict of precomputed distances

        self.XX = np.r_[self.X1,self.X2]
        lamean = np.mean(self.XX[:,0])
        longs  = self.XX[:,1]
        if min(longs)<-150 and max(longs)>150:
            lomean = ((np.mean(longs%360)+180)%360) -180
        else:
            lomean = np.mean(longs)
        self.centroid = np.array([lamean,lomean])
        self.ZZ = np.apply_along_axis(self.to_cartesian,1,self.XX)

        zcent = self.to_cartesian(self.centroid)
        zpole = self.to_cartesian(north_pole)
        self.normal_vect  = self.to_cartesian(self.centroid)
        self.normal_vect = self.normal_vect/np.linalg.norm(self.normal_vect)
        to_pole = zpole - zcent
        self.dir_pole = to_pole - self.normal_vect.dot(to_pole) * self.normal_vect
        self.dir_pole = self.dir_pole/np.linalg.norm(self.dir_pole)
        self.cross_vect = np.cross(self.dir_pole,self.normal_vect)


    def to_cartesian(self,x):
        phi = (90-x[0]) * np.pi/180.
        theta = x[1] * np.pi/180.
        sphi = np.sin(phi)
        return np.array([sphi*np.cos(theta),sphi*np.sin(theta),np.cos(phi)])

    def draw_initial(self,fig=1):
        pylab.figure(fig)
        pylab.clf()
        m = Basemap(llcrnrlon=-190,llcrnrlat=-80,urcrnrlon=190,urcrnrlat=80,projection='mill')
        m.drawcoastlines()
        m.drawcountries()
        x1,y1 = m(self.X1[:,1],self.X1[:,0])
        x2,y2 = m(self.X2[:,1],self.X2[:,0])
        x0,y0 = m(self.centroid[1],self.centroid[0])
        m.scatter(x1,y1,color='red')
        m.scatter(x2,y2,color='blue')
        m.scatter(x0,y0,color='yellow')

    def draw_split(self,fig=2):
        pylab.figure(fig)
        pylab.clf()
        m = Basemap(llcrnrlon=-190,llcrnrlat=-80,urcrnrlon=190,urcrnrlat=80,projection='mill')
        m.drawcoastlines()
        m.drawcountries()
        x1,y1 = m(self.XX[self.split1,1],self.XX[self.split1,0])
        x2,y2 = m(self.XX[self.split2,1],self.XX[self.split2,0])
        x0,y0 = m(self.centroid[1],self.centroid[0])
        m.scatter(x1,y1,color='red')
        m.scatter(x2,y2,color='blue')
        m.scatter(x0,y0,color='yellow')

    def theta_split(self,theta):
        u = np.cos(theta)*self.dir_pole + np.sin(theta)*self.cross_vect
        profits = self.ZZ.dot(u) - min(self.ZZ.dot(u))
        order = np.argsort(profits)[::-1]
        w = 0.
        sub = []
        for i in order:
            w+=self.ww[i]
            sub.append(i)
            if w>750:
                break
        n = len(sub)
        wcand = 0.
        candidates = []
        prof = []
        sizes = []
        for i in order[n:]:
            wcand+=self.ww[i]
            candidates.append(i)
            prof.append(profits[i])
            sizes.append(self.ww[i])
            if wcand>500:
                break

        ks = self.knapsack(sizes,prof,weight_limit-w,0.01)
        wsplit = (w+sum([sizes[i] for i in ks]))
        if (self.SW-wsplit <= weight_limit and wsplit<weight_limit):
            ids = np.r_[sub,order[n:][ks]]
            nids = np.array(list(set(range(len(self.all_its))) - set(ids)))
            self.split1 = ids
            self.split2 = nids

        else:
            print 'no split found: {0},{1}'.format(wsplit,self.SW-wsplit)


    def knapsack(self,sizes,profits,capacity,epsilon):
        P = max(profits)
        n = len(profits)
        sizes = np.floor(np.array(sizes)/epsilon)
        capa = int(capacity/epsilon)

        opt = {}
        for w in range(capa+1):
            opt[-1, w] = 0.
        for i in range(n):
            #print i
            for w in range(capa+1):
                if sizes[i] <= w:
                    opt[i,w] = max(opt[i-1,w-sizes[i]]+profits[i], opt[i-1,w])
                else:
                    opt[i,w] = opt[i-1,w]

        best = []
        k = capa
        i = n-1
        while k>0 and i>=0:
             if opt[i,k]>opt[i-1,k]:
                 best.append(i)
                 k-= sizes[i]
             i-=1

        return best

    def precompute_distances(self):
        n = len(self.XX)
        d = {}
        for i,x in enumerate(self.XX):
            for j,y in enumerate(self.XX):
                if i < j:
                    continue
                elif i == j:
                    d[i,j] = 0.
                else:
                    d[i,j] = haversine(x,y)
                    d[j,i] = d[i,j]
        #north-pole is (-1)
        for i,x in enumerate(self.XX):
            d[-1,i] = haversine(x,north_pole)
            d[i,-1] = d[-1,i]
        d[-1,-1] = 0.

        self.distances = d

    def construct(self,alpha=0.3,beta=0.4):
        if self.distances is None:
            self.precompute_distances()

        n = len(self.XX)
        current_nodes = [-1,-1]
        candidates = range(1,n)
        tour = ([],[])
        wg = [0.,0.]

        while len(candidates)>1:
            if np.random.rand()<0.5:
                trip = 0
            else:
                trip = 1

            current = current_nodes[trip]
            dist_from_current = [(self.distances[current,i]/(self.ww[i]**beta),i) for i in candidates]
            sorted_from_current = np.array(sorted(dist_from_current))

            max_index = np.searchsorted(sorted_from_current[:,0],(1.+alpha)*sorted_from_current[0,0])
            potential_next = [i for i in range(max_index) if self.ww[int(sorted_from_current[i][1])]+wg[trip] <weight_limit]
            if not potential_next:
                print 'failed !' #TODO try with other trip first, and raise Exception
                import pdb;pdb.set_trace()
                return
            next_node = int(sorted_from_current[potential_next[np.random.randint(len(potential_next))]][1])
            candidates.remove(next_node)
            tour[trip].append(next_node)
            wg[trip] += self.ww[next_node]
            current_nodes[trip] = next_node
        return tour,wg,candidates#TODO here: check, it looks that some weight disappears...


"""
#####################
#solution analysis: #
#####################
north_pole = (90,0)
weight_limit = 1000#TMP 1000
sleigh_weight = 10


import pandas as pd
import numpy as np
from haversine import haversine,AVG_EARTH_RADIUS
from mpl_toolkits.basemap import Basemap
import Kmeans as km
import build_clusters as bc
import vrp
import sub_mlp as smlp


gifts = pd.read_csv('../input/gifts.csv')


sol = vrp.Solution('../solutions/sol_greedy_East5000__12504429700.8.csv',gifts)
sum(sol.wlatencies.values())

ts = vrp.TripSplitter(sol,0,3)
ts.draw_initial()

#BOF...
theta = 90
ts.theta_split(theta*np.pi/180.)
ts.draw_split()

ml1 = smlp.MLT(sol.gifts,[i+1 for i in ts.all_its[ts.split1]])
ml2 = smlp.MLT(sol.gifts,[i+1 for i in ts.all_its[ts.split2]])

tr1 = ml1.optimize(disp=2,restart=1)
tr2 = ml2.optimize(disp=2,restart=1)

tr1.wlatency+tr2.wlatency
sol.wlatencies[ts.trip2]+sol.wlatencies[ts.trip1]


#for 998kg west, with theta=-90, there is an improvement...
sol = vrp.Solution('../solutions/sol_west_998_12505488385.5.csv',gifts)
ts = vrp.TripSplitter(sol,1409,1410)

"""
