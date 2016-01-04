north_pole = (90,0)
weight_limit = 1000#TMP 1000
sleigh_weight = 10

import numpy as np
from haversine import haversine,AVG_EARTH_RADIUS
import pandas as pd
import sub_mlp as smlp
from mpl_toolkits.basemap import Basemap
import pylab
import sklearn.neighbors
import scipy.stats

class Solution:
    """
    A class for a solution of the problem solution
    can be initialized with
     * a cluster file ('../clusters/xxx')
     * a csv-solution file ('../solutions/xxx')
     * a cluster dictionary

    the data is stored as a dict tripID ->MLT
    """
    def __init__(self,cluster_or_solution,gifts):
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
        self.update_cluster_dicts()

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

    def write(self,name,only_clusts = None):
        if not name.startswith('../solutions'):
            name = '../solutions/' + name
        f = open(name,'w')
        f.write('GiftId,TripId\n')
        if only_clusts is None:
            for c,v in self.cluster.iteritems():
                for i in v:
                    f.write('{0},{1}\n'.format(i+1,c))
        else:
            for c in only_clusts:
                v = self.cluster[c]
                for i in v:
                    f.write('{0},{1}\n'.format(i+1,c))
        
        f.close()

    def update_cluster_dicts(self):
        self.compute_wgt_per_cluster()
        self.lower_bound_per_cluster(true_weight=False)
        self.compute_latencies()
        self.compute_cluster_sizes()
        self.efficiencies = {k: self.wlatencies[k]/self.bound_per_cluster[k] for k in self.cluster}

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
        self.lomeans = {}
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
            self.lomeans[c] = lomean

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
        self.gifts = solution.gifts
        self.wl1 = solution.wlatencies[self.trip1]
        self.wl2 = solution.wlatencies[self.trip2]
        self.latcy1 = solution.latencies[self.trip1]
        self.latcy2 = solution.latencies[self.trip2]

        #vars for the geometric split
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

    def theta_split(self,theta,thres=950):
        u = np.cos(theta)*self.dir_pole + np.sin(theta)*self.cross_vect
        profits = self.ZZ.dot(u) - min(self.ZZ.dot(u))
        order = np.argsort(profits)[::-1]
        w = 0.
        sub = []
        for i in order:
            w+=self.ww[i]
            sub.append(i)
            if w>min(self.SW-weight_limit,thres):
                break
        n = len(sub)
        w1 = w
        if w1>self.SW/2:
            ids = sub[:]
        else:
            wcand = w1
            candidates = []
            prof = []
            sizes = []
            for i in order[n:]:
                wcand+=self.ww[i]
                candidates.append(i)
                prof.append(profits[i])
                sizes.append(self.ww[i])
                if wcand>self.SW-w1:
                    break

            mpr = min(prof)
            prof = [pr - mpr + 0.01 for pr in prof]
            ks = self.knapsack(sizes,prof,self.SW/2.-w1,0.01)
            wsplit = (w+sum([sizes[i] for i in ks]))

            ids = np.r_[sub,order[n:][ks]]

            print wsplit
            for cand,pr,wg in [(i,prof[j],self.ww[i]) for j,i in enumerate(candidates) if j not in ks]:
                if abs(wsplit+wg-self.SW/2.) < abs(wsplit-self.SW/2):
                    ids = np.r_[ids,[cand]]
                    wsplit += wg

            print wsplit

        if (self.SW-wsplit <= weight_limit and wsplit<weight_limit):
            nids = np.array(list(set(range(len(self.all_its))) - set(ids)))
            self.split1 = ids
            self.split2 = nids
            self.wsplit1 = wsplit
            self.wsplit2 = self.SW - wsplit

        else:
            print 'no split found: {0},{1}'.format(wsplit,self.SW-wsplit)

    def opt_split(self,restart=1):
        ml1 = smlp.MLT(self.gifts,[i+1 for i in self.all_its[self.split1]])
        ml2 = smlp.MLT(self.gifts,[i+1 for i in self.all_its[self.split2]])

        tr1 = ml1.optimize(disp=2,restart=restart)
        tr2 = ml2.optimize(disp=2,restart=restart)

        newcost = tr1.wlatency+tr2.wlatency
        oldcost = self.wl1 + self.wl2
        print 'old: ' + str(oldcost)
        print 'new: ' + str(newcost)
        if newcost<oldcost:
            self.opttrs = [tr1,tr2]

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


class KTour:
    """
    A ktuple of tours for the k-MLT problem, starting and ending at north pole
    """
    def __init__(self,solution,tourIDs,orders=None,max_excess = 0.,mu_init = 500.):
        """
        The constructor takes a Sol, a list of tours and a list of ordered gift IDs.
        """
        if orders is None:
            ind = 0
            orders = []
            for i in tourIDs:
                orders.append(range(ind,ind+len(solution.cluster[i])))
                ind+=len(solution.cluster[i])

        self.solution = solution
        self.tourIDs = tourIDs
        self.gifts = np.array([])
        self.ww = np.array([])
        self.XX = np.array([[],[]]).T
        for i in tourIDs:
            self.gifts = np.r_[self.gifts,solution.cluster[i]]
            self.ww = np.r_[self.ww,solution.gifts.Weight[solution.cluster[i]].values]
            self.XX = np.r_[self.XX,solution.X[solution.cluster[i]]]

        self.ZZ = np.apply_along_axis(self.to_cartesian,1,self.XX)

        assert(len(self.ww)==sum([len(o) for o in orders]))
        self.K = len(orders)

        #wlatencies = tuple([solution.wlatencies[i] for i in tourIDs])
        #latencies = tuple([solution.latencies[i] for i in tourIDs])
        #wsplit = tuple([solution.wgt_per_cluster[i] for i in tourIDs])

        self.N = len(self.ww)
        self.precompute_distances()

        #self.tours = [Tour(self,o,lat,wl,wgt) for o,wl,lat,wgt in zip(orders,wlatencies,latencies,wsplit)]
        self.tours = [Tour(self,o,None,None,None) for o in orders]

        self.Z = sum([t.wlatency for t in self.tours])
        self.L = sum([t.latency for t in self.tours])
        self.beta_dis = 1.
        self.kdtree = sklearn.neighbors.kd_tree.KDTree(self.ZZ)

        self.gift_to_tour = {}
        for k in range(self.K):
            for i in orders[k]:
                self.gift_to_tour[i] = k

        self.max_excess = max_excess
        self.all_granular_neighborhoods()
        self.preprocess_all_tours()

        self.mu_excess = mu_init
        self.excess = sum([max(0.,t.weight-weight_limit) for t in self.tours])
        self.cost = self.Z + self.mu_excess * self.excess

    def to_cartesian(self,x):
        phi = (90-x[0]) * np.pi/180.
        theta = x[1] * np.pi/180.
        sphi = np.sin(phi)
        return np.array([sphi*np.cos(theta),sphi*np.sin(theta),np.cos(phi)])

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

    def preprocess_all_tours(self):
        for t in self.tours:
            t.preprocess_subsequences()

    def all_granular_neighborhoods(self):
        for t in self.tours:
            t.compute_granular_neighborhoods()

    def update_sol(self):
        for tr,t in zip(self.tourIDs,self.tours):
            self.solution.cluster[tr] = list(np.int32(self.gifts[t.order]))
        self.solution.update_cluster_dicts()

    def clark_wright(self):
        savings = {}
        for i in range(self.N):
            for j in range(self.N):
                if i==j:
                    continue
                new = ((sleigh_weight+self.ww[j]) * (self.distances[-1,i]+self.distances[i,j])
                       + self.distances[-1,i]*self.ww[i] + self.distances[-1,j] * sleigh_weight )
                old = self.distances[-1,i] * (self.ww[i] + 2*sleigh_weight) + self.distances[-1,j] * (self.ww[j] + 2*sleigh_weight)
                savings[i,j] = old-new
        sorted_savings = sorted([(sv,ij) for ij,sv in savings.iteritems()])
        tours = []
        ext = []
        wgt = []
        alloc = []
        while True:
            sv,(i,j) = sorted_savings.pop()
            iin = i in alloc
            jin = j in alloc
            if iin and jin:
                continue
            if (not iin) and any([j == e[0] for e in ext]):
                k = [id for id,e in enumerate(ext) if j==e[0]][0]
                if self.ww[i] + wgt[k] < weight_limit:
                    tours[k].insert(0,i)
                    ext[k] = (i,ext[k][1])
                    wgt[k] = self.ww[i] + wgt[k]
                    alloc.append(i)

            elif (not jin) and any([i == e[1] for e in ext]):
                k = [id for id,e in enumerate(ext) if i==e[1]][0]
                if self.ww[j] + wgt[k] < weight_limit:
                    tours[k].append(j)
                    ext[k] = (ext[k][0],j)
                    wgt[k] = self.ww[j] + wgt[k]
                    alloc.append(j)

            elif (not iin) and (not jin) and len(tours)<self.K:
                tours.append([i,j])
                ext.append((i,j))
                wgt.append(self.ww[i] + self.ww[j])
                alloc.extend([i,j])

            if len(alloc) == self.N:
                break

        self.tours = [Tour(self,o,None,None,None) for o in tours]

        self.Z = sum([t.wlatency for t in self.tours])
        self.L = sum([t.latency for t in self.tours])

        self.gift_to_tour = {}
        for k in range(self.K):
            for i in tours[k]:
                self.gift_to_tour[i] = k

        self.all_granular_neighborhoods()
        self.preprocess_all_tours()

        self.excess = sum([max(0.,t.weight-weight_limit) for t in self.tours])
        self.cost = self.Z + self.mu_excess * self.excess


    def RVND(self,inbest = 5, disp=0,start_it = 0,tabu_tenure=3):
        #NLinit = ['relocate','exchange','twoopt']
        NLinit = ['relocate','twoopt']
        NL = NLinit[:]
        it = start_it
        if disp>=2:
            print it,self.cost

        tabu = []
        fbest = self.cost

        excess_it = 0
        no_excess_it = 0

        while NL:
            it+=1
            neighborhood = NL[np.random.randint(len(NL))]

            tab = []
            for args in self.explore_neighborhood(neighborhood):
                sw1,sw2 = self.eval_move(neighborhood,args)
                t1,t2 = args[0], args[1]
                diff_excess = (-(max(0.,self.tours[t1].weight-weight_limit) + max(0.,self.tours[t2].weight-weight_limit))
                                + max(0.,sw1.weight-weight_limit) + max(0.,sw2.weight-weight_limit))
                diff_latency = sw1.wlatency + sw2.wlatency - self.tours[t1].wlatency - self.tours[t2].wlatency
                if tabu_tenure>0:
                    istabu = any([t[:2] in self.all_moves(neighborhood,args) for t in tabu])
                else:
                    istabu = False
                tab.append((diff_latency + self.mu_excess*diff_excess,args,istabu))


            if inbest == 5:
                r = np.random.randint(18)
            elif inbest == 4:
                r = np.random.randint(17) + 1
            elif inbest == 3:
                r = np.random.randint(15) + 3
            elif inbest == 2:
                r = np.random.randint(12) + 6
            elif inbest == 1:
                r = 16

            if tabu_tenure>0:
                stab = sorted([t for t in tab if not(t[2])])
            else:
                stab = sorted(tab)

            if r == 0 and len(stab) >= 5 and stab[4][0] < 1e-3:
                next_move = stab[4]
            elif r<=2 and len(stab) >= 4 and stab[3][0] < 1e-3:
                next_move = stab[3]
            elif r<=5 and len(stab) >= 3 and stab[2][0] < 1e-3:
                next_move = stab[2]
            elif r<=10 and len(stab) >= 2 and stab[1][0] < 1e-3:
                next_move = stab[1]
            elif r<=17 and len(stab) >= 1 and stab[0][0] < 1e-3:
                next_move = stab[0]
            else:
                NL.remove(neighborhood)
                if disp>=2:
                    print it,self.cost,self.excess
                continue

            if tabu_tenure>0:
                revmoves = self.reverse_moves(neighborhood,next_move[1])
                tabu = [(i,k,t-1) for i,k,t in tabu if t>0]
                tabu.extend([(i,k,tabu_tenure) for i,k in revmoves])

            self.move(neighborhood,next_move[1])
            fbest = self.cost

            NL = NLinit[:]
            if self.excess > 0:
                excess_it += 1
                no_excess_it = 0
                if excess_it == 6:
                    self.mu_excess *= 10
                    excess_it = 0
            else:
                no_excess_it += 1
                excess_it = 0
                if no_excess_it == 6:
                    self.mu_excess /= 5
                    no_excess_it = 0

            if disp>=2:
                print it,self.cost,self.excess,neighborhood

        if self.excess > 1e-3:
            self.mu_excess *= 10
            self.cost = self.Z + self.mu_excess * self.excess
            print 'augmenting weight penalty: mu={0}'.format(self.mu_excess)
            self.RVND(inbest=inbest,disp=disp,start_it=it)

    def reverse_moves(self,neighborhood,args):
        if neighborhood == 'relocate':
            k1,k2,i,m,j = args
            return [(self.tours[k1].order[k],k1) for k in range(i,i+m)]
        elif neighborhood == 'exchange':
            k1,k2,i,m,p,j,t = args
            return [(self.tours[k1].order[k],k1) for k in range(i,i+m)] + [self.tours[k2].order[j],k2]
        elif neighborhood == 'twoopt':
            k1,k2,i,j = args
            n1 = self.tours[k1].n
            n2 = self.tours[k2].n
            return [(self.tours[k1].order[k],k1) for k in range(i,n1)] + [(self.tours[k2].order[k],k2) for k in range(j,n2)]

    def all_moves(self,neighborhood,args):
        if neighborhood == 'relocate':
            k1,k2,i,m,j = args
            return [(self.tours[k1].order[k],k2) for k in range(i,i+m)]
        elif neighborhood == 'exchange':
            k1,k2,i,m,p,j,t = args
            return [(self.tours[k1].order[k],k2) for k in range(i,i+m)] + [self.tours[k2].order[j],k1]
        elif neighborhood == 'twoopt':
            k1,k2,i,j = args
            n1 = self.tours[k1].n
            n2 = self.tours[k2].n
            return [(self.tours[k1].order[k],k2) for k in range(i,n1)] + [(self.tours[k2].order[k],k1) for k in range(j,n2)]

    def relocate(self,k1,k2,i,m,j):
        """
        relocates subseq i...i+m-1 of tour k1 at pos j of tour k2
        """
        assert(self.tours[k1].subseq is not None)
        assert(self.tours[k2].subseq is not None)

        n1 = self.tours[k1].n
        n2 = self.tours[k2].n

        assert(i >= 0)
        assert(i+m-1 < n1)
        assert(m>=1)
        assert(j >= 0)
        assert(j <= n2)
        assert(k1 != k2)

        #0 ... i-1  | i   i+m-1 | i+m ... n1-1
        #0 ... j-1 | j ... n2-1

        if i>0:
            seq1 = self.tours[k1].subseq[0,i]
            if i+m < n1:
                seq1 = seq1 + self.tours[k1].subseq[i+m,n1]
        else:
            if i+m < n1:
                seq1 = self.tours[k1].subseq[i+m,n1]
            else:
                seq1 = None#TODO to_tour() must return an empty tour

        seq2 = self.tours[k1].subseq[i,i+m]
        if j > 0:
            seq2 = self.tours[k2].subseq[0,j] + seq2
        if j < n2:
            seq2 = seq2 + self.tours[k2].subseq[j,n2]

        return (seq1.to_tour(),seq2.to_tour())

    def twoopt(self,k1,k2,i,j):
        '''
        i-1->i in tour k1 exchanged with j-1->j in tour k2, so
        the new tours become
        0...i-1 | j ... n2-1
        0...j.1 | i ... n1-1
        '''
        assert(self.tours[k1].subseq is not None)
        assert(self.tours[k2].subseq is not None)

        n1 = self.tours[k1].n
        n2 = self.tours[k2].n

        assert(i>=1)
        assert(j>=1)
        assert(i < n1)
        assert(j < n2)

        seq1 = self.tours[k1].subseq[0,i]
        seq2 = self.tours[k2].subseq[0,j]

        seq1 = seq1 + self.tours[k2].subseq[j,n2]
        seq2 = seq2 + self.tours[k1].subseq[i,n1]

        return (seq1.to_tour(),seq2.to_tour())


    def exchange(self,k1,k2,i,m,p,j,t):
        """
        subseq i...i+m-1 of tr k1 goes before (former) position p of tour k2
              and j goes before (former) position t of tour k1
        """
        #0 ... i-1  | i   i+m-1 | i+m ... n1-1
        #0 ... j-1 | j | j+1 ... n2-1

        assert(self.tours[k1].subseq is not None)
        assert(self.tours[k2].subseq is not None)

        n1 = self.tours[k1].n
        n2 = self.tours[k2].n

        assert(i >= 0)
        assert(i+m-1 < n1)
        assert(m>=1)
        assert(j >= 0)
        assert(j < n2)
        assert(k1 != k2)
        assert(p >= 0)
        assert(p <= n2)
        assert(t >= 0)
        assert(t <= n1)

        seq1 = self.tours[k2].subseq[j,j+1]
        seq2 = self.tours[k1].subseq[i,i+m]

        if t< i:
            if t>0:
                seq1 = self.tours[k1].subseq[0,t] + seq1
            seq1 = seq1 + self.tours[k1].subseq[t,i]
            if i+m < n1:
                seq1 = seq1 + self.tours[k1].subseq[i+m,n1]
        elif t>i+m:
            seq1 = self.tours[k1].subseq[i+m,t] + seq1
            if i>0:
                seq1 = self.tours[k1].subseq[0,i] + seq1
            if t<n1:
                seq1 = seq1 + self.tours[k1].subseq[t,n1]
        else:
            if i>0:
                seq1 = self.tours[k1].subseq[0,i] + seq1
            if i+m < n1:
                seq1 = seq1 + self.tours[k1].subseq[i+m,n1]

        if p< j:
            if p>0:
                seq2 = self.tours[k2].subseq[0,p] + seq2
            seq2 = seq2 + self.tours[k2].subseq[p,j]
            if j+1 < n2:
                seq2 = seq2 + self.tours[k2].subseq[j+1,n2]
        elif p>j+1:
            seq2 = self.tours[k2].subseq[j+1,p] + seq2
            if j>0:
                seq2 = self.tours[k2].subseq[0,j] + seq2
            if p<n2:
                seq2 = seq2 + self.tours[k2].subseq[p,n2]
        else:
            if j>0:
                seq2 = self.tours[k2].subseq[0,j] + seq2
            if j+1<n2:
                seq2 = seq2 + self.tours[k2].subseq[j+1,n2]

        return (seq1.to_tour(),seq2.to_tour())

    def eval_move(self,neighborhood,args):
        if neighborhood == 'relocate':
            sw1,sw2 = self.relocate(*args)
        elif neighborhood == 'exchange':
            sw1,sw2 = self.exchange(*args)
        elif neighborhood == 'twoopt':
            sw1,sw2 = self.twoopt(*args)
        return sw1,sw2

    def move(self,neighborhood,args):
        t1 = args[0]
        t2 = args[1]
        sw1,sw2 = self.eval_move(neighborhood,args)

        #if sw1.wlatency != Tour(self,sw1.order).wlatency:
        #    print sw1.wlatency,Tour(self,sw1.order).wlatency
        #    import pdb;pdb.set_trace()

        ol = self.tours[t1].latency + self.tours[t2].latency
        wol = self.tours[t1].wlatency + self.tours[t2].wlatency

        sw1.preprocess_subsequences()
        sw1.compute_granular_neighborhoods()
        sw2.preprocess_subsequences()
        sw2.compute_granular_neighborhoods()

        diff_excess = ( -(max(0.,self.tours[t1].weight-weight_limit) + max(0.,self.tours[t2].weight-weight_limit))
                                + max(0.,sw1.weight-weight_limit) + max(0.,sw2.weight-weight_limit) )

        self.tours[t1] = sw1
        self.tours[t2] = sw2
        self.L = self.L - ol + sw1.latency + sw2.latency
        self.Z = self.Z - wol + sw1.wlatency + sw2.wlatency
        self.excess += diff_excess
        self.cost = self.Z + self.mu_excess * self.excess

        for i in sw1.order:
            self.gift_to_tour[i] = t1
        for i in sw2.order:
            self.gift_to_tour[i] = t2

    def explore_neighborhood(self,neighborhood,maxm=3):
        if neighborhood == 'relocate':
            for k in range(self.K):
                remk = weight_limit - self.tours[k].weight
                maxw = self.max_excess + remk
                if maxw < 0:
                    continue
                for indi,i in enumerate(self.tours[k].order):
                    for j in self.tours[k].grangb[indi]:
                        tourj = self.gift_to_tour[j]
                        indj = self.tours[tourj].order.index(j)
                        m = 1
                        while self.tours[tourj].subseq[indj,indj+m].wgt < maxw:
                            yield (tourj,k,indj,m,indi+1)
                            m+=1
                            if m > maxm or indj+m > self.tours[tourj].n:
                                break
                            if indj+m == self.tours[tourj].n:
                                break

                        m = 1
                        while self.tours[tourj].subseq[indj-m+1,indj+1].wgt < maxw:
                            ngbm1 = self.tours[tourj].grangb[indj-m+1]
                            if all([self.gift_to_tour[ng]!=k for ng in ngbm1]):
                                yield (tourj,k,indj-m+1,m,indi)
                            m+=1
                            if m > maxm or indj+m > self.tours[tourj].n:
                                break
                            if indj-m+1 == -1:
                                break

        elif neighborhood == 'exchange':
            for k in range(self.K):
                remk = weight_limit - self.tours[k].weight
                maxwk = self.max_excess + remk
                for indi,i in enumerate(self.tours[k].order):
                    capa_k = maxwk +  self.ww[i]
                    for j in self.tours[k].grangb[indi]:
                        tourj = self.gift_to_tour[j]
                        indj = self.tours[tourj].order.index(j)
                        #remj = weight_limit - self.tours[tourj].weight
                        #maxwj = self.max_excess + remj
                        #if self.ww[i] > maxwj:
                        #    continue
                        #ok, so elem indi of tour k will go before/after indj #TODO before/after dep. on latitude
                        for kj in range(self.tours[tourj].n):
                            for mj in range(1,min(maxm+1,self.tours[tourj].n+1-kj)):
                                if self.tours[tourj].subseq[kj,kj+mj].wgt > capa_k:
                                    break
                                if self.tours[tourj].subseq[kj,kj+mj].wgt < self.ww[i]:
                                    continue

                            if ((self.tours[tourj].subseq[kj,kj+mj].wgt > capa_k) or
                                (self.tours[tourj].subseq[kj,kj+mj].wgt < self.ww[i])):
                                    break

                            #ok, subchain kj,mj can go in tour k, but where ?
                            srdis = sorted([(self.distances[self.tours[tourj].order[kj],ki],ki)
                                            for ki in self.tours[tourj].grangb[kj]
                                            if self.gift_to_tour[ki]==k])
                            if not(srdis):
                                break
                            p = srdis[0][1]
                            ip = self.tours[k].order.index(p)

                            yield (tourj,k,kj,mj,ip,indi,indj)

        elif neighborhood == 'twoopt':
            for k in range(self.K):
                nk = self.tours[k].n
                for indi,i in enumerate(self.tours[k].order[:-1]):
                    indip1 = indi + 1
                    ip1 = self.tours[k].order[indip1]
                    for jp1 in self.tours[k].grangb[indi]:
                        tourj = self.gift_to_tour[jp1]
                        nj = self.tours[tourj].n
                        indjp1 = self.tours[tourj].order.index(jp1)
                        if indjp1==0:
                            continue
                        indj = indjp1 - 1
                        if ip1 not in self.tours[tourj].grangb[indj]:
                            continue
                        if ((self.tours[k].subseq[0,indip1].wgt + self.tours[tourj].subseq[indjp1,nj].wgt < weight_limit + self.max_excess) and
                            (self.tours[tourj].subseq[0,indjp1].wgt + self.tours[k].subseq[indip1,nk].wgt < weight_limit + self.max_excess)):
                            yield (k,tourj,indip1,indjp1)

class Tour:
    """
    A tour for the TS problem, startind and ending at north pole
    """
    def __init__(self,kt,order,latency = None, wlatency = None, weight = None):
        """
        The constructor takes a KTour and a list of ordered gift IDs (starting from 0 !).
        """
        self.n = len(order)
        self.kt = kt
        self.order = order
        if latency is None or wlatency is None or weight is None:
            self.compute_latencies()
        else:
            self.latency = latency
            self.wlatency = wlatency
            self.weight = weight

        #subsequence not computed yet.
        self.subseq = None

    def compute_latencies(self):
        latency = 0.
        weighted_latency = 0.
        current_node = -1
        wgg = 0

        for j in self.order:
            latency += self.kt.distances[current_node,j]
            weighted_latency += latency * self.kt.ww[j]
            wgg += self.kt.ww[j]
            current_node = j

        latency += self.kt.distances[current_node,-1]
        weighted_latency += latency * sleigh_weight

        self.wlatency = weighted_latency
        self.latency = latency
        self.weight = wgg

    def __repr__(self):
        return '<Tour of weighted latency '+str(self.wlatency)+' >'

    def preprocess_subsequences(self):
        """
        self.subseq[i,j] is the subseq with nodes order[i],,,.order[j-1]
        """
        self.subseq = {}
        for i in range(self.n):
            for j in range(i+1,self.n+1):
                if j == i+1:
                    self.subseq[i,j] = Subsequence(self.kt,[self.order[i]],0.,0.)
                else:
                    self.subseq[i,j] = self.subseq[i,j-1] + Subsequence(self.kt,[self.order[j-1]],0.,0.)

    def compute_granular_neighborhoods(self):
        radius = self.kt.beta_dis * self.kt.L/float(self.kt.N+self.kt.K) #TODO adapt with remaining capacity ?
        ngb = []
        for i in self.order:
            close_nodes = self.kt.kdtree.query_radius(self.kt.ZZ[i],radius/AVG_EARTH_RADIUS)[0]
            ngb.append([j for j in close_nodes if j not in self.order])
        self.grangb = ngb

    def swap(self,i,j):
        """
        swaps self.order[i] with self.order[j]
        """
        assert(i < j)
        assert(i >= 0)
        assert(j < self.n)
        assert(self.subseq is not None)

        if i == 0:
            seq = self.subseq[j,j+1]
        else:
            seq = self.subseq[0,i] + self.subseq[j,j+1]
        if j>i+1:
            seq = seq + self.subseq[i+1,j]
        seq = seq + self.subseq[i,i+1]
        if j < self.n-1:
            seq = seq + self.subseq[j+1,self.n]
        return seq.to_tour()

    def reinsert(self,i,j,k=1):
        """
        (self.order[i],...,self.order[i+k-1]) are reinserted in the seq, so that new pos of i is j
        """
        assert(self.subseq is not None)
        assert(i >= 0)
        assert(i+k-1 < self.n)
        assert(j >= 0)
        assert(j+k-1 < self.n)
        assert(i!=j)
        if i < j :
            #0  1  2 | i+k ... j+k-1 | i  i+1 i+k-1  | j+k  .. n-2
            seq = self.subseq[i,i+k]
            seq = self.subseq[i+k,j+k] + seq
            if i > 0:
                seq = self.subseq[0,i] + seq
            if j+k < self.n:
                seq = seq + self.subseq[j+k,self.n]
        else:
            #0  1  j-1 | i  i+1 i+k-1 | j ... i-1 | i+k ... n-2
            seq = self.subseq[i,i+k]
            if j>0:
                seq = self.subseq[0,j] + seq
            seq = seq + self.subseq[j,i]
            if i+k < self.n:
                seq = seq + self.subseq[i+k,self.n]

        return seq.to_tour()

    def rev_mid(self,i,j):
        """
        reverses the order of elements from pos i to pos j-1
        """
        assert(self.subseq is not None)
        assert(i < j-1)
        assert(i >= 0)
        assert(j <= self.n)
        if i == 0:
            seq = ~self.subseq[i,j]
        else:
            seq = self.subseq[0,i] + (~self.subseq[i,j])
        if j < self.n:
            seq = seq +  self.subseq[j,self.n]

        return seq.to_tour()

    def iter_neighbours(self,neighboorhood):
        if neighboorhood == 'swap':
            for i in range(self.n-1):
                for j in range(i+1,self.n):
                    yield self.swap(i,j)
        elif neighboorhood[0] == 'R' and neighboorhood[1].isdigit() and len(neighboorhood)==2:
            k = int(neighboorhood[1])
            for i in range(self.n-k+1):
                for j in range(self.n-k+1):
                    if i==j:
                        continue
                    else:
                        yield self.reinsert(i,j,k)
        elif neighboorhood == '2opt':
            for i in range(self.n-2):
                for j in range(i+2,self.n):
                    yield self.rev_mid(i,j)

class Subsequence:
    """
    A subsequence of a tour
    """
    def __init__(self,kt,order,latency=None,wlatency=None,wgt=None):
        self.kt = kt
        self.order = order
        if latency is None or wlatency is None or wgt is None:
            self.compute_latencies()
        else:
            self.latency = latency
            self.wlatency = wlatency
            self.wgt = wgt

    def __repr__(self):
        return '<Subsequence with nodes '+str(self.order)+' >'

    def compute_latencies(self):
        latency = 0.
        weighted_latency = 0.
        current_node = self.order[0]
        wgt = self.kt.ww[current_node]

        for j in self.order[1:]:
            latency += self.kt.distances[current_node,j]
            weighted_latency += latency * self.kt.ww[j]
            wgt += self.kt.ww[j]
            current_node = j

        self.wlatency = weighted_latency
        self.latency = latency
        self.wgt = wgt

    def __add__(self,next_sub):
        """
        +operator: concatenation of two subsequences
        """
        u = self.order[-1]
        v = next_sub.order[0]
        init_distance = self.latency + self.kt.distances[u,v]
        end_weight = next_sub.wgt
        lat = init_distance + next_sub.latency
        wlat = self.wlatency + next_sub.wlatency + init_distance * end_weight
        wg = self.wgt + next_sub.wgt
        return Subsequence(self.kt,self.order+next_sub.order,lat,wlat,wg)

    def __invert__(self):
        """
        ~operator: makes tour in reverse order
        """
        rev = self.order[::-1]
        wlat= self.wgt * self.latency - self.wlatency
        return Subsequence(self.kt,rev,self.latency,wlat,self.wgt)

    def to_tour(self):
        """
        add the north pole at the beginning and at the end of the subsequence
        """
        d1 = self.kt.distances[-1,self.order[0]]
        d2 = self.kt.distances[-1,self.order[-1]]
        lat = self.latency + d1 + d2
        wlat = self.wlatency + d1 * self.wgt + lat * sleigh_weight#(self.latency + d2) * sleigh_weight
        return Tour(self.kt,self.order,lat,wlat,self.wgt)

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
import scipy.stats


gifts = pd.read_csv('../input/gifts.csv')


sol = vrp.Solution('../solutions/sol_East_2000_993_12503567581.7.csv',gifts)
sum(sol.wlatencies.values())

ts = vrp.TripSplitter(sol,1402,1403)
ts.draw_initial()

#BOF...
theta = 90
ts.theta_split(theta*np.pi/180.)
ts.draw_split()

ts.opt_split(restart=8)

sol2 = vrp.Solution('../solutions/sol_East_2000_993_12503567581.7.csv',gifts)
sol2.cluster[ts.trip1] = list(ts.all_its[ts.split1[[(i-1) for i in ts.opttrs[0].order]]])
sol2.cluster[ts.trip2] = list(ts.all_its[ts.split2[[(i-1) for i in ts.opttrs[1].order]]])
sol2.write('solE993+1opt.csv')


#############################
# test neighboorhood search #
#############################

sol = vrp.Solution('../solutions/sol_East_2000_993_12503567581.7.csv',gifts)

#sections sorted per longitude
offset = -180.
sorlongs = [c for lo,c in sorted([(lo if lo>offset else lo+360.,c) for c,lo in sol.lomeans.iteritems()])]
nsec = 5
sections = []
ind = 0
k = 0
while True:
    if k < (nsec - (len(sorlongs) % nsec)) % nsec:
        nn = nsec -1
    else:
        nn = nsec
    sections.append(sorlongs[ind : ind + nn])
    ind += nn
    k += 1
    if ind >= len(sorlongs):
        break


#sec = 17
#kt = vrp.KTour(sol,sections[sec],max_excess=50)
#kt.RVND(inbest=3,disp=2)
#sol.write('../partsol/tmpsol1_'+str(sec),only_clusts = kt.tourIDs)


#285 sections
for sec in range(215,285):
    print '---------------------------'
    print 'SECTION '+str(sec)
    print '---------------------------'
    sol = vrp.Solution('../solutions/sol_East_2000_993_12503567581.7.csv',gifts)
    kt = vrp.KTour(sol,sections[sec],max_excess=20.)
    best = kt.cost
    if kt.excess > 0:
        import pdb;pdb.set_trace()

    sol.write('../partsol/tmpsol1_'+str(sec),only_clusts = kt.tourIDs)
    for b in [1,3,5]:
        print 'restart with best '+str(b)
        sol = vrp.Solution('../solutions/sol_East_2000_993_12503567581.7.csv',gifts)
        kt = vrp.KTour(sol,sections[sec],max_excess=20.)
        kt.RVND(inbest=b,disp=2,tabu_tenure=3)
        if kt.excess < 1e-3 and kt.cost < best:
            kt.update_sol()
            best = kt.cost
            sol.write('../partsol/tmpsol1_'+str(sec),only_clusts = kt.tourIDs)





#TODO multiple restarts with other inbest? with clark_wright for init ?
#TODO update Neighborhood (involving long ?)
#TODO neg moves
#TODO in exchange bef/after dep on lat.
"""
