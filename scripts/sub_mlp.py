north_pole = (90,0)
weight_limit = 100#TMP 1000
sleigh_weight = 10


import pandas as pd
import numpy as np
from haversine import haversine

class MLT:
    """
    A class for a sub minimum latency tour problem
    #TODOPT smarter load  of ids with pandas
    """
    def __init__(self,gifts,ids):

        self.positions = [north_pole]
        self.weights = [sleigh_weight]

        for i in ids:
            gift = gifts.loc[i-1]
            pos = tuple(gift[['Latitude','Longitude']])
            wgt = gift.Weight
            self.weights.append(wgt)
            self.positions.append(pos)

        """
        the number of gifts + the north pole
        """
        self.n = len(self.positions)
        assert(sum(self.weights) < weight_limit + sleigh_weight)

        d = np.zeros((self.n,self.n))
        for i,x in enumerate(self.positions):
            for j,y in enumerate(self.positions):
                if i < j:
                    continue
                elif i == j:
                    d[i,j] = 0.
                else:
                    d[i,j] = haversine(x,y)
                    d[j,i] = d[i,j]
        self.distances = d


    """
    this method is now OBSOLETE with class attribute latency,wlatency of class Tour
    """
    def eval_ordering(self,order):
        """
        order[i] = j means that gift#j (with j>=1) is visited in ith position
        returns weighted_latency,latency (length of the tour)
        """
        latency = 0.
        weighted_latency = 0.
        current_node = 0

        for i,j in enumerate(order):
            latency += self.distances[current_node,j]
            weighted_latency += latency * mlt.weights[j]
            current_node = j

        latency += self.distances[current_node,0]
        weighted_latency += latency * mlt.weights[0]

        return weighted_latency,latency

    def sorted_weights_heuristics(self):
        sorted_weights = sorted([(wgt,id+1) for id,wgt in enumerate(self.weights[1:])],reverse=True)
        return [id for wgt,id in sorted_weights]

    def construction(self,alpha=0.2,beta=0.1):
        """
        construct an initial tour, with greediness parameter alpha, and
        param beta to favor large weights first
        """
        current_node = 0
        candidates = range(1,self.n)
        tour = []
        while True:
            dist_from_current = [(self.distances[current_node,i]/(self.weights[i]**beta),i) for i in candidates]
            sorted_from_current = np.array(sorted(dist_from_current))
            max_index = np.searchsorted(sorted_from_current[:,0],(1.+alpha)*sorted_from_current[0,0])
            next_node = int(sorted_from_current[np.random.randint(max_index)][1])
            candidates.remove(next_node)
            tour.append(next_node)
            current_node = next_node
            if len(candidates)==1:
                break
        return Tour(self,tour+candidates)


class Tour:
    """
    A tour for the MLT problem, startind and ending at north pole
    """
    def __init__(self,mlt,ordered_gifts,latency = None,wlatency = None):
        """
        The constructor takes a MLT problem and a list of ordered gift IDs.
        """
        assert(len(ordered_gifts)==mlt.n-1)

        self.mlt = mlt
        self.order = ordered_gifts
        if latency is None or wlatency is None:
            self.compute_latencies()
        else:
            self.latency = latency
            self.wlatency = wlatency

        #subsequence not computed yet.
        self.subseq = None

    def compute_latencies(self):
        latency = 0.
        weighted_latency = 0.
        current_node = 0

        for j in self.order:
            latency += self.mlt.distances[current_node,j]
            weighted_latency += latency * self.mlt.weights[j]
            current_node = j

        latency += self.mlt.distances[current_node,0]
        weighted_latency += latency * self.mlt.weights[0]

        self.wlatency = weighted_latency
        self.latency = latency

    def __repr__(self):
        return '<Tour of weighted latency '+str(self.wlatency)+' >'

    def preprocess_subsequences(self):
        """
        self.subseq[i,j] is the subseq with nodes order[i],,,.order[j-1]
        """
        self.subseq = {}
        for i in range(self.mlt.n):
            for j in range(i+1,self.mlt.n):
                if j == i+1:
                    self.subseq[i,j] = Subsequence(self.mlt,[self.order[i]],0.,0.)
                else:
                    self.subseq[i,j] = self.subseq[i,j-1] + Subsequence(self.mlt,[self.order[j-1]],0.,0.)

    def swap(self,i,j):
        """
        swaps self.order[i] with self.order[j]
        """
        assert(i < j)
        assert(i >= 0)
        assert(j < t.mlt.n-1)
        assert(self.subseq is not None)
        """useless !?
        oi = self.order[i]
        oj = self.order[j]
        order = self.order[:]
        order[i] = oj
        order[j] = oi
        """
        if i == 0:
            seq = self.subseq[j,j+1]
        else:
            seq = self.subseq[0,i] + self.subseq[j,j+1]
        if j>i+1:
            seq = seq + self.subseq[i+1,j]
        seq = seq + self.subseq[i,i+1]
        if j < self.mlt.n-2:
            seq = seq + self.subseq[j+1,t.mlt.n-1]
        return seq

    def reinsert(self,i,j,k=1):
        """
        (self.order[i],...,self.order[i+k-1]) are reinserted in the seq, so that new pos of i is j
        """
        assert(self.subseq is not None)
        assert(i >= 0)
        assert(i+k-1 < t.mlt.n-1)
        assert(j >= 0)
        assert(j+k-1 < t.mlt.n-1)
        assert(i!=j)
        if i < j :
            seq = self.subseq[i,i+k]
            if j-1 > i:
                seq = self.subseq[i+k,j-1+k] + seq
            if i > 0:
                seq = self.subseq[0,i] + seq
            if j+k < self.mlt.n-1:
                seq = seq + self.subseq[j+k-1,t.mlt.n-1]
        else:
            pass#TODO !!!
        return seq

class Subsequence:
    """
    A subsequence of a tour
    """
    def __init__(self,mlt,order,latency=None,wlatency=None):
        self.mlt = mlt
        self.order = order
        if latency is None or wlatency is None:
            self.compute_latencies()
        else:
            self.latency = latency
            self.wlatency = wlatency

    def __repr__(self):
        return '<Subsequence with nodes '+str(self.order)+' >'

    def compute_latencies(self):
        latency = 0.
        weighted_latency = 0.
        current_node = self.order[0]

        for j in self.order[1:]:
            latency += self.mlt.distances[current_node,j]
            weighted_latency += latency * self.mlt.weights[j]
            current_node = j

        self.wlatency = weighted_latency
        self.latency = latency

    def __add__(self,next_sub):
        """
        concatenation of two subsequences
        """
        u = self.order[-1]
        v = next_sub.order[0]
        init_distance = self.latency + self.mlt.distances[u,v]
        end_weight = sum([self.mlt.weights[i] for i in next_sub.order])
        lat = init_distance + next_sub.latency
        wlat = self.wlatency + next_sub.wlatency + init_distance * end_weight
        return Subsequence(self.mlt,self.order+next_sub.order,lat,wlat)

    def to_tour(self):
        """
        add the north pole at the beginning and at the end of the subsequence
        """
        d1 = self.mlt.distances[0,self.order[0]]
        d2 = self.mlt.distances[0,self.order[-1]]
        lat = self.latency + d1 + d2
        wlat = self.wlatency + d1 * sum(self.mlt.weights) + (self.latency + d2) * self.mlt.weights[0]
        return Tour(self.mlt,self.order,lat,wlat)





#----------------------#
#construct an instance #
#----------------------#
gifts = pd.read_csv('../input/gifts.csv')
center = (27.,60.)
dist_to_center = lambda x: haversine(tuple(x),center)
dists = np.apply_along_axis(dist_to_center,1,gifts[['Latitude','Longitude']].values)
close_gifts = sorted([(d,gft_id_minus_1) for gft_id_minus_1,d in enumerate(dists)])[:600]
wgt = 0.
sub_gift_ids = []
for d,gft_id_minus_1 in close_gifts:
    wgt += gifts.loc[gft_id_minus_1].Weight
    if wgt < weight_limit:
        sub_gift_ids.append(gft_id_minus_1+1)
    else:
        break

#--------------------------#
#eval some (stupid) orders #
#--------------------------#
mlt = MLT(gifts,sub_gift_ids)
mlt.eval_ordering(range(1,mlt.n))
sorted_weight_tour = mlt.sorted_weights_heuristics()
mlt.eval_ordering(sorted_weight_tour)
mlt.eval_ordering([1+i for i in np.random.permutation(mlt.n-1)])

#--------------------------#
# eval one heuristic order #
#--------------------------#
constructed_tour = mlt.construction(0.15)
mlt.eval_ordering(constructed_tour.order)
constructed_tour.wlatency

#----------------------------------------------#
# Check computation with tour and subsequences #
#----------------------------------------------#

seq1 = constructed_tour.order[:3]
seq2 = constructed_tour.order[3:]
seq1 = Subsequence(mlt,seq1)
seq2 = Subsequence(mlt,seq2)
(seq1 + seq2).to_tour()

#------------#
# test swap  #
#------------#
t = constructed_tour
t.preprocess_subsequences()
sw = t.swap(2,5)
sw.to_tour().wlatency
Tour(mlt,sw.order)

#----------------#
# test reinsert  #
#----------------#
t = constructed_tour
t.preprocess_subsequences()
t.order
t.reinsert(2,6)#there is a bug ... TODO correct
