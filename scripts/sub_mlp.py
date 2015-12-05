north_pole = (90,0)
weight_limit = 1000
sleigh_weight = 10


import pandas as pd
import numpy as np
from haversine import haversine

class MLT:
    """
    A class for a sub minimum latency tour problem
    """
    def __init__(self,gifts,ids):

        self.positions = [north_pole]
        self.weights = [10.]

        for i in ids:
            gift = gifts.loc[i-1]
            pos = tuple(gift[['Latitude','Longitude']])
            wgt = gift.Weight
            self.weights.append(wgt)
            self.positions.append(pos)

        self.n = len(self.positions)

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

    def construction(self,alpha):
        current_node = 0
        candidates = range(1,self.n)
        tour = []
        while True:
            dist_from_current = [(self.distances[current_node,i],i) for i in candidates]
            sorted_from_current = np.array(sorted(dist_from_current))
            max_index = np.searchsorted(sorted_from_current[:,0],(1.+alpha)*sorted_from_current[0,0])
            next_node = int(sorted_from_current[np.random.randint(max_index)][1])
            candidates.remove(next_node)
            tour.append(next_node)
            current_node = next_node
            if len(candidates)==1:
                break
        return tour+candidates




gifts = pd.read_csv('../input/gifts.csv')
center = (27.,60.)
dist_to_center = lambda x: haversine(tuple(x),center)
dists = np.apply_along_axis(dist_to_center,1,gifts[['Latitude','Longitude']].values)
close_gifts = sorted([(d,gft_id_minus_1) for gft_id_minus_1,d in enumerate(dists)])[:600]
wgt = 0.
sub_gift_ids = []
for d,gft_id_minus_1 in close_gifts:
    wgt += gifts.loc[gft_id_minus_1].Weight
    if wgt<1000:
        sub_gift_ids.append(gft_id_minus_1+1)
    else:
        break

mlt = MLT(gifts,sub_gift_ids)
mlt.eval_ordering(range(1,mlt.n))
sorted_weight_tour = mlt.sorted_weights_heuristics()
mlt.eval_ordering(sorted_weight_tour)
mlt.eval_ordering([1+i for i in np.random.permutation(mlt.n-1)])

constructed_tour = mlt.construction(1.05)
mlt.eval_ordering(constructed_tour)
