#example:
#
#srun -p opt -A traffic python greedy_clustering.py --output east_2000_975kg --start -10 --penalty 25 --withinnext 2000 &
#
import os.path
import sys
import sub_mlp as subMLT
import gflags
import pandas as pd

FLAGS = gflags.FLAGS
gflags.DEFINE_string('output',None,'name of the cluster-file where the greedy clusters are stored')
gflags.DEFINE_string('direction','east','east or west')
gflags.DEFINE_integer('start',-180,'longitude, in degree, where we start the tours [-180,180]')
gflags.DEFINE_integer('width',12,'angle width where we consider potential clusters')
gflags.DEFINE_integer('penalty',0,'max weight is 1000-penalty')
gflags.DEFINE_integer('withinnext',None,'next point to allocate to be selected within ... next points in the search direction')

gflags.MarkFlagAsRequired('output')
gflags.MarkFlagAsRequired('withinnext')

parse = FLAGS(sys.argv)  # parse flags

print FLAGS.output

north_pole = (90,0)
weight_limit = 1000#TMP 1000
sleigh_weight = 10


import pandas as pd
import numpy as np
from haversine import haversine,AVG_EARTH_RADIUS
from mpl_toolkits.basemap import Basemap
import Kmeans as km
import build_clusters as bc

gifts = pd.read_csv('../input/gifts.csv')
X = gifts[['Latitude','Longitude']].values
cb = bc.Cluster_Builder(X,[],None,gifts.Weight.values)
cb.greedy_for_bound(FLAGS.withinnext,direction=FLAGS.direction,width=FLAGS.width,wgpenalty=FLAGS.penalty,start=FLAGS.start)

import vrp
cl = vrp.Solution(cb.clusters,gifts)
cl.save_cluster(FLAGS.output)

print 'done !'




