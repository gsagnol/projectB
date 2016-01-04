#example:
#
#python vrp_one_sec.py --filename ../solutions/Sol1_after_vrp5_12491931420.4.csv --solution tmp2 --restart 5 --nsec 5  --offset -180 --sec1 0 --sec2 1
#
import os.path
import sys
import sub_mlp as subMLT
import gflags
import pandas as pd
import shlex,subprocess

FLAGS = gflags.FLAGS
gflags.DEFINE_string('filename',None,'name of the cluster-file where the tours to optimize are stored')
gflags.DEFINE_string('solution',None,'name of the solution-file template where the optimal tours')
gflags.DEFINE_integer('restart',4,'number of restart for the optimize heuristic')
gflags.DEFINE_integer('sec1',0,'first sec')
gflags.DEFINE_integer('sec2',0,'last sec')
gflags.DEFINE_integer('nsec',0,'number tours per sect')
gflags.DEFINE_integer('offset',-180,'starting angle')

gflags.MarkFlagAsRequired('solution')
gflags.MarkFlagAsRequired('filename')
gflags.MarkFlagAsRequired('sec1')
gflags.MarkFlagAsRequired('sec2')
gflags.MarkFlagAsRequired('nsec')

parse = FLAGS(sys.argv)  # parse flags

print FLAGS.filename

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

sol = vrp.Solution('../solutions/'+FLAGS.filename,gifts)


#sections sorted per longitude
offset = FLAGS.offset
sorlongs = [c for lo,c in sorted([(lo if lo>offset else lo+360.,c) for c,lo in sol.lomeans.iteritems()])]
nsec = FLAGS.nsec
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
    
bb = range(1,FLAGS.restart+1)
    

for sec in range(FLAGS.sec1,FLAGS.sec2):
    print '---------------------------'
    print 'SECTION '+str(sec)
    print '---------------------------'
    sol = vrp.Solution('../solutions/'+FLAGS.filename,gifts)
    kt = vrp.KTour(sol,sections[sec],max_excess=20.)
    best0 = kt.cost
    best = kt.cost
    if kt.excess > 0:
        import pdb;pdb.set_trace()

    sol.write('../partsol/'+FLAGS.solution+'_'+str(sec)+'.csv',only_clusts = kt.tourIDs)
    for b in bb:
        print 'restart with best '+str(b)
        sol = vrp.Solution('../solutions/'+FLAGS.filename,gifts)
        kt = vrp.KTour(sol,sections[sec],max_excess=20.)
        kt.RVND(inbest=b,disp=2,tabu_tenure=3)
        if kt.excess < 1e-3 and kt.cost < best:
            kt.update_sol()
            best = kt.cost
            print 'opt '+str(best/best0)
            sol.write('../partsol/'+FLAGS.solution+'_'+str(sec)+'.csv',only_clusts = kt.tourIDs)
            
print 'OK,done'
print '-------'