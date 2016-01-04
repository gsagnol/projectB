#example:
#
#python vrp_sections.py --filename ../solutions/Sol1_after_vrp5_12491931420.4.csv --solution tmp2 --restart 5 --nsec 5  --offset -180 --processes 50 --slurm
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
gflags.DEFINE_integer('processes',4,'number of processes used')
gflags.DEFINE_integer('nsec',5,'number of processes used')
gflags.DEFINE_integer('offset',-180,'starting angle')
gflags.DEFINE_bool('slurm',False,'if True, run on slurm with srun command')


gflags.MarkFlagAsRequired('solution')
gflags.MarkFlagAsRequired('filename')
gflags.MarkFlagAsRequired('processes')


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


n = len(sections)
p = FLAGS.processes

for i in range(p):
    a,b =  i*(n//p+1),min((i+1)*(n//p+1),n)
    if a>=b:
        break
    print '**** opt on {0}--{1} ****'.format(a,b)
    command = 'python vrp_one_sec.py'
    if FLAGS.slurm:
        command = 'srun -p opt -A traffic ' + command
    command += ' --filename '+FLAGS.filename
    command += ' --solution '+ FLAGS.solution+'_section_'.format(a,b)
    command += ' --sec1 '+str(a)
    command += ' --sec2 '+str(b)
    command += ' --restart '+ str(FLAGS.restart)
    command += ' --nsec '+ str(FLAGS.nsec)
    command += ' --offset '+ str(FLAGS.offset)
    
    print 'running command:'
    print command

    '''
    com = shlex.split(command)
    print com
    #os.system(command)
    subprocess.Popen(com)
    print 'OK, it runs'
    '''

