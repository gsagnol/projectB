#example:
#
#srun -p opt -A traffic python optimize_tours.py --filename greedy_tkmeans_100_50_1480 --solution sol1.csv --tours 1,100 --range
#
import os.path
import sys
import sub_mlp as subMLT
import gflags
import pandas as pd

FLAGS = gflags.FLAGS
gflags.DEFINE_string('filename',None,'name of the cluster-file where the tours to optimize are stored')
gflags.DEFINE_string('solution_file',None,'name of the solution-file (with.csv) where the optimal tours will be written (order will be change in cluster file as well)')
gflags.DEFINE_list('tours',None,'ids of the tours to optimize')
gflags.DEFINE_bool('range',False,'if True, compute all tours from min(tours) to (included) max(tours)')
gflags.DEFINE_bool('quickopt',False,'if True, Do the optimization using quickopt')
gflags.DEFINE_integer('verbose',1,'verbosity level')
gflags.DEFINE_integer('restart',4,'number of restart for the optimize heuristic')


gflags.MarkFlagAsRequired('solution_file')
gflags.MarkFlagAsRequired('filename')
gflags.MarkFlagAsRequired('tours')

gflags.RegisterValidator('tours',
                         lambda value: all([v.isdigit() for v in value]),
                         message='Flag --tours must be a list of comma separated ints')

gflags.RegisterValidator('solution_file',
                         lambda value: value.endswith('.csv'),
                         message='Flag --solution file must have .csv extension')       

parse = FLAGS(sys.argv)  # parse flags

print FLAGS.filename

f = open('../clusters/'+FLAGS.filename,'r')
clusters = eval(f.read())
f.close()

tours = [int(v) for v in FLAGS.tours]
if range:
    tours = range(min(tours),max(tours)+1)

gifts = pd.read_csv('../input/gifts.csv')
for c in tours:
    print 'Optimizing cluster #'+str(c)
    mlt = subMLT.MLT(gifts,clusters[c])
    if FLAGS.quickopt:
        tr = mlt.quick_opt()
    else:
        tr = mlt.optimize(restart=FLAGS.restart,disp=FLAGS.verbose)
    print 'Found a tour of weighted latency {0}'.format(tr.wlatency)
    new_order = [clusters[c][i-1] for i in tr.order]
    clusters[c] = new_order
    f = open('../clusters/'+FLAGS.filename,'w')
    f.write(str(clusters))
    f.close()


print '---------------'
print 'Optimization done- storing the results in '+'../solutions/'+FLAGS.solution_file
print '---------------'


solfile = '../solutions/'+FLAGS.solution_file
if os.path.isfile(solfile):
    new_sol = pd.read_csv(solfile)
else:
    tmp = open(solfile,'w')
    tmp.write('GiftId,TripId\n')
    tmp.close()
    new_sol = pd.read_csv(solfile)

for c in tours:
    new_sol = new_sol.drop(new_sol[new_sol.TripId==c].index)

import copy
sol0 = copy.deepcopy(new_sol)

for c in tours:
    try:
        ind = max(new_sol.index)+1
    except ValueError:
        ind = 0
    for gft in clusters[c]:
        new_sol.loc[ind] = [int(gft),int(c)]
        ind += 1

def convert_to_integers(x):
    try:
        return x.astype(int)
    except:
        return x

if len(new_sol)-len(sol0) == sum([len(clusters[c]) for c in tours]):
    print 'length OK'
else:
    import pdb;pdb.set_trace()

new_sol.apply(convert_to_integers).to_csv('../solutions/'+FLAGS.solution_file,index=False)

print 'done !'




