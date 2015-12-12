#example:
#
#python opt_each_tour.py --filename greedy_tkmeans_20_20_1438 --solution sol_greedy_tkmeans_20_20_1438 --restart 8 --processes 4 --slurm
#
import os.path
import sys
import sub_mlp as subMLT
import gflags
import pandas as pd
import shlex,subprocess

FLAGS = gflags.FLAGS
gflags.DEFINE_string('filename',None,'name of the cluster-file where the tours to optimize are stored')
gflags.DEFINE_string('solution_file',None,'name of the solution-file template where the optimal tours will be written -- one output for each process (order will be change in cluster file as well)')
gflags.DEFINE_bool('quickopt',False,'if True, Do the optimization using quickopt')
gflags.DEFINE_integer('verbose',1,'verbosity level')
gflags.DEFINE_integer('restart',4,'number of restart for the optimize heuristic')
gflags.DEFINE_integer('processes',4,'number of processes used')
gflags.DEFINE_bool('slurm',False,'if True, run on slurm with srun command')


gflags.MarkFlagAsRequired('solution_file')
gflags.MarkFlagAsRequired('filename')
gflags.MarkFlagAsRequired('processes')


parse = FLAGS(sys.argv)  # parse flags

print FLAGS.filename

f = open('../clusters/'+FLAGS.filename,'r')
clusters = eval(f.read())
f.close()

n = len(clusters)
#TMP
#n=14 #uncomment for testing purpose
p = FLAGS.processes

for i in range(p):
    a,b =  i*(n//p+1),min((i+1)*(n//p+1),n)
    print '**** opt on {0}--{1} ****'.format(a,b)
    command = 'python optimize_tours.py'
    if FLAGS.slurm:
        command = 'srun -p opt -A traffic ' + command
    command += ' --filename '+FLAGS.filename
    command += ' --solution_file '+ FLAGS.solution_file+'_tours_{0}_{1}.csv'.format(a,b)
    command += ' --tours '+str(a)+','+str(b)
    command += ' --range'
    command += ' --verbose '+ str(FLAGS.verbose)
    command += ' --restart '+ str(FLAGS.restart)
    if FLAGS.quickopt:
        command += ' --quickopt'

    print 'running command:'
    print command

    com = shlex.split(command)
    print com
    #os.system(command)
    subprocess.Popen(com)
    print 'OK, it runs'

