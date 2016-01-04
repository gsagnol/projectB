#usage: python merge_sol.py template 
#
# This concatenate all files of the directory solutions starting with template
# in one big solution file, called template_merged.csv

import sys
import os

template = sys.argv[1]
out = template+'_merged.csv'
to_merge = os.popen('ls -a '+template+'*.csv').read().split()

first = to_merge.pop()
os.system('cp '+first+' ../solutions/'+out)

output = open('../solutions/'+out,'a')
for filename in to_merge:
    f = open(filename,'r')
    lines = f.readlines()
    f.close()
    output.writelines(lines[1:])

output.close()


nn = int(os.popen('wc -l ../solutions/'+out).read().split()[0])

if nn==100001:
    print 'merge worked'
    for fn in [first]+to_merge:
        os.system('rm '+fn)
else:
    print 'Error: The merge file didnt have 100000 gifts'