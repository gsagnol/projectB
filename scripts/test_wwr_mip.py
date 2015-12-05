import numpy as np
import scipy.spatial.distance as dist
import networkx as nx
import picos as pic

n = 75

np.random.seed(42)
w = [10]+[np.random.randint(50) for i in range(n)]

x = np.random.randn(n+1,2)
x[0]=[10,0]
d = dist.cdist(x,x)
d = np.c_[d,d[:,0]]
d = np.r_[d,np.atleast_2d(d[0,:])]

G = nx.complete_graph(n)
G = nx.relabel_nodes(G,{i:i+1 for i in range(n)})
G = nx.DiGraph(G)
G.add_edges_from([(0,i) for i in range(1,n+1)])
G.add_edges_from([(i,n+1) for i in range(1,n+1)])


P = pic.Problem()
f = {e:P.add_variable('f['+str(e)+']',1) for e in G.edges()}
P.add_constraint(pic.tools.flow_Constraint(G,f,0,range(1,n+2),w[1:]+[w[0]],None,'G'))
P.set_objective('min',pic.sum([f[i,j]*d[i,j] for i,j in G.edges()],'e'))
P._make_cplex_instance()

c = P.cplex_Instance
c.SOS.add(type='1',SOS=[['f[(0, {0})]_0'.format(i) for i in range(1,n+1)],range(1,n+1)])
for i in range(1,n+1):
    c.SOS.add(type='1',SOS=[['f[({0}, {1})]_0'.format(i,j) for j in range(1,n+2) if j!=i],range(1,n+1)])
    
c.solve()

nam = c.variables.get_names()
ff = c.solution.get_values()
flow = [nam[i].split('(')[1].split(')')[0] for i,fi in enumerate(ff) if fi>0]
successor = {int(f.split(',')[0]):int(f.split(',')[1]) for f in flow}
tour = [0]
nxt=0
for i in range(1,n+1):
    nxt = successor[nxt]
    tour.append(nxt)
print tour