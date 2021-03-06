# Copyright 2010-2014 Google
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Traveling Salesman Sample.

   This is a sample using the routing library python wrapper to solve a
   Traveling Salesman Problem.
   The description of the problem can be found here:
   http://en.wikipedia.org/wiki/Travelling_salesman_problem.
   The optimization engine uses local search to improve solutions, first
   solutions being generated using a cheapest addition heuristic.
   Optionally one can randomly forbid a set of random connections between nodes
   (forbidden arcs).
"""

#python tsp.py --tsp_size=75 --tsp_use_random_matrix=0 (well, for now this is also random...)

import random

from google.apputils import app
import gflags
from ortools.constraint_solver import pywrapcp
import numpy as np

FLAGS = gflags.FLAGS

gflags.DEFINE_integer('tsp_size', 10,
                     'Size of Traveling Salesman Problem instance.')
gflags.DEFINE_boolean('tsp_use_random_matrix', True,
                     'Use random cost matrix.')
gflags.DEFINE_integer('tsp_random_forbidden_connections', 0,
                     'Number of random forbidden connections.')
gflags.DEFINE_integer('tsp_random_seed', 0, 'Random seed.')
gflags.DEFINE_boolean('light_propagation', False, 'Use light propagation')


# Cost/distance functions.
"""
for some reason this does not work, but with class Mat2 it works
"""
np.random.seed(42)
x = 200*np.random.randn(FLAGS.tsp_size,2)
x[0]=[1000,0]

def Distance(i, j):
  """Sample function."""
  # Put your distance code here.
  return int(np.linalg.norm(x[i]-x[j]))

class RandomMatrix(object):
  """Random matrix."""

  def __init__(self, size):
    """Initialize random matrix."""

    rand = random.Random()
    rand.seed(FLAGS.tsp_random_seed)
    distance_max = 100
    self.matrix = {}
    for from_node in xrange(size):
      self.matrix[from_node] = {}
      for to_node in xrange(size):
        if from_node == to_node:
          self.matrix[from_node][to_node] = 0
        else:
          self.matrix[from_node][to_node] = rand.randrange(distance_max)

  def Distance(self, from_node, to_node):
    return self.matrix[from_node][to_node]


class RandomMatrix2(object):
  """Random matrix."""

  def __init__(self, size):
    """Initialize random matrix."""

    rand = random.Random()
    rand.seed(42)
    x = 200*np.random.randn(size,2)
    x[0]=[1000.,0.]
    
       
    self.matrix = {}
    for from_node in xrange(size):
      self.matrix[from_node] = {}
      for to_node in xrange(size):
        if from_node == to_node:
          self.matrix[from_node][to_node] = 0
        else:
          self.matrix[from_node][to_node] = int(np.linalg.norm(x[from_node]-x[to_node]))
        
  def Distance(self, from_node, to_node):
    return self.matrix[from_node][to_node]


def main(_):
  # Create routing model
  if FLAGS.tsp_size > 0:
    # Set a global parameter.
    param = pywrapcp.RoutingParameters()
    param.use_light_propagation = FLAGS.light_propagation
    pywrapcp.RoutingModel.SetGlobalParameters(param)

    ## TSP of size FLAGS.tsp_size
    # Second argument = 1 to build a single tour (it's a TSP).
    # Nodes are indexed from 0 to FLAGS_tsp_size - 1, by default the start of
    # the route is node 0.
    routing = pywrapcp.RoutingModel(FLAGS.tsp_size, 1)

    parameters = pywrapcp.RoutingSearchParameters()
    # Setting first solution heuristic (cheapest addition).
    parameters.first_solution = 'PathCheapestArc'
    # Disabling Large Neighborhood Search, comment out to activate it.
    parameters.no_lns = True
    parameters.no_tsp = False

    # Setting the cost function.
    # Put a callback to the distance accessor here. The callback takes two
    # arguments (the from and to node inidices) and returns the distance between
    # these nodes.
    #...
    if FLAGS.tsp_use_random_matrix:
      matrix = RandomMatrix(FLAGS.tsp_size)
      matrix_callback = matrix.Distance
      routing.SetArcCostEvaluatorOfAllVehicles(matrix_callback)
    else:
      matrix = RandomMatrix2(FLAGS.tsp_size)
      matrix_callback = matrix.Distance
      routing.SetArcCostEvaluatorOfAllVehicles(matrix_callback)
    # Forbid node connections (randomly).
    rand = random.Random()
    rand.seed(FLAGS.tsp_random_seed)
    forbidden_connections = 0
    while forbidden_connections < FLAGS.tsp_random_forbidden_connections:
      from_node = rand.randrange(FLAGS.tsp_size - 1)
      to_node = rand.randrange(FLAGS.tsp_size - 1) + 1
      if routing.NextVar(from_node).Contains(to_node):
        print 'Forbidding connection ' + str(from_node) + ' -> ' + str(to_node)
        routing.NextVar(from_node).RemoveValue(to_node)
        forbidden_connections += 1

    # Solve, returns a solution if any.
    assignment = routing.SolveWithParameters(parameters, None)
    if assignment:
      # Solution cost.
      print assignment.ObjectiveValue()
      # Inspect solution.
      # Only one route here; otherwise iterate from 0 to routing.vehicles() - 1
      route_number = 0
      node = routing.Start(route_number)
      route = ''
      while not routing.IsEnd(node):
        route += str(node) + ' -> '
        node = assignment.Value(routing.NextVar(node))
      route += '0'
      print route
    else:
      print 'No solution found.'
  else:
    print 'Specify an instance greater than 0.'

if __name__ == '__main__':
  app.run()
