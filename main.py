import random

from evrp.model import *
from evrp.operation import *
from evrp.evolution import *

random.seed(10)

file = 'data/small_evrptw_instances/Cplex10er/c101C10.txt'
#file = 'data/small_evrptw_instances/Cplex15er/c103C15.txt'
#file = 'data/evrptw_instances/c101_21.txt'

model = Model(file, max_vehicle=100)
model.read_data()
assert Operation.test_model(model)

evo = DEMA_Evolution(model)
evo.main()
evo.output_to_file()
