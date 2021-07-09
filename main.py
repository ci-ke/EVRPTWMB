import random

from evrp.model import *
from evrp.operation import *
from evrp.evolution import *

#random.seed(2021)

file = 'data/small_evrptw_instances/Cplex5er/c101C5.txt'
#file = 'data/small_evrptw_instances/Cplex10er/c101C10.txt'
#file = 'data/small_evrptw_instances/Cplex15er/c103C15.txt'
#file = 'data/evrptw_instances/c101_21.txt'

model = Model(file, max_vehicle=100)
model.read_data()
# model.set_negative_demand(2)
assert Operation.test_model(model)

evo = DEMA_Evolution(model)
evo.main()
evo.output_to_file()
