import random

from evrp.model import *
from evrp.operation import *
from evrp.evolution import *

random.seed(2021)

file5 = 'data/small_evrptw_instances/Cplex5er/c101C5.txt'
file10 = 'data/small_evrptw_instances/Cplex10er/c101C10.txt'
file15 = 'data/small_evrptw_instances/Cplex15er/rc103C15.txt'
file100 = 'data/evrptw_instances/c101_21.txt'

model = Model(file15)
model.read_data()
# model.set_negative_demand(2)
assert Operation.test_model(model)

try:
    evo = DEMA(model)
    evo.main()
    Operation.output_to_file(model, evo.S_best)

except KeyboardInterrupt:
    print(evo.S_best)
    Operation.output_to_file(model, evo.S_best, '_ahead')
