import random

from evrp.model import *
from evrp.operation import *
from evrp.evolution import *

random.seed(1)

#file = 'data/small_evrptw_instances/Cplex10er/c101C10.txt'
file = 'data/evrptw_instances/c101_21.txt'
model = Model(file, max_vehicle=100)
model.read_data()

evo = DEMA_Evolution(model)
S = evo.main()
print(S)
