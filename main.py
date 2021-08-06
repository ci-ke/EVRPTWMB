import random
import sys
import traceback

from evrp.model import *
from evrp.operation import *
from evrp.evolution import *


def process_input(in_str: list):
    file_type = in_str[1]  # n s5 s10 s15
    map_name = in_str[2]  # c101 r201
    mode = in_str[3]  # n(new) c(continue)
    if len(in_str) == 5:
        if in_str[4] == '.':
            suffix = ''
        else:
            suffix = in_str[4]  # _ahead

    if file_type == 's5':
        folder = 'data/small_evrptw_instances/Cplex5er/'
        filename = map_name+'C5.txt'
    elif file_type == 's10':
        folder = 'data/small_evrptw_instances/Cplex10er/'
        filename = map_name+'C10.txt'
    elif file_type == 's15':
        folder = 'data/small_evrptw_instances/Cplex15er/'
        filename = map_name+'C15.txt'
    elif file_type == 'n':
        folder = 'data/evrptw_instances/'
        filename = map_name+'_21.txt'

    filepath = folder+filename

    if mode == 'n':
        icecube = None
    elif mode == 'c':
        icecube = pickle.load(open('result/{}_evo{}.pickle'.format(filename.split('.')[0], suffix), 'rb'))

    return filepath, icecube


def run():
    filepath, icecube = process_input(sys.argv)
    model = Model(filepath)
    model.read_data()
    model.set_negative_demand(2)
    assert Operation.test_model(model)

    try:
        evo = DEMA(model, maxiter_evo=300, size=20)
        evo.main(icecube)
        Operation.output_to_file(model, evo.S_best)
        Operation.freeze_evo(evo, model)

    except:
        traceback.print_exc()
        print(evo.S_best)
        Operation.output_to_file(model, evo.S_best, '_ahead')
        Operation.freeze_evo(evo, model, '_ahead')


if __name__ == '__main__':
    # random.seed(2021)
    run()
    # python main.py s5 c101 n
    # python main.py s5 c101 c _ahead
