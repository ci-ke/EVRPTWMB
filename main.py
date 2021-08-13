import random
import sys

from evrp.model import *
from evrp.operation import *
from evrp.evolution import *
from evrp.util import *


def run(input_list: list) -> None:
    file_path, file_type, icecube = Util.process_input(input_list)
    model = Model(file_path, file_type)
    if file_type in ['n', 's5', 's10', 's15']:
        model.read_data()
        # model.set_negative_demand(2)
    elif file_type in ['tw']:
        model.read_data_as_VRPTW()
    else:
        raise Exception('impossible')
    assert Operation.test_model(model)

    try:
        evo = DEMA(model, maxiter_evo=300, size=20)
        evo.main(icecube)
        evo.output_to_file()
        evo.freeze_evo()

    except BaseException as e:
        print(evo.S_best)
        evo.output_to_file('_ahead')
        evo.freeze_evo('_ahead')
        raise e


if __name__ == '__main__':
    # random.seed(2021)
    run(sys.argv)
    # python main.py s5 c101 n
    # python main.py s5 c101 c _ahead
