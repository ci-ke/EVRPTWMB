import random
import sys

from evrp.model import *
from evrp.operation import *
from evrp.evolution import *
from evrp.util import *


def run(input_list: list) -> None:
    filepath, icecube = Util.process_input(input_list)
    model = Model(filepath)
    model.read_data()
    # model.set_negative_demand(2)
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
