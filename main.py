import random
import sys

from evrp.model import *
from evrp.operation import *
from evrp.evolution import *
from evrp.util import *


def run(input_list: list) -> None:
    file_path, file_type, icecube, negative_demand, save_suffix, control = Util.process_input(input_list)
    model = Model(file_path, file_type, negative_demand)
    model.read_data()
    assert Operation.test_model(model)

    try:
        evo = DEMA(model, maxiter_evo=300, size=20)
        evo.main(control, icecube)
        evo.output_to_file(save_suffix)
        evo.freeze_evo(save_suffix)

    except BaseException as e:
        print(evo.S_best)
        evo.output_to_file('{}_ahead'.format(save_suffix))
        evo.freeze_evo('{}_ahead'.format(save_suffix))
        raise e


if __name__ == '__main__':
    # random.seed(2021)
    run(sys.argv)
