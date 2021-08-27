import random
import sys

from evrp.model import *
from evrp.operation import *
from evrp.evolution import *
from evrp.util import *


def run(input_list: list) -> None:
    file_path, file_type, icecube, negative_demand = Util.process_input(input_list)
    model = Model(file_path, file_type, negative_demand)
    model.read_data()
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
    # python main.py e/s5/s10/s15/tw/p c101 0(每几个设为负，0为不设置) n(new)
    # python main.py e/s5/s10/s15/tw/p c101 0(每几个设为负，0为不设置) c(continue) _ahead(文件名中evo后面的，扩展名前面的)
