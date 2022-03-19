from Model import *
from util import *
from operation import *
from evolu import *
for ds in range(1, 11):
    for maindir, subdir, file_name_list in os.walk('data'):
        if maindir == 'data\evrptw_instances':
            for file in file_name_list:
                if file[-10] =='2':
                    lj = os.path.join(maindir, file)
                    model = Util.set_model(['e', lj], 2)
                    evo = DEMA1(model, '')
                    evo.main()
        if maindir == 'data\solomon':
            for file in file_name_list:
                if file[-7] =='2':
                    lj = os.path.join(maindir, file)
                    model = Util.set_model(['tw', lj])
                    evo = DEMA1(model, '')
                    evo.main()
        if maindir == 'data\p':
            for file in file_name_list:
                lj = os.path.join(maindir, file)
                for i in range(3):
                    if i == 0:
                        ne = 2
                    elif i == 1:
                        ne = 4
                    elif i == 2:
                        ne = 10
                    model = Util.set_model(['p', lj], ne)
                    evo = DEMA1(model, '')
                    evo.main()
        if maindir == 'data\jd':
            model = Util.set_model(['jd', ''], 2)
            evo = DEMA1(model, '')
            evo.main()




