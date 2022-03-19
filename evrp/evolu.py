import os
import pickle
import collections
import math

from Model import *
from util import *
from operation import Modification,Operation
import pandas as pd

class Evolution(metaclass=ABCMeta):
    result=None

    def __init__(self):
        if not os.path.exists('result.pkl'):
            result=[]
            for maindir,subdir,file_name_list in os.walk('data'):
                if maindir=='data\solomon':
                    for file in file_name_list:
                        if file[-7]=='2':
                            result.append(file[-9:-4].upper())
                if maindir=='data\p':
                    for file in file_name_list:
                        result.extend([file[-7:-4]+'_02',file[-7:-4]+'_04',file[-7:-4]+'_10'])
                if maindir=='data\jd':
                    result.append('jd')
                if maindir=='data\evrptw_instances':
                    for file in file_name_list:
                        if file[-10]=='2':
                            result.append('E'+file[-12:-7].upper()+'L')
            col=['BKS_obj1','BKS_obj2','S_best_obj1','S_best_obj2','S_best','count']
            col.extend(['last'+str(i) for i in range(1,11)])
            self.result=pd.DataFrame(index=result,columns=col)
            self.result['count']=0
            self.result.to_pickle('result.csv')
        else:
            self.result=pd.read_pickle('result.pkl')

    def output_to_file(self, suffix: str = '') -> None:
        if not os.path.exists('result'):
            os.mkdir('result')
        if not os.path.exists('result/'+self.model.file_type):
            os.mkdir('result/'+self.model.file_type)
        filename = self.model.data_file.split('/')[-1].split('.')[0]
        output_file = open('result/{}/{}{}{}.txt'.format(self.model.file_type, filename, '' if self.model.negative_demand == 0 else '_neg'+str(self.model.negative_demand), suffix), 'a')
        output_file.write(str(self.S_best)+'\n'+str(self.S_best.sum_distance())+'\n'+str(self.S_best.feasible_detail(self.model))+'\n\n')
        output_file.close()

    def freeze_evo(self, suffix: str = '') -> None:
        if not os.path.exists('result'):
            os.mkdir('result')
        if not os.path.exists('result/'+self.model.file_type):
            os.mkdir('result/'+self.model.file_type)
        filename = self.model.data_file.split('/')[-1].split('.')[0]

        num = 1
        base_pickle_filepath = 'result/{}/{}{}_evo{}.pickle'.format(self.model.file_type, filename, '' if self.model.negative_demand == 0 else '_neg'+str(self.model.negative_demand), suffix)
        pickle_filepath = base_pickle_filepath
        while os.path.exists(pickle_filepath):
            pickle_filepath = base_pickle_filepath[:-7]+str(num)+base_pickle_filepath[-7:]
            num += 1

        pickle_file = open(pickle_filepath, 'wb')
        pickle.dump(self.freeze(), pickle_file)
        pickle_file.close()

    def update_result(self):
        solution=self.S_best.copy()
        assert isinstance(solution,Solution)
        if solution.feasible(self.model):
            if self.model.file_type == 'tw':
                if self.model.data_file[-9]!='r':
                    file=self.model.data_file[-8:-4]
                else:
                    file = self.model.data_file[-9:-4]
                file=file.upper()
                s=self.result['S_best'][file]
                s=None if pd.isnull(s) else s
                if self.compare_better(solution,s):
                    self.result['S_best'][file]=solution.copy()
                    self.result['S_best_obj1'][file]=len(solution)
                    self.result['S_best_obj2'][file] =solution.sum_distance()
                self.result['count'][file]+=1
                cs=self.result['count'][file]%10
                if cs!=0:
                    self.result['last'+str(cs)][file]=(len(solution),solution.sum_distance())
                else:
                    self.result['last10'][file] = (len(solution), solution.sum_distance())
            elif self.model.file_type == 'p':
                if self.model.negative_demand!=10:
                    file = self.model.data_file[-7:-4]+'_0'+str(self.model.negative_demand)
                else:
                    file = self.model.data_file[-7:-4] + '_10'
                s = self.result['S_best'][file]
                if (pd.isnull(s)):
                    self.result['S_best'][file] = solution.copy()
                    self.result['S_best_obj2'][file] = solution.sum_distance()
                elif (s.sum_distance()>solution.sum_distance()):
                    self.result['S_best'][file]=solution.copy()
                    self.result['S_best_obj2'][file] = solution.sum_distance()
                self.result['count'][file] += 1
                cs = self.result['count'][file] % 10
                if cs != 0:
                    self.result['last' + str(cs)][file] = (None, solution.sum_distance())
                else:
                    self.result['last10'][file] = (None, solution.sum_distance())
            elif self.model.file_type=='jd':
                s = self.result['S_best']['jd']
                s = None if pd.isnull(s) else s
                if self.compare_better(solution,s):
                    self.result['S_best']['jd']=solution.copy()
                    self.result['S_best_obj1']['jd'] = len(solution)
                    self.result['S_best_obj2']['jd'] = solution.sum_distance()
                self.result['count']['jd'] += 1
                cs = self.result['count']['jd'] % 10
                if cs != 0:
                    self.result['last' + str(cs)]['jd'] = (len(solution), solution.sum_distance())
                else:
                    self.result['last10']['jd'] = (len(solution), solution.sum_distance())
            elif self.model.file_type=='e':
                if self.model.data_file[-12]!='r':
                    file=self.model.data_file[-11:-7]
                else:
                    file = self.model.data_file[-12:-7]
                file = 'E'+file.upper()+'L'
                s = self.result['S_best'][file]
                s = None if pd.isnull(s) else s
                if self.compare_better(solution, s):
                    self.result['S_best'][file] = solution.copy()
                    self.result['S_best_obj1'][file] = len(solution)
                    self.result['S_best_obj2'][file] = solution.sum_distance()
                self.result['count'][file] += 1
                cs = self.result['count'][file] % 10
                if cs != 0:
                    self.result['last' + str(cs)][file] = (len(solution), solution.sum_distance())
                else:
                    self.result['last10'][file] = (len(solution), solution.sum_distance())
        self.result.to_pickle('result.pkl')
        return


class VNS_TS(Evolution):
    # 构造属性
    model = None

    vns_neighbour_Rts = 4
    vns_neighbour_max = 5
    eta_feas = 10
    eta_dist = 40
    Delta_SA = 0.08

    penalty = [10, 10, 10]
    penalty_min = (0.5, 0.5, 0.5)
    penalty_max = (5000, 5000, 5000)
    delta = 1.2
    eta_penalty = 2

    nu_min = 15
    nu_max = 30
    lambda_div = 1.0
    eta_tabu = 100
    # 状态属性
    vns_neighbour = []
    frequency = {}
    possible_arc = {}
    SA_dist = None
    SA_feas = None
    penalty_update_flag = []
    S_best = None
    max_ini_routes = None

    def __init__(self, model: Model, **param) -> None:
        self.model = model
        for key, value in param.items():
            assert hasattr(self, key)
            setattr(self, key, value)

        self.SA_dist = Util.SA(self.Delta_SA, self.eta_dist)
        self.SA_feas = Util.SA(self.Delta_SA, self.eta_feas)
        self.penalty_update_flag = [collections.deque(maxlen=self.eta_penalty), collections.deque(maxlen=self.eta_penalty), collections.deque(maxlen=self.eta_penalty)]
        self.calculate_possible_arc()
        self.max_ini_routes=max(1,math.ceil((np.sum(np.array([i.demand for i in model.customers if i.demand>0])))/model.vehicle.capacity),math.ceil((np.sum(np.array([abs(i.demand) for i in model.customers if i.demand<0])))/model.vehicle.capacity))
        print(len(self.possible_arc))

    @staticmethod
    def penalty_capacity(route: Route, vehicle: Vehicle) -> float:
        if route.arrive_load_weight is None:
            route.cal_load_weight(vehicle)
        penalty = max(route.arrive_load_weight[0]-vehicle.capacity, 0)
        neg_demand_cus = []
        for i, cus in enumerate(route.visit):
            if cus.demand < 0:
                neg_demand_cus.append(i)
        for i in neg_demand_cus:
            penalty += max(route.arrive_load_weight[i]-vehicle.capacity, 0)
        return penalty

    @staticmethod
    def penalty_time(route: Route, vehicle: Vehicle) -> float:
        if route.arrive_time is None:
            route.cal_arrive_time(vehicle)
        late_time = route.arrive_time-np.array([cus.over_time for cus in route.visit])
        late_time[late_time<0]=0
        if_late = np.where(late_time > 0)[0]
        if len(if_late) > 0:
            #return late_time[if_late[0]
            return np.sum(late_time)
        else:
            return 0.0

    @staticmethod
    def penalty_battery(route: Route, vehicle: Vehicle) -> float:
        if route.arrive_remain_battery is None:
            route.cal_remain_battery(vehicle)
        return np.abs(np.sum(route.arrive_remain_battery, where=route.arrive_remain_battery < 0))

    @staticmethod
    def get_objective_route(route: Route, vehicle: Vehicle, penalty: list) -> float:
        if route.no_customer():
            return 0
        return route.sum_distance()+penalty[0]*VNS_TS.penalty_capacity(route, vehicle)+penalty[1]*VNS_TS.penalty_time(route, vehicle)+penalty[2]*VNS_TS.penalty_battery(route, vehicle)

    @staticmethod
    def get_objective(solution: Solution, model: Model, penalty: list) -> float:
        ret = 0
        for route in solution.routes:
            ret += VNS_TS.get_objective_route(route, model.vehicle, penalty)
        return ret

    def calculate_possible_arc(self) -> None:
        self.model.find_nearest_station()
        all_node_list = [self.model.depot]+self.model.rechargers+self.model.customers
        for node1 in all_node_list:
            for node2 in all_node_list:
                if (isinstance(node1, Depot) and isinstance(node2, Recharger) and (node1.x == node2.x and node1.y == node2.y)) or (isinstance(node1, Recharger) and isinstance(node2, Depot) and (node1.x == node2.x and node1.y == node2.y)):
                    continue
                if not node1 == node2:
                    distance = node1.distance_to(node2)
                    if isinstance(node1, Customer) and isinstance(node2, Customer) and (node1.demand+node2.demand) > self.model.vehicle.capacity:
                        continue
                    if node1.ready_time+node1.service_time+distance > node2.over_time:
                        continue
                    if node1.ready_time+node1.service_time+distance+node2.service_time+node2.distance_to(self.model.depot) > self.model.depot.over_time:
                        continue
                    if len(self.model.rechargers) != 0 and isinstance(node1, Customer) and isinstance(node2, Customer):
                        recharger1 = self.model.nearest_station[node1][0]
                        recharger2 = self.model.nearest_station[node2][0]
                        if self.model.vehicle.battery_cost_speed*(node1.distance_to(recharger1)+distance+node2.distance_to(recharger2)) > self.model.vehicle.max_battery:
                            continue
                    if distance == 0:
                        distance = 0.0000001
                    self.possible_arc[(node1, node2)] = distance

    def select_possible_arc(self, N: int) -> list:
        selected_arc = []
        keys = list(self.possible_arc.keys())
        while len(keys) > 0 and len(selected_arc) < N:
            values = [self.possible_arc[key] for key in keys]
            values = np.array(values)
            values = 1/values
            #values = values/np.sum(values)
            select = Util.wheel_select(values)
            selected_arc.append(keys[select])
            del keys[select]
        return selected_arc

    def update_penalty(self, S: Solution) -> None:
        self.penalty_update_flag[0].append(S.feasible_capacity(self.model))
        self.penalty_update_flag[1].append(S.feasible_time(self.model))
        self.penalty_update_flag[2].append(S.feasible_battery(self.model))
        for i in range(len(self.penalty)):
            if self.penalty_update_flag[i].count(False) == self.eta_penalty:
                self.penalty[i] += self.delta
                if self.penalty[i] > self.penalty_max[i]:
                    self.penalty[i] = self.penalty_max[i]
            elif self.penalty_update_flag[i].count(True) == self.eta_penalty:
                self.penalty[i] /= self.delta
                if self.penalty[i] < self.penalty_min[i]:
                    self.penalty[i] = self.penalty_min[i]

    def update_frequency(self, soloution: Solution) -> None:
        for which, route in enumerate(soloution.routes):
            for where in range(1, len(route.visit)-1):
                left, right = Operation.find_left_right_station(route, where)
                if (route.visit[where], which, left, right) in self.frequency:
                    self.frequency[(route.visit[where], which, left, right)] += 1
                else:
                    self.frequency[(route.visit[where], which, left, right)] = 1

    def random_create(self) -> Solution:
        x = random.uniform(self.model.get_map_bound()[0], self.model.get_map_bound()[1])
        y = random.uniform(self.model.get_map_bound()[2], self.model.get_map_bound()[3])
        choose = self.model.customers[:]
        choose.sort(key=lambda cus: Util.cal_angle_AoB((self.model.depot.x, self.model.depot.y), (x, y), (cus.x, cus.y)))
        routes = []
        building_route_visit = [self.model.depot, self.model.depot]

        choose_index = 0
        while choose_index < len(choose):
            allow_insert_place = list(range(1, len(building_route_visit)))

            while True:
                min_increase_dis = float('inf')
                decide_insert_place = None
                for insert_place in allow_insert_place:
                    increase_dis = choose[choose_index].distance_to(building_route_visit[insert_place-1])+choose[choose_index].distance_to(building_route_visit[insert_place])-building_route_visit[insert_place-1].distance_to(building_route_visit[insert_place])
                    if increase_dis < min_increase_dis:
                        min_increase_dis=increase_dis
                        decide_insert_place = insert_place
                if len(allow_insert_place) == 1:
                    break
                elif (isinstance(building_route_visit[decide_insert_place-1], Customer) and isinstance(building_route_visit[decide_insert_place], Customer)) and (building_route_visit[decide_insert_place-1].ready_time <= choose[choose_index].ready_time and choose[choose_index].ready_time <= building_route_visit[decide_insert_place].ready_time):
                    break
                elif (isinstance(building_route_visit[decide_insert_place-1], Customer) and not isinstance(building_route_visit[decide_insert_place], Customer)) and building_route_visit[decide_insert_place-1].ready_time <= choose[choose_index].ready_time:
                    break
                elif (not isinstance(building_route_visit[decide_insert_place-1], Customer) and isinstance(building_route_visit[decide_insert_place], Customer)) and choose[choose_index].ready_time <= building_route_visit[decide_insert_place]:
                    break
                elif not isinstance(building_route_visit[decide_insert_place-1], Customer) and not isinstance(building_route_visit[decide_insert_place], Customer):
                    break
                else:
                    allow_insert_place.remove(decide_insert_place)
                    continue

            building_route_visit.insert(decide_insert_place, choose[choose_index])

            try_route = Route(building_route_visit)
            if try_route.feasible_capacity(self.model.vehicle)[0] and try_route.feasible_battery(self.model.vehicle)[0]:
                del choose[choose_index]
            else:
                if len(routes) < self.max_ini_routes-1:
                    del building_route_visit[decide_insert_place]
                    if len(building_route_visit) == 2:
                        choose_index += 1
                    else:
                        routes.append(Route(building_route_visit))
                        building_route_visit = [self.model.depot, self.model.depot]
                elif len(routes) == self.max_ini_routes-1:
                    del choose[choose_index]

        routes.append(Route(building_route_visit[:-1]+choose+[self.model.depot]))

        return Solution(routes)

    def create_vns_neighbour(self, Rts: int, max: int) -> list:
        assert Rts >= 2 and max >= 1
        self.vns_neighbour = []
        for R in range(2, Rts+1):
            for m in range(1, max+1):
                self.vns_neighbour.append((R, m))

    def tabu_search(self, S: Solution) -> Solution:
        best_S = S
        #best_val = VNS_TS.get_objective(S, self.model, self.penalty)
        select_arc = self.select_possible_arc(100)
        tabu_list = {}
        input_len=len(S)
        for _ in range(self.eta_tabu):
            local_best_S = None
            local_best_act = None
            for arc in select_arc:
                for neighbor_opt in [Modification.two_opt_star_arc, Modification.relocate_arc, Modification.exchange_arc, Modification.stationInRe_arc]:
                    neighbor_sol, neighbor_act = neighbor_opt(self.model, S, *arc)
                    for sol in neighbor_sol:
                        assert sol.serve_all_customer(self.model)
                    for sol, act in zip(neighbor_sol, neighbor_act):
                        if tabu_list.get(act, 0) == 0:
                            if self.compare_better(sol, local_best_S):
                                local_best_S = sol
                                local_best_act = act
                                #tabu_list[local_best_act] = random.randint(self.nu_min, self.nu_max)
            for key in tabu_list:
                if tabu_list[key] >= 1:
                    tabu_list[key] -= 1
            tabu_list[local_best_act] = random.randint(self.nu_min, self.nu_max)
            if self.compare_better(local_best_S, best_S):
                best_S = local_best_S
            S = local_best_S
        assert len(best_S)<=input_len
        return best_S

    def compare_better(self, solution1: Solution, solution2: Solution) -> bool:
        if solution2 is None:
            return True
        s1_val = VNS_TS.get_objective(solution1, self.model, self.penalty)
        s2_val = VNS_TS.get_objective(solution2, self.model, self.penalty)
        if solution1.feasible(self.model) and solution2.feasible(self.model):
            if len(solution1) < len(solution2) or (len(solution1) == len(solution2) and s1_val < s2_val):
                # if solution1.get_actual_routes() < solution2.get_actual_routes() or (solution1.get_actual_routes() == solution2.get_actual_routes() and s1_val < s2_val):
                return True
        elif solution1.feasible(self.model) and not solution2.feasible(self.model):
            return True
        elif not solution1.feasible(self.model) and solution2.feasible(self.model):
            return False
        elif not solution1.feasible(self.model) and not solution2.feasible(self.model):
            if s1_val < s2_val:
                return True
            else:
                return False

    def acceptSA_feas(self, S2: Solution, S: Solution, i) -> bool:
        S2_objective = VNS_TS.get_objective(S2, self.model, self.penalty)
        S_objective = VNS_TS.get_objective(S, self.model, self.penalty)
        if random.random() < self.SA_feas.probability(S2_objective, S_objective, i):
            return True
        return False

    def acceptSA_dist(self, S2: Solution, S: Solution, i) -> bool:
        S2_objective = VNS_TS.get_objective(S2, self.model, self.penalty)
        S_objective = VNS_TS.get_objective(S, self.model, self.penalty)
        if random.random() < self.SA_dist.probability(S2_objective, S_objective, i):
            return True
        return False

    def main(self) -> Solution:
        self.create_vns_neighbour(self.vns_neighbour_Rts, self.vns_neighbour_max)
        S = self.random_create()
        #p=self.model.customers.copy()
        #random.shuffle(p)
        #S=Solution([Route([self.model.depot]+p+[self.model.depot])])
        k = 0
        i = 0
        feasibilityPhase = True
        while feasibilityPhase or i < self.eta_dist:
            if self.compare_better(S, self.S_best):
                self.S_best = S
            self.update_penalty(S)

            print(i, S.feasible(self.model), len(S), VNS_TS.get_objective(S, self.model, self.penalty))
            print(S)
            S1 = Modification.cyclic_exchange(S, self.model, *self.vns_neighbour[k])
            #print(S1)
            S2 = self.tabu_search(S1)
            #print(S2)
            if self.compare_better(S2, S) or (feasibilityPhase and self.acceptSA_feas(S2, S, i)) or (not feasibilityPhase and self.acceptSA_dist(S2, S, i)):
                S = S2
                k = 0
            else:
                k = (k+1) % len(self.vns_neighbour)

            if feasibilityPhase:
                if not S.feasible(self.model):
                    if i == self.eta_feas:
                        S.add_empty_route(self.model)
                        print('增加车辆数至',len(S))
                        num=len(S)
                        self.max_ini_routes=num
                        S = self.random_create()
                        while num > len(S):
                            S.add_empty_route(self.model)
                        #random.shuffle(p)
                        #seq=np.cumsum(np.array([0]+[len(self.model.customers)/len(S)]*len(S)))
                        #s=[]
                        #for j in range(len(seq) - 1):
                        #    if j != len(seq) - 2:
                        #        s.append(
                        #            Route([self.model.depot] + p[int(round(seq[j])):int(round(seq[j + 1]))] + [self.model.depot]))
                        #    else:
                        #        s.append(Route([self.model.depot] + p[int(round(seq[j])):] + [self.model.depot]))
                        #S=Solution(s)
                        self.penalty=[10,10,10]
                        self.penalty_update_flag = [collections.deque(maxlen=self.eta_penalty), collections.deque(maxlen=self.eta_penalty), collections.deque(maxlen=self.eta_penalty)]
                        i = -1
                else:
                    feasibilityPhase = False
                    print('确定可行最小车辆数',len(S))
                    i = -1
            i += 1

        return S


class DEMA(Evolution):
    # 构造属性
    model = None
    penalty = (15, 5, 10)
    maxiter_evo = 100
    size = 30
    infeasible_proportion = 0.25
    sigma = (1, 5, 10)
    theta = 0.7
    maxiter_tabu_mul = 4
    max_neighbour_mul = 3
    tabu_len = 4
    local_search_step = 10
    charge_modify_step = 14
    # 状态属性
    last_local_search = 0
    last_charge_modify = 0
    S_best = None
    min_cost = float('inf')

    def __init__(self, model: Model, **param) -> None:
        self.model = model
        for key, value in param.items():
            assert hasattr(self, key)
            setattr(self, key, value)
        assert self.size >= 4

    @ staticmethod
    def get_objective_route(route: Route, vehicle: Vehicle, penalty: tuple) -> float:
        return route.sum_distance()+penalty[0]*VNS_TS.penalty_capacity(route, vehicle)+penalty[1]*VNS_TS.penalty_time(route, vehicle)+penalty[2]*VNS_TS.penalty_battery(route, vehicle)

    @ staticmethod
    def get_objective(solution: Solution, model: Model, penalty: tuple) -> float:
        if solution.objective is None:
            ret = 0
            for route in solution.routes:
                ret += DEMA.get_objective_route(route, model.vehicle, penalty)
            solution.objective = ret
            return ret
        else:
            return solution.objective

    @ staticmethod
    def overlapping_degree(solution1: Solution, solution2: Solution) -> float:
        sol1arcs = []
        sol2arcs = []
        for route in solution1.routes:
            for i in range(len(route.visit)-1):
                sol1arcs.append((route.visit[i], route.visit[i+1]))
        for route in solution2.routes:
            for i in range(len(route.visit)-1):
                sol2arcs.append((route.visit[i], route.visit[i+1]))
        num = 0
        for arc in sol1arcs:
            if arc in sol2arcs:
                num += 2
        return num/(len(sol1arcs)+len(sol2arcs))

    @ staticmethod
    def overlapping_degree_population(solution: Solution, population: list) -> float:
        sum = 0
        for p in population:
            sum += DEMA.overlapping_degree(solution, p)
        return sum/len(population)

    def random_create(self) -> Solution:
        x = random.uniform(self.model.get_map_bound()[0], self.model.get_map_bound()[1])
        y = random.uniform(self.model.get_map_bound()[2], self.model.get_map_bound()[3])
        choose = self.model.customers[:]
        choose.sort(key=lambda cus: Util.cal_angle_AoB((self.model.depot.x, self.model.depot.y), (x, y), (cus.x, cus.y)))
        routes = []
        building_route_visit = [self.model.depot, self.model.depot]

        choose_index = 0
        while choose_index < len(choose):
            allow_insert_place = list(range(1, len(building_route_visit)))

            while True:
                min_increase_dis = float('inf')
                decide_insert_place = None
                for insert_place in allow_insert_place:
                    increase_dis = choose[choose_index].distance_to(building_route_visit[insert_place-1])+choose[choose_index].distance_to(building_route_visit[insert_place])-building_route_visit[insert_place-1].distance_to(building_route_visit[insert_place])
                    if increase_dis < min_increase_dis:
                        min_increase_dis = increase_dis
                        decide_insert_place = insert_place
                if len(allow_insert_place) == 1:
                    break
                elif (isinstance(building_route_visit[decide_insert_place-1], Customer) and isinstance(building_route_visit[decide_insert_place], Customer)) and (building_route_visit[decide_insert_place-1].ready_time <= choose[choose_index].ready_time and choose[choose_index].ready_time <= building_route_visit[decide_insert_place].ready_time):
                    break
                elif (isinstance(building_route_visit[decide_insert_place-1], Customer) and not isinstance(building_route_visit[decide_insert_place], Customer)) and building_route_visit[decide_insert_place-1].ready_time <= choose[choose_index].ready_time:
                    break
                elif (not isinstance(building_route_visit[decide_insert_place-1], Customer) and isinstance(building_route_visit[decide_insert_place], Customer)) and choose[choose_index].ready_time <= building_route_visit[decide_insert_place].ready_time:
                    break
                elif not isinstance(building_route_visit[decide_insert_place-1], Customer) and not isinstance(building_route_visit[decide_insert_place], Customer):
                    break
                else:
                    allow_insert_place.remove(decide_insert_place)
                    continue

            building_route_visit.insert(decide_insert_place, choose[choose_index])

            try_route = Route(building_route_visit)
            if try_route.feasible_capacity(self.model.vehicle)[0] and try_route.feasible_time(self.model.vehicle)[0]:
                # del choose[choose_index]
                choose_index += 1
            else:
                del building_route_visit[decide_insert_place]
                assert len(building_route_visit) != 2
                routes.append(Route(building_route_visit))
                building_route_visit = [self.model.depot, self.model.depot]

        routes.append(Route(building_route_visit))

        return Solution(routes)

    def greedy_ini(self, model: Model):
        solu = []
        first = model.customers[0]
        for i in model.customers:
            if i.ready_time < first.ready_time:
                first = i
        s = Route([model.depot, first, model.depot])
        pool = model.customers.copy()
        del pool[np.where((np.array(pool) == first) == 1)[0][0]]
        while len(pool) != 0:
            best_place = None;
            best_customer = None;
            best_value = float('inf')
            for place in list(range(1, len(s))):
                for i in pool:
                    if i.demand > 0:
                        before = s.visit[:place];
                        before.append(i)
                        ss = Route(before + s.visit[place:])
                        if ss.feasible_capacity(model.vehicle)[0] & ss.feasible_time(model.vehicle)[0]:
                            distance_b = sum(np.array([before[index].distance_to(before[index + 1]) for index in
                                                       list(range(0, len(before) - 1))]))
                            value = i.demand * distance_b
                        else:
                            value = float('inf')
                    else:
                        after = s.visit[place:];
                        after = [i] + after
                        ss = Route(s.visit[:place] + after)
                        if ss.feasible_capacity(model.vehicle)[0] & ss.feasible_time(model.vehicle)[0]:
                            distance_a = sum(np.array([after[index].distance_to(after[index + 1]) for index in
                                                       list(range(0, len(after) - 1))]))
                            value = abs(i.demand) * distance_a
                        else:
                            value = float('inf')

                    if value < best_value:
                        best_value = value
                        best_place = place
                        best_customer = i
            if not best_value == float('inf'):
                s.add_node(model.vehicle, best_place, best_customer)
                del pool[np.where((np.array(pool) == best_customer) == 1)[0][0]]
                if len(pool) == 0:
                    solu.append(s)
            else:
                solu.append(s)
                one = random.sample(pool, 1)[0]
                s = Route([model.depot, one, model.depot])
                del pool[np.where((np.array(pool) == one) == 1)[0][0]]
                if len(pool) == 0:
                    solu.append(s)
        #sss=Solution(solu)
        #assert sss.serve_all_customer(model)
        greedy_solu = []
        for i in range(len(solu)):
            greedy_solu += Modification.charging_modification_for_route(solu[i], model)
        greedy_solu = Solution(greedy_solu)
        return greedy_solu

    def initialization(self) -> list:
        population = []
        while len(population) < self.size:
            reroll = False
            times = 0
            sol = self.random_create()
            #sol = self.greedy_ini(self.model)
            assert sol.serve_all_customer(self.model)
            while True:
                if times > 10:
                    reroll = True
                    break
                fes_dic = sol.feasible_detail(self.model)
                for _, value in fes_dic.items():
                    if value[1] == 'battery':
                        sol = Modification.charging_modification(sol, self.model)
                        assert sol.serve_all_customer(self.model)
                        times += 1
                        break
                    if value[1] == 'time':
                        sol = Modification.fix_time(sol, self.model)
                        assert sol.serve_all_customer(self.model)
                        times += 1
                        break
                else:#可行
                    sol.renumber_id()
                    break
            if reroll:
                reroll = False
                continue
            population.append(sol)
        return population

    def ACO_GM(self, P: list) -> list:
        cross_score = [0.0, 0.0]
        cross_call_times = [0, 0]
        cross_weigh = [0.0, 0.0]

        fes_P = []
        infes_P = []
        for sol in P:
            if sol.feasible(self.model):
                fes_P.append(sol)
            else:
                infes_P.append(sol)
        fes_P.sort(key=lambda sol: DEMA.get_objective(sol, self.model, self.penalty))

        obj_value = []
        for sol in infes_P:
            overlapping_degree = DEMA.overlapping_degree_population(sol, P)
            objective = DEMA.get_objective(sol, self.model, self.penalty)
            obj_value.append([objective, overlapping_degree])
        infes_P = Util.pareto_sort(infes_P, obj_value)

        P = fes_P+infes_P
        choose = Util.binary_tournament(len(P))
        P_parent = []
        for i in choose:
            P_parent.append(P[i])
        P_child = []
        all_cost = [DEMA.get_objective(sol, self.model, self.penalty) for sol in P]
        while len(P_child) < self.size:
            for i in range(2):
                if cross_call_times[i] != 0:
                    cross_weigh[i] = self.theta*np.pi/cross_call_times[i]+(1-self.theta)*cross_weigh[i]
            if cross_weigh[0] == 0 and cross_weigh[1] == 0:
                sel_prob = np.array([0.5, 0.5])
            else:
                sel_prob = np.array(cross_weigh)/np.sum(np.array(cross_weigh))
            sel = Util.wheel_select(sel_prob)

            if sel == 0:
                S_parent = random.choice(P_parent)
                S = Modification.ACO_GM_cross1(S_parent, self.model)
                assert S.serve_all_customer(self.model)
            elif sel == 1:
                S_parent, S2 = random.sample(P_parent, 2)
                S = Modification.ACO_GM_cross2(S_parent, S2, self.model)
                assert S.serve_all_customer(self.model)

            cross_call_times[sel] += 1
            cost = DEMA.get_objective(S, self.model, self.penalty)
            if cost < all(all_cost):
                cross_score[sel] += self.sigma[0]
            elif cost < DEMA.get_objective(S_parent, self.model, self.penalty):
                cross_score[sel] += self.sigma[1]
            else:
                cross_score[sel] += self.sigma[2]

            P_child.append(S)

        # self.penalty = penalty_save
        return P_child

    def ISSD(self, P: list, iter: int) -> list:
        SP1 = []
        SP2 = []
        for sol in P:
            if sol.feasible(self.model):
                SP1.append(sol)
            else:
                SP2.append(sol)
        SP1.sort(key=lambda sol: DEMA.get_objective(sol, self.model, self.penalty))
        obj_value = []
        for sol in SP2:
            overlapping_degree = DEMA.overlapping_degree_population(sol, P)
            objective = DEMA.get_objective(sol, self.model, self.penalty)
            obj_value.append([objective, overlapping_degree])
        SP2 = Util.pareto_sort(SP2, obj_value)
        #sp1up = int((iter/self.maxiter_evo)*self.size)
        sp1up = int((1-self.infeasible_proportion)*self.size)
        sp2up = self.size-sp1up
        P = SP1[:sp1up]+SP2[:sp2up]
        SP1 = SP1[sp1up:]
        SP2 = SP2[sp2up:]
        for sol in SP1:
            if len(P) < self.size:
                P.append(sol)
        for sol in SP2:
            if len(P) < self.size:
                P.append(sol)
        assert len(P) == self.size

        # for sol in P:
        #    sol.clear_status()

        return P

    def tabu_search_vnsts(self, solution: Solution) -> Solution:
        if getattr(self, 'vnsts', None) is None:
            self.vnsts = VNS_TS(self.model)
            self.vnsts.penalty = self.penalty
        sol = self.vnsts.tabu_search(solution)
        return sol

    def tabu_search_abandon(self, solution: Solution, iter_num: int, neighbor_num: int) -> Solution:
        best_sol = solution
        best_val = DEMA.get_objective(solution, self.model, self.penalty)
        tabu_list = {}
        # delta = collections.deque([float('inf')]*10, maxlen=10)
        for iter in range(iter_num):
            print('tabu {} {}'.format(iter, best_val))
            actions = []
            while len(actions) < neighbor_num:
                act = random.choice(['exchange', 'relocate', 'two-opt', 'stationInRe'])
                if act == 'exchange':
                    target = Modification.exchange_choose(solution)
                    if ('exchange', *target) not in actions:
                        actions.append(('exchange', *target))
                elif act == 'relocate':
                    target = Modification.relocate_choose(solution)
                    if ('relocate', *target) not in actions:
                        actions.append(('relocate', *target))
                elif act == 'two-opt':
                    target = Modification.two_opt_choose(solution)
                    if ('two-opt', *target) not in actions:
                        actions.append(('two-opt', *target))
                elif act == 'stationInRe':
                    target = Modification.stationInRe_choose(solution, self.model)
                    if ('stationInRe', *target) not in actions:
                        actions.append(('stationInRe', *target))
            local_best_sol = solution
            local_best_val = DEMA.get_objective(solution, self.model, self.penalty)
            local_best_action = (None,)
            for action in actions:
                tabu_status = tabu_list.get(action, 0)
                if tabu_status == 0:
                    if action[0] == 'exchange':
                        try_sol = Modification.exchange_action(solution, *action[1:])
                    elif action[0] == 'relocate':
                        try_sol = Modification.relocate_action(solution, *action[1:])
                    elif action[0] == 'two-opt':
                        try_sol = Modification.two_opt_action(solution, *action[1:])
                    elif action[0] == 'stationInRe':
                        try_sol = Modification.stationInRe_action(solution, *action[1:])
                    try_val = DEMA.get_objective(try_sol, self.model, self.penalty)
                    if try_val < local_best_val:
                        local_best_sol = try_sol
                        local_best_val = try_val
                        local_best_action = action
            for key in tabu_list:
                if tabu_list[key] > 0:
                    tabu_list[key] -= 1
            if local_best_action[0] == 'exchange':
                tabu_list[('exchange', *local_best_action[1:])] = self.tabu_len
                tabu_list[('exchange', *local_best_action[3:5], *local_best_action[1:3])] = self.tabu_len
            elif local_best_action[0] == 'relocate':
                tabu_list[('relocate', *local_best_action[1:])] = self.tabu_len
            elif local_best_action[0] == 'two-opt':
                tabu_list[('two-opt', *local_best_action[1:])] = self.tabu_len
                tabu_list[('two-opt', local_best_action[1], local_best_action[3], local_best_action[2])] = self.tabu_len
            if local_best_val < best_val:
                best_sol = local_best_sol
                best_val = local_best_val

            solution = local_best_sol

            # delta.append(DEMA.get_objective(solution, self.model, self.penalty)-local_best_val)
            # should_break = True
            # for i in delta:
            #    if i > 0.00001:
            #        should_break = False
            #    break
            # if should_break:
            #    break

        return best_sol

    def MVS(self, P: list, iter: int) -> list:
        self.last_local_search += 1
        self.last_charge_modify += 1
        if self.last_local_search >= self.local_search_step:
            retP = []
            for i, sol in enumerate(P):
                print(iter, 'tabu', i)
                retP.append(self.tabu_search_vnsts(sol))
            self.last_local_search = 0
            return retP
        elif self.last_charge_modify >= self.charge_modify_step:
            retP = []
            for i, sol in enumerate(P):
                print(iter, 'charge', i)
                retP.append(Modification.charging_modification(sol, self.model))
            self.last_charge_modify = 0
            return retP
        return P

    def update_S(self, P: list) -> None:
        for S in P:
            if S.feasible(self.model):
                cost = DEMA.get_objective(S, self.model, self.penalty)
                num = len(S.routes)
                # cost = S.sum_distance()
                if self.S_best is None:
                    self.S_best = S
                    self.min_cost = cost
                elif not self.S_best is None and num < len(self.S_best.routes):
                    self.S_best = S
                    self.min_cost = cost
                elif not self.S_best is None and num == len(self.S_best.routes):
                    if cost < self.min_cost:
                        self.S_best = S
                        self.min_cost = cost

    def main(self, control: tuple = (True, True), icecube: list = None) -> tuple:
        if icecube is None:
            P = self.initialization()
        else:
            self.model, self.S_best, self.min_cost, P = icecube
        self.update_S(P)
        for iter in range(self.maxiter_evo):
            print(iter, len(self.S_best), self.min_cost)
            if control[0]:
                P_child = self.ACO_GM(P)
                P = self.ISSD(P+P_child, iter)
            if control[1]:
                P = self.MVS(P, iter)
            self.update_S(P)
            self.P = P
        return self.S_best, self.min_cost

    def freeze(self) -> list:
        return [self.model, self.S_best, self.min_cost, self.P]


class DEMA1(Evolution):
    model=None
    max_iteration=100
    S_best=None
    min_cost=None
    popsize=None
    penalty=None
    theta=0.7
    sigma = (1, 5, 10)
    infeasible_proportion = 0.25
    file_dic=None
    P=None
    possible_arc= {}
    max_repair_in=5

    def __init__(self,model:Model,dic_file):
        self.model=model
        self.penalty=(15, 10, 10)
        self.popsize=10
        self.S_best=None
        self.ls=LS(model,dic_file,1)
        self.ls.eta_feas=self.max_iteration
        self.ls.SA_feas = Util.SA(self.ls.Delta_SA, self.ls.eta_feas)
        self.ls.penalty=self.penalty
        self.ls.max_iter=1
        self.calculate_possible_arc()
        self.ls.possible_arc=self.possible_arc
        self.ls.popsize=self.popsize
        self.ls.jjcs1=10
        self.ls.jjcs2=20
        self.ls.DEMA_max_iter=self.max_iteration
        self.file_dic=dic_file

    def rw_result(self):
        location = r'result.pkl'
        if not os.path.exists(location):
            result = []
            for maindir, subdir, file_name_list in os.walk('evrp'):
                if maindir == 'data\solomon':
                    for file in file_name_list:
                        if file[-7] == '2':
                            result.append(file[-9:-4].upper())
                if maindir == 'data\p':
                    for file in file_name_list:
                        result.extend([file[-7:-4] + '_02', file[-7:-4] + '_04', file[-7:-4] + '_10'])
                if maindir == 'data\jd':
                    result.append('jd')
                if maindir == 'data\evrptw_instances':
                    for file in file_name_list:
                        if file[-10] == '2':
                            result.append('E' + file[-12:-7].upper() + 'L')
            col = ['BKS_obj1', 'BKS_obj2', 'S_best_obj1', 'S_best_obj2', 'S_best', 'count']
            col.extend(['last' + str(i) for i in range(1, 11)])
            self.result = pd.DataFrame(index=result, columns=col)
            self.result['count'] = 0
            self.result.to_pickle(r'result.pkl')
        else:
            self.result = pd.read_pickle(location)
        solution = self.S_best.copy()
        assert isinstance(solution, Solution)
        if solution.feasible(self.model):
            if self.model.file_type == 'tw':
                if self.model.data_file[-9] != 'r':
                    file = self.model.data_file[-8:-4]
                else:
                    file = self.model.data_file[-9:-4]
                file = file.upper()
                s = self.result['S_best'][file]
                s = None if pd.isnull(s) else s
                if self.compare_better(solution, s):
                    self.result['S_best'][file] = solution.copy()
                    self.result['S_best_obj1'][file] = len(solution)
                    self.result['S_best_obj2'][file] = solution.sum_distance()
                self.result['count'][file] += 1
                cs = self.result['count'][file] % 10
                if cs != 0:
                    self.result['last' + str(cs)][file] = (len(solution), solution.sum_distance())
                else:
                    self.result['last10'][file] = (len(solution), solution.sum_distance())
            elif self.model.file_type == 'p':
                if self.model.negative_demand != 10:
                    file = self.model.data_file[-7:-4] + '_0' + str(self.model.negative_demand)
                else:
                    file = self.model.data_file[-7:-4] + '_10'
                s = self.result['S_best'][file]
                if (pd.isnull(s)):
                    self.result['S_best'][file] = solution.copy()
                    self.result['S_best_obj2'][file] = solution.sum_distance()
                elif (s.sum_distance() > solution.sum_distance()):
                    self.result['S_best'][file] = solution.copy()
                    self.result['S_best_obj2'][file] = solution.sum_distance()
                self.result['count'][file] += 1
                cs = self.result['count'][file] % 10
                if cs != 0:
                    self.result['last' + str(cs)][file] = (None, solution.sum_distance())
                else:
                    self.result['last10'][file] = (None, solution.sum_distance())
            elif self.model.file_type == 'jd':
                s = self.result['S_best']['jd']
                s = None if pd.isnull(s) else s
                if self.compare_better(solution, s):
                    self.result['S_best']['jd'] = solution.copy()
                    self.result['S_best_obj1']['jd'] = len(solution)
                    self.result['S_best_obj2']['jd'] = solution.sum_distance()
                self.result['count']['jd'] += 1
                cs = self.result['count']['jd'] % 10
                if cs != 0:
                    self.result['last' + str(cs)]['jd'] = (len(solution), solution.sum_distance())
                else:
                    self.result['last10']['jd'] = (len(solution), solution.sum_distance())
            elif self.model.file_type == 'e':
                if self.model.data_file[-12] != 'r':
                    file = self.model.data_file[-11:-7]
                else:
                    file = self.model.data_file[-12:-7]
                file = 'E' + file.upper() + 'L'
                s = self.result['S_best'][file]
                s = None if pd.isnull(s) else s
                if self.compare_better(solution, s):
                    self.result['S_best'][file] = solution.copy()
                    self.result['S_best_obj1'][file] = len(solution)
                    self.result['S_best_obj2'][file] = solution.sum_distance()
                self.result['count'][file] += 1
                cs = self.result['count'][file] % 10
                if cs != 0:
                    self.result['last' + str(cs)][file] = (len(solution), solution.sum_distance())
                else:
                    self.result['last10'][file] = (len(solution), solution.sum_distance())
        self.result.to_pickle(location)
        return

    def random_create(self) -> Solution:
        x = random.uniform(self.model.get_map_bound()[0], self.model.get_map_bound()[1])
        y = random.uniform(self.model.get_map_bound()[2], self.model.get_map_bound()[3])
        choose = self.model.customers[:]
        choose.sort(
            key=lambda cus: Util.cal_angle_AoB((self.model.depot.x, self.model.depot.y), (x, y), (cus.x, cus.y)))
        routes = []
        building_route_visit = [self.model.depot, self.model.depot]

        choose_index = 0
        while choose_index < len(choose):
            allow_insert_place = list(range(1, len(building_route_visit)))

            while True:
                min_increase_dis = float('inf')
                decide_insert_place = None
                for insert_place in allow_insert_place:
                    increase_dis = choose[choose_index].distance_to(building_route_visit[insert_place - 1]) + choose[
                        choose_index].distance_to(building_route_visit[insert_place]) - building_route_visit[
                                       insert_place - 1].distance_to(building_route_visit[insert_place])
                    if increase_dis < min_increase_dis:
                        min_increase_dis = increase_dis
                        decide_insert_place = insert_place
                if len(allow_insert_place) == 1:
                    break
                elif (isinstance(building_route_visit[decide_insert_place - 1], Customer) and isinstance(
                        building_route_visit[decide_insert_place], Customer)) and (
                        building_route_visit[decide_insert_place - 1].ready_time <= choose[choose_index].ready_time and
                        choose[choose_index].ready_time <= building_route_visit[decide_insert_place].ready_time):
                    break
                elif (isinstance(building_route_visit[decide_insert_place - 1], Customer) and not isinstance(
                        building_route_visit[decide_insert_place], Customer)) and building_route_visit[
                    decide_insert_place - 1].ready_time <= choose[choose_index].ready_time:
                    break
                elif (not isinstance(building_route_visit[decide_insert_place - 1], Customer) and isinstance(
                        building_route_visit[decide_insert_place], Customer)) and choose[choose_index].ready_time <= \
                        building_route_visit[decide_insert_place].ready_time:
                    break
                elif not isinstance(building_route_visit[decide_insert_place - 1], Customer) and not isinstance(
                        building_route_visit[decide_insert_place], Customer):
                    break
                else:
                    allow_insert_place.remove(decide_insert_place)
                    continue

            building_route_visit.insert(decide_insert_place, choose[choose_index])

            try_route = Route(building_route_visit)
            if try_route.feasible_capacity(self.model.vehicle)[0] and try_route.feasible_time(self.model.vehicle)[0]:
                # del choose[choose_index]
                choose_index += 1
            else:
                del building_route_visit[decide_insert_place]
                assert len(building_route_visit) != 2
                routes.append(Route(building_route_visit))
                building_route_visit = [self.model.depot, self.model.depot]

        routes.append(Route(building_route_visit))

        return Solution(routes)

    def calculate_possible_arc(self) -> None:
        self.model.find_nearest_station()
        all_node_list = [self.model.depot]+self.model.rechargers+self.model.customers
        for node1 in all_node_list:
            for node2 in all_node_list:
                if (isinstance(node1, Depot) and isinstance(node2, Recharger) and (node1.x == node2.x and node1.y == node2.y)) or (isinstance(node1, Recharger) and isinstance(node2, Depot) and (node1.x == node2.x and node1.y == node2.y)):
                    continue
                if not node1 == node2:
                    distance = node1.distance_to(node2)
                    if isinstance(node1, Customer) and isinstance(node2, Customer) and (node1.demand+node2.demand) > self.model.vehicle.capacity:
                        continue
                    if node1.ready_time+node1.service_time+distance > node2.over_time:
                        continue
                    if node1.ready_time+node1.service_time+distance+node2.service_time+node2.distance_to(self.model.depot) > self.model.depot.over_time:
                        continue
                    if len(self.model.rechargers) != 0 and isinstance(node1, Customer) and isinstance(node2, Customer):
                        recharger1 = self.model.nearest_station[node1][0]
                        recharger2 = self.model.nearest_station[node2][0]
                        if self.model.vehicle.battery_cost_speed*(node1.distance_to(recharger1)+distance+node2.distance_to(recharger2)) > self.model.vehicle.max_battery:
                            continue
                    if distance == 0:
                        distance = 0.0000001
                    self.possible_arc[(node1, node2)] = distance

    def ini(self):
        pop=[]
        while len(pop)<self.popsize:
            reroll = False
            times = 0
            sol = self.random_create()
            assert sol.serve_all_customer(self.model)
            while True:
                if times > 10:
                    reroll = True
                    break
                fes_dic = sol.feasible_detail(self.model)
                for _, value in fes_dic.items():
                    if value[1] == 'battery':
                        sol = Modification.charging_modification(sol, self.model,self)
                        sol=self.repair(sol.copy())
                        assert sol.serve_all_customer(self.model)
                        times += 1
                        break
                    if value[1] == 'time':
                        sol = Modification.fix_time(sol, self.model)
                        assert sol.serve_all_customer(self.model)
                        times += 1
                        break
                else:  # 可行
                    sol.renumber_id()
                    break
            if reroll:
                reroll = False
                continue
            pop.append(sol)
        return pop

    def compare_better(self, solution1: Solution, solution2: Solution) -> bool:
        if solution2 is None:
            return True
        s1_val = VNS_TS.get_objective(solution1, self.model, self.penalty)
        s2_val = VNS_TS.get_objective(solution2, self.model, self.penalty)
        if not self.model.file_type == 'p':
            if solution1.feasible(self.model) and solution2.feasible(self.model):
                if len(solution1) < len(solution2) or (len(solution1) == len(solution2) and s1_val < s2_val):
                    # if solution1.get_actual_routes() < solution2.get_actual_routes() or (solution1.get_actual_routes() == solution2.get_actual_routes() and s1_val < s2_val):
                    return True
                else:
                    return False
            elif solution1.feasible(self.model) and not solution2.feasible(self.model):
                return True
            elif not solution1.feasible(self.model) and solution2.feasible(self.model):
                return False
            elif not solution1.feasible(self.model) and not solution2.feasible(self.model):
                if s1_val < s2_val:
                    return True
                else:
                    return False
        else:
            if solution1.feasible(self.model) and solution2.feasible(self.model):
                if s1_val < s2_val:
                    return True
                else:
                    return False
            elif solution1.feasible(self.model) and not solution2.feasible(self.model):
                return True
            elif not solution1.feasible(self.model) and solution2.feasible(self.model):
                return False
            elif not solution1.feasible(self.model) and not solution2.feasible(self.model):
                if s1_val < s2_val:
                    return True
                else:
                    return False

    def update_S(self, P: list) -> None:
        for S in P:
            if S.feasible(self.model):
                cost = DEMA.get_objective(S, self.model, self.penalty)
                num = len(S.routes)
                # cost = S.sum_distance()
                if self.S_best is None:
                    self.S_best = S
                    self.min_cost = cost
                elif not self.S_best is None and num < len(self.S_best.routes):
                    self.S_best = S
                    self.min_cost = cost
                elif not self.S_best is None and num == len(self.S_best.routes):
                    if cost < self.min_cost:
                        self.S_best = S
                        self.min_cost = cost

    def repair(self,solution:Solution):
        if solution.feasible(self.model):
            return solution
        solution=solution.copy()
        new_routes=[]
        for i in range(len(solution)):
            if solution[i].feasible(self.model.vehicle)[0]:
                new_routes.append(solution[i].copy())
                continue
            route=solution[i].copy()
            if not (route.feasible_time(self.model.vehicle)[0] and route.feasible_capacity(self.model.vehicle)[0]):
                r=Route([j for j in route.visit if isinstance(j,(Customer,Depot))])
                if (not route.feasible_time(self.model.vehicle)[0]) and route.feasible_capacity(self.model.vehicle)[0]:
                    if r.feasible_time(self.model.vehicle)[0]:
                        pass
                    else:
                        max_try = 0
                        while (not r.feasible_time(self.model.vehicle)[0]) and (not (max_try > len(r) * 10)):
                            cus_index = r.feasible_time(self.model.vehicle)[1]
                            cus = r.visit[cus_index]
                            if isinstance(cus,Depot):
                                break
                            sol = []
                            for i in range(cus_index):
                                r1 = r.copy()
                                if i == 0:
                                    continue
                                if self.possible_arc.get((r1.visit[i - 1], cus), 0) and self.possible_arc.get(
                                        (cus, r1.visit[i]), 0):
                                    r1.del_node(self.model.vehicle, cus_index)
                                    r1.add_node(self.model.vehicle, i, cus)
                                    if (r1.feasible_time(self.model.vehicle)[1] != i) and \
                                            r1.feasible_capacity(self.model.vehicle)[0]:
                                        r1.cal_arrive_time(self.model.vehicle)
                                        sol.append(
                                            (r1, len(np.where(
                                                np.array([cus.over_time for cus in r1.visit]) > r1.arrive_time)[0])))
                            if len(sol) != 0:
                                sol.sort(key=lambda x: x[1], reverse=1)
                                hx = [j[0] for j in sol if j[1] == sol[0][1]]
                                r = random.choice(hx)
                            else:
                                break
                            max_try = max_try + 1
                elif (not route.feasible_capacity(self.model.vehicle)[0]) and route.feasible_time(self.model.vehicle)[0]:
                    if (np.sum(np.array([i.demand for i in r.visit if i.demand > 0])) > self.model.vehicle.capacity) or (
                            np.sum(
                                np.array([abs(i.demand) for i in r.visit if i.demand < 0])) > self.model.vehicle.capacity):
                        new_routes.append(r)
                        continue
                    else:
                        max_try = 0
                        while (not r.feasible_capacity(self.model.vehicle)[0]) and (not (max_try > len(r) * 10)):
                            cus_index = r.feasible_capacity(self.model.vehicle)[1]
                            cus = r.visit[cus_index]  # 该客户必为pickup客户
                            sol = []
                            for i in range(cus_index + 2, len(r)):
                                r1 = r.copy()
                                r1.cal_load_weight(self.model.vehicle)
                                load1 = r1.arrive_load_weight[cus_index - 1]
                                varyload = np.sum(np.array([j.demand for j in r1.visit[cus_index:i]]))
                                if self.possible_arc.get((r1.visit[i - 1], cus), 0) and self.possible_arc.get(
                                        (cus, r1.visit[i]),
                                        0) and (
                                        varyload >= -(self.model.vehicle.capacity - load1)):
                                    r1.add_node(self.model.vehicle, i, cus)
                                    r1.del_node(self.model, cus_index)
                                    if (r1.feasible_capacity(self.model.vehicle)[1] != (i - 1)) and \
                                            r1.feasible_time(self.model.vehicle)[
                                                0]:
                                        r1.cal_load_weight(self.model.vehicle)
                                        sol.append((r1, np.sum(r1.arrive_load_weight > self.model.vehicle.capacity)))
                            if len(sol) != 0:
                                sol.sort(key=lambda x: x[1])
                                hx = [j[0] for j in sol if j[1] == sol[0][1]]
                                r = random.choice(hx)
                            else:
                                break
                            max_try = max_try + 1
                else:
                    if (np.sum(np.array([i.demand for i in r.visit if i.demand > 0])) > self.model.vehicle.capacity) or (
                            np.sum(
                                np.array([abs(i.demand) for i in r.visit if i.demand < 0])) > self.model.vehicle.capacity):
                        new_routes.append(r)
                        continue
                    else:
                        max_try = 0
                        while not ((r.feasible_capacity(self.model.vehicle)[0] and r.feasible_time(self.model.vehicle)[0]) or (
                                max_try > len(r) * 5)):
                            cus_index1 = r.feasible_time(self.model.vehicle)[1]
                            cus_index2 = r.feasible_capacity(self.model.vehicle)[1]
                            if cus_index1 == cus_index2:
                                break
                            cus_index = min(cus_index1, cus_index2)
                            if cus_index == cus_index1:  # 修复时间窗
                                cus = r.visit[cus_index]
                                if isinstance(cus, Depot):
                                    break
                                sol = []
                                for i in range(cus_index):
                                    r1 = r.copy()
                                    if i == 0:
                                        continue
                                    if self.possible_arc.get((r1.visit[i - 1], cus), 0) and self.possible_arc.get(
                                            (cus, r1.visit[i]), 0):
                                        r1.del_node(self.model.vehicle, cus_index)
                                        r1.add_node(self.model.vehicle, i, cus)
                                        if (r1.feasible_time(self.model.vehicle)[1] != i):
                                            r1.cal_arrive_time(self.model.vehicle)
                                            sol.append((r1, len(
                                                np.where(
                                                    np.array([cus.over_time for cus in r1.visit]) > r1.arrive_time)[
                                                    0])))
                                if len(sol) != 0:
                                    sol.sort(key=lambda x: x[1], reverse=1)
                                    hx = [j[0] for j in sol if j[1] == sol[0][1]]
                                    r = random.choice(hx)
                                else:
                                    break
                                max_try = max_try + 1

                            else:  # 修复载重约束
                                cus = r.visit[cus_index]  # 该客户必为pickup客户
                                sol = []
                                for i in range(cus_index + 2, len(r)):
                                    r1 = r.copy()
                                    r1.cal_load_weight(self.model.vehicle)
                                    load1 = r1.arrive_load_weight[cus_index - 1]
                                    varyload = np.sum(np.array([j.demand for j in r1.visit[cus_index:i]]))
                                    if self.possible_arc.get((r1.visit[i - 1], cus), 0) and self.possible_arc.get(
                                            (cus, r1.visit[i]), 0) and (varyload >= -(self.model.vehicle.capacity - load1)):
                                        r1.add_node(self.model.vehicle, i, cus)
                                        r1.del_node(self.model, cus_index)
                                        if (r1.feasible_capacity(self.model.vehicle)[1] != (i - 1)):
                                            r1.cal_load_weight(self.model.vehicle)
                                            sol.append((r1, np.sum(r1.arrive_load_weight > self.model.vehicle.capacity)))
                                if len(sol) != 0:
                                    sol.sort(key=lambda x: x[1])
                                    hx = [j[0] for j in sol if j[1] == sol[0][1]]
                                    r = random.choice(hx)
                                else:
                                    break
                                max_try = max_try + 1
            else:
                r=route.copy()
            if r.feasible_time(self.model.vehicle)[0] and r.feasible_capacity(self.model.vehicle)[0]:
                if not r.feasible_battery(self.model.vehicle)[0]:
                    o=0
                    max_try=0
                    while not (r.feasible(self.model.vehicle)[0] or (max_try>2*len(r))):
                        r.find_charge_station()
                        # 尝试增加充电桩
                        if not r.feasible_battery(self.model.vehicle)[0]:
                            r.cal_arrive_time(self.model.vehicle)
                            wait_time = np.array([j.ready_time for j in r.visit]) - r.arrive_time
                            wait_time=Util.map_100(wait_time)
                            wait_time[0]=1/1000000
                            if True:
                                #wt = np.zeros(len(wait_time))
                                #for i in range(len(wait_time)):
                                    #wt[i] = max(min(wait_time[i:]), 0)

                                node_index = Util.wheel_select(wait_time)
                                if node_index != 0:
                                    pos_sta = Modification.find_feasible_station_between(self.model, r.visit[node_index - 1],
                                                                                         r.visit[node_index], self)
                                    value = VNS_TS.get_objective_route(r, self.model.vehicle, self.penalty)
                                    station = None
                                    for i in range(len(pos_sta)):
                                        r.add_node(self.model.vehicle, node_index, pos_sta[i])
                                        o1=r.feasible(self.model.vehicle)[0]
                                        value1 = VNS_TS.get_objective_route(r, self.model.vehicle, self.penalty)
                                        if value > value1:
                                            if not (o==1 and o1==0):
                                                value = value1
                                                station = pos_sta[i]
                                                r.del_node(self.model.vehicle, node_index)
                                            else:
                                                r.del_node(self.model.vehicle, node_index)
                                        else:
                                            if (o==0 and o1==1):
                                                value = value1
                                                station = pos_sta[i]
                                                r.del_node(self.model.vehicle, node_index)
                                            else:
                                                r.del_node(self.model.vehicle, node_index)
                                    if station is not None:
                                        r.add_node(self.model.vehicle, node_index, station)
                                        if r.feasible(self.model.vehicle)[0]:
                                            o=True
                                            pass
                                        else:
                                            r.del_node(self.model.vehicle, node_index)
                        # 尝试减少充电桩
                        r.find_charge_station()
                        if len(r.rechargers) > 0:
                            r.cal_remain_battery
                            value = VNS_TS.get_objective_route(r, self.model.vehicle, self.penalty)
                            sc = np.zeros(len(r.rechargers))
                            for i in range(len(r.rechargers)):
                                if i != (len(r.rechargers)) - 1:
                                    sc[i] = np.sum(r.arrive_remain_battery[r.rechargers[i]:r.rechargers[i + 1]] > 0)
                                else:
                                    if r.arrive_remain_battery[-1] >= 0 and len(sc) >= 2:
                                        sc[i] = math.ceil(np.mean(sc[0:-1]))
                                    else:
                                        sc[i] = np.sum(r.arrive_remain_battery[r.rechargers[i]:] > 0)
                            node_index = Util.wheel_select(len(r) / (sc + 1))
                            station = r.visit[r.rechargers[node_index]]
                            index_sta = r.rechargers[node_index]
                            r.del_node(self.model.vehicle, index_sta)
                            o1=r.feasible(self.model.vehicle)[0]
                            if (value > VNS_TS.get_objective_route(r, self.model.vehicle, self.penalty)):
                                if not(o==1 and o1==0):
                                    o=r.feasible(self.model.vehicle)[0]
                                    pass
                                else:
                                    r.add_node(self.model.vehicle, index_sta, station)
                            else:
                                if (o==0 and o1==1):
                                    o=1
                                    pass
                                else:
                                    r.add_node(self.model.vehicle, index_sta, station)
                        if not r.feasible_battery(self.model.vehicle)[0]:
                            #   StationInRe_new
                            value = VNS_TS.get_objective_route(r, self.model.vehicle, self.penalty)
                            r.cal_remain_battery(self.model.vehicle)
                            left_fail_index = np.where(r.arrive_remain_battery < 0)[0][0]
                            battery = r.arrive_remain_battery - r.arrive_remain_battery[left_fail_index]
                            right_over_index = np.where(battery >= self.model.vehicle.max_battery)[0][-1]
                            cusgroup=r.visit[right_over_index:left_fail_index + 1]
                            jl = [sum([c.distance_to(s) for c in cusgroup if isinstance(c,(Customer,Depot))]) for s in self.model.rechargers]
                            jl=np.array(jl)
                            jl[jl>np.quantile(jl,0.3)]=1000000
                            station = self.model.rechargers[Util.wheel_select(1 / (jl+1))]
                            #station = self.model.rechargers[Util.wheel_select(np.array([s.distance_to(r.visit[left_fail_index]) for s in self.model.rechargers]))]
                            pos_insert_location = []
                            idis=[]
                            for i in range(1, len(r)):
                                if self.possible_arc.get((station, r.visit[i]), 0):
                                    pos_insert_location.append(i)
                                    idis.append(station.distance_to(r.visit[i-1])+station.distance_to(r.visit[i])-r.visit[i-1].distance_to(r.visit[i]))
                            if len(pos_insert_location) != 0:
                                insert_location = random.choice(pos_insert_location)
                                if r.visit[insert_location - 1] == station:
                                    #insert_location = random.choice(pos_insert_location)
                                    r.del_node(self.model.vehicle, insert_location - 1)
                                    o1 = r.feasible(self.model.vehicle)[0]
                                    value1 = VNS_TS.get_objective_route(r, self.model.vehicle, self.penalty)
                                    if value > value1:
                                        if not (o==1 and o1==0):
                                            o=r.feasible(self.model.vehicle)[0]
                                            pass
                                        else:
                                            r.add_node(self.model.vehicle, insert_location - 1, station)
                                    else:
                                        if (o==0 and o1==1):
                                            o=1
                                            pass
                                        else:
                                            r.add_node(self.model.vehicle, insert_location - 1, station)
                                else:
                                    idis = np.array(idis)
                                    if np.sum(idis<0)!=0:
                                        print(idis)
                                        print(station)
                                        print(r)
                                        p=np.where(idis<0)[0][0]
                                        print(r.visit[p],r.visit[p+1])
                                    insert_location = pos_insert_location[Util.wheel_select(1 / (idis + 1))]
                                    r.add_node(self.model.vehicle, insert_location, station)
                                    o1 = r.feasible(self.model.vehicle)[0]
                                    value1 = VNS_TS.get_objective_route(r, self.model.vehicle, self.penalty)
                                    if value > value1:
                                        if not (o == 1 and o1 == 0):
                                            o = r.feasible(self.model.vehicle)[0]
                                            pass
                                        else:
                                            r.del_node(self.model.vehicle, insert_location)
                                    else:
                                        if (o==0 and o1==1):
                                            o=1
                                            pass
                                        else:
                                            r.del_node(self.model.vehicle, insert_location)
                        max_try+=1
                    new_routes.append(r)
                else:
                    new_routes.append(r)
            else:
                new_routes.append(r)
        S=Solution(new_routes)
        assert S.serve_all_customer(self.model)
        return S

    def extra_repair(self,solution:Solution):
        #s=solution.copy()
        max_iteration=5
        solution=solution.copy()
        if len(solution.feasible_detail(self.model))>len(solution)/3:
            return solution
        max_try=min(max_iteration,len(solution.feasible_detail(self.model)))
        my_try=0
        fes_count = sum([r.feasible(self.model.vehicle)[0] for r in solution.routes])
        value = VNS_TS.get_objective(solution, self.model, self.penalty)
        while (fes_count!=len(solution)) and (my_try<=max_try):
            infes_r=[]
            for j in range(len(solution)):
                if not solution[j].feasible(self.model.vehicle)[0]:
                    infes_r.append(j)
            i=random.choice(infes_r)
            c,t,b=100,100,100
            if not solution[i].feasible_capacity(self.model.vehicle)[0]:
                c=solution[i].feasible_capacity(self.model.vehicle)[1]
            if not solution[i].feasible_time(self.model.vehicle)[0]:
                t=solution[i].feasible_time(self.model.vehicle)[1]
            if not solution[i].feasible_battery(self.model.vehicle)[0]:
                b=solution[i].feasible_battery(self.model.vehicle)[1]
                while isinstance(solution[i].visit[b],(Recharger,Depot)):
                    b-=1
            infes_index=min(c,t,b)
            infes_node=solution[i].visit[infes_index]
            search_list=[]
            for key in self.possible_arc.keys():
                if key[0]==infes_node:
                    search_list.append((key,self.possible_arc[key]))
                elif key[1]==infes_node:
                    search_list.append((key,self.possible_arc[key]))
            search_list.sort(key=lambda x:x[1])
            search_list=search_list[0:max(round(len(search_list)/10),1)]
            opt = [Modification.exchange_arc, Modification.two_opt_star_arc, Modification.relocate_arc]
            yyh=0
            for arc,_ in search_list:
                if arc[0] != infes_node:
                    opt1=opt[0:2]
                else:
                    opt1=[opt[2]]
                for neighbor_opt in opt1:
                    neighbor_sol, _ = neighbor_opt(self.model, solution, *arc)
                    for sol in neighbor_sol:
                        assert sol.serve_all_customer(self.model)
                    for sol in neighbor_sol:
                        if sol.feasible(self.model):
                            #print('!')
                            return sol
                        else:
                            sol_fes_count=sum([r.feasible(self.model.vehicle)[0] for r in sol.routes])
                            if sol_fes_count>fes_count:
                                solution=sol.copy()
                                fes_count=sol_fes_count
                                value=VNS_TS.get_objective(solution,self.model,self.penalty)
                                yyh=1
                                break
                            else:
                                if sol_fes_count==fes_count:
                                    value1=VNS_TS.get_objective(sol,self.model,self.penalty)
                                    if value1<value:
                                        solution=sol.copy()
                                        value = VNS_TS.get_objective(solution, self.model, self.penalty)
                    if yyh:
                        break
                if yyh:
                    break
            my_try+=1
        return solution

    def GA(self,P:list):
        cross_score = [0.0, 0.0]
        cross_call_times = [0, 0]
        cross_weigh = [0.0, 0.0]
        fes_P = []
        infes_P = []
        for sol in P:
            if sol.feasible(self.model):
                fes_P.append(sol)
            else:
                infes_P.append(sol)
        fes_P.sort(key=lambda sol:(len(sol),VNS_TS.get_objective(sol, self.model, self.penalty)))
        infes_P.sort(key=lambda sol:VNS_TS.get_objective(sol, self.model, self.penalty))
        #obj_value = []
        #for sol in infes_P:
        #    overlapping_degree = DEMA.overlapping_degree_population(sol, P)
        #    objective = DEMA.get_objective(sol, self.model, self.penalty)
        #    obj_value.append([objective, overlapping_degree])
        #infes_P = Util.pareto_sort(infes_P, obj_value)

        P = fes_P + infes_P
        choose = Util.binary_tournament(len(P))
        P_parent = []
        for i in choose:
            P_parent.append(P[i])
        P_child = []
        all_cost = [VNS_TS.get_objective(sol, self.model, self.penalty) for sol in P]
        while len(P_child) < self.popsize:
            for i in range(2):
                if cross_call_times[i] != 0:
                    cross_weigh[i] = self.theta * np.pi / cross_call_times[i] + (1 - self.theta) * cross_weigh[i]
            if cross_weigh[0] == 0 and cross_weigh[1] == 0:
                sel_prob = np.array([0.5, 0.5])
            else:
                sel_prob = np.array(cross_weigh) / np.sum(np.array(cross_weigh))
            sel = Util.wheel_select(sel_prob)

            if sel == 0:
                S_parent = random.choice(P_parent)
                S = Modification.ACO_GM_cross1(S_parent, self.model)
                assert S.serve_all_customer(self.model)
            elif sel == 1:
                S_parent, S2 = random.sample(P_parent, 2)
                S = Modification.ACO_GM_cross2(S_parent, S2, self.model)
                assert S.serve_all_customer(self.model)

            cross_call_times[sel] += 1
            cost = DEMA.get_objective(S, self.model, self.penalty)
            if cost < all(all_cost):
                cross_score[sel] += self.sigma[0]
            elif cost < DEMA.get_objective(S_parent, self.model, self.penalty):
                cross_score[sel] += self.sigma[1]
            else:
                cross_score[sel] += self.sigma[2]

            P_child.append(S)
        return P_child

    def ISSD(self, P: list, iter: int) -> list:
        SP1 = []
        SP2 = []
        infeasible_proportion=self.infeasible_proportion-iter*self.infeasible_proportion/self.max_iteration
        for sol in P:
            if sol.feasible(self.model):
                SP1.append(sol)
            else:
                SP2.append(sol)
        SP1.sort(key=lambda sol: (len(sol),DEMA.get_objective(sol, self.model, self.penalty)))
        obj_value = []
        for sol in SP2:
            overlapping_degree = DEMA.overlapping_degree_population(sol, P)
            objective = DEMA.get_objective(sol, self.model, self.penalty)
            obj_value.append([objective, overlapping_degree])
        SP2 = Util.pareto_sort(SP2, obj_value)
        sp1up = int((1-infeasible_proportion)*self.popsize)
        sp2up = self.popsize-sp1up
        P = SP1[:sp1up]+SP2[:sp2up]
        SP1 = SP1[sp1up:]
        SP2 = SP2[sp2up:]
        for sol in SP1:
            if len(P) < self.popsize:
                P.append(sol)
        for sol in SP2:
            if len(P) < self.popsize:
                P.append(sol)
        assert len(P) == self.popsize

        # for sol in P:
        #    sol.clear_status()

        return P

    def main(self):
        P=self.ini()
        self.P=P
        self.update_S(P)
        for iter in range(self.max_iteration):
            self.ls.itera=iter
            popchild=self.GA(P)
            repair_num_in=0
            repair_num_ex=0
            feas_num=0
            for i in range(len(popchild)):
                if not popchild[i].feasible(self.model):
                    for _ in range(self.max_repair_in):
                        popchild[i]=self.repair(popchild[i])
                        if popchild[i].feasible(self.model):
                            repair_num_in+=1
                            break
                    else:
                        popchild[i]=self.extra_repair(popchild[i])
                        if popchild[i].feasible(self.model):
                            repair_num_ex+=1
                else:
                    feas_num+=1
            print("feas_num:",feas_num,' ','repair_num_in:',repair_num_in,' ','repair_num_ex:',repair_num_ex)
            P=self.ISSD(P+popchild,iter)
            #self.update_S(P)
            #self.P = P
            #continue
            #print('!')
            for i in range(len(P)):
                s,o1,o2=self.ls.main(P[i])
                P[i]=s.copy()
                print(i,s.feasible(self.model),o1,o2)
            self.update_S(P)
            self.P=P
            print('当前代数:',iter,self.S_best.feasible(self.model),len(self.S_best),self.min_cost)
        self.rw_result()
        print('模型:'+self.model.data_file+'已完成')
        return

class LS(Evolution):
    model=None
    penalty=None   #cap,time,battery
    max_iter=40
    S_best=None
    max_extra_search=10
    possible_arc = {}
    eta_feas = 10
    eta_dist = 40
    Delta_SA = 0.08
    max_ini_routes=100
    rdic=None
    penalty_update_flag=None
    eta_penalty=2
    penalty_min=(0.5,0.5,0.5)
    penalty_max=(5000,5000,5000)
    delta=1.2
    with_DEMA=False
    S_jj=None
    S_out=None
    itera=None
    popsize=1
    DEMA_max_iter=1
    jjcs1=10
    jjcs2=20

    def __init__(self,model:Model,dic_file,with_DEMA):
        self.model=model
        if os.path.exists(dic_file+'.npy'):
            self.rdic=np.load(dic_file+'.npy',allow_pickle=True)
            self.rdic=self.rdic.item()
        else:
            self.rdic={}
        if not with_DEMA:
            #self.SA_dist = Util.SA(self.Delta_SA, self.eta_dist)
            self.SA_feas = Util.SA(self.Delta_SA, self.eta_feas)
            self.calculate_possible_arc()
            for key in self.rdic.keys():
                if (self.rdic[key][0]) and (self.rdic[key][1]>self.jjcs1):
                    self.rdic[key][1] = self.jjcs1
                elif (not self.rdic[key][0]) and (self.rdic[key][1]>self.jjcs2):
                    self.rdic[key][1] = self.jjcs2

        self.S_jj=None
        self.S_out = None
        self.penalty_update_flag = [collections.deque(maxlen=self.eta_penalty),
                                    collections.deque(maxlen=self.eta_penalty),
                                    collections.deque(maxlen=self.eta_penalty)]

    def calculate_possible_arc(self) -> None:
        self.model.find_nearest_station()
        all_node_list = [self.model.depot]+self.model.rechargers+self.model.customers
        for node1 in all_node_list:
            for node2 in all_node_list:
                if (isinstance(node1, Depot) and isinstance(node2, Recharger) and (node1.x == node2.x and node1.y == node2.y)) or (isinstance(node1, Recharger) and isinstance(node2, Depot) and (node1.x == node2.x and node1.y == node2.y)):
                    continue
                if not node1 == node2:
                    distance = node1.distance_to(node2)
                    if isinstance(node1, Customer) and isinstance(node2, Customer) and (node1.demand+node2.demand) > self.model.vehicle.capacity:
                        continue
                    if node1.ready_time+node1.service_time+distance > node2.over_time:
                        continue
                    if node1.ready_time+node1.service_time+distance+node2.service_time+node2.distance_to(self.model.depot) > self.model.depot.over_time:
                        continue
                    if len(self.model.rechargers) != 0 and isinstance(node1, Customer) and isinstance(node2, Customer):
                        recharger1 = self.model.nearest_station[node1][0]
                        recharger2 = self.model.nearest_station[node2][0]
                        if self.model.vehicle.battery_cost_speed*(node1.distance_to(recharger1)+distance+node2.distance_to(recharger2)) > self.model.vehicle.max_battery:
                            continue
                    if distance == 0:
                        distance = 0.0000001
                    self.possible_arc[(node1, node2)] = distance

    def select_possible_arc(self, solution:Solution, N: int) -> list:
        not_select=[]
        for i in range(len(solution)):
            for j in range(len(solution.routes[i])-1):
                not_select.append((solution[i].visit[j],solution[i].visit[j+1]))
        selected_arc = []
        keys = list(self.possible_arc.keys())
        while len(keys) > 0 and len(selected_arc) < N:
            values = [self.possible_arc[key] for key in keys]
            values = np.array(values)
            values = 1/values
            select = Util.wheel_select(values)
            if keys[select] not in not_select:
                selected_arc.append(keys[select])
                del keys[select]
            else:
                del keys[select]
        return selected_arc

    def compare_better(self, solution1: Solution, solution2: Solution) -> bool:
        if solution2 is None:
            return True
        s1_val = VNS_TS.get_objective(solution1, self.model, self.penalty)
        s2_val = VNS_TS.get_objective(solution2, self.model, self.penalty)
        if solution1.feasible(self.model) and solution2.feasible(self.model):
            if len(solution1) < len(solution2) or (len(solution1) == len(solution2) and s1_val < s2_val):
                return True
        elif solution1.feasible(self.model) and not solution2.feasible(self.model):
            return True
        elif not solution1.feasible(self.model) and solution2.feasible(self.model):
            return False
        elif not solution1.feasible(self.model) and not solution2.feasible(self.model):
            if s1_val < s2_val:
                return True
            else:
                return False

    def main_extra(self,solution:Solution,iteration:int):
        best_S = solution
        s=solution.copy()
        local_best_S = s
        local_value=VNS_TS.get_objective(local_best_S,self.model,self.penalty)
        opt = [Modification.two_opt_star_arc, Modification.relocate_arc, Modification.exchange_arc,
             Modification.stationInRe_arc, Modification.time_flow, Modification.connection_two_routes]
        for i in range(self.max_extra_search):
            select_arc = self.select_possible_arc(local_best_S, round(3*len(self.possible_arc)/(self.max_iter*self.max_extra_search*self.DEMA_max_iter)))
            for arc in select_arc:
                for neighbor_opt in opt:
                    if neighbor_opt not in [Modification.time_flow, Modification.connection_two_routes]:
                        neighbor_sol,_ = neighbor_opt(self.model, local_best_S, *arc)
                        for sol in neighbor_sol:
                            assert sol.serve_all_customer(self.model)
                        for sol in neighbor_sol:
                            value=VNS_TS.get_objective(sol,self.model,self.penalty)
                            if not self.model.file_type=='p':
                                if (random.random()<self.SA_feas.probability(value,local_value,iteration)) or ((len(sol)<len(local_best_S)) and sol.feasible(self.model)):
                                    local_best_S = sol.copy()
                                    local_value=value
                            else:
                                if self.compare_better(sol,local_best_S):
                                    local_best_S = sol.copy()
                                    local_value = value
                                elif (random.random() < self.SA_feas.probability(value, local_value, iteration)) and sol.feasible(self.model):
                                    local_best_S = sol.copy()
                                    local_value = value
                    else:
                        neighbor_sol = neighbor_opt(self.model, local_best_S, self)
                        assert neighbor_sol.serve_all_customer(self.model)
                        if self.compare_better(neighbor_sol,local_best_S):
                            local_best_S = neighbor_sol.copy()
                            local_value = VNS_TS.get_objective(neighbor_sol, self.model, self.penalty)
            if self.compare_better(local_best_S, best_S):
                best_S = local_best_S
        return best_S

    def check_input_main_extra(self,solution:Solution):
        solution=solution.copy()
        yyh=0
        for i in range(len(solution)):
            route=solution[i].copy()
            r=Route([j for j in route.visit if isinstance(j,(Customer,Depot))])
            #r = Modification.fix_time_cap(self.model, r)
            if not (r.feasible_time(self.model.vehicle)[0] and r.feasible_capacity(self.model.vehicle)[0]):
                r = self.fix_time_cap(self.model, r)
                if not (r.feasible_time(self.model.vehicle)[0] and r.feasible_capacity(self.model.vehicle)[0]):
                    return False,None
                else:
                    rl=solution.routes.copy()
                    p=rl.index(route)
                    del rl[p]
                    rl.insert(p,r)
                    solution=Solution(rl)
                    yyh=1
        if yyh:
            return True,solution
        else:
            return True,solution

    def main_inter(self,solution:Solution):
        penalty=list(self.penalty)
        solution=solution.copy()
        new_routes=[]
        for route in solution.routes:
            route=route.copy()#
            r_onlycusid=[cus.id for cus in route.visit if isinstance(cus,Customer)]
            r_onlycusid.sort()
            r_onlycusid=tuple(r_onlycusid)
            kg=0
            if self.rdic.get(r_onlycusid,0):
                if self.rdic.get(r_onlycusid)[0] and (self.rdic.get(r_onlycusid)[1]>0):
                    new_routes.append(self.rdic.get(r_onlycusid)[0])
                    self.rdic.get(r_onlycusid)[1]-=1
                    continue
                else:
                    if self.rdic.get(r_onlycusid)[0] and (self.rdic.get(r_onlycusid)[1]==0):
                        kg=1
                        pass
                    elif self.rdic.get(r_onlycusid)[1]==0:
                        pass
                    else:
                        new_routes.append(self.rdic.get(r_onlycusid)[2])
                        self.rdic.get(r_onlycusid)[1] -= 1
                        continue
            jjb = []
            if not kg:
                routelist=self.try_diffseq(self.model,route,jjb,self.rdic.get(r_onlycusid,0))
            else:
                routelist = self.try_diffseq(self.model, route, jjb,0)
                routelist.append(self.rdic.get(r_onlycusid)[0])
            routelist_elecase = [None] * len(routelist)
            score=1000000*np.ones(len(routelist))
            for i in range(len(routelist)):
                pos_route=routelist[i].copy()#
                after_elecase=self.SR(self.model,pos_route)
                routelist_elecase[i]=after_elecase.copy()#
                score[i]=VNS_TS.get_objective_route(after_elecase,self.model.vehicle,penalty)
            best_route=routelist_elecase[np.where(score==min(score))[0][0]]
            if best_route.feasible(self.model.vehicle)[0]:
                new_routes.append(best_route)
                idlist=[cus.id for cus in best_route.visit if isinstance(cus,Customer)]
                idlist.sort()
                idlist=tuple(idlist)
                self.rdic[idlist]=[best_route,self.jjcs1]
                #self.rdic[tuple([cus.id for cus in best_route.visit if isinstance(cus,Customer)].sort())]=(best_route,10)
            else:
                #考虑电耗与载重有关且是MB
                value=VNS_TS.get_objective_route(best_route,self.model.vehicle,penalty)
                yyh=0
                for i in range(len(routelist)):
                    if self.model.file_type in ['e','s5','s10','s15']:
                        routelist_after_fixele=Modification.fix_ele(self.model,routelist[i],10,jjb)
                    else:
                        continue
                    for j in range(len(routelist_after_fixele)):
                        r=routelist_after_fixele[j].copy()#
                        r1 = self.SR(self.model, r)
                        if r1.feasible(self.model.vehicle)[0]:
                            new_routes.append(r1)
                            yyh=1
                            break
                        else:
                            value1=VNS_TS.get_objective_route(r1,self.model.vehicle,penalty)
                            if value>value1:
                                best_route=r1.copy()
                                value=value1
                            jjb.append(r)
                    if yyh:
                        break
                else:
                    idlist=[cus.id for cus in best_route.visit if isinstance(cus, Customer)]
                    idlist.sort()
                    idlist=tuple(idlist)
                    self.rdic[idlist]=[False, self.jjcs2, best_route]
                    new_routes.append(best_route)

        return Solution(new_routes)

    def compare_better(self, solution1: Solution, solution2: Solution) -> bool:
        if solution2 is None:
            return True
        s1_val = VNS_TS.get_objective(solution1, self.model, self.penalty)
        s2_val = VNS_TS.get_objective(solution2, self.model, self.penalty)
        if not self.model.file_type=='p':
            if solution1.feasible(self.model) and solution2.feasible(self.model):
                if len(solution1) < len(solution2) or (len(solution1) == len(solution2) and s1_val < s2_val):
                    # if solution1.get_actual_routes() < solution2.get_actual_routes() or (solution1.get_actual_routes() == solution2.get_actual_routes() and s1_val < s2_val):
                    return True
                else:
                    return False
            elif solution1.feasible(self.model) and not solution2.feasible(self.model):
                return True
            elif not solution1.feasible(self.model) and solution2.feasible(self.model):
                return False
            elif not solution1.feasible(self.model) and not solution2.feasible(self.model):
                if s1_val < s2_val:
                    return True
                else:
                    return False
        else:
            if solution1.feasible(self.model) and solution2.feasible(self.model):
                if s1_val < s2_val:
                    return True
                else:
                    return False
            elif solution1.feasible(self.model) and not solution2.feasible(self.model):
                return True
            elif not solution1.feasible(self.model) and solution2.feasible(self.model):
                return False
            elif not solution1.feasible(self.model) and not solution2.feasible(self.model):
                if s1_val < s2_val:
                    return True
                else:
                    return False

    def random_create(self) -> Solution:
        x = random.uniform(self.model.get_map_bound()[0], self.model.get_map_bound()[1])
        y = random.uniform(self.model.get_map_bound()[2], self.model.get_map_bound()[3])
        choose = self.model.customers[:]
        choose.sort(key=lambda cus: Util.cal_angle_AoB((self.model.depot.x, self.model.depot.y), (x, y), (cus.x, cus.y)))
        routes = []
        building_route_visit = [self.model.depot, self.model.depot]

        choose_index = 0
        while choose_index < len(choose):
            allow_insert_place = list(range(1, len(building_route_visit)))

            while True:
                min_increase_dis = float('inf')
                decide_insert_place = None
                for insert_place in allow_insert_place:
                    increase_dis = choose[choose_index].distance_to(building_route_visit[insert_place-1])+choose[choose_index].distance_to(building_route_visit[insert_place])-building_route_visit[insert_place-1].distance_to(building_route_visit[insert_place])
                    if increase_dis < min_increase_dis:
                        decide_insert_place = insert_place
                if len(allow_insert_place) == 1:
                    break
                elif (isinstance(building_route_visit[decide_insert_place-1], Customer) and isinstance(building_route_visit[decide_insert_place], Customer)) and (building_route_visit[decide_insert_place-1].ready_time <= choose[choose_index].ready_time and choose[choose_index].ready_time <= building_route_visit[decide_insert_place].ready_time):
                    break
                elif (isinstance(building_route_visit[decide_insert_place-1], Customer) and not isinstance(building_route_visit[decide_insert_place], Customer)) and building_route_visit[decide_insert_place-1].ready_time <= choose[choose_index].ready_time:
                    break
                elif (not isinstance(building_route_visit[decide_insert_place-1], Customer) and isinstance(building_route_visit[decide_insert_place], Customer)) and choose[choose_index].ready_time <= building_route_visit[decide_insert_place]:
                    break
                elif not isinstance(building_route_visit[decide_insert_place-1], Customer) and not isinstance(building_route_visit[decide_insert_place], Customer):
                    break
                else:
                    allow_insert_place.remove(decide_insert_place)
                    continue

            building_route_visit.insert(decide_insert_place, choose[choose_index])

            try_route = Route(building_route_visit)
            if try_route.feasible_capacity(self.model.vehicle)[0] and try_route.feasible_battery(self.model.vehicle)[0]:
                del choose[choose_index]
            else:
                if len(routes) < self.max_ini_routes-1:
                    del building_route_visit[decide_insert_place]
                    if len(building_route_visit) == 2:
                        choose_index += 1
                    else:
                        routes.append(Route(building_route_visit))
                        building_route_visit = [self.model.depot, self.model.depot]
                elif len(routes) == self.max_ini_routes-1:
                    del choose[choose_index]

        routes.append(Route(building_route_visit[:-1]+choose+[self.model.depot]))

        return Solution(routes)

    # 尝试通过改变客户顺序修正时间窗约束和容量约束,对于不考虑充电也违背时间或载重约束的路径，经修复后，无充电桩
    def fix_time_cap(self,model: Model, route: Route) -> Route:

        r = Route([i for i in route.visit if isinstance(i, (Customer, Depot))])
        if r.feasible_time(model.vehicle)[0] and r.feasible_capacity(model.vehicle)[0]:
            return route
        elif (not r.feasible_time(model.vehicle)[0]) and r.feasible_capacity(model.vehicle)[0]:
            #r.cal_arrive_time(model.vehicle)
            #p1 = VNS_TS(model)
            max_try = 0
            # overlist=np.where(np.array([cus.over_time for cus in r.visit if cus.isinstance(Customer)])>r.arrive_time)[0]
            while (not r.feasible_time(model.vehicle)[0]) and (not (max_try > len(r) * 10)):
                cus_index = r.feasible_time(model.vehicle)[1]
                cus = r.visit[cus_index]
                if isinstance(cus, Depot):
                    break
                sol = []
                for i in range(cus_index):
                    r1 = r.copy()
                    if i == 0:
                        continue
                    if self.possible_arc.get((r1.visit[i - 1], cus), 0) and self.possible_arc.get((cus, r1.visit[i]), 0):
                        r1.del_node(model.vehicle, cus_index)
                        r1.add_node(model.vehicle, i, cus)
                        if (r1.feasible_time(model.vehicle)[1] != i) and r1.feasible_capacity(model.vehicle)[0]:
                            r1.cal_arrive_time(model.vehicle)
                            sol.append(
                                (r1, len(np.where(np.array([cus.over_time for cus in r1.visit]) > r1.arrive_time)[0])))
                if len(sol) != 0:
                    sol.sort(key=lambda x: x[1],reverse=1)
                    hx = [j[0] for j in sol if j[1] == sol[0][1]]
                    r = random.choice(hx)
                else:
                    return r
                    break
                max_try = max_try + 1
        elif (not r.feasible_capacity(model.vehicle)[0]) and r.feasible_time(model.vehicle)[0]:
            if (np.sum(np.array([i.demand for i in r.visit if i.demand > 0])) > model.vehicle.capacity) or (
                    np.sum(np.array([abs(i.demand) for i in r.visit if i.demand < 0])) > model.vehicle.capacity):
                pass
            else:
                max_try = 0
                while (not r.feasible_capacity(model.vehicle)[0]) and (not (max_try > len(r) * 10)):
                    cus_index = r.feasible_capacity(model.vehicle)[1]
                    cus = r.visit[cus_index]  # 该客户必为pickup客户
                    sol = []
                    for i in range(cus_index + 2, len(r)):
                        r1 = r.copy()
                        r1.cal_load_weight(model.vehicle)
                        load1 = r1.arrive_load_weight[cus_index - 1]
                        varyload = np.sum(np.array([j.demand for j in r1.visit[cus_index:i]]))
                        if self.possible_arc.get((r1.visit[i - 1], cus), 0) and self.possible_arc.get((cus, r1.visit[i]),
                                                                                                  0) and (
                                varyload >= -(model.vehicle.capacity - load1)):
                            r1.add_node(model.vehicle, i, cus)
                            r1.del_node(model, cus_index)
                            if (r1.feasible_capacity(model.vehicle)[1] != (i - 1)) and r1.feasible_time(model.vehicle)[
                                0]:
                                r1.cal_load_weight(model.vehicle)
                                sol.append((r1, np.sum(r1.arrive_load_weight > model.vehicle.capacity)))
                    if len(sol) != 0:
                        sol.sort(key=lambda x: x[1])
                        hx = [j[0] for j in sol if j[1] == sol[0][1]]
                        r = random.choice(hx)
                    else:
                        return r
                        break
                    max_try = max_try + 1
        else:
            if (np.sum(np.array([i.demand for i in r.visit if i.demand > 0])) > model.vehicle.capacity) or (
                    np.sum(np.array([abs(i.demand) for i in r.visit if i.demand < 0])) > model.vehicle.capacity):
                pass
            else:
                max_try = 0
                while not ((r.feasible_capacity(model.vehicle)[0] and r.feasible_time(model.vehicle)[0]) or (
                        max_try > len(r) * 10)):
                    cus_index1 = r.feasible_time(model.vehicle)[1]
                    cus_index2 = r.feasible_capacity(model.vehicle)[1]
                    if (cus_index1 is None) or (cus_index2 is None):
                        r=self.fix_time_cap(self.model,r.copy())
                        break
                    if cus_index1 == cus_index2:
                        break
                    cus_index = min(cus_index1, cus_index2)
                    if cus_index == cus_index1:  # 修复时间窗
                        cus = r.visit[cus_index]
                        if isinstance(cus,Depot):
                            break
                        sol = []
                        for i in range(cus_index):
                            r1 = r.copy()
                            if i == 0:
                                continue
                            if self.possible_arc.get((r1.visit[i - 1], cus), 0) and self.possible_arc.get(
                                    (cus, r1.visit[i]), 0):
                                r1.del_node(model.vehicle, cus_index)
                                r1.add_node(model.vehicle, i, cus)
                                if (r1.feasible_time(model.vehicle)[1] != i):
                                    r1.cal_arrive_time(model.vehicle)
                                    sol.append((r1, len(
                                        np.where(np.array([cus.over_time for cus in r1.visit]) > r1.arrive_time)[0])))
                        if len(sol) != 0:
                            sol.sort(key=lambda x: x[1],reverse=1)
                            hx = [j[0] for j in sol if j[1] == sol[0][1]]
                            r = random.choice(hx)
                        else:
                            return r
                            break
                        max_try = max_try + 1

                    else:  # 修复载重约束
                        cus = r.visit[cus_index]  # 该客户必为pickup客户
                        sol = []
                        for i in range(cus_index + 2, len(r)):
                            r1 = r.copy()
                            r1.cal_load_weight(model.vehicle)
                            load1 = r1.arrive_load_weight[cus_index - 1]
                            varyload = np.sum(np.array([j.demand for j in r1.visit[cus_index:i]]))
                            if self.possible_arc.get((r1.visit[i - 1], cus), 0) and self.possible_arc.get(
                                    (cus, r1.visit[i]), 0) and (varyload >= -(model.vehicle.capacity - load1)):
                                r1.add_node(model.vehicle, i, cus)
                                r1.del_node(model, cus_index)
                                if (r1.feasible_capacity(model.vehicle)[1] != (i - 1)):
                                    r1.cal_load_weight(model.vehicle)
                                    sol.append((r1, np.sum(r1.arrive_load_weight > model.vehicle.capacity)))
                        if len(sol) != 0:
                            sol.sort(key=lambda x: x[1])
                            hx = [j[0] for j in sol if j[1] == sol[0][1]]
                            r = random.choice(hx)
                        else:
                            return r
                        max_try = max_try + 1
        return r

    # 使用局部算子尝试其他可行（cap、time）的客户顺序，产生以路径为元素的列表，并按照路径距离排序
    def try_diffseq(self, model: Model, route: Route, jjb:list, k):
        para_s=1
        para_r=1.05
        if route.feasible(model.vehicle)[0] and route.feasible_capacity(model.vehicle)[0]:
            if not k:
                maxc = 1
            elif k[0]:
                maxc=1
            else:
                maxc=5
            r = Route([i for i in route.visit if isinstance(i, (Customer, Depot))])
            if len(r)==3:
                return [r]
            jl = r.sum_distance()
            if route.feasible_battery(model.vehicle)[0]:
                sol=[(route.copy(),route.sum_distance())]
            else:
                sol = [(r, jl)]
            jjb.append(r)
            it = 0
            max_try = len(r) * 5  # 设置连续未"优化"则停止的代数
            #p1 = LS(model)
            # pos_arc = list(p1.possible_arc.keys())
            while it < max_try:
                #   swap,relocate
                if len(sol)>=maxc:
                    break
                r1 = r.copy()
                jl1=1000000
                nodelist = random.sample(r1.visit[1:-1], 2)
                index1 = r1.visit.index(nodelist[0])
                index2 = r1.visit.index(nodelist[1])
                if abs(index1-index2)!=1:
                    if self.possible_arc.get((r1.visit[index1 - 1], nodelist[1]), 0) and self.possible_arc.get(
                            (nodelist[1], r1.visit[index1 + 1]), 0) \
                            and self.possible_arc.get((r1.visit[index2 - 1], nodelist[0]), 0) and self.possible_arc.get(
                        (nodelist[0], r1.visit[index1 + 1]), 0):
                        newv = r1.visit.copy()
                        newv[index2], newv[index1] = nodelist[0], nodelist[1]
                        r1 = Route(newv)
                        if r1 not in jjb:
                            if r1.feasible_time(model.vehicle)[0] and r1.feasible_capacity(model.vehicle)[0]:
                                jl1 = r1.sum_distance()
                                if jl1 < para_s*jl:
                                    #r = r1.copy()
                                    sol.append((r1, jl1))
                                    jjb.append(r1)
                                    it = 0
                                    #continue
                                else:
                                    r1=r.copy()
                            else:
                                r1=r.copy()
                        else:
                            r1 = r.copy()
                else:
                    if self.possible_arc.get((r1.visit[min(index1,index2)-1],r1.visit[max(index1,index2)]),0) and \
                            self.possible_arc.get((r1.visit[min(index1,index2)],r1.visit[max(index1,index2)+1]),0) and \
                            self.possible_arc.get((r1.visit[max(index1, index2)], r1.visit[min(index1, index2)]), 0):
                        newv = r1.visit.copy()
                        newv[index2], newv[index1] = nodelist[0], nodelist[1]
                        r1 = Route(newv)
                        if r1 not in jjb:
                            if r1.feasible_time(model.vehicle)[0] and r1.feasible_capacity(model.vehicle)[0]:
                                jl1 = r1.sum_distance()
                                if jl1 < para_s * jl:
                                    # r = r1.copy()
                                    sol.append((r1, jl1))
                                    jjb.append(r1)
                                    it = 0
                                    # continue
                                else:
                                    r1 = r.copy()
                            else:
                                r1 = r.copy()
                        else:
                            r1 = r.copy()
                node = random.choice(r1.visit[1:-1])
                index3 = r1.visit.index(node)
                for i in range(len(r1)):
                    if len(sol) >= maxc:
                        break
                    else:
                        r2 = r1.copy()
                        if (i == 0) or (i == index3) or (i == (index3 + 1)):
                            continue
                        if self.possible_arc.get((r2.visit[i - 1], node), 0) and self.possible_arc.get((node, r2.visit[i]),
                                                                                                   0):
                            if i < index3:
                                r2.del_node(model.vehicle, index3)
                                r2.add_node(model.vehicle, i,node)
                            else:
                                r2.add_node(model.vehicle, i,node)
                                r2.del_node(model.vehicle, index3)
                            if r2 not in jjb:
                                if r2.feasible_time(model.vehicle)[0] and r2.feasible_capacity(model.vehicle)[0]:
                                    jl1 = r2.sum_distance()
                                    if jl1 < para_r*jl:
                                        #r = r1.copy()
                                        sol.append((r2, jl1))
                                        jjb.append(r2)
                                        it = 0
                if jl1<jl:
                    r=r1.copy()
                it+=1
            #sol.sort(key=lambda x: x[1])
            return [i[0] for i in sol]
        else:
            return [route]

    # 为一个确定的客户顺序考虑最优的充电方案
    def SR(self,model: Model, route: Route):
        if route.feasible(model.vehicle)[0]:  # 路径可行，且无充电桩
            route.find_charge_station()
            if len(route.rechargers) == 0:
                return route
        if (not route.feasible_battery(model.vehicle)[0]) and route.feasible_capacity(
                model.vehicle)[0] and route.feasible_time(model.vehicle)[0]:
            # 只是不满足电量约束
            route.find_charge_station()
            if len(route.rechargers) == 0:  # 只是不满足电量约束，且无充电桩
                r = route.copy()
                while not r.feasible(model.vehicle)[0]:
                    r.cal_remain_battery(model.vehicle)
                    left_fail_index = np.where(r.arrive_remain_battery < 0)[0][0]  # 第一个缺电点索引
                    battery = r.arrive_remain_battery - r.arrive_remain_battery[left_fail_index]
                    right_over_index = np.where(battery >= model.vehicle.max_battery)[0][-1]
                    right_insert = list(range(right_over_index + 1, left_fail_index + 1))
                    choose = []
                    for i in list(reversed(right_insert)):
                        choose.append((i, model.find_near_station_between(r.visit[i], r.visit[i - 1])))
                    for pair in choose:
                        r.add_node(model.vehicle, pair[0], pair[1])
                        if r.feasible_battery(model.vehicle)[0] & r.feasible_time(model.vehicle)[0]:
                            break
                        elif (not r.feasible_battery(model.vehicle)[0]) & r.feasible_time(model.vehicle)[0]:
                            r.cal_remain_battery(model.vehicle)
                            if (r.arrive_remain_battery[left_fail_index + 1] >= 0) & (
                                    r.arrive_remain_battery[pair[0]] >= 0):
                                break  # 继续while
                            else:
                                r.del_node(model.vehicle, pair[0])
                        else:
                            r.del_node(model.vehicle, pair[0])
                    else:
                        break
                route = r.copy()
        r = route.copy()
        o=r.feasible_battery(model.vehicle)[0]
        for _ in range(len(route) * 1):
            r.find_charge_station()
            # 尝试增加充电桩
            if not r.feasible_battery(model.vehicle)[0]:
                r.cal_arrive_time(model.vehicle)
                #wait_time = np.array([j.over_time for j in r.visit]) - r.arrive_time
                wait_time = np.array([j.ready_time for j in r.visit]) - r.arrive_time
                wait_time=Util.map_100(wait_time)
                wait_time[0]=1/100000
                if True:
                    #wt = np.zeros(len(wait_time))
                    #for i in range(len(wait_time)):
                        #wt[i] = max(min(wait_time[i:]), 0)
                    #node_index = Util.wheel_select(wt)
                    node_index = Util.wheel_select(wait_time)
                    if node_index != 0:
                        pos_sta = Modification.find_feasible_station_between(model, r.visit[node_index - 1],
                                                                             r.visit[node_index],self)
                        value = VNS_TS.get_objective_route(r, model.vehicle, self.penalty)

                        station = None
                        for i in range(len(pos_sta)):
                            r.add_node(model.vehicle, node_index, pos_sta[i])
                            o1=r.feasible(model.vehicle)[0]
                            value1 = VNS_TS.get_objective_route(r, model.vehicle, self.penalty)
                            if value > value1:
                                if not (o==1 and o1==0):
                                    value = value1
                                    station = pos_sta[i]
                                    r.del_node(model.vehicle, node_index)
                                else:
                                    r.del_node(model.vehicle, node_index)
                            else:
                                if (o==0 and o1==1):
                                    value = value1
                                    station = pos_sta[i]
                                    r.del_node(model.vehicle, node_index)
                                    pass
                                else:
                                    r.del_node(model.vehicle, node_index)
                        if station is not None:
                            r.add_node(model.vehicle, node_index, station)
                            if r.feasible(model.vehicle)[0]:
                                o=True
                                pass
                            else:
                                r.del_node(model.vehicle, node_index)
            # 尝试减少充电桩
            r.find_charge_station()
            if len(r.rechargers)>0:
                r.cal_remain_battery
                value = VNS_TS.get_objective_route(r, model.vehicle, self.penalty)
                sc = np.zeros(len(r.rechargers))
                for i in range(len(r.rechargers)):
                    if i != (len(r.rechargers)) - 1:
                        sc[i] = np.sum(r.arrive_remain_battery[r.rechargers[i]:r.rechargers[i + 1]] > 0)
                    else:
                        if r.arrive_remain_battery[-1] >= 0 and len(sc) >= 2:
                            sc[i] = math.ceil(np.mean(sc[0:-1]))
                        else:
                            sc[i] = np.sum(r.arrive_remain_battery[r.rechargers[i]:] > 0)
                node_index = Util.wheel_select(len(r) / (sc + 1))
                station = r.visit[r.rechargers[node_index]]
                index_sta=r.rechargers[node_index]
                r.del_node(model.vehicle, index_sta)
                o1=r.feasible(model.vehicle)[0]
                if value > VNS_TS.get_objective_route(r, model.vehicle, self.penalty):
                    if (o==1 and o1==0):
                        r.add_node(model.vehicle, index_sta, station)
                    else:
                        o = r.feasible(model.vehicle)[0]
                        pass
                else:
                    if (o==0 and o1==1):
                        o=1
                    else:
                        r.add_node(model.vehicle, index_sta, station)
            # 尝试替换充电桩
            r.find_charge_station()
            if len(r.rechargers)>0:
                value = VNS_TS.get_objective_route(r, model.vehicle, self.penalty)
                res1 = random.sample(list(r.rechargers), 1)[0]
                resn1 = r.visit[res1]
                r.del_node(model.vehicle, res1)
                # 查找被移除的充电桩周围的充电桩
                zws = [(j, j.distance_to(resn1)) for j in model.rechargers if j != resn1]
                zws.sort(key=lambda j: j[1])
                zws = zws[0:math.ceil(len(model.rechargers) / 5)]
                station = None
                for j in range(len(r) - 1):
                    ss = Modification.find_feasible_station_between(model, r.visit[j], r.visit[j + 1],self)
                    s = list(set(ss) & set(zws))
                    if s:
                        for k in range(len(s)):
                            r.add_node(model.vehicle, j + 1, s[k])
                            o1=r.feasible(model.vehicle)[0]
                            value1 = VNS_TS.get_objective_route(r, model.vehicle, self.penalty)
                            if value > value1:
                                if not (o==1 and o1==0):
                                    value = value1
                                    station = (s[k], r.visit[j], r.visit[j + 2])
                                    r.del_node(model.vehicle, j + 1)
                                else:
                                    r.del_node(model.vehicle, j + 1)
                            else:
                                if (o==0 and o1==1):
                                    value = value1
                                    station = (s[k], r.visit[j], r.visit[j + 2])
                                    r.del_node(model.vehicle, j + 1)
                                else:
                                    r.del_node(model.vehicle, j + 1)
                else:
                    r.add_node(model.vehicle, res1, resn1)
                if station is not None:
                    r.del_node(model.vehicle, res1)
                    if isinstance(station[2], Customer):
                        r.add_node(model.vehicle, r.visit.index(station[2]), station[0])
                        o= r.feasible(model.vehicle)[0]
                    elif isinstance(station[1], Customer):
                        r.add_node(model.vehicle, r.visit.index(station[1]) + 1, station[0])
                        o = r.feasible(model.vehicle)[0]
                    else:
                        r.add_node(model.vehicle, res1, resn1)
            #   StationInRe_new
            value = VNS_TS.get_objective_route(r, model.vehicle, self.penalty)
            if r.feasible_battery(model.vehicle)[0]:
                point=random.randint(1,len(r)-2)
                length=random.randint(2,len(r)-point)
                cusgroup=r.visit[point:point+length]
                jl=[sum([j.distance_to(i) for j in cusgroup]) for i in model.rechargers]
                station = model.rechargers[Util.wheel_select(1/(np.array(jl)+1))]
            else:
                left_fail_index = np.where(r.arrive_remain_battery < 0)[0][0]
                length = random.randint(1, left_fail_index)
                cusgroup = r.visit[left_fail_index-length:left_fail_index]
                jl = [sum([j.distance_to(i) for j in cusgroup]) for i in model.rechargers]
                station = model.rechargers[Util.wheel_select(1 / (np.array(jl) + 1))]
            pos_insert_location = []
            for i in range(1, len(r)):
                if self.possible_arc.get((station, r.visit[i]), 0):
                    pos_insert_location.append(i)
            if len(pos_insert_location) != 0:
                insert_location = random.choice(pos_insert_location)
                if r.visit[insert_location - 1] == station:
                    r.del_node(model.vehicle, insert_location - 1)
                    o1 = r.feasible(model.vehicle)[0]
                    value1 = VNS_TS.get_objective_route(r, model.vehicle, self.penalty)
                    if value > value1:
                        if not (o==1 and o1==0):
                            o = r.feasible(model.vehicle)[0]
                            pass
                        else:
                            r.add_node(model.vehicle, insert_location - 1, station)
                    else:
                        if (o==0 and o1==1):
                            o = 1
                            pass
                        else:
                            r.add_node(model.vehicle, insert_location - 1, station)
                else:
                    r.add_node(model.vehicle, insert_location, station)
                    o1 = r.feasible(model.vehicle)[0]
                    value1 = VNS_TS.get_objective_route(r, model.vehicle, self.penalty)
                    if value > value1:
                        if not (o == 1 and o1 == 0):
                            o = r.feasible(model.vehicle)[0]
                            pass
                        else:
                            r.del_node(model.vehicle, insert_location)
                    else:
                        if (o==0 and o1==1):
                            o = 1
                        else:
                            r.del_node(model.vehicle, insert_location)

        return r

    def main(self,solution:Solution):
        max_try=15
        max_extra_count=5
        self.S_best=solution.copy()
        for itera in range(self.max_iter):
            kx = 0
            count_not_kx=0
            extra_count=1
            while not kx:
                S_extra=self.main_extra(solution,self.itera)
                #self.S_out=S_extra.copy()
                print('完成径间搜索')
                solution_fix_cap_time=self.check_input_main_extra(S_extra)
                print('check_fix')
                while not solution_fix_cap_time[0]:
                    p = 0
                    if extra_count >= max_extra_count:
                        for i in range(10):
                            S_extra = Modification.fix_time(S_extra, self.model)
                            if S_extra.feasible_capacity(self.model) and S_extra.feasible_time(self.model):
                                solution_fix_cap_time = (True, S_extra)
                                p = 1
                                break
                        else:
                            S_extra.add_empty_route(self.model)
                    if p:
                        break
                    S_extra = self.main_extra(S_extra,self.itera)
                    self.S_out = S_extra.copy()
                    solution_fix_cap_time = self.check_input_main_extra(S_extra)
                    extra_count+=1
                    print('完成径间搜索及check_fix次数',extra_count)
                #self.S_jj=solution_fix_cap_time[1]
                S_inter=self.main_inter(solution_fix_cap_time[1])
                solution = S_inter.copy()
                kx=S_inter.feasible(self.model)
                if kx:
                    if self.compare_better(solution,self.S_best):
                        self.S_best=solution
                else:
                    count_not_kx+=1
                if count_not_kx>max_try:
                    break
        return self.S_best,len(self.S_best),VNS_TS.get_objective(self.S_best,self.model,self.penalty)

    def main1(self,file_dic):   #此函数用于单独跑LS
        solution=self.random_create()
        for itera in range(self.max_iter):
            kx = 0
            extra_count=1
            while not kx:
                S_extra=self.main_extra(solution,itera)
                self.S_out=S_extra.copy()##########
                solution_fix_cap_time=self.check_input_main_extra(S_extra)
                while not solution_fix_cap_time[0]:
                    S_extra = self.main_extra(S_extra,itera)
                    self.S_out = S_extra.copy()#########
                    solution_fix_cap_time = self.check_input_main_extra(S_extra)
                    extra_count+=1
                    print(extra_count)#######
                self.S_jj=solution_fix_cap_time[1]#########
                S_inter=self.main_inter(solution_fix_cap_time[1])
                solution = S_inter.copy()
                kx=S_inter.feasible(self.model)
                if kx:
                    if self.compare_better(solution,self.S_best):
                        self.S_best=solution
                print(kx,itera)
                print('使用车辆数：'+str(len(self.S_best)))
                print('目标值：'+str(VNS_TS.get_objective(self.S_best,self.model,self.penalty)))
                print('径间搜索次数：'+str(extra_count))
        np.save(file_dic+'.npy',self.rdic)
        if os.path.exists('s_'+file_dic+'.npy'):
            s = np.load('s_'+file_dic+'.npy', allow_pickle=1)
            s = Solution(list(s))
            if self.compare_better(self.S_best,s):
                np.save('s_' + file_dic + '.npy', self.S_best)
        else:
            np.save('s_'+file_dic+'.npy',self.S_best)
        return self.S_best,len(self.S_best),VNS_TS.get_objective(self.S_best,self.model,self.penalty)






