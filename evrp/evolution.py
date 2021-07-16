import os
import pickle
import collections

from .model import *
from .util import *
from .operation import *


class VNS_TS:
    # 构造属性
    model = None

    vns_neighbour_Rts = 4
    vns_neighbour_max = 5
    eta_feas = 50
    eta_dist = 20
    Delta_SA = 0.08

    penalty_0 = (10, 10, 10)
    penalty_min = tuple()
    penalty_max = tuple()
    delta = 0.0
    eta_penalty = 0

    nu_min = 0
    nu_max = 0
    lambda_div = 0.0
    eta_tabu = 0
    # 计算属性
    vns_neighbour = []

    def __init__(self, model: Model, **param) -> None:
        self.model = model
        for key, value in param.items():
            assert hasattr(self, key)
            setattr(self, key, value)

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
        if_late = np.where(late_time > 0)[0]
        if len(if_late) > 0:
            return late_time[if_late[0]]
        else:
            return 0.0

    @staticmethod
    def penalty_battery(route: Route, vehicle: Vehicle) -> float:
        if route.arrive_remain_battery is None:
            route.cal_remain_battery(vehicle)
        return np.abs(np.sum(route.arrive_remain_battery, where=route.arrive_remain_battery < 0))

    @staticmethod
    def get_objective_route(route: Route, vehicle: Vehicle, penalty: tuple) -> float:
        return route.sum_distance()+penalty[0]*VNS_TS.penalty_capacity(route, vehicle)+penalty[1]*VNS_TS.penalty_time(route, vehicle)+penalty[2]*VNS_TS.penalty_battery(route, vehicle)

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
            if try_route.feasible_weight(self.model.vehicle) and try_route.feasible_battery(self.model.vehicle):
                del choose[choose_index]
            else:
                if len(routes) < self.model.max_vehicle-1:
                    del building_route_visit[decide_insert_place]
                    if len(building_route_visit) == 2:
                        choose_index += 1
                    else:
                        routes.append(Route(building_route_visit))
                        building_route_visit = [self.model.depot, self.model.depot]
                elif len(routes) == self.model.max_vehicle-1:
                    del choose[choose_index]

        routes.append(Route(building_route_visit[:-1]+choose+[self.model.depot]))

        return Solution(routes)

    def create_vns_neighbour(self, Rts: int, max: int) -> list:
        assert Rts >= 2 and max >= 1
        self.vns_neighbour = []
        for R in range(2, Rts+1):
            for m in range(1, max+1):
                self.vns_neighbour.append((R, m))

    def tabu_search(self, S: Solution, eta_tabu: int) -> Solution:
        return S

    def main(self) -> Solution:
        self.create_vns_neighbour(self.vns_neighbour_Rts, self.vns_neighbour_max)
        S = self.random_create_vnsts()
        k = 0
        i = 0
        feasibilityPhase = True
        acceptSA = Util.SA(self.Delta_SA, self.eta_dist)
        while feasibilityPhase or i < self.eta_dist:
            S1 = Operation.cyclic_exchange(S, *self.vns_neighbour[k])
            S2 = self.tabu_search(S1, self.eta_tabu)
            if random.random() < acceptSA.probability(DEMA.get_objective(S2, self.model, self.penalty_0), DEMA.get_objective(S, self.model, self.penalty_0), i):
                S = S2
                print(i, S)
                print(S.feasible(self.model), S.sum_distance())
                k = 0
            else:
                k = (k+1) % len(self.vns_neighbour)
            if feasibilityPhase:
                if not S.feasible(self.model):
                    if i == self.eta_feas:
                        S.addVehicle(self.model)
                        i -= 1
                else:
                    feasibilityPhase = False
                    i -= 1
            i += 1
        return S


class DEMA:
    # 构造属性
    model = None
    penalty = (15, 5, 10)
    maxiter_evo = 100
    size = 30
    infeasible_proportion = 0.25
    sigma = (1, 5, 10)
    theta = 0.7
    maxiter_tabu_mul = 4
    max_neighbour_mul = 5
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

    @staticmethod
    def get_objective_route(route: Route, vehicle: Vehicle, penalty: tuple) -> float:
        return route.sum_distance()+penalty[0]*VNS_TS.penalty_capacity(route, vehicle)+penalty[1]*VNS_TS.penalty_time(route, vehicle)+penalty[2]*VNS_TS.penalty_battery(route, vehicle)

    @staticmethod
    def get_objective(solution: Solution, model: Model, penalty: tuple) -> float:
        if solution.objective == 0:
            ret = 0
            for route in solution.routes:
                ret += DEMA.get_objective_route(route, model.vehicle, penalty)
            solution.objective = ret
            return ret
        else:
            return solution.objective

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
            if try_route.feasible_weight(self.model.vehicle)[0] and try_route.feasible_time(self.model.vehicle)[0]:
                # del choose[choose_index]
                choose_index += 1
            else:
                del building_route_visit[decide_insert_place]
                assert len(building_route_visit) != 2
                routes.append(Route(building_route_visit))
                building_route_visit = [self.model.depot, self.model.depot]

        routes.append(Route(building_route_visit))

        return Solution(routes)

    def abondon_random_create(self) -> Solution:
        choose = self.model.customers[:]
        random.shuffle(choose)
        routes = []
        building_route_visit = [self.model.depot]
        i = 0
        while i < len(choose):
            try_route = Route(building_route_visit+[choose[i], self.model.depot])
            if try_route.feasible_weight(self.model.vehicle) and try_route.feasible_time(self.model.vehicle):
                if i == len(choose)-1:
                    routes.append(try_route)
                    break
                building_route_visit.append(choose[i])
                i += 1
            else:
                building_route_visit.append(self.model.depot)
                routes.append(Route(building_route_visit))
                building_route_visit = [self.model.depot]
        return Solution(routes)

    def initialization(self) -> list:
        population = []
        for _ in range(self.size):
            sol = self.random_create()
            while True:
                fes_dic = sol.feasible_detail(self.model)
                for _, value in fes_dic.items():
                    if value[1] == 'battery':
                        sol = Operation.charging_modification(sol, self.model)
                        break
                    if value[1] == 'time':
                        sol = Operation.fix_time(sol, self.model)
                        break
                else:
                    break
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
            overlapping_degree = Operation.overlapping_degree_population(sol, P)
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
            if len(P_child) == int((1-self.infeasible_proportion)*self.size):
                penalty_save = self.penalty
                self.penalty = (0, 0, 0)
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
                S = Operation.ACO_GM_cross1(S_parent)
            elif sel == 1:
                S_parent, S2 = random.sample(P_parent, 2)
                S = Operation.ACO_GM_cross2(S_parent, S2)

            cross_call_times[sel] += 1
            cost = DEMA.get_objective(S, self.model, self.penalty)
            if cost < all(all_cost):
                cross_score[sel] += self.sigma[0]
            elif cost < DEMA.get_objective(S_parent, self.model, self.penalty):
                cross_score[sel] += self.sigma[1]
            else:
                cross_score[sel] += self.sigma[2]

            P_child.append(S)

        self.penalty = penalty_save
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
            overlapping_degree = Operation.overlapping_degree_population(sol, P)
            objective = DEMA.get_objective(sol, self.model, self.penalty)
            obj_value.append([objective, overlapping_degree])
        SP2 = Util.pareto_sort(SP2, obj_value)
        sp1up = int((iter/self.maxiter_evo)*self.size)
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

        for sol in P:
            sol.clear_status()

        return P

    def tabu_search(self, solution: Solution) -> Solution:
        best_sol = solution
        best_val = DEMA.get_objective(solution, self.model, self.penalty)
        tabu_list = {}
        #delta = collections.deque([float('inf')]*10, maxlen=10)
        for iter in range(int(self.maxiter_tabu_mul*len(self.model.customers))):
            print('tabu {} {}'.format(iter, best_val))
            actions = []
            while len(actions) < int(self.max_neighbour_mul*len(self.model.customers)):
                act = random.choice(['exchange', 'relocate', 'two-opt', 'stationInRe'])
                if act == 'exchange':
                    target = Operation.exchange_choose(solution)
                    if ('exchange', *target) not in actions:
                        actions.append(('exchange', *target))
                elif act == 'relocate':
                    target = Operation.relocate_choose(solution)
                    if ('relocate', *target) not in actions:
                        actions.append(('relocate', *target))
                elif act == 'two-opt':
                    target = Operation.two_opt_choose(solution)
                    if ('two-opt', *target) not in actions:
                        actions.append(('two-opt', *target))
                elif act == 'stationInRe':
                    target = Operation.stationInRe_choose(solution, self.model)
                    if ('stationInRe', *target) not in actions:
                        actions.append(('stationInRe', *target))
            local_best_sol = solution
            local_best_val = DEMA.get_objective(solution, self.model, self.penalty)
            local_best_action = (None,)
            for action in actions:
                tabu_status = tabu_list.get(action, 0)
                if tabu_status == 0:
                    if action[0] == 'exchange':
                        try_sol = Operation.exchange_action(solution, *action[1:])
                    elif action[0] == 'relocate':
                        try_sol = Operation.relocate_action(solution, *action[1:])
                    elif action[0] == 'two-opt':
                        try_sol = Operation.two_opt_action(solution, *action[1:])
                    elif action[0] == 'stationInRe':
                        try_sol = Operation.stationInRe_action(solution, *action[1:])
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

            #delta.append(DEMA.get_objective(solution, self.model, self.penalty)-local_best_val)
            #should_break = True
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
                print(iter, i)
                retP.append(self.tabu_search(sol))
            self.last_local_search = 0
            return retP
        elif self.last_charge_modify >= self.charge_modify_step:
            retP = []
            for i, sol in enumerate(P):
                print(iter, 'charge', i)
                retP.append(Operation.charging_modification(sol, self.model))
            self.last_charge_modify = 0
            return retP
        return P

    def update_S(self, P: list) -> tuple:
        for S in P:
            if S.feasible(self.model):
                cost = DEMA.get_objective(S, self.model, self.penalty)
                num = len(S.routes)
                #cost = S.sum_distance()
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

    def main(self) -> tuple:
        P = self.initialization()
        self.update_S(P)
        for iter in range(self.maxiter_evo):
            print(iter, self.min_cost)
            P_child = self.ACO_GM(P)
            P = self.ISSD(P+P_child, iter)
            P = self.MVS(P, iter)
            self.update_S(P)
        return self.S_best, self.min_cost

    def output_to_file(self, suffix: str = '') -> None:
        if not os.path.exists('result'):
            os.mkdir('result')
        filename = self.model.data_file.split('/')[-1].split('.')[0]
        output_file = open('result/'+filename+suffix+'.txt', 'a')
        output_file.write(str(self.S_best)+'\n'+str(self.min_cost)+'\n\n')
        output_file.close()

        pickle_file = open('result/'+filename+suffix+'.pickle', 'wb')
        pickle.dump(self.S_best, pickle_file)
        pickle_file.close()
