from .model import *
from .util import *
from .operation import *


class VNS_TS_Evolution:
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
            assert(hasattr(self, key))
            setattr(self, key, value)

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
            if random.random() < acceptSA.probability(S2.get_objective(self.model, self.penalty_0), S.get_objective(self.model, self.penalty_0), i):
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


class DEMA_Evolution:
    # 构造属性
    model = None
    penalty = (15, 5, 10)
    maxiter_evo = 10
    size = 30
    cross_prob = 0.7
    infeasible_proportion = 0.25
    sigma = (1, 5, 10)
    theta = 0.5
    maxiter_tabu_multiply = 4
    max_neighbour_multiply = 3
    tabu_len = 4
    # 状态属性
    cross_score = [0, 0]
    cross_call_times = [0, 0]
    cross_weigh = [0, 0]

    def __init__(self, model: Model, **param) -> None:
        self.model = model
        for key, value in param.items():
            assert(hasattr(self, key))
            setattr(self, key, value)

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
            if try_route.feasible_weight(self.model.vehicle) and try_route.feasible_time(self.model.vehicle):
                #del choose[choose_index]
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
            population.append(self.random_create())
        return population

    def ACO_GM(self, P: list) -> list:
        pass

    def main(self):
        pass
