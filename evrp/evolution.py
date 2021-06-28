from operator import mod
from .model import *
from . import util


class Operation:
    @staticmethod
    def cyclic_exchange(solution: Solution, Rts: int, max: int) -> Solution:
        if len(solution.routes) <= Rts:
            sel = list(range(len(solution.routes)))
        else:
            sel = random.sample(range(len(solution.routes)), Rts)
        actual_select = [None]*len(solution.routes)
        for route_i in sel:
            actual_select[route_i] = solution.routes[route_i].random_segment_range(max)
        ret_sol = solution.copy()
        for sel_i in range(len(sel)):
            ret_sol.routes[sel[sel_i]].visit[actual_select[sel[sel_i]][0]:actual_select[sel[sel_i]][1]] = solution.routes[sel[(sel_i+1) % len(sel)]].visit[actual_select[sel[(sel_i+1) % len(sel)]][0]:actual_select[sel[(sel_i+1) % len(sel)]][1]]

        for route in ret_sol.routes:
            if len(route.visit) == 2:
                ret_sol.routes.remove(route)

        return ret_sol

    @staticmethod
    def two_opt(solution: Solution, first_which: int, first_where: int, second_which: int, second_where: int) -> Solution:
        assert first_where >= 0 and first_where <= len(solution.routes[first_which].visit)-2
        assert second_where >= 0 and second_where <= len(solution.routes[second_which].visit)-2
        ret_sol = solution.copy()
        ret_sol.routes[first_which].visit[first_where+1:] = solution.routes[second_which].visit[second_where+1:]
        ret_sol.routes[second_which].visit[second_where+1:] = solution.routes[first_which].visit[first_where+1:]

        if len(ret_sol.routes[first_which].visit) == 2:
            del ret_sol.routes[first_which]
        elif len(ret_sol.routes[second_which].visit) == 2:
            del ret_sol.routes[second_which]

        return ret_sol

    @staticmethod
    def relocate(solution: Solution, which: int, where: int) -> Solution:
        assert where >= 1 and where <= len(solution.routes[which].visit)-2
        if len(solution.routes[which].visit) > 3:
            new_which = random.randint(0, len(solution.routes)-1)  # random.choice(range(len(solution.routes)))
        else:
            choose = list(range(len(solution.routes)))
            choose.remove(which)
            new_which = random.choice(choose)
        ret_sol = solution.copy()
        if new_which != which:
            new_where = random.randint(1, len(solution.routes[new_which].visit)-1)  # random.choice(range(1, len(solution.routes[new_which].visit)))
            ret_sol[new_which].visit.insert(new_where, solution.routes[which].visit[where])
            del ret_sol.routes[which].visit[where]
            if len(ret_sol.routes[which].visit) == 2:
                del ret_sol.routes[which]
        else:
            choose = list(range(1, len(solution.routes[new_which].visit)))
            choose.remove(where)
            choose.remove(where+1)
            new_where = random.choice(choose)
            ret_sol[which].visit.insert(new_where, solution.routes[which].visit[where])
            if new_where > where:
                del ret_sol.routes[which].visit[where]
            else:
                del ret_sol.routes[which].visit[where+1]
        return ret_sol

    @staticmethod
    def exchange(solution: Solution) -> Solution:
        ret_sol = solution.copy()
        while True:
            which1 = random.randint(0, len(solution.routes)-1)  # random.choice(range(len(solution.routes)))
            which2 = random.randint(0, len(solution.routes)-1)
            while which1 == which2:
                if len(solution.routes[which1].visit) <= 3:
                    which1 = random.randint(0, len(solution.routes)-1)
                    which2 = random.randint(0, len(solution.routes)-1)
                else:
                    break
            if which1 == which2:
                where1, where2 = random.sample(range(1, len(solution.routes[which1].visit)-1), 2)
                if isinstance(solution.routes[which1].visit[where1], Recharger) or isinstance(solution.routes[which2].visit[where2], Recharger):
                    continue
                ret_sol.routes[which1].visit[where1] = solution.routes[which2].visit[where2]
                ret_sol.routes[which2].visit[where2] = solution.routes[which1].visit[where1]
            else:
                where1 = random.randint(1, len(solution.routes[which1].visit)-2)  # random.choice(list(range(1, len(solution.routes[which1].visit)-1)))
                where2 = random.randint(1, len(solution.routes[which2].visit)-2)  # random.choice(list(range(1, len(solution.routes[which2].visit)-1)))
                if isinstance(solution.routes[which1].visit[where1], Recharger) or isinstance(solution.routes[which2].visit[where2], Recharger):
                    continue
                ret_sol.routes[which1].visit[where1] = solution.routes[which2].visit[where2]
                ret_sol.routes[which2].visit[where2] = solution.routes[which1].visit[where1]
            break
        return ret_sol

    @staticmethod
    def stationInRe(solution: Solution) -> Solution:
        pass

    def choose_best_insert(solution: Solution, node: Node, route_indexes: list) -> tuple:
        min_increase_dis_to_route = float('inf')
        to_route = None
        insert_place_to_route = None
        for route_index in route_indexes:
            min_increase_dis = float('inf')
            insert_place = None
            for place in range(1, len(solution.routes[route_index])):
                increase_dis = node.distance_to(solution.routes[route_index].visit[place-1])+node.distance_to(solution.routes[route_index].visit[place])-solution.routes[route_index].visit[place-1].distance_to(solution.routes[route_index].visit[place])
                if increase_dis < min_increase_dis:
                    min_increase_dis = increase_dis
                    insert_place = place
            if min_increase_dis < min_increase_dis_to_route:
                min_increase_dis_to_route = min_increase_dis
                to_route = route_index
                insert_place_to_route = insert_place
        return to_route, insert_place_to_route

    @staticmethod
    def ACO_GM_cross1(solution: Solution) -> Solution:
        if len(solution.routes) > 1:
            solution = solution.copy()
            avg_dis = np.zeros(len(solution.routes), dtype=float)
            for i, route in enumerate(solution.routes):
                avg_dis[i] = route.avg_distance()
            avg_dis = avg_dis/np.sum(avg_dis)
            select = util.wheel_select(avg_dis)
            rest_routes_index = list(range(len(solution.routes)))
            rest_routes_index.remove(select)
            for node in solution.routes[select].visit[1:-1]:
                to_route, insert_place_to_route = Operation.choose_best_insert(solution, node, rest_routes_index)
                solution.routes[to_route].visit.insert(insert_place_to_route, node)
            del solution.routes[select]
        return solution

    @staticmethod
    def ACO_GM_cross2(solution1: Solution, solution2: Solution) -> Solution:
        solution1 = solution1.copy()
        min_dis = float('inf')
        min_route = None
        for route in solution2.routes:
            dis = route.sum_distance()
            if dis < min_dis:
                min_dis = dis
                min_route = route
        for node in min_route.visit[1:-1]:
            for route in solution1.routes:
                if node in route:
                    route.visit.remove(node)
                    if len(route) == 2:
                        solution1.routes.remove(route)
                    break
        for node in min_route.visit[1:-1]:
            to_route, insert_place_to_route = Operation.choose_best_insert(solution1, node, list(range(len(solution1.routes))))
            solution1.routes[to_route].visit.insert(insert_place_to_route, node)
        return solution1


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
        choose.sort(key=lambda cus: util.cal_angle_AoB((self.model.depot.x, self.model.depot.y), (x, y), (cus.x, cus.y)))
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
        acceptSA = util.SA(self.Delta_SA, self.eta_dist)
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
