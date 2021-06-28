import random
import numpy as np
from abc import ABCMeta

from . import util


class Node(metaclass=ABCMeta):
    # 构造属性
    id = 0
    x = 0.0
    y = 0.0
    demand = 0.0
    ready_time = 0.0
    over_time = 0.0
    service_time = 0.0

    def __init__(self, id: int, x: float, y: float) -> None:
        self.id = id
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return '{} {} at {}'.format(type(self).__name__, self.id, id(self))

    def distance_to(self, node: object) -> float:
        assert isinstance(node, Node)
        return ((self.x-node.x)**2+(self.y-node.y)**2)**0.5


class Depot(Node):
    def __init__(self, id: int, x: float, y: float, over_time: float) -> None:
        super().__init__(id, x, y)
        self.demand = 0
        self.ready_time = 0
        self.over_time = over_time
        self.service_time = 0


class Customer(Node):
    def __init__(self, id: int, x: float, y: float, demand: float, ready_time: float, over_time: float, service_time: float) -> None:
        super().__init__(id, x, y)
        assert(over_time >= ready_time)
        self.demand = demand
        self.ready_time = ready_time
        self.over_time = over_time
        self.service_time = service_time


class Recharger(Node):
    def __init__(self, id: int, x: float, y: float, over_time: float) -> None:
        super().__init__(id, x, y)
        self.demand = 0
        self.ready_time = 0
        self.over_time = over_time
        self.service_time = 0


class Vehicle:
    capacity = 0.0
    max_battery = 0.0
    net_weight = 0.0
    velocity = 0.0
    battery_cost_speed = 0.0
    charge_speed = 0.0

    def __init__(self, capacity: float = 0, max_battery: float = 0, net_weight: float = 0, velocity: float = 0, battery_cost_speed: float = 0, charge_speed: float = 0) -> None:
        self.capacity = capacity
        self.max_battery = max_battery
        self.net_weight = net_weight
        self.velocity = velocity
        self.battery_cost_speed = battery_cost_speed
        self.charge_speed = charge_speed


class Route:
    # 构造属性
    visit = []
    # 计算属性
    arrive_load_weight = None  # 到达并服务后载货量 向量
    arrive_remain_battery = None  # 刚到达时剩余电量 向量
    arrive_time = None  # 刚到达时的时刻 向量
    adjacent_distance = None  # 两点距离 向量
    rechargers = None  # 充电桩索引 向量

    def __init__(self, visit: list) -> None:
        assert(isinstance(visit[0], Depot) and isinstance(visit[-1], Depot))
        self.visit = visit

    def __str__(self) -> str:
        retstr = 'Route: D{}'.format(self.visit[0].id)
        for node in self.visit[1:]:
            if isinstance(node, Depot):
                retstr += ' -> D{}'.format(node.id)
            elif isinstance(node, Customer):
                retstr += ' -> C{}'.format(node.id)
            elif isinstance(node, Recharger):
                retstr += ' -> R{}'.format(node.id)
        return retstr

    def __getitem__(self, index: int) -> Node:
        return self.visit[index]

    def __eq__(self, other: object) -> bool:
        return self.visit == other.visit

    def copy(self) -> object:
        return Route(self.visit[:])

    def sum_distance(self) -> float:
        if self.adjacent_distance is None:
            self.cal_adjacent_distance()
        return np.sum(self.adjacent_distance)

    def find_charge_station(self) -> None:
        visit = np.array(self.visit)
        test_recharger = np.vectorize(lambda node: isinstance(node, Recharger))
        self.rechargers = np.where(test_recharger(visit))[0]

    def cal_adjacent_distance(self) -> None:
        self.adjacent_distance = np.array(list(map(lambda i: self.visit[i].distance_to(self.visit[i+1]), range(len(self.visit)-1))))

    def cal_load_weight(self, vehicle: Vehicle) -> None:
        demand = np.array([cus.demand for cus in self.visit])
        start_load_weight = np.sum(demand, where=demand > 0)
        if start_load_weight > vehicle.capacity:
            self.arrive_load_weight = np.array([start_load_weight])
            return
        demand_vary = np.cumsum(demand)
        self.arrive_load_weight = start_load_weight-demand_vary

    def cal_remain_battery_without_consider_weight(self, vehicle: Vehicle) -> None:
        if self.adjacent_distance is None:
            self.cal_adjacent_distance()
        if self.rechargers is None:
            self.find_charge_station()
        adjacent_consume_battery = np.zeros(len(self.visit))
        adjacent_consume_battery[1:] = self.adjacent_distance*vehicle.battery_cost_speed
        arrive_consume_battery = np.cumsum(adjacent_consume_battery)
        self.arrive_remain_battery = vehicle.max_battery-arrive_consume_battery
        for i in self.rechargers:
            self.arrive_remain_battery[i+1:] += vehicle.max_battery-self.arrive_remain_battery[i]

    def cal_remain_battery_consider_weight(self, vehicle: Vehicle) -> None:
        if self.adjacent_distance is None:
            self.cal_adjacent_distance()
        if self.rechargers is None:
            self.find_charge_station()
        if self.arrive_load_weight is None:
            self.cal_load_weight(vehicle)
        adjacent_consume_battery = np.zeros(len(self.visit))
        adjacent_consume_battery[1:] = (self.arrive_load_weight[:-1]+vehicle.net_weight)*self.adjacent_distance*vehicle.battery_cost_speed
        arrive_consume_battery = np.cumsum(adjacent_consume_battery)
        self.arrive_remain_battery = vehicle.max_battery-arrive_consume_battery
        for i in self.rechargers:
            self.arrive_remain_battery[i+1:] += vehicle.max_battery-self.arrive_remain_battery[i]

    cal_remain_battery = cal_remain_battery_without_consider_weight

    def cal_arrive_time(self, vehicle: Vehicle) -> None:
        if self.adjacent_distance is None:
            self.cal_adjacent_distance()
        ready_time = np.array([node.ready_time for node in self.visit])
        service_time = np.array([node.service_time for node in self.visit])
        if self.rechargers is None:
            self.find_charge_station()
        if len(self.rechargers != 0):
            self.cal_remain_battery(vehicle)
            for i in self.rechargers:
                service_time[i] += (vehicle.max_battery-self.arrive_remain_battery[i])/vehicle.charge_speed
        arrive_service_time = np.cumsum(service_time)
        arrive_before_service_time = np.zeros(len(self.visit))
        arrive_before_service_time[1:] = arrive_service_time[:-1]
        adjacent_consume_time = np.zeros(len(self.visit))
        adjacent_consume_time[1:] = self.adjacent_distance/vehicle.velocity
        arrive_consume_time = np.cumsum(adjacent_consume_time)
        arrive_time = arrive_consume_time + arrive_before_service_time

        done = 0
        while True:
            need_process = np.where(arrive_time < ready_time)[0]
            if len(need_process) == done:
                break
            i = need_process[done]
            arrive_time[i+1:] += ready_time[i]-arrive_time[i]
            done += 1
        # while True in (arrive_time < ready_time):
        #    i = np.where(arrive_time < ready_time)[0][0]
        #    arrive_time[i:] += ready_time[i]-arrive_time[i]
        self.arrive_time = arrive_time

    def feasible_weight(self, vehicle: Vehicle) -> bool:
        if self.arrive_load_weight is None:
            self.cal_load_weight(vehicle)
        if True in (self.arrive_load_weight > vehicle.capacity):  # 这里不加括号会有错误，in的优先级高
            return False
        else:
            return True

    def feasible_battery(self, vehicle: Vehicle) -> bool:
        if self.arrive_remain_battery is None:
            self.cal_remain_battery(vehicle)
        if True in (self.arrive_remain_battery < 0):
            return False
        else:
            return True

    def feasible_time(self, vehicle: Vehicle) -> bool:
        if self.arrive_time is None:
            self.cal_arrive_time(vehicle)
        over_time = np.array([node.over_time for node in self.visit])
        if True in (self.arrive_time > over_time):
            return False
        else:
            return True

    def feasible(self, vehicle: Vehicle) -> tuple:
        if not self.feasible_weight(vehicle):
            return False, 'capacity'
        if not self.feasible_time(vehicle):
            return False, 'time'
        if not self.feasible_battery(vehicle):
            return False, 'battery'
        return True, ''

    def abandoned_feasible(self, vehicle: Vehicle) -> tuple:
        if self.arrive_load_weight is None:
            self.cal_load_weight(vehicle)
        loaded_weight = self.arrive_load_weight[0]
        if loaded_weight > vehicle.capacity:
            return False, (0, 'capacity', loaded_weight)
        remain_battery = vehicle.max_battery
        at_time = 0
        for loc_index, dest in enumerate(self.visit[1:]):
            next_loaded_weight = loaded_weight-dest.demand
            if next_loaded_weight > vehicle.capacity:  # 检查容量限制
                return False, (loc_index, 'capacity', loaded_weight)

            distance = self.visit[loc_index].distance_to(dest)
            next_remain_battery = remain_battery-(vehicle.battery_cost_speed*distance*(loaded_weight+vehicle.net_weight))
            if next_remain_battery < 0:  # 检查电池限制
                return False, (loc_index, 'battery', remain_battery)

            next_at_time = at_time+distance/vehicle.velocity
            if next_at_time > dest.over_time:  # 检查超时
                return False, (loc_index, 'time', at_time)
            elif next_at_time < dest.ready_time:  # 提前到达情况
                next_at_time = dest.ready_time

            loaded_weight = next_loaded_weight
            remain_battery = next_remain_battery
            at_time = next_at_time  # 成功到达目的地

            if isinstance(dest, Customer):
                at_time += dest.service_time  # 服务时间
            elif isinstance(dest, Recharger):
                at_time += (vehicle.max_battery-remain_battery)/vehicle.charge_speed  # 充电时间
                remain_battery = vehicle.max_battery
        return True, (loaded_weight, remain_battery, at_time)

    def penalty_capacity(self, vehicle: Vehicle) -> float:
        if self.arrive_load_weight is None:
            self.cal_load_weight(vehicle)
        penalty = max(self.arrive_load_weight[0]-vehicle.capacity, 0)
        neg_demand_cus = []
        for i, cus in enumerate(self.visit):
            if cus.demand < 0:
                neg_demand_cus.append(i)
        for i in neg_demand_cus:
            penalty += max(self.arrive_load_weight[i]-vehicle.capacity, 0)
        return penalty

    def penalty_time(self, vehicle: Vehicle) -> float:
        if self.arrive_time is None:
            self.cal_arrive_time(vehicle)
        late_time = self.arrive_time-np.array([cus.over_time for cus in self.visit])
        if_late = np.where(late_time > 0)[0]
        if len(if_late) > 0:
            return late_time[if_late[0]]
        else:
            return 0.0

    def penalty_battery(self, vehicle: Vehicle) -> float:
        if self.arrive_remain_battery is None:
            self.cal_remain_battery(vehicle)
        return np.abs(np.sum(self.arrive_remain_battery, where=self.arrive_remain_battery < 0))

    def get_objective(self, vehicle: Vehicle, alpha: float, beta: float, gamma: float) -> float:
        return self.sum_distance()+alpha*self.penalty_capacity(vehicle)+beta*self.penalty_time(vehicle)+gamma*self.penalty_battery(vehicle)

    def clear_status(self) -> None:
        self.arrive_load_weight = None
        self.arrive_remain_battery = None
        self.arrive_time = None
        self.adjacent_distance = None
        self.rechargers = None

    def random_segment_range(self, max: int) -> tuple:
        actual_max = min(len(self.visit)-2, max)
        length = random.randint(0, actual_max)
        start_point = random.choice(range(1, len(self.visit)-length))
        end_point = start_point+length
        return (start_point, end_point)


class Model:
    # 构造属性
    data_file = ''
    file_type = ''

    vehicle = Vehicle()
    max_vehicle = 0
    # 计算属性
    depot = None
    customers = []
    rechargers = []

    def __init__(self, data_file: str = '', file_type: str = '', **para) -> None:
        self.data_file = data_file
        self.file_type = file_type
        for key, value in para.items():
            assert(hasattr(self, key))
            setattr(self, key, value)

    def read_data(self) -> None:
        assert len(self.data_file) != 0
        with open(self.data_file) as f:
            meet_empty_line = False
            for line in f.readlines()[1:]:
                if line == '\n':
                    meet_empty_line = True
                    continue
                if not meet_empty_line:
                    if line[0] in 'DSC':
                        name, type_, x, y, demand, ready_time, over_time, service_time = line.split()
                        if type_ == 'd':
                            self.depot = Depot(int(name[1:]), float(x), float(y), float(over_time))
                        elif type_ == 'f':
                            self.rechargers.append(Recharger(int(name[1:]), float(x), float(y), float(over_time)))
                        elif type_ == 'c':
                            self.customers.append(Customer(int(name[1:]), float(x), float(y), float(demand), float(ready_time), float(over_time), float(service_time)))
                    else:
                        assert('wrong file')
                else:
                    end_num = float(line[line.find('/')+1:-2])
                    if line[0] == 'Q':
                        self.vehicle.max_battery = end_num
                    elif line[0] == 'C':
                        self.vehicle.capacity = end_num
                    elif line[0] == 'r':
                        self.vehicle.battery_cost_speed = end_num
                    elif line[0] == 'g':
                        self.vehicle.charge_speed = end_num
                    elif line[0] == 'v':
                        self.vehicle.velocity = end_num
                    else:
                        assert('wrong file')

    def get_map_bound(self) -> tuple:
        cus_x = [cus.x for cus in self.customers]
        cus_y = [cus.y for cus in self.customers]
        cus_min_x = min(cus_x)
        cus_max_x = max(cus_x)
        cus_min_y = min(cus_y)
        cus_max_y = max(cus_y)
        rec_x = [rec.x for rec in self.rechargers]
        rec_y = [rec.y for rec in self.rechargers]
        rec_min_x = min(rec_x)
        rec_max_x = max(rec_x)
        rec_min_y = min(rec_y)
        rec_max_y = max(rec_y)
        min_x = min(cus_min_x, rec_min_x, self.depot.x)
        max_x = max(cus_max_x, rec_max_x, self.depot.x)
        min_y = min(cus_min_y, rec_min_y, self.depot.y)
        max_y = max(cus_max_y, rec_max_y, self.depot.y)
        return min_x, max_x, min_y, max_y


class Solution:
    # 构造属性
    routes = []

    def __init__(self, routes: list) -> None:
        self.routes = routes

    def __str__(self) -> str:
        retstr = 'Solution: {} route{}'.format(len(self.routes), 's' if len(self.routes) != 1 else '')
        for route in self.routes:
            retstr += '\n'+' '*4+str(route)
        return retstr

    def __getitem__(self, index: int) -> Route:
        return self.routes[index]

    def __eq__(self, other: object) -> bool:
        if len(self.routes) != len(other.routes):
            return False
        self.arrange()
        for a, b in zip(self.routes, other.routes):
            if a != b:
                return False
        return True

    def copy(self) -> object:
        return Solution([route.copy() for route in self.routes])

    def arrange(self) -> None:
        self.routes.sort(key=lambda route: (route.visit[1].id, route.visit[1].x, route.visit[1].y))

    def sum_distance(self) -> float:
        return sum(map(lambda route: route.sum_distance(), self.routes))

    def feasible(self, model: Model) -> bool:
        if len(self.routes) > model.max_vehicle:
            return False
        route_test_result = map(lambda route: route.feasible(model.vehicle)[0], self.routes)
        if False in route_test_result:
            return False
        else:
            return True

    def get_objective(self, model: Model, penalty: tuple) -> float:
        ret = 0
        for route in self.routes:
            ret += route.get_objective(model.vehicle, penalty[0], penalty[1], penalty[2])
        return ret

    def clear_status(self) -> None:
        for route in self.routes:
            route.clear_status()

    def cyclic_exchange(self, Rts: int, max: int) -> object:
        if len(self.routes) <= Rts:
            sel = list(range(len(self.routes)))
        else:
            sel = random.sample(range(len(self.routes)), Rts)
        actual_select = [None]*len(self.routes)
        for route_i in sel:
            actual_select[route_i] = self.routes[route_i].random_segment_range(max)
        ret_sol = self.copy()
        for sel_i in range(len(sel)):
            ret_sol.routes[sel[sel_i]].visit[actual_select[sel[sel_i]][0]:actual_select[sel[sel_i]][1]] = self.routes[sel[(sel_i+1) % len(sel)]].visit[actual_select[sel[(sel_i+1) % len(sel)]][0]:actual_select[sel[(sel_i+1) % len(sel)]][1]]

        for route in ret_sol.routes:
            if len(route.visit) == 2:
                ret_sol.routes.remove(route)

        return ret_sol

    def two_opt(self, first_which: int, first_where: int, second_which: int, second_where: int) -> object:
        assert first_where >= 0 and first_where <= len(self.routes[first_which].visit)-2
        assert second_where >= 0 and second_where <= len(self.routes[second_which].visit)-2
        ret_sol = self.copy()
        ret_sol.routes[first_which].visit[first_where+1:] = self.routes[second_which].visit[second_where+1:]
        ret_sol.routes[second_which].visit[second_where+1:] = self.routes[first_which].visit[first_where+1:]

        if len(ret_sol.routes[first_which].visit) == 2:
            del ret_sol.routes[first_which]
        elif len(ret_sol.routes[second_which].visit) == 2:
            del ret_sol.routes[second_which]

        return ret_sol

    def relocate(self, which: int, where: int) -> object:
        assert where >= 1 and where <= len(self.routes[which].visit)-2
        if len(self.routes[which].visit) > 3:
            new_which = random.choice(range(len(self.routes)))
        else:
            choose = list(range(len(self.routes)))
            choose.remove(which)
            new_which = random.choice(choose)
        ret_sol = self.copy()
        if new_which != which:
            new_where = random.choice(range(1, len(self.routes[new_which].visit)))
            ret_sol[new_which].visit.insert(new_where, self.routes[which].visit[where])
            del ret_sol.routes[which].visit[where]
            if len(ret_sol.routes[which].visit) == 2:
                del ret_sol.routes[which]
        else:
            choose = list(range(1, len(self.routes[new_which].visit)))
            choose.remove(where)
            choose.remove(where+1)
            new_where = random.choice(choose)
            ret_sol[which].visit.insert(new_where, self.routes[which].visit[where])
            if new_where > where:
                del ret_sol.routes[which].visit[where]
            else:
                del ret_sol.routes[which].visit[where+1]
        return ret_sol

    def exchange(self) -> object:
        ret_sol = self.copy()
        while True:
            which1 = random.choice(range(len(self.routes)))
            which2 = random.choice(range(len(self.routes)))
            while which1 == which2:
                if len(self.routes[which1].visit) <= 3:
                    which1 = random.choice(range(len(self.routes)))
                    which2 = random.choice(range(len(self.routes)))
                else:
                    break
            if which1 == which2:
                where1, where2 = random.sample(range(1, len(self.routes[which1].visit)-1), 2)
                if isinstance(self.routes[which1].visit[where1], Recharger) or isinstance(self.routes[which2].visit[where2], Recharger):
                    continue
                ret_sol.routes[which1].visit[where1] = self.routes[which2].visit[where2]
                ret_sol.routes[which2].visit[where2] = self.routes[which1].visit[where1]
            else:
                where1 = random.choice(list(range(1, len(self.routes[which1].visit)-1)))
                where2 = random.choice(list(range(1, len(self.routes[which2].visit)-1)))
                if isinstance(self.routes[which1].visit[where1], Recharger) or isinstance(self.routes[which2].visit[where2], Recharger):
                    continue
                ret_sol.routes[which1].visit[where1] = self.routes[which2].visit[where2]
                ret_sol.routes[which2].visit[where2] = self.routes[which1].visit[where1]
            break
        return ret_sol

    def stationInRe(self) -> object:
        pass

    def addVehicle(self, model: Model) -> None:
        if len(self.routes) < model.max_vehicle:
            self.routes.append(Route([self.routes[0].visit[0], self.routes[0].visit[0]]))


class Evolution:
    # 构造属性
    model = None

    vns_neighbour_Rts = 0
    vns_neighbour_max = 0
    eta_feas = 0
    eta_dist = 0
    Delta_SA = 0.0

    penalty_0 = tuple()
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

    def initialization(self) -> list:
        population = []
        for _ in range(self.size):
            population.append(self.random_create())
        return population

    def create_vns_neighbour(self, Rts: int, max: int) -> list:
        assert Rts >= 2 and max >= 1
        self.vns_neighbour = []
        for R in range(2, Rts+1):
            for m in range(1, max+1):
                self.vns_neighbour.append((R, m))

    def tabu_search(self, S: Solution, eta_tabu: int) -> Solution:
        return S

    def VNS_TS(self) -> Solution:
        self.create_vns_neighbour(self.vns_neighbour_Rts, self.vns_neighbour_max)
        S = self.random_create()
        k = 0
        i = 0
        feasibilityPhase = True
        acceptSA = util.SA(self.Delta_SA, self.eta_dist)
        while feasibilityPhase or i < self.eta_dist:
            S1 = S.cyclic_exchange(*self.vns_neighbour[k])
            S2 = self.tabu_search(S1, self.eta_tabu)
            if random.random() < acceptSA.probability(S2.get_objective(self.model, self.penalty_0), S.get_objective(self.model, self.penalty_0), i):
                S = S2
                #print(i, S)
                # print(S.feasible(self.model),S.sum_distance())
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
