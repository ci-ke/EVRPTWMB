from math import modf
import random
import numpy as np
from abc import ABCMeta


class Node(metaclass=ABCMeta):
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

    def __init__(self, capacity: float, max_battery: float, net_weight: float, velocity: float, battery_cost_speed: float, charge_speed: float) -> None:
        self.capacity = capacity
        self.max_battery = max_battery
        self.net_weight = net_weight
        self.velocity = velocity
        self.battery_cost_speed = battery_cost_speed
        self.charge_speed = charge_speed


class Route:
    visit = []

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
        return Route(self.visit_list[:])

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

    def cal_remain_battery(self, vehicle: Vehicle) -> None:
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
            self.arrive_remain_battery[i+1:] += vehicle.max_battery

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


class Model:
    data_file = ''
    file_type = ''

    vehicle = None
    max_vehicle = 0

    depot = None
    customers = []
    rechargers = []

    def __init__(self, data_file: str = '', file_type: str = '', capacity: float = 0, max_battery: float = 0, net_weight: float = 0, velocity: float = 0, battery_cost_speed: float = 0, charge_speed: float = 0, max_vehicle: int = 0, depot: Depot = None, customers: list = [], rechargers: list = []) -> None:
        self.data_file = data_file
        self.file_type = file_type
        self.vehicle = Vehicle(capacity, max_battery, net_weight, velocity, battery_cost_speed, charge_speed)
        self.max_vehicle = max_vehicle
        self.depot = depot
        self.customers = customers
        self.rechargers = rechargers

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


class Solution:
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


class Evolution:
    model = None
    size = 0

    def __init__(self, model: Model, size: int) -> None:
        self.model = model
        self.size = size

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
