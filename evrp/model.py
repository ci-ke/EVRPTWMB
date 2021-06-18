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

    def distance_to(self, node):
        return ((self.x-node.x)**2+(self.y-node.y)**2)**0.5


class Depot(Node):
    def __init__(self, id: int, x: float, y: float, over_time: float) -> None:
        super().__init__(id, x, y)
        self.demand = 0
        self.ready_time = 0
        self.over_time = over_time
        self.service_time = 0

    def __repr__(self) -> str:
        return 'Depot {} at {}'.format(self.id, id(self))


class Customer(Node):
    def __init__(self, id: int, x: float, y: float, demand: float, ready_time: float, over_time: float, service_time: float) -> None:
        super().__init__(id, x, y)
        self.demand = demand
        self.ready_time = ready_time
        self.over_time = over_time
        self.service_time = service_time

    def __repr__(self) -> str:
        return 'Customer {} at {}'.format(self.id, id(self))


class Recharger(Node):
    def __init__(self, id: int, x: float, y: float, over_time: float) -> None:
        super().__init__(id, x, y)
        self.demand = 0
        self.ready_time = 0
        self.over_time = over_time
        self.service_time = 0

    def __repr__(self) -> str:
        return 'Recharger {} at {}'.format(self.id, id(self))


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

    def copy(self):
        return Vehicle(self.capacity, self.max_battery, self.net_weight, self.battery_cost_speed, self.charge_speed)


class Route:
    visit = []

    def __init__(self, visit: list) -> None:
        self.visit = visit

    def __getitem__(self, index):
        return self.visit[index]

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

    def copy(self):
        return Route(self.visit_list[:])

    def distance(self):
        return sum(map(lambda i: self.visit[i].distance_to(self.visit[i+1]), range(len(self.visit)-1)))

    def feasible(self, vehicle: Vehicle, ini_load_weigh: float):
        if ini_load_weigh == None:
            loaded_weight = vehicle.capacity
        else:
            assert(ini_load_weigh <= vehicle.capacity)
            loaded_weight = ini_load_weigh
        remain_battery = vehicle.max_battery
        at_time = 0
        for loc_index, dest in enumerate(self.visit[1:]):
            next_loaded_weight = loaded_weight-dest.demand
            if next_loaded_weight > vehicle.capacity or next_loaded_weight < 0:  # 检查容量限制
                return False, loc_index, 'capacity', loaded_weight

            distance = self.visit[loc_index].distance_to(dest)
            next_remain_battery = remain_battery-(vehicle.battery_cost_speed*distance*(loaded_weight+vehicle.net_weight))
            if next_remain_battery < 0:  # 检查电池限制
                return False, loc_index, 'battery', remain_battery

            next_at_time = at_time+distance/vehicle.velocity
            if next_at_time > dest.over_time:  # 检查超时
                return False, loc_index, 'time', at_time
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
        return True, loaded_weight, remain_battery, at_time


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

    def read_data(self):
        assert len(self.data_file) != 0


class Solution:
    routes = []
    ini_load_weight = []

    def __init__(self, routes: list, ini_load_weight: list = None) -> None:
        self.routes = routes
        if ini_load_weight != None:
            self.ini_load_weight = ini_load_weight
        else:
            self.ini_load_weight = [None]*len(self.routes)

    def __getitem__(self, index):
        return self.routes[index]

    def __str__(self) -> str:
        retstr = 'Solution: {} route{}'.format(len(self.routes), 's' if len(self.routes) != 1 else '')
        for route in self.routes:
            retstr += '\n'+' '*4+str(route)
        return retstr

    def copy(self):
        return Solution([route.copy() for route in self.routes])

    def distance(self):
        return sum(map(lambda route: route.distance(), self.routes))

    def feasible(self, model: Model):
        if len(self.routes) > model.max_vehicle:
            return False
        route_test_result = map(lambda route_load: route_load[0].feasible(model.vehicle, route_load[1]), zip(self.routes, self.ini_load_weight))
        if False in map(lambda x: x[0], route_test_result):
            return False
        else:
            return True
