from abc import ABCMeta


class Node(metaclass=ABCMeta):
    id = 0
    x = 0.0
    y = 0.0
    demand = 0.0
    ready_time = 0.0
    due_date = 0.0
    service_time = 0.0

    def __init__(self, id: int, x: float, y: float) -> None:
        self.id = id
        self.x = x
        self.y = y

    def distance_to(self, node):
        return ((self.x-node.x)**2+(self.y-node.y)**2)**0.5


class Depot(Node):
    def __init__(self, id: int, x: float, y: float, due_date: float) -> None:
        super().__init__(id, x, y)
        self.demand = 0
        self.ready_time = 0
        self.due_date = due_date
        self.service_time = 0


class Customer(Node):
    def __init__(self, id: int, x: float, y: float, demand: float, ready_time: float, due_date: float, service_time: float) -> None:
        super().__init__(id, x, y)
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time


class Recharger(Node):
    def __init__(self, id: int, x: float, y: float, due_date: float) -> None:
        super().__init__(id, x, y)
        self.demand = 0
        self.ready_time = 0
        self.due_date = due_date
        self.service_time = 0


class Route:
    visit = []

    def __init__(self, visit: list) -> None:
        self.visit = visit

    def copy(self):
        return Route(self.visit_list[:])


class Vehicle:
    capacity = 0.0
    max_battery = 0.0
    net_weight = 0.0
    battery_cost_speed = 0.0
    charge_speed = 0.0
    on_route = None

    loaded_weight = 0.0
    remain_battery = 0.0
    location = None
    at_time = 0.0

    def __init__(self, capacity: float, max_battery: float, net_weight: float, battery_cost_speed: float, charge_speed: float, on_route: Route, loaded_weight: float, remain_battery: float, location: Node, at_time: float = 0) -> None:
        self.capacity = capacity
        self.max_battery = max_battery
        self.net_weight = net_weight
        self.battery_cost_speed = battery_cost_speed
        self.charge_speed = charge_speed
        self.on_route = on_route
        self.loaded_weight = loaded_weight
        self.remain_battery = remain_battery
        self.location = location
        self.at_time = at_time

    def copy(self):
        return Vehicle(self.capacity, self.max_battery, self.net_weight, self.battery_cost_speed, self.charge_speed, self.on_route, self.loaded_weight, self.remain_battery, self.location, self.at_time)


class Solution:
    routes = []

    def __init__(self, routes: list) -> None:
        self.routes = routes

    def copy(self):
        return Solution([route.copy() for route in self.routes])


class Model:
    data_file = ''
    file_type = ''

    capacity = 0.0
    max_battery = 0.0
    net_weight = 0.0
    battery_cost_speed = 0.0
    charge_speed = 0.0
    max_vehicle = 0

    depot = None
    customers = []
    rechargers = []

    def __init__(self, data_file: str = '', file_type: str = '', capacity: float = 0, max_battery: float = 0, net_weight: float = 0, battery_cost_speed: float = 0, charge_speed: float = 0, max_vehicle: int = 0, depot: Depot = None, customers: list = [], rechargers: list = []) -> None:
        self.data_file = data_file
        self.file_type = file_type

        self.capacity = capacity
        self.max_battery = max_battery
        self.net_weight = net_weight
        self.battery_cost_speed = battery_cost_speed
        self.charge_speed = charge_speed
        self.max_vehicle = max_vehicle

        self.depot = depot
        self.customers = customers
        self.rechargers = rechargers

    def read_data(self):
        assert len(self.data_file) != 0
