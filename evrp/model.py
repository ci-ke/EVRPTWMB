import random
import numpy as np
import pandas as pd
from abc import ABCMeta


class Node(metaclass=ABCMeta):
    # 构造属性
    id = 0
    x = 0.0
    y = 0.0
    demand = 0.0
    ready_time = 0.0
    over_time = 0.0
    service_time = 0.0

    use_matrix = False
    dis_matrix = None

    def __init__(self, id: int, x: float, y: float, use_matrix: bool, dis_matrix: object) -> None:
        self.id = id
        self.x = x
        self.y = y
        self.use_matrix = use_matrix
        self.dis_matrix = dis_matrix

    def __repr__(self) -> str:
        # return '{} {} at {}'.format(type(self).__name__, self.id, id(self))
        return '{} {}'.format(type(self).__name__, self.id)

    def __hash__(self) -> str:
        return hash(self.__class__.__name__+str(self.id))

    def __eq__(self, other) -> bool:
        if type(self) != type(other):
            return False
        if self.id == other.id:
            return True
        else:
            return False

    def distance_to(self, node: object) -> float:
        assert isinstance(node, Node)
        if not self.use_matrix:
            return ((self.x-node.x)**2+(self.y-node.y)**2)**0.5
        else:
            if self.id == node.id:
                return 0
            elif self.id < node.id:
                return self.dis_matrix[((self.dis_matrix['x'] == self.id) & (self.dis_matrix['y'] == node.id))].iloc[0]['distance']
            else:
                return self.dis_matrix[((self.dis_matrix['y'] == self.id) & (self.dis_matrix['x'] == node.id))].iloc[0]['distance']


class Depot(Node):
    def __init__(self, id: int, x: float, y: float, over_time: float, use_matrix: bool = False, dis_matrix: object = None) -> None:
        super().__init__(id, x, y, use_matrix, dis_matrix)
        self.demand = 0.0
        self.ready_time = 0.0
        self.over_time = over_time
        self.service_time = 0.0


class Customer(Node):
    def __init__(self, id: int, x: float, y: float, demand: float, ready_time: float, over_time: float, service_time: float, use_matrix: bool = False, dis_matrix: object = None) -> None:
        super().__init__(id, x, y, use_matrix, dis_matrix)
        assert over_time >= ready_time
        self.demand = demand
        self.ready_time = ready_time
        self.over_time = over_time
        self.service_time = service_time


class Recharger(Node):
    def __init__(self, id: int, x: float, y: float, over_time: float, use_matrix: bool = False, dis_matrix: object = None) -> None:
        super().__init__(id, x, y, use_matrix, dis_matrix)
        self.demand = 0.0
        self.ready_time = 0.0
        self.over_time = over_time
        self.service_time = 0.0


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
        assert isinstance(visit[0], Depot) and isinstance(visit[-1], Depot)
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

    def __len__(self) -> int:
        return len(self.visit)

    def __eq__(self, other: object) -> bool:
        return self.visit == other.visit

    def copy(self) -> object:
        ret = Route(self.visit[:])
        if self.arrive_load_weight is not None:
            ret.arrive_load_weight = self.arrive_load_weight.copy()
        if self.arrive_remain_battery is not None:
            ret.arrive_remain_battery = self.arrive_remain_battery.copy()
        if self.arrive_time is not None:
            ret.arrive_time = self.arrive_time.copy()
        if self.adjacent_distance is not None:
            ret.adjacent_distance = self.adjacent_distance.copy()
        if self.rechargers is not None:
            ret.rechargers = self.rechargers.copy()
        return ret

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
        # if start_load_weight > vehicle.capacity:
        #    self.arrive_load_weight = np.array([start_load_weight])
        #    return
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
                service_time[i] += (vehicle.max_battery-self.arrive_remain_battery[i])*vehicle.charge_speed
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

    def cal_arrive_time_after_index(self, vehicle: Vehicle, index: int) -> None:
        '''
        需要保证index点到达时间正确
        '''
        ready_time = np.array([node.ready_time for node in self.visit[index:]])
        service_time = np.array([node.service_time for node in self.visit[index:]])
        for i in np.extract(self.rechargers >= index, self.rechargers):
            i = int(i)
            service_time[i-index] += (vehicle.max_battery-self.arrive_remain_battery[i])*vehicle.charge_speed
        arrive_service_time = np.cumsum(service_time)
        arrive_before_service_time = np.zeros(len(self.visit)-index)
        arrive_before_service_time[:] = self.arrive_time[index]
        arrive_before_service_time[1:] += arrive_service_time[:-1]
        adjacent_consume_time = np.zeros(len(self.visit)-index)
        adjacent_consume_time[1:] = self.adjacent_distance[index:]/vehicle.velocity
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
        self.arrive_time[index:] = arrive_time

    def feasible_capacity(self, vehicle: Vehicle) -> tuple:
        if self.arrive_load_weight is None:
            self.cal_load_weight(vehicle)
        if True in (self.arrive_load_weight > vehicle.capacity):  # 这里不加括号会有错误，in的优先级高
            return False, np.where(self.arrive_load_weight > vehicle.capacity)[0][0]
        else:
            return True, None

    def feasible_battery(self, vehicle: Vehicle) -> tuple:
        if self.arrive_remain_battery is None:
            self.cal_remain_battery(vehicle)
        if True in (self.arrive_remain_battery < 0):
            return False, np.where(self.arrive_remain_battery < 0)[0][0]
        else:
            return True, None

    def feasible_time(self, vehicle: Vehicle) -> tuple:
        if self.arrive_time is None:
            self.cal_arrive_time(vehicle)
        over_time = np.array([node.over_time for node in self.visit])
        if True in (self.arrive_time > over_time):
            return False, np.where(self.arrive_time > over_time)[0][0]
        else:
            return True, None

    def feasible(self, vehicle: Vehicle) -> tuple:
        if not self.feasible_capacity(vehicle)[0]:
            return False, 'capacity', self.feasible_capacity(vehicle)[1]
        if not self.feasible_time(vehicle)[0]:
            return False, 'time', self.feasible_time(vehicle)[1]
        if not self.feasible_battery(vehicle)[0]:
            return False, 'battery', self.feasible_battery(vehicle)[1]
        return True, '', None

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

    def clear_status(self) -> None:
        self.arrive_load_weight = None
        self.arrive_remain_battery = None
        self.arrive_time = None
        self.adjacent_distance = None
        self.rechargers = None

    def random_segment_range(self, max: int) -> tuple:
        if len(self.visit) == 2:
            return(1, 1)  # a[1:1]=x 表示a.insert(1,x)
        actual_max = min(len(self.visit)-2, max)
        length = random.randint(0, actual_max)
        start_point = random.randint(1, len(self.visit)-1-length)  # random.choice(range(1, len(self.visit)-length))
        end_point = start_point+length
        return (start_point, end_point)

    def avg_distance(self) -> float:
        cus_num = 0
        for cus in self.visit:
            if isinstance(cus, Customer):
                cus_num += 1
        return self.sum_distance()/cus_num

    def no_customer(self) -> bool:
        if len(self.visit) == 2:
            return True
        else:
            for node in self.visit[1:-1]:
                if isinstance(node, Customer):
                    return False
            return True

    def remove_successive_recharger(self, vehicle: Vehicle) -> None:
        i = 1
        while i < len(self.visit)-1:
            if isinstance(self.visit[i], Recharger) and isinstance(self.visit[i-1], Recharger) and self.visit[i].id == self.visit[i-1].id:
                #del self.visit[i]
                self.del_node(vehicle, i)
            i += 1

    def remove_depot_to_recharger0(self, vehicle: Vehicle) -> None:
        while isinstance(self.visit[1], Recharger) and self.visit[1].x == self.visit[0].x and self.visit[1].y == self.visit[0].y:
            #del self.visit[1]
            self.del_node(vehicle, 1)
        while isinstance(self.visit[-2], Recharger) and self.visit[-2].x == self.visit[0].x and self.visit[-2].y == self.visit[0].y:
            #del self.visit[-2]
            self.del_node(vehicle, len(self.visit)-2)

    def add_node(self, vehicle: Vehicle, i: int, node: Node) -> None:
        '''
        在i之前插入点
        '''
        i = int(i)
        assert 1 <= i and i <= len(self.visit)-1
        # 插入访问节点
        self.visit.insert(i, node)
        self.clear_status()
        return

        # 更新两点距离
        self.adjacent_distance[i-1] = node.distance_to(self.visit[i-1])
        new_distance = node.distance_to(self.visit[i+1])
        self.adjacent_distance = np.insert(self.adjacent_distance, i, new_distance)

        if isinstance(node, Customer):
            # 更新载重
            if node.demand >= 0:
                new_node_load_weight = self.arrive_load_weight[i-1]
                self.arrive_load_weight[:i] += node.demand
            else:
                new_node_load_weight = self.arrive_load_weight[i-1]-node.demand
                self.arrive_load_weight[i] -= node.demand
            self.arrive_load_weight = np.insert(self.arrive_load_weight, i, new_node_load_weight)

            # 更新电量
            first_cost_battery = vehicle.battery_cost_speed*self.adjacent_distance[i-1]
            second_cost_battery = vehicle.battery_cost_speed*self.adjacent_distance[i]
            if isinstance(self.visit[i-1], Recharger):
                new_node_arrive_battery = vehicle.max_battery-first_cost_battery
            else:
                new_node_arrive_battery = self.arrive_remain_battery[i-1]-first_cost_battery
            new_node_next_arrive_battery = new_node_arrive_battery-second_cost_battery
            difference = new_node_next_arrive_battery-self.arrive_remain_battery[i]
            next_station = np.extract(self.rechargers >= i, self.rechargers)
            if len(next_station) == 0:
                self.arrive_remain_battery[i:] += difference
            else:
                self.arrive_remain_battery[i:int(next_station[0])+1] += difference
            self.arrive_remain_battery = np.insert(self.arrive_remain_battery, i, new_node_arrive_battery)

            # 更新电站位置
            self.rechargers[np.where(self.rechargers >= i)] += 1

        elif isinstance(node, Recharger):
            # 更新载重
            new_node_load_weight = self.arrive_load_weight[i-1]
            self.arrive_load_weight = np.insert(self.arrive_load_weight, i, new_node_load_weight)

            # 更新电量
            first_cost_battery = vehicle.battery_cost_speed*self.adjacent_distance[i-1]
            second_cost_battery = vehicle.battery_cost_speed*self.adjacent_distance[i]
            if isinstance(self.visit[i-1], Recharger):
                new_node_arrive_battery = vehicle.max_battery-first_cost_battery
            else:
                new_node_arrive_battery = self.arrive_remain_battery[i-1]-first_cost_battery
            new_node_next_arrive_battery = vehicle.max_battery-second_cost_battery
            difference = new_node_next_arrive_battery-self.arrive_remain_battery[i]
            next_station = np.extract(self.rechargers >= i, self.rechargers)
            if len(next_station) == 0:
                self.arrive_remain_battery[i:] += difference
            else:
                self.arrive_remain_battery[i:int(next_station[0])+1] += difference
            self.arrive_remain_battery = np.insert(self.arrive_remain_battery, i, new_node_arrive_battery)

            # 更新电站位置
            self.rechargers[np.where(self.rechargers >= i)] += 1
            insert_places = np.where(self.rechargers < i)[0]
            if len(insert_places) != 0:
                insert_place = insert_places[-1]+1
            else:
                insert_place = 0
            self.rechargers = np.insert(self.rechargers, insert_place, i)

        else:
            raise Exception('impossible')

        # 插入正确到达时间
        travel_time = self.adjacent_distance[i-1]/vehicle.velocity
        if isinstance(self.visit[i-1], Recharger):
            travel_time_with_service = travel_time + vehicle.charge_speed*(vehicle.max_battery-self.arrive_remain_battery[i-1])
            new_node_arrive_time = self.arrive_time[i-1]+travel_time_with_service
        else:
            travel_time_with_service = travel_time + self.visit[i-1].service_time
            if self.arrive_time[i-1] >= self.visit[i-1].ready_time:
                new_node_arrive_time = self.arrive_time[i-1]+travel_time_with_service
            else:
                new_node_arrive_time = self.visit[i-1].ready_time+travel_time_with_service
        self.arrive_time = np.insert(self.arrive_time, i, new_node_arrive_time)

        # 更新之后的到达时间
        self.cal_arrive_time_after_index(vehicle, i)
        # self.cal_arrive_time(vehicle)

    def del_node(self, vehicle: Vehicle, i: int) -> None:
        '''
        删除i
        '''
        i = int(i)
        assert 1 <= i and i <= len(self.visit)-1
        del self.visit[i]
        self.clear_status()
        return

        # 删除相邻距离
        self.adjacent_distance = np.delete(self.adjacent_distance, i)
        self.adjacent_distance[i-1] = self.visit[i-1].distance_to(self.visit[i+1])

        if isinstance(self.visit[i], Customer):
            # 更新载重
            if self.visit[i].demand >= 0:
                self.arrive_load_weight[:i] -= self.visit[i].demand
            else:
                self.arrive_load_weight[i:] += self.visit[i].demand
            self.arrive_load_weight = np.delete(self.arrive_load_weight, i)

        elif isinstance(self.visit[i], Recharger):
            # 更新载重
            self.arrive_load_weight = np.delete(self.arrive_load_weight, i)
            # 更新电站位置
            self.rechargers = np.delete(self.rechargers, np.where(self.rechargers == i))

        else:
            raise Exception('impossible')

        # 更新电量
        self.arrive_remain_battery = np.delete(self.arrive_remain_battery, i)
        if isinstance(self.visit[i-1], Recharger):
            i_next_battery = vehicle.max_battery-self.adjacent_distance[i-1]*vehicle.battery_cost_speed
        else:
            i_next_battery = self.arrive_remain_battery[i-1]-self.adjacent_distance[i-1]*vehicle.battery_cost_speed
        difference = i_next_battery-self.arrive_remain_battery[i]
        next_station = np.extract(self.rechargers > i, self.rechargers)
        if len(next_station) == 0:
            self.arrive_remain_battery[i:] += difference
        else:
            self.arrive_remain_battery[i:int(next_station[0])+1] += difference

        # 更新电站位置
        self.rechargers[np.where(self.rechargers > i)] -= 1

        # 删除访问节点
        del self.visit[i]

        # 更新之后的到达时间
        self.arrive_time = np.delete(self.arrive_time, i)
        self.cal_arrive_time_after_index(vehicle, i-1)
        # self.cal_arrive_time(vehicle)

    def replace_node(self, vehicle: Vehicle, i: int, node: Node) -> None:
        i = int(i)
        assert 1 <= i and i <= len(self.visit)-1
        self.visit[i] = node
        self.clear_status()

    def add_nodes(self, vehicle: Vehicle, i: int, node_list: list) -> None:
        assert 1 <= i and i <= len(self.visit)-1
        i = int(i)
        self.visit = self.visit[:i]+node_list+self.visit[i:]
        self.clear_status()

    def del_nodes(self, vehicle: Vehicle, start: int, end: int) -> None:
        start = int(start)
        end = int(end)
        del self.visit[start:end]
        self.clear_status()

    def replace_nodes(self, vehicle: Vehicle, start: int, end: int, node_list: list) -> None:
        start = int(start)
        end = int(end)
        self.visit[start:end] = node_list
        self.clear_status()


class Model:
    # 构造属性
    data_file = ''
    file_type = ''
    negative_demand = 0
    vehicle = Vehicle()
    max_vehicle = 0
    depot = None
    customers = []
    rechargers = []
    # 计算属性
    nearest_station = {}

    def __init__(self, data_file: str = '', file_type: str = '', negative_demand=0, **para) -> None:
        self.data_file = data_file
        self.file_type = file_type
        self.negative_demand = negative_demand
        for key, value in para.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    def read_data(self) -> None:
        if self.file_type in ['e', 's5', 's10', 's15']:
            self.__read_data_normal()
        elif self.file_type == 'tw':
            self.__read_data_solomon()
        elif self.file_type == 'p':
            self.__read_data_p()
        elif self.file_type == 'jd':
            self.__read_data_jd()
        else:
            raise Exception('impossible')
        self.set_negative_demand(self.negative_demand)

    def __read_data_normal(self) -> None:
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
                        raise Exception('wrong file')
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
                        raise Exception('wrong file')

    def get_map_bound(self) -> tuple:
        cus_x = [cus.x for cus in self.customers]
        cus_y = [cus.y for cus in self.customers]
        cus_min_x = min(cus_x)
        cus_max_x = max(cus_x)
        cus_min_y = min(cus_y)
        cus_max_y = max(cus_y)
        if len(self.rechargers) != 0:
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
        else:
            min_x = min(cus_min_x, self.depot.x)
            max_x = max(cus_max_x, self.depot.x)
            min_y = min(cus_min_y, self.depot.y)
            max_y = max(cus_max_y, self.depot.y)
        return min_x, max_x, min_y, max_y

    def find_nearest_station(self) -> None:
        if len(self.rechargers) == 0:
            return
        self.nearest_station = {}
        station00 = None
        for station in self.rechargers:
            if station.x == self.depot.x and station.y == self.depot.y:
                station00 = station
                break
        self.nearest_station[self.depot] = sorted(self.rechargers, key=lambda rec: self.depot.distance_to(rec))
        if not station00 is None:
            self.nearest_station[self.depot].remove(station00)
        for cus in self.customers:
            self.nearest_station[cus] = sorted(self.rechargers, key=lambda rec: cus.distance_to(rec))
        for station in self.rechargers:
            other_staion = self.rechargers[:]
            other_staion.remove(station)
            other_staion.sort(key=lambda rec: station.distance_to(rec))
            self.nearest_station[station] = other_staion

    def find_near_station_between(self, node1: Node, node2: Node) -> Recharger:
        min_dis = float('inf')
        min_station = None
        for station in self.rechargers:
            if station == node1 or station == node2 or (isinstance(node1, Depot) and station.x == node1.x and station.y == node1.y) or (isinstance(node2, Depot) and station.x == node2.x and station.y == node2.y):
                continue
            dis = node1.distance_to(station)+node2.distance_to(station)
            if dis < min_dis:
                min_dis = dis
                min_station = station
        return min_station

    def set_negative_demand(self, every: int) -> None:
        if every == 0:
            return
        for i, cus in enumerate(self.customers):
            if i % every == every-1:
                cus.demand = -cus.demand

    def get_customer(self, id: int) -> Customer:
        for customer in self.customers:
            if customer.id == id:
                return customer
        raise Exception('no this customer')

    def get_recharger(self, id: int) -> Recharger:
        for recharger in self.rechargers:
            if recharger.id == id:
                return recharger
        raise Exception('no this recharger')

    def create_empty_route(self) -> Route:
        ret = Route([self.depot, self.depot])
        ret.arrive_load_weight = np.array([0.0, 0.0])  # 到达并服务后载货量 向量
        ret.arrive_remain_battery = np.array([self.vehicle.max_battery, self.vehicle.max_battery])  # 刚到达时剩余电量 向量
        ret.arrive_time = np.array([0.0, 0.0])  # 刚到达时的时刻 向量
        ret.adjacent_distance = np.array([0.0])  # 两点距离 向量
        ret.rechargers = np.array([])  # 充电桩索引 向量
        return ret

    def __read_data_solomon(self):
        fp = open(self.data_file)
        self.vehicle = Vehicle(max_battery=100, battery_cost_speed=0, velocity=1)
        self.customers = []
        self.rechargers = []
        for num, line in enumerate(fp.readlines()):
            if num == 4:
                self.vehicle.capacity = float(line.split()[1])
            if num <= 8:  # 跳过无关行
                continue
            cus_no, x_coord, y_coord, demand, ready_time, over_time, service_time = [float(x) for x in line.split()]
            if cus_no == 0:
                self.depot = Depot(int(cus_no), x_coord, y_coord, over_time)
            else:
                cus = Customer(int(cus_no), x_coord, y_coord, demand, ready_time, over_time, service_time)
                self.customers.append(cus)
        fp.close()

    def __read_data_p(self) -> None:
        fp = open(self.data_file)
        self.vehicle = Vehicle(max_battery=100, battery_cost_speed=0, velocity=1)
        self.customers = []
        self.rechargers = []
        for num, line in enumerate(fp.readlines()):
            if line == '\n':
                continue
            if num == 1:
                depot_over_time = float(line.split()[0])
                if depot_over_time == 0:
                    depot_over_time = float('inf')
                self.vehicle.capacity = float(line.split()[1])
            if num <= 1:  # 跳过无关行
                continue
            cus_no, x_coord, y_coord, service_time, demand, *_ = [float(x) for x in line.split()]
            if cus_no == 0:
                self.depot = Depot(int(cus_no), x_coord, y_coord, over_time=depot_over_time)
            else:
                if service_time == 0:
                    service_time = 10
                cus = Customer(int(cus_no), x_coord, y_coord, demand, ready_time=0, over_time=float('inf'), service_time=service_time)
                self.customers.append(cus)
        fp.close()

    def __read_data_jd(self) -> None:
        dis_matrix_ori = pd.read_csv('data/jd/input_distance-time.txt', names=['id', 'x', 'y', 'distance', 'time'])
        dis_matrix_ori[['x', 'y']] = dis_matrix_ori[['x', 'y']].astype('int')
        dis_matrix_ori['distance'] = dis_matrix_ori['distance'].astype('float')
        node_matrix = pd.read_excel('data/jd/input_node.xlsx', sheet_name=['Customer_data'])['Customer_data']
        self.customers = []
        self.rechargers = []
        for cus_info in node_matrix.itertuples():
            node_id = int(cus_info[1])
            node_type = int(cus_info[2])
            x = float(cus_info[3])
            y = float(cus_info[4])
            demand = str(cus_info[5])
            if node_type == 1:
                self.depot = Depot(node_id, x, y, over_time=float('inf'), use_matrix=True)
            elif node_type == 2:
                self.customers.append(Customer(node_id, x, y, float(demand), ready_time=0, over_time=float('inf'), service_time=0, use_matrix=True))
            elif node_type == 3:
                self.customers.append(Customer(node_id, x, y, -float(demand), ready_time=0, over_time=float('inf'), service_time=0, use_matrix=True))
            elif node_type == 4:
                self.rechargers.append(Recharger(node_id, x, y, over_time=float('inf'), use_matrix=True))
        vehicle_matrix = pd.read_excel('data/jd/input_vehicle_type.xlsx', sheet_name=['Vehicle_data'])['Vehicle_data']
        capacity = float(vehicle_matrix.iloc[1].iloc[3])
        max_battery = float(vehicle_matrix.iloc[1].iloc[5])
        self.vehicle = Vehicle(capacity=capacity, max_battery=max_battery, battery_cost_speed=1, velocity=1)
        rng = random.Random(100)
        self.customers = rng.sample(self.customers, 200)
        self.rechargers = rng.sample(self.rechargers, 40)

        dis_matrix = pd.DataFrame(columns=['x', 'y', 'distance'])
        selected_id = [0]
        for node in self.customers+self.rechargers:
            selected_id.append(node.id)
        dis_matrix = dis_matrix_ori[dis_matrix_ori['x'].isin(selected_id) & dis_matrix_ori['y'].isin(selected_id) & (dis_matrix_ori['x'] < dis_matrix_ori['y'])].loc[:, ['x', 'y', 'distance']]
        dis_matrix.index = range(len(dis_matrix))
        #dis_matrix[['x', 'y']] = dis_matrix[['x', 'y']].astype('int')
        #dis_matrix['distance'] = dis_matrix['distance'].astype('float')

        self.depot.dis_matrix = dis_matrix
        for cus in self.customers:
            cus.dis_matrix = dis_matrix
        for rec in self.rechargers:
            rec.dis_matrix = dis_matrix
        # print(dis_matrix)
        #print(dis_matrix[((dis_matrix['x'] == 50012) & (dis_matrix['y'] == 50026))].iloc[0]['distance'])
        # exit()


class Solution:
    # 构造属性
    routes = []
    # 状态属性
    id = []
    next_id = 0
    objective = None

    def __init__(self, routes: list) -> None:
        assert isinstance(routes[0], Route)
        self.routes = routes
        self.id = list(range(len(routes)))
        self.next_id = len(routes)

    def __str__(self) -> str:
        retstr = 'Solution: {} route{}'.format(len(self.routes), 's' if len(self.routes) != 1 else '')
        for i, route in zip(self.id, self.routes):
            #retstr += '\n'+' '*4+str(route)[:5]+'_'+str(i)+str(route)[5:]
            retstr += '\n    {}_{}{}'.format(str(route)[:5], i, str(route)[5:])
        return retstr

    def __getitem__(self, index: int) -> Route:
        return self.routes[index]

    def __len__(self) -> int:
        return len(self.routes)

    def __eq__(self, other: object) -> bool:
        if len(self.routes) != len(other.routes):
            return False
        self.arrange()
        for a, b in zip(self.routes, other.routes):
            if a != b:
                return False
        return True

    def copy(self) -> object:
        ret = Solution([route.copy() for route in self.routes])
        ret.id = self.id[:]
        ret.next_id = self.next_id
        return ret

    def arrange(self) -> None:
        self.routes.sort(key=lambda route: (route.visit[1].id, route.visit[1].x, route.visit[1].y))

    def sum_distance(self) -> float:
        return sum(map(lambda route: route.sum_distance(), self.routes))

    def feasible(self, model: Model) -> bool:
        # if len(self.routes) > model.max_vehicle:
        #    return False
        for route in self.routes:
            if not route.feasible(model.vehicle)[0]:
                return False
        return True

    def feasible_detail(self, model: Model) -> tuple:
        ret_dict = {}
        for i, route in enumerate(self.routes):
            result = route.feasible(model.vehicle)
            if result[0] == False:
                ret_dict[i] = result
        return ret_dict

    def feasible_capacity(self, model: Model) -> bool:
        for route in self.routes:
            if not route.feasible_capacity(model.vehicle)[0]:
                return False
        return True

    def feasible_time(self, model: Model) -> bool:
        for route in self.routes:
            if not route.feasible_time(model.vehicle)[0]:
                return False
        return True

    def feasible_battery(self, model: Model) -> bool:
        for route in self.routes:
            if not route.feasible_battery(model.vehicle)[0]:
                return False
        return True

    def clear_status(self) -> None:
        self.objective = None
        for route in self.routes:
            route.clear_status()

    def add_empty_route(self, model: Model) -> None:
        self.routes.append(model.create_empty_route())
        self.id.append(self.next_id)
        self.next_id += 1

    def remove_empty_route(self) -> None:
        i = 0
        while i < len(self.routes):
            if self.routes[i].no_customer():
                del self.routes[i]
                del self.id[i]
                continue
            i += 1

    def remove_route_index(self, index: int) -> None:
        del self.routes[index]
        del self.id[index]

    def remove_route_object(self, route: Route) -> None:
        index = self.routes.index(route)
        del self.routes[index]
        del self.id[index]

    def add_route(self, route: Route) -> None:
        self.routes.append(route)
        self.id.append(self.next_id)
        self.next_id += 1

    def renumber_id(self) -> None:
        self.id = list(range(len(self.routes)))
        self.next_id = len(self.routes)

    def get_id_from_route(self, route: Route) -> int:
        index = self.routes.index(route)
        return self.id[index]

    def get_route_from_id(self, id: int) -> Route:
        index = self.id.index(id)
        return self.routes[index]

    def serve_all_customer(self, model: Model) -> bool:
        served_cus_list = []
        for route in self.routes:
            for node in route.visit[1:-1]:
                if isinstance(node, Customer):
                    served_cus_list.append(node.id)
        served_cus_set = set(served_cus_list)
        if len(served_cus_list) != len(served_cus_set):
            return False
        if len(served_cus_set) == len(model.customers):
            return True
        else:
            return False
