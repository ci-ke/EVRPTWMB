from Model import *
from util import *
import math
import random

class Modification:
    @staticmethod
    def penalty_capacity(route: Route, vehicle: Vehicle) -> float:
        if route.arrive_load_weight is None:
            route.cal_load_weight(vehicle)
        penalty = max(route.arrive_load_weight[0] - vehicle.capacity, 0)
        neg_demand_cus = []
        for i, cus in enumerate(route.visit):
            if cus.demand < 0:
                neg_demand_cus.append(i)
        for i in neg_demand_cus:
            penalty += max(route.arrive_load_weight[i] - vehicle.capacity, 0)
        return penalty

    @staticmethod
    def penalty_time(route: Route, vehicle: Vehicle) -> float:
        if route.arrive_time is None:
            route.cal_arrive_time(vehicle)
        late_time = route.arrive_time - np.array([cus.over_time for cus in route.visit])
        late_time[late_time < 0] = 0
        if_late = np.where(late_time > 0)[0]
        if len(if_late) > 0:
            return np.sum(late_time)
            #return late_time[if_late[0]]
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
        return route.sum_distance() + penalty[0] * Modification.penalty_capacity(route, vehicle) + penalty[
            1] * Modification.penalty_time(route, vehicle) + penalty[2] * Modification.penalty_battery(route, vehicle)

    @staticmethod
    def get_objective(solution: Solution, model: Model, penalty: list) -> float:
        ret = 0
        for route in solution.routes:
            ret += Modification.get_objective_route(route, model.vehicle, penalty)
        return ret

    @staticmethod
    def cyclic_exchange(solution: Solution, model: Model, Rts: int, max: int) -> Solution:
        input_len=len(solution)
        if len(solution.routes) <= Rts:
            sel = list(range(len(solution.routes)))
        else:
            sel = random.sample(range(len(solution.routes)), Rts)
        actual_select = [None]*len(solution.routes)
        for route_i in sel:
            actual_select[route_i] = solution.routes[route_i].random_segment_range(max)

        ret_sol = solution.copy()
        for sel_i in range(len(sel)):
            #ret_sol.routes[sel[sel_i]].visit[actual_select[sel[sel_i]][0]:actual_select[sel[sel_i]][1]] = solution.routes[sel[(sel_i+1) % len(sel)]].visit[actual_select[sel[(sel_i+1) % len(sel)]][0]:actual_select[sel[(sel_i+1) % len(sel)]][1]]
            ret_sol.routes[sel[sel_i]].replace_nodes(model.vehicle, actual_select[sel[sel_i]][0], actual_select[sel[sel_i]][1], solution.routes[sel[(sel_i+1) % len(sel)]].visit[actual_select[sel[(sel_i+1) % len(sel)]][0]:actual_select[sel[(sel_i+1) % len(sel)]][1]])

        # ret_sol.remove_empty_route()

        for route in ret_sol.routes:
            route.remove_depot_to_recharger0(model.vehicle)
            route.remove_successive_recharger(model.vehicle)
        assert len(ret_sol)<=input_len
        return ret_sol

    @staticmethod
    def two_opt_star_action(solution: Solution, model: Model, first_which: int, first_where: int, second_which: int, second_where: int) -> Solution:
        assert first_which != second_which
        ret_sol = solution.copy()
        #ret_sol.routes[first_which].visit[first_where+1:] = solution.routes[second_which].visit[second_where+1:]
        #ret_sol.routes[second_which].visit[second_where+1:] = solution.routes[first_which].visit[first_where+1:]
        ret_sol.routes[first_which].replace_nodes(model.vehicle, first_where+1, len(ret_sol.routes[first_which].visit), solution.routes[second_which].visit[second_where+1:])
        ret_sol.routes[second_which].replace_nodes(model.vehicle, second_where+1, len(ret_sol.routes[second_which].visit), solution.routes[first_which].visit[first_where+1:])

        ret_sol.routes[first_which].remove_depot_to_recharger0(model.vehicle)
        ret_sol.routes[first_which].remove_successive_recharger(model.vehicle)
        ret_sol.routes[second_which].remove_depot_to_recharger0(model.vehicle)
        ret_sol.routes[second_which].remove_successive_recharger(model.vehicle)

        ret_sol.remove_empty_route()

        return ret_sol

    @staticmethod
    def relocate_choose(solution: Solution) -> tuple:
        which = random.randint(0, len(solution.routes)-1)
        where = random.randint(1, len(solution.routes[which].visit)-2)
        if len(solution.routes[which].visit) > 3:
            new_which = random.randint(0, len(solution.routes)-1)
        else:
            choose = list(range(len(solution.routes)))
            choose.remove(which)
            new_which = random.choice(choose)

        if new_which != which:
            new_where = random.randint(1, len(solution.routes[new_which].visit)-1)
        else:
            choose = list(range(1, len(solution.routes[new_which].visit)))
            choose.remove(where)
            choose.remove(where+1)
            new_where = random.choice(choose)

        return which, where, new_which, new_where

    @staticmethod
    def relocate_action(solution: Solution, model: Model, which: int, where: int, new_which: int, new_where: int) -> Solution:
        ret_sol = solution.copy()
        if new_which != which:
            #ret_sol[new_which].visit.insert(new_where, solution.routes[which].visit[where])
            ret_sol[new_which].add_node(model.vehicle, new_where, solution.routes[which].visit[where])
            #del ret_sol.routes[which].visit[where]
            ret_sol.routes[which].del_node(model.vehicle, where)
            # if ret_sol.routes[which].no_customer():
            #    ret_sol.remove_route_index(which)
        else:
            #ret_sol[which].add_node(new_where, solution.routes[which].visit[where])
            ret_sol[new_which].add_node(model.vehicle, new_where, solution.routes[which].visit[where])
            if new_where > where:
                #del ret_sol.routes[which].visit[where]
                ret_sol.routes[which].del_node(model.vehicle, where)
            else:
                #del ret_sol.routes[which].visit[where+1]
                ret_sol.routes[which].del_node(model.vehicle, where+1)

        ret_sol.remove_empty_route()
        for route in ret_sol.routes:
            route.remove_depot_to_recharger0(model.vehicle)
            route.remove_successive_recharger(model.vehicle)

        return ret_sol

    @staticmethod
    def exchange_choose(solution: Solution) -> tuple:
        while True:
            which1 = random.randint(0, len(solution.routes)-1)
            which2 = random.randint(0, len(solution.routes)-1)
            while which1 == which2:
                if len(solution.routes[which1].visit) <= 3:
                    which1 = random.randint(0, len(solution.routes)-1)
                    which2 = random.randint(0, len(solution.routes)-1)
                else:
                    break
            if which1 == which2:
                where1, where2 = random.sample(range(1, len(solution.routes[which1].visit)-1), 2)
                # if isinstance(solution.routes[which1].visit[where1], Recharger) or isinstance(solution.routes[which2].visit[where2], Recharger):
                #    continue
                return which1, where1, which2, where2
            else:
                where1 = random.randint(1, len(solution.routes[which1].visit)-2)
                where2 = random.randint(1, len(solution.routes[which2].visit)-2)
                # if isinstance(solution.routes[which1].visit[where1], Recharger) or isinstance(solution.routes[which2].visit[where2], Recharger):
                #    continue
                return which1, where1, which2, where2

    @staticmethod
    def exchange_action(solution: Solution, model: Model, which1: int, where1: int, which2: int, where2: int) -> Solution:
        ret_sol = solution.copy()
        #ret_sol.routes[which1].visit[where1] = solution.routes[which2].visit[where2]
        ret_sol.routes[which1].replace_node(model.vehicle, where1, solution.routes[which2].visit[where2])
        #ret_sol.routes[which2].visit[where2] = solution.routes[which1].visit[where1]
        ret_sol.routes[which2].replace_node(model.vehicle, where2, solution.routes[which1].visit[where1])

        ret_sol.remove_empty_route()
        for route in ret_sol:
            route.remove_depot_to_recharger0(model.vehicle)
            route.remove_successive_recharger(model.vehicle)

        return ret_sol

    @staticmethod
    def stationInRe_choose(solution: Solution, model: Model) -> tuple:
        which = random.randint(0, len(solution.routes)-1)
        where = random.randint(1, len(solution.routes[which].visit)-2)
        while not isinstance(solution.routes[which].visit[where], Customer):
            which = random.randint(0, len(solution.routes)-1)
            where = random.randint(1, len(solution.routes[which].visit)-2)
        recharger = random.choice(model.rechargers)
        depot = solution.routes[0].visit[0]
        while recharger.x == depot.x and recharger.y == depot.y and (where == 1 or where == len(solution.routes[which].visit)-2):
            recharger = random.choice(model.rechargers)
        return recharger, which, where

    @staticmethod
    def stationInRe_action(solution: Solution, model: Model, recharger: Recharger, which: int, where: int) -> Solution:
        ret_sol = solution.copy()
        if ret_sol.routes[which].visit[where-1]==recharger:
            #del ret_sol.routes[which].visit[where-1]
            ret_sol.routes[which].del_node(model.vehicle, where-1)
        else:
            #ret_sol.routes[which].visit.insert(where, recharger)
            ret_sol.routes[which].add_node(model.vehicle, where, recharger)
        # ret_sol.routes[which].remove_depot_to_recharger0()
        ret_sol.remove_empty_route()
        return ret_sol

    @staticmethod
    def two_opt_choose(solution: Solution) -> tuple:
        which = random.randint(0, len(solution.routes)-1)
        while len(solution.routes[which].visit) < 5:
            which = random.randint(0, len(solution.routes)-1)
        where1, where2 = random.sample(range(1, len(solution.routes[which].visit)-1), 2)
        while where2-where1 < 2:
            where1, where2 = random.sample(range(1, len(solution.routes[which].visit)-1), 2)
        return which, where1, where2

    @staticmethod
    def two_opt_action(solution: Solution, model: Model, which: int, where1: int, where2: int) -> Solution:
        ret_sol = solution.copy()
        #ret_sol.routes[which].visit[where1:where2+1] = reversed(solution.routes[which].visit[where1:where2+1])
        # ret_sol.routes[which].clear_status()
        ret_sol.routes[which].replace_nodes(model.vehicle, where1, where2+1, reversed(solution.routes[which].visit[where1:where2+1]))

        for route in ret_sol.routes:
            if isinstance(route.visit[1], Recharger) and route.visit[1].x == route.visit[0].x and route.visit[1].y == route.visit[0].y:
                #del route.visit[1]
                route.del_node(model.vehicle, 1)
            if isinstance(route.visit[-2], Recharger) and route.visit[-2].x == route.visit[0].x and route.visit[-2].y == route.visit[0].y:
                #del route.visit[-2]
                route.del_node(model.vehicle, len(route.visit)-2)

        ret_sol.routes[which].remove_successive_recharger(model.vehicle)

        return ret_sol

    @staticmethod
    def ACO_GM_cross1(solution: Solution, model: Model) -> Solution:
        solution = solution.copy()
        new_routes=[]
        for i in range(len(solution)):
            new_routes.append(Route([cus for cus in solution[i].visit if isinstance(cus,(Customer,Depot))]))
        solution=Solution(new_routes)
        if len(solution.routes) > 1:
            avg_dis = np.zeros(len(solution.routes), dtype=float)
            waste_time = 10000*np.ones(len(solution.routes), dtype=float)
            for i, route in enumerate(solution.routes):
                #r=Route([cus for cus in route.visit if isinstance(cus,(Customer,Depot))])
                avg_dis[i] = route.avg_distance()
                if not model.file_type=='p':
                    route.cal_arrive_time(model.vehicle)
                    lj=np.array([isinstance(cus,Customer) for cus in route.visit])
                    wt=np.array([cus.ready_time for cus in route.visit if isinstance(cus,Customer)])-route.arrive_time[lj]
                    wt[wt<0]=0
                    waste_time[i]=np.sum(wt)
            avg_dis = avg_dis/np.sum(avg_dis)
            if random.random()<0.5 and np.sum(waste_time)!=0 and (not model.file_type=='p'):
                select = Util.wheel_select(waste_time)
            else:
                select = Util.wheel_select(avg_dis)
            rest_routes_index = list(range(len(solution.routes)))
            rest_routes_index.remove(select)
            visit_list = solution.routes[select].visit[1:-1]
            visit_list=[cus for cus in visit_list if isinstance(cus,Customer)]
            visit_list.sort(key=lambda cus:random.uniform(0.8,1.2)*abs(cus.demand)*cus.distance_to(model.depot),reverse=1)
            #random.shuffle(visit_list)
            for node in visit_list:
                to_route, insert_place_to_route = Operation.choose_best_insert(model.vehicle, solution, node, rest_routes_index)
                #solution.routes[to_route].visit.insert(insert_place_to_route, node)
                solution.routes[to_route].add_node(model.vehicle, insert_place_to_route, node)
            solution.remove_route_index(select)
        solution.clear_status()
        return solution

    @staticmethod
    def ACO_GM_cross2(solution1: Solution, solution2: Solution, model: Model) -> Solution:
        solution1 = solution1.copy()
        new_routes = []
        for i in range(len(solution1)):
            new_routes.append(Route([cus for cus in solution1[i].visit if isinstance(cus, (Customer, Depot))]))
        solution1 = Solution(new_routes)

        avg_dis_reciprocal = np.zeros(len(solution2.routes), dtype=float)
        for i, route in enumerate(solution2.routes):
            avg_dis_reciprocal[i] = 1/route.avg_distance()
        avg_dis_reciprocal = avg_dis_reciprocal/np.sum(avg_dis_reciprocal)
        select = Util.wheel_select(avg_dis_reciprocal)

        visit_cus_list = []
        for node in solution2.routes[select].visit[1:-1]:
            if isinstance(node, Customer):
                visit_cus_list.append(node)

        for route in solution1.routes:
            i = 1
            while i < len(route.visit)-1:
                if route.visit[i] in visit_cus_list:
                    #del route.visit[i]
                    route.del_node(model.vehicle, i)
                else:
                    i += 1
        visit_cus_list.sort(key=lambda cus: random.uniform(0.8,1.2)*abs(cus.demand) * cus.distance_to(model.depot), reverse=1)
        #random.shuffle(visit_cus_list)
        for node in visit_cus_list:
            to_route, insert_place_to_route = Operation.choose_best_insert(model.vehicle,solution1, node, list(range(len(solution1.routes))))
            #solution1.routes[to_route].visit.insert(insert_place_to_route, node)
            solution1.routes[to_route].add_node(model.vehicle, insert_place_to_route, node)
        solution1.remove_empty_route()
        solution1.clear_status()
        return solution1

    @staticmethod
    def charging_modification(solution: Solution, model: Model,evo) -> Solution:
        solution = solution.copy()
        ready_to_remove = []
        for route in solution.routes:
            if route.feasible_capacity(model.vehicle)[0] and route.feasible_time(model.vehicle)[0] and not route.feasible_battery(model.vehicle)[0]:
                left_fail_index = np.where(route.arrive_remain_battery < 0)[0][0]#第一个缺电点索引
                left = np.where(route.rechargers < left_fail_index)[0]
                if len(left) != 0:
                    left = left[-1]#记录缺电前一个充电站的索引
                else:
                    left = 0#仓库
                right_over_index = len(route.visit)
                if route.arrive_remain_battery[-1] < 0:
                    battery = route.arrive_remain_battery-route.arrive_remain_battery[-1]
                    right_over_index = np.where(battery > model.vehicle.max_battery)[0][-1]
                # left - left_fail, right_over - end
                left_insert = list(range(left+1, left_fail_index+1))
                right_insert = list(range(right_over_index+1, len(route.visit)))
                common_insert = list(set(left_insert) & set(right_insert))

                assert len(left_insert) != 0

                if len(common_insert) == 0 and len(right_insert) != 0:
                    left_choose = []
                    right_choose = []
                    for node_i in left_insert:
                        left_choose.append((node_i, model.find_near_station_between(route.visit[node_i], route.visit[node_i-1])))
                    for node_i in right_insert:
                        right_choose.append((node_i, model.find_near_station_between(route.visit[node_i], route.visit[node_i-1])))
                    for left in left_choose:
                        for right in right_choose:
                            #route.visit.insert(left[0], left[1])
                            # route.visit.insert(right[0]+1, right[1])  # 因为左侧插入了一个
                            # route.clear_status()
                            route.add_node(model.vehicle, left[0], left[1])
                            route.add_node(model.vehicle, right[0]+1, right[1])
                            if route.feasible_battery(model.vehicle)[0]:
                                break
                            else:
                                #del route.visit[right[0]+1]
                                #del route.visit[left[0]]
                                route.del_node(model.vehicle, right[0]+1)
                                route.del_node(model.vehicle, left[0])
                        else:
                            continue
                        break
                    else:
                        assert len(route.visit) != 3, "unreasonable model"
                        cut_point = left_insert[-1]
                        if cut_point == len(route.visit)-1:
                            cut_point -= 1
                        if cut_point == 1:
                            cut_point += 1
                        solution.add_route(Route(route.visit[0:cut_point]+[model.depot]))
                        assert len(solution[-1].visit) != 2
                        solution.add_route(Route([model.depot]+route.visit[cut_point:]))
                        assert len(solution[-1].visit) != 2
                        ready_to_remove.append(route)
                elif len(common_insert) == 0 and len(right_insert) == 0:
                    choose = []
                    for node_i in left_insert:
                        choose.append((node_i, model.find_near_station_between(route.visit[node_i], route.visit[node_i-1])))
                    for pair in choose:
                        #route.visit.insert(pair[0], pair[1])
                        # route.clear_status()
                        route.add_node(model.vehicle, pair[0], pair[1])
                        if route.feasible_battery(model.vehicle)[0]:
                            break
                        else:
                            #del route.visit[pair[0]]
                            route.del_node(model.vehicle, pair[0])
                    else:
                        assert len(route.visit) != 3, "unreasonable model"
                        cut_point = left_insert[-1]
                        if cut_point == len(route.visit)-1:
                            cut_point -= 1
                        if cut_point == 1:
                            cut_point += 1
                        solution.add_route(Route(route.visit[0:cut_point]+[model.depot]))
                        assert len(solution[-1].visit) != 2
                        solution.add_route(Route([model.depot]+route.visit[cut_point:]))
                        assert len(solution[-1].visit) != 2
                        ready_to_remove.append(route)
                elif len(common_insert) != 0:
                    common_insert.sort()
                    choose = []
                    for node_i in common_insert:
                        choose.append((node_i, model.find_near_station_between(route.visit[node_i], route.visit[node_i-1])))
                    for pair in choose:
                        #route.visit.insert(pair[0], pair[1])
                        # route.clear_status()
                        route.add_node(model.vehicle, pair[0], pair[1])
                        if route.feasible_battery(model.vehicle)[0]:
                            break
                        else:
                            #del route.visit[pair[0]]
                            route.del_node(model.vehicle, pair[0])
                    else:
                        if len(route.visit) == 3:#
                            while not route.feasible_battery(model.vehicle)[0]:
                                route=evo.ls.SR(model,route)
                        else:
                            #"unreasonable model"
                            cut_point = common_insert[-1]
                            if cut_point == len(route.visit)-1:
                                cut_point -= 1
                            if cut_point == 1:
                                cut_point += 1
                            solution.add_route(Route(route.visit[0:cut_point]+[model.depot]))
                            assert len(solution[-1].visit) != 2
                            solution.add_route(Route([model.depot]+route.visit[cut_point:]))
                            assert len(solution[-1].visit) != 2
                            ready_to_remove.append(route)
                else:
                    raise Exception('impossible')

        for route in ready_to_remove:
            solution.remove_route_object(route)

        solution.remove_empty_route()

        return solution

    @staticmethod
    def charging_modification_for_route(route: Route, model: Model):
        # 对于不满足电量约束的路径进行插入充电桩的操作
        r = route.copy()
        #num=sum(np.array([isinstance(i,Customer) for i in r.visit]))
        while not r.feasible(model.vehicle)[0]:
            r.cal_remain_battery(model.vehicle)
            left_fail_index = np.where(r.arrive_remain_battery < 0)[0][0]  # 第一个缺电点索引
            # left = np.where(r.rechargers < left_fail_index)[0]
            # if len(left) != 0:
            #    left = left[-1]  # 记录缺电前一个充电站的索引
            # else:
            #    left = 0  # 仓库
            # right_over_index = left
            battery = r.arrive_remain_battery - r.arrive_remain_battery[left_fail_index]
            right_over_index = np.where(battery >= model.vehicle.max_battery)[0][-1]
            # left_insert = list(range(left + 1, left_fail_index + 1))
            right_insert = list(range(right_over_index + 1, left_fail_index + 1))
            # common_insert = list(set(left_insert) & set(right_insert))
            choose = []
            for i in list(reversed(right_insert)):
                choose.append((i, model.find_near_station_between(r.visit[i], r.visit[i - 1])))
            for pair in choose:
                r.add_node(model.vehicle, pair[0], pair[1])
                if r.feasible_battery(model.vehicle)[0] & r.feasible_time(model.vehicle)[0]:
                    break
                elif (not r.feasible_battery(model.vehicle)[0]) & r.feasible_time(model.vehicle)[0]:
                    r.cal_remain_battery(model.vehicle)
                    if (r.arrive_remain_battery[left_fail_index + 1] >= 0) & (r.arrive_remain_battery[pair[0]] >= 0):
                        break  # 继续while
                    else:
                        r.del_node(model.vehicle, pair[0])
                else:
                    r.del_node(model.vehicle, pair[0])
            else:  # 所有插入均不可行：切割路径
                assert len(r.visit) != 3, "unreasonable model"
                cut_point = left_fail_index
                if cut_point == len(r.visit) - 1:  # 缺电点为仓库
                    cut_point -= 1
                if cut_point == 1:
                    cut_point += 1
                r1 = (Route(r.visit[0:cut_point] + [model.depot]))
                assert len(r1.visit) != 2
                r2 = (Route([model.depot] + r.visit[cut_point:]))
                assert len(r2.visit) != 2
                if r1.feasible(model.vehicle)[0] & r2.feasible(model.vehicle)[0]:
                    r = [r1, r2]
                    break
                elif (not r1.feasible(model.vehicle)[0]) & r2.feasible(model.vehicle)[0]:
                    if not r1.feasible_battery(model.vehicle)[0]:
                        r = Modification.charging_modification_for_route(r1, model) + [r2]
                    else:
                        r = Modification.fix_time_for_route(r1, model) + [r2]
                    break
                elif r1.feasible(model.vehicle)[0] & (not r2.feasible(model.vehicle)[0]):
                    if not r2.feasible_battery(model.vehicle)[0]:
                        r = [r1] + Modification.charging_modification_for_route(r2, model)
                    else:
                        r = [r1] + Modification.fix_time_for_route(r2, model)
                    break
                else:
                    # r = charging_modification_for_route(r1, model) + charging_modification_for_route(r2, model)
                    if (not r1.feasible_battery(model.vehicle)) & (not r2.feasible_battery(model.vehicle)):
                        r = Modification.charging_modification_for_route(r1, model) + Modification.charging_modification_for_route(r2, model)
                    elif (not r1.feasible_time(model.vehicle)[0]) & (r1.feasible_battery(model.vehicle)[0]) & (
                    r2.feasible_time(model.vehicle)[0]):
                        r = Modification.fix_time_for_route(r1, model) + Modification.charging_modification_for_route(r2, model)
                    elif (not r2.feasible_time(model.vehicle)[0]) & (r2.feasible_battery(model.vehicle)[0]) & (
                    r1.feasible_time(model.vehicle)[0]):
                        r = Modification.fix_time_for_route(r2, model) + Modification.charging_modification_for_route(r1, model)
                    else:
                        r = Modification.fix_time_for_route(r1, model) + Modification.fix_time_for_route(r2, model)
                    break
        if type(r) != list:
            r = [r]
        #num1=0
        #for i in r:
        #    num1 += sum(np.array([isinstance(j, Customer) for j in i.visit]))
        #if num!=num1:
        #    print('客户缺失！')
        return r

    @staticmethod
    def fix_time_for_route(route: Route, model: Model):
        r = route.copy()
        if r.feasible_time(model.vehicle)[0] == False:
            cut = r.feasible_time(model.vehicle)[1]
            if cut == len(r.visit) - 1:
                cut -= 1
            new_route = Route([model.depot] + r.visit[cut:])
            r.replace_nodes(model.vehicle, cut, len(r.visit), [model.depot])
            r=[r]+[new_route]
            return r
        else:
            return [r]

    @staticmethod
    def fix_time(solution: Solution, model: Model) -> Solution:
        solution = solution.copy()

        for route in solution.routes:
            if route.feasible_time(model.vehicle)[0] == False:
                cut = route.feasible_time(model.vehicle)[1]
                if cut == len(route.visit)-1:
                    cut -= 1
                new_route = [model.depot]+route.visit[cut:]
                #route.visit[cut:] = [model.depot]
                # route.clear_status()
                route.replace_nodes(model.vehicle, cut, len(route.visit), [model.depot])
                solution.add_route(Route(new_route))

        solution.remove_empty_route()

        return solution

    @staticmethod
    def two_opt_star_arc(model: Model, solution: Solution, node1: Node, node2: Node) -> tuple:
        '''
        a后的边，与b前的边，去掉，交换后半部分再连起来，路间
        '''
        input_len=len(solution)
        assert not node1==node2
        if isinstance(node1, Customer) and isinstance(node2, Customer):
            which1, where1, which2, where2 = Operation.find_two_customer(solution, node1, node2)
            if which1 == which2:
                return [], []
            else:
                sol = Modification.two_opt_star_action(solution, model, which1, where1, which2, where2-1)
                assert len(sol) <= input_len,str(sol)
                #act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                #return [sol], [act]
                return [sol],None
        elif isinstance(node1, Customer) and isinstance(node2, Recharger):
            which1, where1, recharger2_which_where = Operation.find_customer_recharger(solution, node1, node2)
            ret_sol = []
            ret_act = []
            for which2, where2 in recharger2_which_where:
                if which1 != which2:
                    sol = Modification.two_opt_star_action(solution, model, which1, where1, which2, where2-1)
                    ret_sol.append(sol)
                    assert len(sol)<=input_len,str(sol)
                    #act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                    #ret_act.append(act)
            return ret_sol, ret_act
        elif isinstance(node1, Customer) and isinstance(node2, Depot):
            which1, where1 = Operation.find_customer(solution, node1)
            if where1 == len(solution.routes[which1].visit)-2:
                return [], []
            ret_sol = []
            ret_act = []
            solution = solution.copy()
            #solution.add_empty_route(model)
            which2 = 0
            while which2 < len(solution.routes):
                if which1 != which2:
                    where2 = len(solution.routes[which2])-1
                    sol = Modification.two_opt_star_action(solution, model, which1, where1, which2, where2-1)
                    ret_sol.append(sol)
                    assert len(sol) <= input_len,str(sol)
                    #act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                    #ret_act.append(act)
                which2 += 1
            return ret_sol, ret_act
        elif isinstance(node1, Recharger) and isinstance(node2, Customer):
            which2, where2, recharger1_which_where = Operation.find_customer_recharger(solution, node2, node1)
            ret_sol = []
            ret_act = []
            for which1, where1 in recharger1_which_where:
                if which1 != which2:
                    sol = Modification.two_opt_star_action(solution, model, which1, where1, which2, where2-1)
                    ret_sol.append(sol)
                    assert len(sol) <= input_len,str(sol)
                    #act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                    #ret_act.append(act)
            return ret_sol, ret_act
        elif isinstance(node1, Recharger) and isinstance(node2, Recharger):
            recharger1_which_where, recharger2_which_where = Operation.find_two_recharger(solution, node1, node2)
            ret_sol = []
            ret_act = []
            for which1, where1 in recharger1_which_where:
                for which2, where2 in recharger2_which_where:
                    if which1 != which2:
                        sol = Modification.two_opt_star_action(solution, model, which1, where1, which2, where2-1)
                        ret_sol.append(sol)
                        assert len(sol) <= input_len,str(sol)
                        #act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                        #ret_act.append(act)
            return ret_sol, ret_act
        elif isinstance(node1, Recharger) and isinstance(node2, Depot):
            recharger1_which_where = Operation.find_recharger(solution, node1)
            ret_sol = []
            ret_act = []
            #solution = solution.copy()
            #solution.add_empty_route(model)
            for which1, where1 in recharger1_which_where:
                if where1 == len(solution.routes[which1].visit)-2:
                    continue
                which2 = 0
                while which2 < len(solution.routes):
                    if which1 != which2:
                        where2 = len(solution.routes[which2])-1
                        sol = Modification.two_opt_star_action(solution, model, which1, where1, which2, where2-1)
                        ret_sol.append(sol)
                        assert len(sol) <= input_len,str(sol)
                        #act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                        #ret_act.append(act)
                    which2 += 1
            return ret_sol, ret_act
        elif isinstance(node1, Depot) and isinstance(node2, Customer):
            which2, where2 = Operation.find_customer(solution, node2)
            if where2 == 1:
                return [], []
            ret_sol = []
            ret_act = []
            solution = solution.copy()
            #solution.add_empty_route(model)
            which1 = 0
            while which1 < len(solution.routes):
                if which1 != which2:
                    sol = Modification.two_opt_star_action(solution, model, which1, 0, which2, where2-1)
                    ret_sol.append(sol)
                    assert len(sol) <= input_len,str(sol)
                    #act = ((node1, node2), solution.id[which1], node1, Operation.find_right_station(solution.routes[which1], 0))
                    #ret_act.append(act)
                which1 += 1
            return ret_sol, ret_act
        elif isinstance(node1, Depot) and isinstance(node2, Recharger):
            recharger2_which_where = Operation.find_recharger(solution, node2)
            ret_sol = []
            ret_act = []
            solution = solution.copy()
            #solution.add_empty_route(model)
            which1 = 0
            while which1 < len(solution.routes):
                for which2, where2 in recharger2_which_where:
                    if where2 == 1:
                        continue
                    if which1 != which2:
                        sol = Modification.two_opt_star_action(solution, model, which1, 0, which2, where2-1)
                        ret_sol.append(sol)
                        assert len(sol) <= input_len,str(sol)
                        #act = ((node1, node2), solution.id[which1], node1, Operation.find_right_station(solution.routes[which1], 1))
                        #ret_act.append(act)
                which1 += 1
            return ret_sol, ret_act

    @staticmethod
    def relocate_arc(model: Model, solution: Solution, node1: Node, node2: Node) -> tuple:
        '''
        去掉a，插入到b前，路间路内，客户与电站
        node1插入node2前
        '''
        assert not node1==node2
        if isinstance(node1, Depot):
            return [], []
        elif isinstance(node1, Customer) and isinstance(node2, Customer):
            which1, where1, which2, where2 = Operation.find_two_customer(solution, node1, node2)
            if which1 == which2 and where2 == where1+1:
                return [], []
            sol = Modification.relocate_action(solution, model, which1, where1, which2, where2)
            #act = ((node1, node2), solution.id[which2], *Operation.find_left_right_station(solution.routes[which2], where2))
            #return [sol], [act]
            return [sol], None
        elif isinstance(node1, Customer) and isinstance(node2, Depot):
            which1, where1 = Operation.find_customer(solution, node1)
            ret_sol = []
            ret_act = []
            if len(solution.routes[which1].visit) != 3:
                solution = solution.copy()
                #solution.add_empty_route(model)
            which2 = 0
            while which2 < len(solution.routes):
                if not (which2 == which1 and where1 == len(solution.routes[which2].visit)-2):
                    where2 = len(solution.routes[which2].visit)-1
                    sol = Modification.relocate_action(solution, model, which1, where1, which2, where2)
                    #act = ((node1, node2), solution.id[which2], Operation.find_left_station(solution.routes[which2], where2), node2)
                    ret_sol.append(sol)
                    #ret_act.append(act)
                which2 += 1
            return ret_sol, ret_act
        elif isinstance(node1, Customer) and isinstance(node2, Recharger):
            which1, where1, recharger2_which_where = Operation.find_customer_recharger(solution, node1, node2)
            ret_sol = []
            ret_act = []
            for which2, where2 in recharger2_which_where:
                if not (which1 == which2 and where2 == where1+1):
                    sol = Modification.relocate_action(solution, model, which1, where1, which2, where2)
                    ret_sol.append(sol)
                    #act = ((node1, node2), solution.id[which2], *Operation.find_left_right_station(solution.routes[which2], where2))
                    #ret_act.append(act)
            return ret_sol, ret_act
        elif isinstance(node1, Recharger) and isinstance(node2, Customer):
            which2, where2, recharger1_which_where = Operation.find_customer_recharger(solution, node2, node1)
            ret_sol = []
            ret_act = []
            for which1, where1 in recharger1_which_where:
                if not (which1 == which2 and where2 == where1+1):
                    sol = Modification.relocate_action(solution, model, which1, where1, which2, where2)
                    ret_sol.append(sol)
                    #act = ((node1, node2), solution.id[which2], *Operation.find_left_right_station(solution.routes[which2], where2))
                    #ret_act.append(act)
            return ret_sol, ret_act
        elif isinstance(node1, Recharger) and isinstance(node2, Depot):
            if node1.x == node2.x and node1.y == node2.y:
                return [], []
            recharger1_which_where = Operation.find_recharger(solution, node1)
            ret_sol = []
            ret_act = []
            for which1, where1 in recharger1_which_where:
                which2 = 0
                while which2 < len(solution.routes):
                    if not (which2 == which1 and where1 == len(solution.routes[which2].visit)-2):
                        where2 = len(solution.routes[which2].visit)-1
                        sol = Modification.relocate_action(solution, model, which1, where1, which2, where2)
                        ret_sol.append(sol)
                        #act = ((node1, node2), solution.id[which2], Operation.find_left_station(solution.routes[which2], where2), node2)
                        #ret_act.append(act)
                    which2 += 1
            return ret_sol, ret_act
        elif isinstance(node1, Recharger) and isinstance(node2, Recharger):
            recharger1_which_where, recharger2_which_where = Operation.find_two_recharger(solution, node1, node2)
            ret_sol = []
            ret_act = []
            for which1, where1 in recharger1_which_where:
                for which2, where2 in recharger2_which_where:
                    if not (which1 == which2 and where1+1 == where2):
                        sol = Modification.relocate_action(solution, model, which1, where1, which2, where2)
                        ret_sol.append(sol)
                        #act = ((node1, node2), solution.id[which2], *Operation.find_left_right_station(solution.routes[which2], where2))
                        #ret_act.append(act)
            return ret_sol, ret_act

    @staticmethod
    def exchange_arc(model: Model, solution: Solution, node1: Node, node2: Node) -> tuple:
        '''
        a后与b交换，路间路内，只有客户
        node2和node1后面的节点交换
        '''
        assert not node1==node2
        if isinstance(node2, Customer):
            if isinstance(node1, Customer):
                which1, where1, which2, where2 = Operation.find_two_customer(solution, node1, node2)
                if (not isinstance(solution.routes[which1].visit[where1+1], Customer)) or where1 == len(solution.routes[which1])-2 or (which1 == which2 and where2 == where1+1):
                    return [], []
                else:
                    sol = Modification.exchange_action(solution, model, which1, where1+1, which2, where2)
                    #act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                    #return [sol], [act]
                    return [sol], None
            elif isinstance(node1, Depot):
                which2, where2 = Operation.find_customer(solution, node2)
                ret_sol = []
                ret_act = []
                which1 = 0
                while which1 < len(solution.routes):
                    if isinstance(solution.routes[which1].visit[1], Customer) and not (where2 == 1 and which1 == which2):
                        sol = Modification.exchange_action(solution, model, which1, 1, which2, where2)
                        ret_sol.append(sol)
                        #act = ((node1, node2), solution.id[which1], node1, Operation.find_right_station(solution.routes[which1], 0))
                        #ret_act.append(act)
                    which1 += 1
                return ret_sol, ret_act
            elif isinstance(node1, Recharger):
                which2, where2, recharger1_which_where = Operation.find_customer_recharger(solution, node2, node1)
                ret_sol = []
                ret_act = []
                for which1, where1 in recharger1_which_where:
                    if isinstance(solution.routes[which1].visit[where1+1], Customer) and where1 != len(solution.routes[which1])-2 and not (which1 == which2 and where2 == where1+1):
                        sol = Modification.exchange_action(solution, model, which1, where1+1, which2, where2)
                        ret_sol.append(sol)
                        #act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                        #ret_act.append(act)
                return ret_sol, ret_act
            else:
                raise Exception('impossible')
        else:
            return [], []

    @staticmethod
    def stationInRe_arc(model: Model, solution: Solution, node1: Recharger, node2: Node) -> tuple:
        assert not node1==node2
        if not isinstance(node1, Recharger):
            return [], []
        if isinstance(node2, Customer):
            which2, where2 = Operation.find_customer(solution, node2)
            if where2 == 1:
                depot = solution.routes[0].visit[0]
                if node1.x == depot.x and node1.y == depot.y:
                    return [], []
            sol = Modification.stationInRe_action(solution, model, node1, which2, where2)
            #act = ((node1, node2), solution.id[which2], *Operation.find_left_right_station(solution.routes[which2], where2))
            #return [sol], [act]
            return [sol], None
        elif isinstance(node2, Depot):
            if node1.x == node2.x and node1.y == node2.y:
                return []
            ret_sol = []
            ret_act = []
            cur_which = 0
            while cur_which < len(solution.routes):
                cur_where = len(solution.routes[cur_which].visit)-1
                sol = Modification.stationInRe_action(solution, model, node1, cur_which, cur_where)
                ret_sol.append(sol)
                #act = ((node1, node2), solution.id[cur_which], Operation.find_left_station(solution.routes[cur_which], cur_where), node2)
                #ret_act.append(act)
                cur_which += 1
            return ret_sol, ret_act
        elif isinstance(node2, Recharger):
            recharger2_which_where = Operation.find_recharger(solution, node2)
            ret_sol = []
            ret_act = []
            for which2, where2 in recharger2_which_where:
                if where2 == 1:
                    depot = solution.routes[0].visit[0]
                    if node1.x == depot.x and node1.y == depot.y:
                        continue
                sol = Modification.stationInRe_action(solution, model, node1, which2, where2)
                ret_sol.append(sol)
                #act = ((node1, node2), solution.id[which2], *Operation.find_left_right_station(solution.routes[which2], where2))
                #ret_act.append(act)
            return ret_sol, ret_act
        else:
            raise Exception('impossible')

    @staticmethod
    def find_feasible_station_between(model:Model, node1: Node, node2: Node,evo):
        #p1=VNS_TS(model)
        pos_arc=list(evo.possible_arc.keys())
        s1=[i[1] for i in pos_arc if (i[0]==node1) and isinstance(i[1],Recharger)]
        s2=[i[0] for i in pos_arc if (i[1]==node2) and isinstance(i[0],Recharger)]
        s=list(set(s1)&set(s2))
        return s

    @staticmethod
    #给定客户顺序且路径可行，优化充电桩的位置
    #遗弃
    def schedule_sta_for_route(route:Route,model:Model):
        route=route.copy()
        route.find_charge_station()
        if len(route.rechargers)!=0:
            if route.feasible_battery()[0]:
                r1=Route([i for i in route.visit if isinstance(i,Customer)])
                dlxh1=model.vehicle.max_battery-r1.arrive_remain_battery[-1]
                dlxh=np.sum(model.vehicle.max_battery-route.arrive_remain_battery[route.rechargers+[len(route)-1]])
                if (dlxh<=model.vehicle.max_battery) or (dlxh1>=0):
                    #移除充电桩
                    for i in route.rechargers:
                        route.del_node(model.vehicle,i)
                elif (math.ceil(dlxh/model.vehicle.max_battery)<=len(route.rechargers)+1) and (math.ceil(dlxh/model.vehicle.max_battery)==math.ceil(dlxh1/model.vehicle.max_battery)+1):
                    #尝试移除或转换充电桩
                    jl=route.sum_distance()
                    for i in range((len(route)-2)**2):
                        route.find_charge_station()
                        res=random.sample(route.rechargers,1)[0]
                        resn=route.visit[res]
                        route.del_node(model.vehicle,res)
                        if route.feasible(model.vehicle)[0]:
                            jl=route.sum_distance()
                        else:
                            route.add_node(model.vehicle,res,resn)

                        route.find_charge_station()
                        jl = route.sum_distance()
                        res1=random.sample(route.rechargers,1)[0]
                        resn1 = route.visit[res1]
                        route.del_node(model.vehicle, res1)
                        #查找被移除的充电桩周围的充电桩
                        zws=[(j,j.distance_to(resn1)) for j in model.rechargers if j!=resn1]
                        zws.sort(key=lambda j:j[1])
                        zws=zws[0:math.ceil(len(model.rechargers)/8)]
                        yyh=0
                        for j in range(len(route)-1):
                            ss=Modification.find_feasible_station_between(model,route.visit[j],route.visit[j+1])
                            s=list(set(ss)&set(zws))
                            if s:
                                for k in range(len(s)):
                                    route.add_node(model.vehicle,j+1,s[k])
                                    if route.feasible(model.vehicle)[0] and (route.sum_distance()<jl):
                                        jl=route.sum_distance()
                                        yyh=1
                                        break
                                    else:
                                        route.del_node(model.vehicle,j+1)
                            if yyh:
                                break
                        else:
                            route.add_node(model.vehicle,res1,resn1)
        return route

    @staticmethod
    #根据仓库的时间窗，将配送过程分为10个时区，每个时区内的客户进行swap和relocate操作
    def time_flow(model:Model,solution:Solution,evo):
        solution=solution.copy()
        if not model.time_dic==None:
            time_num = [len(model.time_dic[i]) for i in range(len(model.time_dic))]
        if random.random()>0.7 and (not model.time_dic==None):
        #swap
            x=random.sample(list(np.where(np.array(time_num)>=2)[0]),1)[0]
            p=random.sample(model.time_dic[x],random.randint(2,max(int(len(model.time_dic[x])/2),2)))
            for j in range(0,len(p),2):
                which1,where1=Operation.find_customer(solution,p[j%len(p)])
                which2,where2=Operation.find_customer(solution,p[(j+1)%len(p)])
                solution=Modification.exchange_action(solution,model,which1,where1,which2,where2)
        #relocate
        else:
            if not model.time_dic==None:
                x = random.sample(list(np.where(np.array(time_num) >= 2)[0]), 1)[0]
                cus=random.sample(model.time_dic[x],1)[0]
                c=model.time_dic[x].copy()
                if x!=0:
                    c.extend(model.time_dic[x-1])
                c.remove(cus)
                c.sort(key=lambda x:x.distance_to(cus))
                c=c[0:min(len(c),10)]
            else:
                cus=random.choice(model.customers)
                c=model.customers.copy()
                c.remove(cus)
                c.sort(key=lambda x: x.distance_to(cus))
                c = c[0:min(len(c), 10)]
            s_best=solution.copy()
            o=s_best.feasible(model)
            value=float('inf')
            for j in range(len(c)):
                which2, where2 = Operation.find_customer(solution, cus)
                which1, where1 = Operation.find_customer(solution, c[j])
                s = Modification.relocate_action(solution,model,which1,where1,which2,where2)
                o1=s.feasible(model)
                value1 = Modification.get_objective(s, model, evo.penalty)
                if len(s)<len(s_best) and s.feasible(model) and (model.file_type!='p'):
                    value = value1
                    s_best = s.copy()
                    o=1
                else:
                    if value1<value:
                        if not (o==1 and o1==0):
                            value=value1
                            s_best=s.copy()
                            o=s_best.feasible(model)
                    elif (o==0 and o1==1):
                        value = value1
                        s_best = s.copy()
                        o=1
            solution=s_best.copy()
            solution.remove_empty_route()
        return solution

    @staticmethod
    #尝试连接两条路径
    #vrptw
    def connection_two_routes(model:Model,solution:Solution,evo):
        if len(solution)<2 or model.file_type=='p':
            return solution
        else:
            solution=solution.copy()
            co=0
            routes_index=random.sample(list(range(len(solution))),2)
            r1=solution[routes_index[0]]
            r2=solution[routes_index[1]]
            if r1.feasible(model.vehicle)[0] and r2.feasible(model.vehicle)[0]:
                zb=Modification.get_objective_route(r1,model.vehicle,evo.penalty)+Modification.get_objective_route(r2,model.vehicle,evo.penalty)
                if (evo.possible_arc.get((r1.visit[-2],r2.visit[1]),0)) or (evo.possible_arc.get((r2.visit[-2],r1.visit[1]),0)):
                    if (evo.possible_arc.get((r1.visit[-2],r2.visit[1]),0)) and (evo.possible_arc.get((r2.visit[-2],r1.visit[1]),0)):
                        r12=Route(r1.visit[0:-1]+r2.visit[1:])
                        r21 = Route(r2.visit[0:-1] + r1.visit[1:])
                        zb_r12=Modification.get_objective_route(r12,model.vehicle,evo.penalty)
                        zb_r21 = Modification.get_objective_route(r21, model.vehicle, evo.penalty)
                        if zb_r12<zb_r21:
                            if r12.feasible(model.vehicle)[0]:
                                co=1
                                sol=r12
                                solution.remove_route_object(r1)
                                solution.remove_route_object(r2)
                                solution.add_route(r12)
                        else:
                            if r21.feasible(model.vehicle)[0]:
                                co=1
                                sol=r21
                                solution.remove_route_object(r1)
                                solution.remove_route_object(r2)
                                solution.add_route(r21)
                    elif (evo.possible_arc.get((r1.visit[-2],r2.visit[1]),0)):
                        r12 = Route(r1.visit[0:-1] + r2.visit[1:])
                        #zb_r12 = Modification.get_objective_route(r12, model.vehicle, evo.penalty)
                        if r12.feasible(model.vehicle)[0]:
                            co = 1
                            sol=r12
                            solution.remove_route_object(r1)
                            solution.remove_route_object(r2)
                            solution.add_route(r12)
                    else:
                        r21 = Route(r2.visit[0:-1] + r1.visit[1:])
                        #zb_r21 = Modification.get_objective_route(r21, model.vehicle, evo.penalty)
                        if r21.feasible(model.vehicle)[0]:
                            co = 1
                            sol=r21
                            solution.remove_route_object(r1)
                            solution.remove_route_object(r2)
                            solution.add_route(r21)
                    if len(model.rechargers)!=0:
                        re1=Model.find_near_station_between(model,r1.visit[-2],r2.visit[1])
                        re2 = Model.find_near_station_between(model,r2.visit[-2], r1.visit[1])
                        sol1=Route(r1.visit[0:-1]+[re1]+r2.visit[1:])
                        zb_s1=Modification.get_objective_route(sol1, model.vehicle, evo.penalty)
                        sol2=Route(r2.visit[0:-1]+[re2]+r1.visit[1:])
                        zb_s2 = Modification.get_objective_route(sol2, model.vehicle, evo.penalty)
                        if zb_s1<zb_s2:
                            if co:
                                if zb_s1<Modification.get_objective_route(sol, model.vehicle, evo.penalty):
                                    solution.remove_route_object(sol)
                                    solution.add_route(sol1)
                            else:
                                if sol1.feasible(model.vehicle)[0]:
                                    solution.remove_route_object(r1)
                                    solution.remove_route_object(r2)
                                    solution.add_route(sol1)
                        else:
                            if co:
                                if zb_s2 < Modification.get_objective_route(sol, model.vehicle, evo.penalty):
                                    solution.remove_route_object(sol)
                                    solution.add_route(sol2)
                            else:
                                if sol2.feasible(model.vehicle)[0]:
                                    solution.remove_route_object(r1)
                                    solution.remove_route_object(r2)
                                    solution.add_route(sol2)
                else:
                    if len(model.rechargers) != 0:
                        r1.cal_arrive_time(model.vehicle)
                        r2.cal_arrive_time(model.vehicle)
                        r1.cal_load_weight(model.vehicle)
                        r2.cal_load_weight(model.vehicle)
                        if not (r1.arrive_time[-2]+r1.visit[-2].service_time+r1.visit[-2].distance_to(r2.visit[1])/model.vehicle.velocity>min([j.over_time for j in r2.visit if isinstance(j,Customer)])):
                            if np.sum(r1.arrive_load_weight[-1]+r2.arrive_load_weight>model.vehicle.capacity)==0:
                                #尝试插入充电桩或不插入
                                sta=Modification.find_feasible_station_between(model,r1.visit[-2],r2.visit[1],evo)
                                #co=0
                                for j in range(len(sta)+1):
                                    if j==len(sta):
                                        rr=Route(r1.visit[0:-1]+r2.visit[1:])
                                        zb1 = Modification.get_objective_route(rr, model.vehicle, evo.penalty)
                                    else:
                                        rr=Route(r1.visit[0:-1]+[sta[j]]+r2.visit[1:])
                                        zb1=Modification.get_objective_route(rr, model.vehicle, evo.penalty)
                                    if (rr.feasible(model.vehicle)[0]) and (co==0):
                                        solution.remove_route_object(r1)
                                        solution.remove_route_object(r2)
                                        solution.add_route(rr)
                                        zb=zb1
                                        sol=rr
                                        co=1
                                    elif (zb>zb1) and co:
                                        solution.remove_route_object(sol)
                                        sol=rr
                                        solution.add_route(rr)
                                        zb = zb1
                        elif not (r2.arrive_time[-2]+r2.visit[-2].service_time+r2.visit[-2].distance_to(r1.visit[1])/model.vehicle.velocity>min([j.over_time for j in r1.visit if isinstance(j,Customer)])):
                            if np.sum(r2.arrive_load_weight[-1] + r1.arrive_load_weight > model.vehicle.capacity) == 0:
                                sta = Modification.find_feasible_station_between(model, r2.visit[-2], r1.visit[1],evo)
                                #co = 0
                                for j in range(len(sta)+1):
                                    if j == len(sta):
                                        rr = Route(r2.visit[0:-1] + r1.visit[1:])
                                        zb1 = Modification.get_objective_route(rr, model.vehicle, evo.penalty)
                                    else:
                                        rr = Route(r2.visit[0:-1] + [sta[j]] + r1.visit[1:])
                                        zb1 = Modification.get_objective_route(rr, model.vehicle, evo.penalty)
                                    if (rr.feasible(model.vehicle)[0]) and (co == 0):
                                        solution.remove_route_object(r1)
                                        solution.remove_route_object(r2)
                                        solution.add_route(rr)
                                        zb = zb1
                                        sol = rr
                                        co = 1
                                    elif (zb > zb1) and co:
                                        solution.remove_route_object(sol)
                                        solution.add_route(rr)
                                        sol=rr
                                        zb = zb1
                return solution
            else:
                return solution

    @staticmethod
    #权衡距离与电量约束
    def fix_ele(model:Model,route:Route,max_count:int,jjb:list):
        r=Route([i for i in route.visit if isinstance(i,(Customer,Depot))])
        sol=[]  #存储改变客户顺序后的路径
        if r.feasible_time(model.vehicle)[0] and r.feasible_capacity(model.vehicle)[0]:
            cuslist=[i for i in r.visit if isinstance(i,Customer)]
            cuslist.sort(key=lambda x:abs(x.demand),reverse=1)
            for cus in cuslist:
                cus_index = r.visit.index(cus)
                sol1=[]
                if cus.demand>0:
                    for i in range(1,cus_index):
                        r1 = r.copy()
                        r1.del_node(model.vehicle,cus_index)
                        r1.add_node(model.vehicle,i,cus)
                        if r1.feasible_time(model.vehicle)[0]:
                            sol.append((r1,r1.sum_distance()))
                if cus.demand<0:
                    for i in range(cus_index+2,len(r)):
                        r1=r.copy()
                        r1.add_node(model.vehicle,i,cus)
                        r1.del_node(model.vehicle,cus_index)
                        if r1.feasible_time(model.vehicle)[0]:
                            sol.append((r1,r1.sum_distance()))
            if len(sol)!=0:
                sol.sort(key=lambda x:x[1])
                sol1=[i[0] for i in sol]
                for i in range(len(jjb)):
                    if sol1.count(jjb[i])!=0:
                        o=sol1.index(jjb[i])
                        del sol1[o]
                        del sol[o]
                if len(sol)!=0:
                    routelist=sol1[0:min(max_count, len(sol1))]
                    jjb.extend(routelist)
                    return routelist
                else:
                    return [route]
            else:
                return [route]
        else:
            print('fix_ele：输入的路径违背cap或time约束!')
            print(route)
            return [route]
        return routelist

class Operation:
    @staticmethod
    def choose_best_insert(vehicle:Vehicle, solution: Solution, node: Node, route_indexes: list) -> tuple:
        penalty=[1000000,100,0]
        min_increase_dis_to_route = float('inf')
        to_route = None
        insert_place_to_route = None
        for route_index in route_indexes:
            route=solution.routes[route_index].copy()
            min_increase_dis = float('inf')
            insert_place = None
            for place in range(1, len(solution.routes[route_index])):
                #increase_dis = node.distance_to(solution.routes[route_index].visit[place-1])+node.distance_to(solution.routes[route_index].visit[place])-solution.routes[route_index].visit[place-1].distance_to(solution.routes[route_index].visit[place])
                r=route.copy()
                r.add_node(vehicle,place,node)
                increase_dis =Modification.get_objective_route(r,vehicle,penalty)-Modification.get_objective_route(route,vehicle,penalty)
                if increase_dis < min_increase_dis:
                    min_increase_dis = increase_dis
                    insert_place = place
            if min_increase_dis < min_increase_dis_to_route:
                min_increase_dis_to_route = min_increase_dis
                to_route = route_index
                insert_place_to_route = insert_place
        return to_route, insert_place_to_route

    @staticmethod
    def create_test_solution(model: Model) -> Solution:
        routes = []
        for cus in model.customers:
            routes.append(Route([model.depot, cus, model.depot]))
        return Solution(routes)

    @staticmethod
    def test_model(model: Model) -> bool:
        S = Operation.create_test_solution(model)
        result = S.feasible_detail(model)
        for value in result.values():
            if value[1] != 'battery':
                return False
        S = Modification.charging_modification(S, model)
        result = S.feasible_detail(model)
        if len(result) != 0:
            return False
        return True

    @staticmethod
    def find_left_right_station(route: Route, where: int) -> tuple:
        if where == 0:
            left = route.visit[0]
        else:
            left_where = where-1
            while not(isinstance(route.visit[left_where], Recharger)) and not(isinstance(route.visit[left_where], Depot)):
                left_where -= 1
            left = route.visit[left_where]
        if where == len(route.visit)-1:
            right = route.visit[-1]
        else:
            right_where = where+1
            while not(isinstance(route.visit[right_where], Recharger)) and not(isinstance(route.visit[right_where], Depot)):
                right_where += 1
            right = route.visit[right_where]
        return (left, right)

    @staticmethod
    def find_left_station(route: Route, where: int) -> Node:
        if where == 0:
            left = route.visit[0]
        else:
            left_where = where-1
            while not(isinstance(route.visit[left_where], Recharger)) and not(isinstance(route.visit[left_where], Depot)):
                left_where -= 1
            left = route.visit[left_where]
        return left

    @staticmethod
    def find_right_station(route: Route, where: int) -> Node:
        if where == len(route.visit)-1:
            right = route.visit[-1]
        else:
            right_where = where+1
            while not(isinstance(route.visit[right_where], Recharger)) and not(isinstance(route.visit[right_where], Depot)):
                right_where += 1
            right = route.visit[right_where]
        return right

    @staticmethod
    def find_customer(solution: Solution, node: Customer) -> tuple:
        flag = True
        cur_which = 0
        while cur_which < len(solution.routes) and flag:
            cur_where = 1
            while cur_where < len(solution.routes[cur_which].visit)-1 and flag:
                cur_node = solution.routes[cur_which].visit[cur_where]
                if cur_node==node:
                    which = cur_which
                    where = cur_where
                    flag = False
                cur_where += 1
            cur_which += 1
        return which, where

    @staticmethod
    def find_recharger(solution: Solution, node: Recharger) -> list:
        recharger_which_where = []
        cur_which = 0
        while cur_which < len(solution.routes):
            cur_where = 1
            while cur_where < len(solution.routes[cur_which].visit)-1:
                cur_node = solution.routes[cur_which].visit[cur_where]
                if cur_node==node:
                    recharger_which_where.append((cur_which, cur_where))
                cur_where += 1
            cur_which += 1
        return recharger_which_where

    @staticmethod
    def find_two_customer(solution: Solution, node1: Customer, node2: Customer) -> tuple:
        flag = 0
        which = 0
        while which < len(solution.routes) and flag != 2:
            where = 1
            while where < len(solution.routes[which].visit)-1 and flag != 2:
                node = solution.routes[which].visit[where]
                if node==node1:
                    which1 = which
                    where1 = where
                    flag += 1
                elif node==node2:
                    which2 = which
                    where2 = where
                    flag += 1
                where += 1
            which += 1
        return which1, where1, which2, where2

    @staticmethod
    def find_two_recharger(solution: Solution, node1: Recharger, node2: Recharger) -> tuple:
        recharger1_which_where = []
        recharger2_which_where = []
        which = 0
        while which < len(solution.routes):
            where = 1
            while where < len(solution.routes[which].visit)-1:
                node = solution.routes[which].visit[where]
                if node==node1:
                    recharger1_which_where.append((which, where))
                elif node==node2:
                    recharger2_which_where.append((which, where))
                where += 1
            which += 1
        return recharger1_which_where, recharger2_which_where

    @staticmethod
    def find_customer_recharger(solution: Solution, node1: Customer, node2: Recharger) -> tuple:
        recharger2_which_where = []
        which = 0
        while which < len(solution.routes):
            where = 1
            while where < len(solution.routes[which].visit)-1:
                node = solution.routes[which].visit[where]
                if node==node1:
                    which1 = which
                    where1 = where
                elif node==node2:
                    recharger2_which_where.append((which, where))
                where += 1
            which += 1
        return which1, where1, recharger2_which_where



