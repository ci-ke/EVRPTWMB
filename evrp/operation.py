from .model import *
from .util import *


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

        ret_sol.remove_empty_route()

        return ret_sol

    @staticmethod
    def two_opt_star_action(solution: Solution, first_which: int, first_where: int, second_which: int, second_where: int) -> Solution:
        assert first_where >= 0 and first_where <= len(solution.routes[first_which].visit)-2
        assert second_where >= 0 and second_where <= len(solution.routes[second_which].visit)-2
        ret_sol = solution.copy()
        ret_sol.routes[first_which].visit[first_where+1:] = solution.routes[second_which].visit[second_where+1:]
        ret_sol.routes[second_which].visit[second_where+1:] = solution.routes[first_which].visit[first_where+1:]

        if ret_sol.routes[first_which].no_customer():
            del ret_sol.routes[first_which]
        elif ret_sol.routes[second_which].no_customer():
            del ret_sol.routes[second_which]

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
    def relocate_action(solution: Solution, which: int, where: int, new_which: int, new_where: int) -> Solution:
        ret_sol = solution.copy()
        if new_which != which:
            ret_sol[new_which].visit.insert(new_where, solution.routes[which].visit[where])
            del ret_sol.routes[which].visit[where]
            if ret_sol.routes[which].no_customer():
                del ret_sol.routes[which]
        else:
            ret_sol[which].visit.insert(new_where, solution.routes[which].visit[where])
            if new_where > where:
                del ret_sol.routes[which].visit[where]
            else:
                del ret_sol.routes[which].visit[where+1]
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
    def exchange_action(solution: Solution, which1: int, where1: int, which2: int, where2: int) -> Solution:
        ret_sol = solution.copy()
        ret_sol.routes[which1].visit[where1] = solution.routes[which2].visit[where2]
        ret_sol.routes[which2].visit[where2] = solution.routes[which1].visit[where1]
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
    def two_opt_action(solution: Solution, which: int, where1: int, where2: int) -> Solution:
        ret_sol = solution.copy()
        ret_sol.routes[which].visit[where1:where2+1] = reversed(solution.routes[which].visit[where1:where2+1])
        for route in ret_sol.routes:
            if isinstance(route.visit[1], Recharger) and route.visit[1].x == route.visit[0].x and route.visit[1].y == route.visit[0].y:
                del route.visit[1]
            if isinstance(route.visit[-2], Recharger) and route.visit[-2].x == route.visit[0].x and route.visit[-2].y == route.visit[0].y:
                del route.visit[-2]

        return ret_sol

    @staticmethod
    def stationInRe(solution: Solution) -> Solution:
        return solution

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
            select = Util.wheel_select(avg_dis)
            rest_routes_index = list(range(len(solution.routes)))
            rest_routes_index.remove(select)
            visit_list = solution.routes[select].visit[1:-1]
            random.shuffle(visit_list)
            for node in visit_list:
                if isinstance(node, Customer):
                    to_route, insert_place_to_route = Operation.choose_best_insert(solution, node, rest_routes_index)
                    solution.routes[to_route].visit.insert(insert_place_to_route, node)
            del solution.routes[select]
        solution.clear_status()
        return solution

    @staticmethod
    def ACO_GM_cross2(solution1: Solution, solution2: Solution) -> Solution:
        solution1 = solution1.copy()
        avg_dis_reciprocal = np.zeros(len(solution2.routes), dtype=float)
        for i, route in enumerate(solution2.routes):
            avg_dis_reciprocal[i] = 1/route.avg_distance()
        avg_dis_reciprocal = avg_dis_reciprocal/np.sum(avg_dis_reciprocal)
        select = Util.wheel_select(avg_dis_reciprocal)
        for node in solution2.routes[select].visit[1:-1]:
            if isinstance(node, Customer):
                for route in solution1.routes:
                    if node in route:
                        route.visit.remove(node)
                        break
        visit_list = solution2.routes[select].visit[1:-1]
        random.shuffle(visit_list)
        for node in visit_list:
            if isinstance(node, Customer):
                to_route, insert_place_to_route = Operation.choose_best_insert(solution1, node, list(range(len(solution1.routes))))
                solution1.routes[to_route].visit.insert(insert_place_to_route, node)
        solution1.remove_empty_route()
        solution1.clear_status()
        return solution1

    @staticmethod
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

    @staticmethod
    def overlapping_degree_population(solution: Solution, population: list) -> float:
        sum = 0
        for p in population:
            sum += Operation.overlapping_degree(solution, p)
        return sum/len(population)

    # @staticmethod
    # def similarity_degree(solution: Solution, population: list) -> float:
    #    solarcs = []
    #    for route in solution.routes:
    #        for i in range(len(route.visit)-1):
    #            solarcs.append((route.visit[i], route.visit[i+1]))
    #    poparcs = []
    #    for popsol in population:
    #        for route in popsol.routes:
    #            for i in range(len(route.visit)-1):
    #                poparcs.append((route.visit[i], route.visit[i+1]))
    #    up = 0
    #    down = 0
    #    for arc in solarcs:
    #        if arc in poparcs:
    #            down += 1
    #            up += poparcs.count(arc)
    #    return up/down

    @staticmethod
    def find_near_station_between(node1: Node, node2: Node, model: Model) -> Recharger:
        min_dis = float('inf')
        min_station = None
        for station in model.rechargers:
            if station == node1 or station == node2 or (isinstance(node1, Depot) and station.x == node1.x and station.y == node1.y) or (isinstance(node2, Depot) and station.x == node2.x and station.y == node2.y):
                continue
            dis = node1.distance_to(station)+node2.distance_to(station)
            if dis < min_dis:
                min_dis = dis
                min_station = station
        return min_station

    @staticmethod
    def charging_modification(solution: Solution, model: Model) -> Solution:
        solution = solution.copy()
        ready_to_remove = []
        for route in solution.routes:
            if route.feasible_weight(model.vehicle)[0] and route.feasible_time(model.vehicle)[0] and not route.feasible_battery(model.vehicle)[0]:
                left_fail_index = np.where(route.arrive_remain_battery < 0)[0][0]
                left = np.where(route.rechargers < left_fail_index)[0]
                if len(left) != 0:
                    left = left[-1]
                else:
                    left = 0
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
                        left_choose.append((node_i, Operation.find_near_station_between(route.visit[node_i], route.visit[node_i-1], model)))
                    for node_i in right_insert:
                        right_choose.append((node_i, Operation.find_near_station_between(route.visit[node_i], route.visit[node_i-1], model)))
                    for left in left_choose:
                        for right in right_choose:
                            route.visit.insert(left[0], left[1])
                            route.visit.insert(right[0]+1, right[1])  # 因为左侧插入了一个
                            route.clear_status()
                            if route.feasible_battery(model.vehicle)[0]:
                                break
                            else:
                                del route.visit[right[0]+1]
                                del route.visit[left[0]]
                        else:
                            continue
                        break
                    else:
                        assert len(route.visit) != 3, "unreasonable model"
                        cut_point = left_insert[-1]
                        if cut_point == len(route.visit)-1:
                            cut_point -= 1
                        solution.routes.append(Route(route.visit[0:cut_point]+[model.depot]))
                        assert len(solution[-1].visit) != 2
                        solution.routes.append(Route([model.depot]+route.visit[cut_point:]))
                        assert len(solution[-1].visit) != 2
                        ready_to_remove.append(route)
                elif len(common_insert) == 0 and len(right_insert) == 0:
                    choose = []
                    for node_i in left_insert:
                        choose.append((node_i, Operation.find_near_station_between(route.visit[node_i], route.visit[node_i-1], model)))
                    for pair in choose:
                        route.visit.insert(pair[0], pair[1])
                        route.clear_status()
                        if route.feasible_battery(model.vehicle)[0]:
                            break
                        else:
                            del route.visit[pair[0]]
                    else:
                        assert len(route.visit) != 3, "unreasonable model"
                        cut_point = left_insert[-1]
                        if cut_point == len(route.visit)-1:
                            cut_point -= 1
                        solution.routes.append(Route(route.visit[0:cut_point]+[model.depot]))
                        assert len(solution[-1].visit) != 2
                        solution.routes.append(Route([model.depot]+route.visit[cut_point:]))
                        assert len(solution[-1].visit) != 2
                        ready_to_remove.append(route)
                elif len(common_insert) != 0:
                    common_insert.sort()
                    choose = []
                    for node_i in common_insert:
                        choose.append((node_i, Operation.find_near_station_between(route.visit[node_i], route.visit[node_i-1], model)))
                    for pair in choose:
                        route.visit.insert(pair[0], pair[1])
                        route.clear_status()
                        if route.feasible_battery(model.vehicle)[0]:
                            break
                        else:
                            del route.visit[pair[0]]
                    else:
                        assert len(route.visit) != 3, "unreasonable model"
                        cut_point = common_insert[-1]
                        if cut_point == len(route.visit)-1:
                            cut_point -= 1
                        solution.routes.append(Route(route.visit[0:cut_point]+[model.depot]))
                        assert len(solution[-1].visit) != 2
                        solution.routes.append(Route([model.depot]+route.visit[cut_point:]))
                        assert len(solution[-1].visit) != 2
                        ready_to_remove.append(route)
                else:
                    raise Exception('impossible')
        for route in ready_to_remove:
            solution.routes.remove(route)

        solution.remove_empty_route()

        return solution

    def create_test_solution(model: Model) -> Solution:
        routes = []
        for cus in model.customers:
            routes.append(Route([model.depot, cus, model.depot]))
        return Solution(routes)

    def test_model(model: Model) -> bool:
        S = Operation.create_test_solution(model)
        result = S.feasible_detail(model)
        for value in result.values():
            if value[1] != 'battery':
                return False
        S = Operation.charging_modification(S, model)
        result = S.feasible_detail(model)
        if len(result) != 0:
            return False
        return True

    def fix_time(solution: Solution, model: Model) -> Solution:
        solution = solution.copy()

        for route in solution.routes:
            if route.feasible_time(model.vehicle)[0] == False:
                cut = route.feasible_time(model.vehicle)[1]
                new_route = [model.depot]+route.visit[cut:]
                route.visit[cut-1:] = [model.depot]
                route.clear_status()
                solution.routes.append(Route(new_route))

        solution.remove_empty_route()

        return solution
