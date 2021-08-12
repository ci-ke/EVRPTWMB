from .model import *
from .util import *


class Modification:
    @staticmethod
    def cyclic_exchange(solution: Solution, model: Model, Rts: int, max: int) -> Solution:
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
        if ret_sol.routes[which].visit[where-1].eq(recharger):
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
        if len(solution.routes) > 1:
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
                    #solution.routes[to_route].visit.insert(insert_place_to_route, node)
                    solution.routes[to_route].add_node(model.vehicle, insert_place_to_route, node)
            solution.remove_route_index(select)
        solution.clear_status()
        return solution

    @staticmethod
    def ACO_GM_cross2(solution1: Solution, solution2: Solution, model: Model) -> Solution:
        solution1 = solution1.copy()
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

        random.shuffle(visit_cus_list)
        for node in visit_cus_list:
            to_route, insert_place_to_route = Operation.choose_best_insert(solution1, node, list(range(len(solution1.routes))))
            #solution1.routes[to_route].visit.insert(insert_place_to_route, node)
            solution1.routes[to_route].add_node(model.vehicle, insert_place_to_route, node)
        solution1.remove_empty_route()
        solution1.clear_status()
        return solution1

    @staticmethod
    def charging_modification(solution: Solution, model: Model) -> Solution:
        solution = solution.copy()
        ready_to_remove = []
        for route in solution.routes:
            if route.feasible_capacity(model.vehicle)[0] and route.feasible_time(model.vehicle)[0] and not route.feasible_battery(model.vehicle)[0]:
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
                        assert len(route.visit) != 3, "unreasonable model"
                        cut_point = common_insert[-1]
                        if cut_point == len(route.visit)-1:
                            cut_point -= 1
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
        assert not node1.eq(node2)
        if isinstance(node1, Customer) and isinstance(node2, Customer):
            which1, where1, which2, where2 = Operation.find_two_customer(solution, node1, node2)
            if which1 == which2:
                return [], []
            else:
                sol = Modification.two_opt_star_action(solution, model, which1, where1, which2, where2-1)
                act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                return [sol], [act]
        elif isinstance(node1, Customer) and isinstance(node2, Recharger):
            which1, where1, recharger2_which_where = Operation.find_customer_recharger(solution, node1, node2)
            ret_sol = []
            ret_act = []
            for which2, where2 in recharger2_which_where:
                if which1 != which2:
                    sol = Modification.two_opt_star_action(solution, model, which1, where1, which2, where2-1)
                    ret_sol.append(sol)
                    act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                    ret_act.append(act)
            return ret_sol, ret_act
        elif isinstance(node1, Customer) and isinstance(node2, Depot):
            which1, where1 = Operation.find_customer(solution, node1)
            if where1 == len(solution.routes[which1].visit)-2:
                return [], []
            ret_sol = []
            ret_act = []
            solution = solution.copy()
            solution.add_empty_route(model)
            which2 = 0
            while which2 < len(solution.routes):
                if which1 != which2:
                    where2 = len(solution.routes[which2])-1
                    sol = Modification.two_opt_star_action(solution, model, which1, where1, which2, where2-1)
                    ret_sol.append(sol)
                    act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                    ret_act.append(act)
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
                    act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                    ret_act.append(act)
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
                        act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                        ret_act.append(act)
            return ret_sol, ret_act
        elif isinstance(node1, Recharger) and isinstance(node2, Depot):
            recharger1_which_where = Operation.find_recharger(solution, node1)
            ret_sol = []
            ret_act = []
            solution = solution.copy()
            solution.add_empty_route(model)
            for which1, where1 in recharger1_which_where:
                if where1 == len(solution.routes[which1].visit)-2:
                    continue
                which2 = 0
                while which2 < len(solution.routes):
                    if which1 != which2:
                        where2 = len(solution.routes[which2])-1
                        sol = Modification.two_opt_star_action(solution, model, which1, where1, which2, where2-1)
                        ret_sol.append(sol)
                        act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                        ret_act.append(act)
                    which2 += 1
            return ret_sol, ret_act
        elif isinstance(node1, Depot) and isinstance(node2, Customer):
            which2, where2 = Operation.find_customer(solution, node2)
            if where2 == 1:
                return [], []
            ret_sol = []
            ret_act = []
            solution = solution.copy()
            solution.add_empty_route(model)
            which1 = 0
            while which1 < len(solution.routes):
                if which1 != which2:
                    sol = Modification.two_opt_star_action(solution, model, which1, 0, which2, where2-1)
                    ret_sol.append(sol)
                    act = ((node1, node2), solution.id[which1], node1, Operation.find_right_station(solution.routes[which1], 0))
                    ret_act.append(act)
                which1 += 1
            return ret_sol, ret_act
        elif isinstance(node1, Depot) and isinstance(node2, Recharger):
            recharger2_which_where = Operation.find_recharger(solution, node2)
            ret_sol = []
            ret_act = []
            solution = solution.copy()
            solution.add_empty_route(model)
            which1 = 0
            while which1 < len(solution.routes):
                for which2, where2 in recharger2_which_where:
                    if where2 == 1:
                        continue
                    if which1 != which2:
                        sol = Modification.two_opt_star_action(solution, model, which1, 0, which2, where2-1)
                        ret_sol.append(sol)
                        act = ((node1, node2), solution.id[which1], node1, Operation.find_right_station(solution.routes[which1], 1))
                        ret_act.append(act)
                which1 += 1
            return ret_sol, ret_act

    @staticmethod
    def relocate_arc(model: Model, solution: Solution, node1: Node, node2: Node) -> tuple:
        '''
        去掉a，插入到b前，路间路内，客户与电站
        '''
        assert not node1.eq(node2)
        if isinstance(node1, Depot):
            return [], []
        elif isinstance(node1, Customer) and isinstance(node2, Customer):
            which1, where1, which2, where2 = Operation.find_two_customer(solution, node1, node2)
            if which1 == which2 and where2 == where1+1:
                return [], []
            sol = Modification.relocate_action(solution, model, which1, where1, which2, where2)
            act = ((node1, node2), solution.id[which2], *Operation.find_left_right_station(solution.routes[which2], where2))
            return [sol], [act]
        elif isinstance(node1, Customer) and isinstance(node2, Depot):
            which1, where1 = Operation.find_customer(solution, node1)
            ret_sol = []
            ret_act = []
            if len(solution.routes[which1].visit) != 3:
                solution = solution.copy()
                solution.add_empty_route(model)
            which2 = 0
            while which2 < len(solution.routes):
                if not (which2 == which1 and where1 == len(solution.routes[which2].visit)-2):
                    where2 = len(solution.routes[which2].visit)-1
                    sol = Modification.relocate_action(solution, model, which1, where1, which2, where2)
                    act = ((node1, node2), solution.id[which2], Operation.find_left_station(solution.routes[which2], where2), node2)
                    ret_sol.append(sol)
                    ret_act.append(act)
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
                    act = ((node1, node2), solution.id[which2], *Operation.find_left_right_station(solution.routes[which2], where2))
                    ret_act.append(act)
            return ret_sol, ret_act
        elif isinstance(node1, Recharger) and isinstance(node2, Customer):
            which2, where2, recharger1_which_where = Operation.find_customer_recharger(solution, node2, node1)
            ret_sol = []
            ret_act = []
            for which1, where1 in recharger1_which_where:
                if not (which1 == which2 and where2 == where1+1):
                    sol = Modification.relocate_action(solution, model, which1, where1, which2, where2)
                    ret_sol.append(sol)
                    act = ((node1, node2), solution.id[which2], *Operation.find_left_right_station(solution.routes[which2], where2))
                    ret_act.append(act)
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
                        act = ((node1, node2), solution.id[which2], Operation.find_left_station(solution.routes[which2], where2), node2)
                        ret_act.append(act)
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
                        act = ((node1, node2), solution.id[which2], *Operation.find_left_right_station(solution.routes[which2], where2))
                        ret_act.append(act)
            return ret_sol, ret_act

    @staticmethod
    def exchange_arc(model: Model, solution: Solution, node1: Node, node2: Node) -> tuple:
        '''
        a后与b交换，路间路内，只有客户
        '''
        assert not node1.eq(node2)
        if isinstance(node2, Customer):
            if isinstance(node1, Customer):
                which1, where1, which2, where2 = Operation.find_two_customer(solution, node1, node2)
                if (not isinstance(solution.routes[which1].visit[where1+1], Customer)) or where1 == len(solution.routes[which1])-2 or (which1 == which2 and where2 == where1+1):
                    return [], []
                else:
                    sol = Modification.exchange_action(solution, model, which1, where1+1, which2, where2)
                    act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                    return [sol], [act]
            elif isinstance(node1, Depot):
                which2, where2 = Operation.find_customer(solution, node2)
                ret_sol = []
                ret_act = []
                which1 = 0
                while which1 < len(solution.routes):
                    if isinstance(solution.routes[which1].visit[1], Customer) and not (where2 == 1 and which1 == which2):
                        sol = Modification.exchange_action(solution, model, which1, 1, which2, where2)
                        ret_sol.append(sol)
                        act = ((node1, node2), solution.id[which1], node1, Operation.find_right_station(solution.routes[which1], 0))
                        ret_act.append(act)
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
                        act = ((node1, node2), solution.id[which1], *Operation.find_left_right_station(solution.routes[which1], where1))
                        ret_act.append(act)
                return ret_sol, ret_act
            else:
                raise Exception('impossible')
        else:
            return [], []

    @staticmethod
    def stationInRe_arc(model: Model, solution: Solution, node1: Recharger, node2: Node) -> tuple:
        assert not node1.eq(node2)
        if not isinstance(node1, Recharger):
            return [], []
        if isinstance(node2, Customer):
            which2, where2 = Operation.find_customer(solution, node2)
            if where2 == 1:
                depot = solution.routes[0].visit[0]
                if node1.x == depot.x and node1.y == depot.y:
                    return [], []
            sol = Modification.stationInRe_action(solution, model, node1, which2, where2)
            act = ((node1, node2), solution.id[which2], *Operation.find_left_right_station(solution.routes[which2], where2))
            return [sol], [act]
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
                act = ((node1, node2), solution.id[cur_which], Operation.find_left_station(solution.routes[cur_which], cur_where), node2)
                ret_act.append(act)
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
                act = ((node1, node2), solution.id[which2], *Operation.find_left_right_station(solution.routes[which2], where2))
                ret_act.append(act)
            return ret_sol, ret_act
        else:
            raise Exception('impossible')


class Operation:
    @staticmethod
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
                if cur_node.eq(node):
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
                if cur_node.eq(node):
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
                if node.eq(node1):
                    which1 = which
                    where1 = where
                    flag += 1
                elif node.eq(node2):
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
                if node.eq(node1):
                    recharger1_which_where.append((which, where))
                elif node.eq(node2):
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
                if node.eq(node1):
                    which1 = which
                    where1 = where
                elif node.eq(node2):
                    recharger2_which_where.append((which, where))
                where += 1
            which += 1
        return which1, where1, recharger2_which_where
