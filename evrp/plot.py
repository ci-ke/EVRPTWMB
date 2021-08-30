import matplotlib.pyplot as plt

from .model import *


class Plot:
    @staticmethod
    def map(model: Model):
        plt.scatter(model.depot.x, model.depot.y, marker='*', c='red', s=150)
        for cus in model.customers:
            plt.scatter(cus.x, cus.y, marker='.', c='blue')
        for rec in model.rechargers:
            plt.scatter(rec.x, rec.y, marker='^', c='green')
        plt.show()

    @staticmethod
    def map_sol(icecube):
        model = icecube[0]
        solution = icecube[1]
        print(len(solution))

        assert isinstance(model, Model)
        assert isinstance(solution, Solution)

        node_matrix = pd.read_excel('data/jd/input_node.xlsx', sheet_name=['Customer_data'])['Customer_data']

        selected_id = [0]
        for node in model.customers+model.rechargers:
            selected_id.append(node.id)

        for cus_info in node_matrix.itertuples():
            node_id = int(cus_info[1])
            node_type = int(cus_info[2])
            x = float(cus_info[3])
            y = float(cus_info[4])
            if node_id in selected_id:
                if node_type == 1:
                    model.depot.x = x
                    model.depot.y = y
                elif node_type == 2 or node_type == 3:
                    model.get_customer(node_id).x = x
                    model.get_customer(node_id).y = y
                elif node_type == 4:
                    model.get_recharger(node_id).x = x
                    model.get_recharger(node_id).y = y

        plt.scatter(model.depot.x, model.depot.y, marker='*', c='red', s=150)
        for cus in model.customers:
            plt.scatter(cus.x, cus.y, marker='.', c='blue', s=75)
        for rec in model.rechargers:
            plt.scatter(rec.x, rec.y, marker='^', c='green', s=75)

        for route in solution:
            x = []
            y = []
            for node in route:
                x.append(node.x)
                y.append(node.y)
            plt.plot(x, y)

        plt.show()
