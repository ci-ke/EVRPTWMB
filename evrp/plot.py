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
