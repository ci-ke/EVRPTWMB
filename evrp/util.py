import random
import numpy as np
import geatpy as ea


class Util:
    @staticmethod
    def cal_angle_AoB(A: tuple, o: tuple, B: tuple) -> float:
        oA = np.array([A[0]-o[0], A[1]-o[1]])
        oB = np.array([B[0]-o[0], B[1]-o[1]])
        oA_norm = np.linalg.norm(oA)
        oB_norm = np.linalg.norm(oB)
        rad = np.arccos(oA.dot(oB)/(oA_norm*oB_norm))
        deg = np.rad2deg(rad)
        return deg

    class SA:
        T0 = 0.0
        delta_T = 0.0

        def __init__(self, Delta_SA: float, max_iter: int) -> None:
            self.T0 = Delta_SA/np.log(2)
            self.Delta_T = (self.T0-0.0001)/0.8*max_iter

        def probability(self, V_new: float, V: float, iter: int) -> float:
            return np.exp(-(V_new-V)/(self.T0-self.delta_T*iter))

    @staticmethod
    def wheel_select(elements: np.array) -> int:
        sum_elements = np.cumsum(elements)
        select = random.uniform(0, sum_elements[-1])
        sum_elements -= select
        choose = np.where(sum_elements > 0)[0][0]
        return choose

    @staticmethod
    def dominate(a: list, b: list) -> bool:
        have_small = False
        for x, y in zip(a, b):
            if not have_small and x < y:
                have_small = True
            if x > y:
                return False
        if have_small:
            return True
        else:
            return False

    @staticmethod
    def binary_tournament(length: int) -> list:
        '''
        length代表参加竞争的人数，在列表中的索引代表排名\n
        返回获胜者的索引
        '''
        pk = list(range(length))
        random.shuffle(pk)
        pk_win = []
        for i in range(0, length-1, 2):
            if pk[i] < pk[i+1]:
                pk_win.append(pk[i])
            else:
                pk_win.append(pk[i+1])
        return pk_win

    @staticmethod
    def pareto_sort(P: list, objv: list, needNum: int = None, needLevel: int = None):
        if len(objv) <= 1:
            return P
        objv = np.array(objv)
        levels, criLevel = ea.ndsortESS(objv, needNum, needLevel)
        dis = ea.crowdis(objv, levels)
        sortP = []
        for lv in range(1, criLevel):
            indexs = np.where(levels == lv)[0]
            indexs_sorted = sorted(indexs, key=lambda x: dis[x], reverse=True)
            for i in indexs_sorted:
                sortP.append(P[i])
        indexs = np.where(levels == criLevel)[0]
        indexs_sorted = sorted(indexs, key=lambda x: dis[x], reverse=True)
        if needNum is None:
            needNum = len(P)
        for i in indexs_sorted:
            if len(sortP) < needNum:
                sortP.append(P[i])
        return sortP
