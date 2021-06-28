import random
import numpy as np


def cal_angle_AoB(A: tuple, o: tuple, B: tuple):
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

    def probability(self, S_new: float, S: float, iter: int) -> float:
        return np.exp(-(S_new-S)/(self.T0-self.delta_T*iter))


def wheel_select(elements: np.array) -> int:
    sum_elements = np.cumsum(elements)
    select = random.uniform(0, sum_elements[-1])
    sum_elements -= select
    choose = np.where(sum_elements > 0)[0][0]
    return choose
