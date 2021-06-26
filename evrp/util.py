import numpy as np


def cal_angle_AoB(A: tuple, o: tuple, B: tuple):
    oA = np.array([A[0]-o[0], A[1]-o[1]])
    oB = np.array([B[0]-o[0], B[1]-o[1]])
    oA_norm = np.linalg.norm(oA)
    oB_norm = np.linalg.norm(oB)
    rad = np.arccos(oA.dot(oB)/(oA_norm*oB_norm))
    deg = np.rad2deg(rad)
    return deg
