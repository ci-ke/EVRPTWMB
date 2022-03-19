import random
import pickle
import numpy as np
#import geatpy as ea
import argparse
from Model import *
import os

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
        Delta_T = 0.0

        def __init__(self, Delta_SA: float, max_iter: int) -> None:
            self.T0 = Delta_SA/np.log(2)
            self.Delta_T = (self.T0-0.0001)/max_iter

        def probability(self, V_new: float, V: float, iter: int) -> float:
            return np.exp(-(V_new-V)/(100*(self.T0-self.Delta_T*iter)))

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
    def map_100(a):
        a1=np.max(a)
        a2=np.min(a)
        length=a1-a2
        assert length!=0
        for i in range(len(a)):
            a[i]=(a[i]-a2)/length*100
        return a


    #@staticmethod
    #def pareto_sort(P: list, objv: list, needNum: int = None, needLevel: int = None):
    #    if len(objv) <= 1:
    #        return P
    #    objv = np.array(objv)
    #    levels, criLevel = ea.ndsortESS(objv, needNum, needLevel)
    #    dis = ea.crowdis(objv, levels)
    #    sortP = []
    #    for lv in range(1, criLevel):
    #        indexs = np.where(levels == lv)[0]
    #        indexs_sorted = sorted(indexs, key=lambda x: dis[x], reverse=True)
    #        for i in indexs_sorted:
    #            sortP.append(P[i])
    #    indexs = np.where(levels == criLevel)[0]
    #    indexs_sorted = sorted(indexs, key=lambda x: dis[x], reverse=True)
    #    if needNum is None:
    #        needNum = len(P)
    #    for i in indexs_sorted:
    #       if len(sortP) < needNum:
    #            sortP.append(P[i])
    #   return sortP

    @staticmethod
    def pareto_sort(P: list, objv: list):
        assert len(P) == len(objv)
        c = []
        m = np.zeros([len(P), len(P)])
        for i in range(len(objv)):
            for j in range(i + 1, len(objv)):
                if ((objv[i][1] < objv[j][1]) & (objv[i][0] < objv[j][0])) or (
                        (objv[i][1] <= objv[j][1]) & (objv[i][0] < objv[j][0])) or (
                        (objv[i][1] < objv[j][1]) & (objv[i][0] <= objv[j][0])):
                    m[i][j] = 1
                    m[j][i] = 0
                elif ((objv[i][1] > objv[j][1]) & (objv[i][0] > objv[j][0])) or (
                        (objv[i][1] >= objv[j][1]) & (objv[i][0] > objv[j][0])) or (
                        (objv[i][1] > objv[j][1]) & (objv[i][0] >= objv[j][0])):
                    m[i][j] = 0
                    m[j][i] = 1
                else:
                    m[i][j] = 0
                    m[j][i] = 0
        index = []
        while len(c) != len(P):
            for i in range(len(objv)):
                if sum(m[:, i]) == 0:
                    index.append(i)
            m[:, index] = 1
            m[index, :] = 0
            index.sort(key=lambda j: objv[j][1])
            for i in range(len(index)):
                c.append(P[index[i]])
            index = []
        assert len(c) == len(P)
        return c

    @staticmethod
    def set_model(o:list,n:int=0):
        '''
        建立模型
        :param o: o[0]:数据集类型,o[1]:数据集路径
        :return:
        '''
        if o[0]=='tw':
            model=Model(o[1], 'tw')
        elif o[0]=='p':
            model=Model(o[1],'p',n)
        elif o[0]=='jd':
            model=Model('','jd',2)
        elif o[0] in ['e','s5','s10','s15']:
            model=Model(o[1],o[0],n)
        model.read_data()
        return model



    @staticmethod
    def process_input(input_list: list):
        parser = argparse.ArgumentParser()
        new_run = parser.add_mutually_exclusive_group()
        control_run = parser.add_mutually_exclusive_group()

        parser.add_argument('file_type', metavar='file_type', choices=['e', 's5', 's10', 's15', 'tw', 'p', 'jd'])
        parser.add_argument('map_name', nargs='?')

        new_run.add_argument('-n', '--new', action='store_true')
        new_run.add_argument('-c', '--continue', action='store_true', dest='conti')

        parser.add_argument('-d', '--negtive_demand', type=int, default=0)
        parser.add_argument('-r', '--read_suffix', default='')
        parser.add_argument('-s', '--save_suffix', default='')

        control_run.add_argument('--ga', action='store_true')
        control_run.add_argument('--ts', action='store_true')

        args = parser.parse_args(input_list[1:])

        file_type = args.file_type
        map_name = args.map_name
        if file_type != 'jd' and map_name is None:
            print('must give a map name')
            exit()
        if args.new:
            mode = 'n'
        elif args.conti:
            mode = 'c'
        else:
            mode = 'n'
        read_suffix = args.read_suffix
        save_suffix = args.save_suffix
        if len(save_suffix) != 0:
            save_suffix = '_'+save_suffix
        negative_demand = args.negtive_demand

        if args.ga:
            control = (True, False)
            save_suffix = '_ga'+save_suffix
        elif args.ts:
            control = (False, True)
            save_suffix = '_ts'+save_suffix
        else:
            control = (True, True)

        if file_type == 's5':
            folder = 'data/small_evrptw_instances/Cplex5er/'
            filename = map_name+'C5.txt'
        elif file_type == 's10':
            folder = 'data/small_evrptw_instances/Cplex10er/'
            filename = map_name+'C10.txt'
        elif file_type == 's15':
            folder = 'data/small_evrptw_instances/Cplex15er/'
            filename = map_name+'C15.txt'
        elif file_type == 'e':
            folder = 'data/evrptw_instances/'
            filename = map_name+'_21.txt'
        elif file_type == 'tw':
            folder = 'data/solomon/'
            filename = map_name+'.txt'
        elif file_type == 'p':
            folder = 'data/p/'
            filename = map_name+'.txt'
        elif file_type == 'jd':
            folder = 'data/jd/'
            filename = 'jd.txt'

        filepath = folder+filename

        if mode == 'n':
            icecube = None
        elif mode == 'c':
            icecube = pickle.load(open('result/{}/{}{}_evo{}.pickle'.format(file_type, filename.split('.')[0], '' if negative_demand == 0 else '_neg'+str(negative_demand), read_suffix), 'rb'))

        return filepath, file_type, icecube, negative_demand, save_suffix, control
