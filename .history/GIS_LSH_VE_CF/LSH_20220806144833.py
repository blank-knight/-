# -*- coding: utf-8 -*-
'''
    第一种加密方案得嵌入，失败，运行时间太久。
'''
from pickletools import read_uint1
from tracemalloc import start
from grapheme import length
from matplotlib import style
import numpy as np
from numba import jit
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as MAE
import copy
import time
import random
from phe import paillier
from K_means_train import K_means_trian
from vector_encrypt import vector_encrypt
public_key,private_key = paillier.generate_paillier_keypair()


def hash_function(num=4,nbits=8,d=2):
    '''
        构建LSH哈希映射函数，d:数据维度，nbits:编码后的bit位数，num:哈希表个数，采用均匀分布
        注意：这里的np.random.rand产生的随机数是一样的，伪随机，建议设置随机数种子试试
    '''
    plane_norms_groups = np.empty([num,nbits,d])
    for i in range(num):
        plane_norms_groups[i] = np.random.rand(nbits, d) - .5 # 每一张hash table里的均匀分布向量都不一样
    return plane_norms_groups



def LSH(data,plane_norms_groups,nbits,num):
    '''
        进行hash映射，hash后返回的用户索引是随机的，没有顺序的
    '''
    print("当前的nbits为:",nbits)
    value_id = []
    value = data.values
    value = np.delete(value, -1, axis=1) # 最后一列是location，这里不需要，就将其删除
    value_id = value[:,0]
    # 进行hash映射
    value_dot_groups = np.empty([num,data.shape[0],nbits])
    for i in range(num):
        print(value[:,1::].shape)
        print(plane_norms_groups[i].T.shape)
        value_dot_groups[i] = np.dot(value[:,1::], plane_norms_groups[i].T) # value_dot_group是按照用户顺序进行映射的，所以前面的用户可能是相同的
    # 哈希映射后编码
    value_dot_groups = value_dot_groups > 0 # 注：value_dot_groups本身就是num张hash表映射后的结果
    value_dot_groups = value_dot_groups.astype(int)

    lshTable = {}
    i = 0
    for vectors in value_dot_groups: # value_dot_groups:4x111166x8  vectors:111166x8
        buckets = {}
        for j in range(vectors.shape[0]):
            # 防止一个桶里有重复用户, 这里有个小问题，一个用户可能会被分到多个桶里，
            # 而同一个桶里不能有重复用户，所以最后bucket用户数相加总是大于等于总用户数
            hash_str = ''.join(vectors[j].astype(str)) # hash_str就是当前用户映射到的hash桶
            if hash_str not in buckets.keys():
                buckets[hash_str] = []
            if value_id[j] not in buckets[hash_str]: 
                buckets[hash_str].append(value_id[j]) # 将索引添加进去
            else:
                continue    
        # 这里对用户进行排序，之后np索引时就不需要根据用户索引来查找数据了
        for k in buckets.keys():  
            buckets[k] = sorted(buckets[k])
        lshTable[i] = buckets
        i += 1
    return lshTable





class sim_cal():
    def __init__(self,user_mx) -> None:
        self.user_mx = user_mx


    def calSim(self,userId1,userId2):
        '''
            计算相似度
        '''
        # 两个用户经纬度矩阵信息
        user1Items,user2Items = self.splicing(userId1,userId2)
        #两个物品共同用户
        user1Items = self.std(user1Items.astype('float'))
        user2Items = self.std(user2Items.astype('float'))
        # print("当前用户是:{0}和{1}".format(userId1,userId2))
        res = (userId2,1/(1+math.sqrt(np.sum((user1Items-user2Items)**2))/user1Items.shape[0])) 

        return res

    def encrypt_calSim(self,userId1,userId2):
        '''
            计算相似度
        '''
        # 两个用户经纬度矩阵信息
        start_time = time.time()
        user1Items,user2Items = self.splicing(userId1,userId2)
        #两个物品共同用户
        user1Items_row = user1Items.shape[0]
        user2Items_row = user2Items.shape[0]
        if user1Items_row > user2Items_row: # 将短的向量取平均经纬度进行填充扩容
            user2Items = self.fill(user2Items,user1Items.shape[0])
        else:
            user1Items = self.fill(user1Items,user2Items.shape[0])
        user1Items = self.std(user1Items.astype('float'))
        user2Items = self.std(user2Items.astype('float'))
        end_time = time.time()
        m = user1Items.shape[0]
        n = m
        w = 16
        S = vec_encrypt.generate_key(w,m,n)
        T = vec_encrypt.get_T(n)
        c1,S = mx_encrypt(user1Items,w,m,n,T)
        c2,S = mx_encrypt(user2Items,w,m,n,T)
        total_time = end_time-start_time
        start_time = time.time()
        res = (userId2,1/(1+math.sqrt(np.sum(vec_encrypt.mx_decrypt((c1-c2),S,w)**2))/user1Items.shape[0])) 
        end_time = time.time()
        total_time += (end_time-start_time)
        return res,total_time

