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
    # flag = 1
    for i in range(num):
        plane_norms_groups[i] = random_point(np.zeros([nbits,d])) # 每一张hash table里的均匀分布向量都不一样
        # round_draw(plane_norms_groups[i])
        # line_draw(plane_norms_groups[i])
    # print(plane_norms_groups)
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
    # for i in range(num): # 遍历num个hash表
    #     buckets = {}
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