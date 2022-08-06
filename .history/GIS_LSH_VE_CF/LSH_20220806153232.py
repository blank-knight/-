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
import time
from K_means_train import K_means_trian
from vector_encrypt import vector_encrypt
import math


def time_cal(func):
    '''
        计算函数运行时间的装饰器
    '''
    def inner(*args,**kwargs):
        start = time.time()
        func(*args,**kwargs)
        end = time.time()
        print('用时:{}秒'.format(end-start))
    return inner

class LSH():
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

    def sig_mae(self,x,y):
        '''
            MAE是处理向量，sig_mae处理单个值
        '''
        return abs(x-y)

    def lsh_detect(self,lsh_table):
        '''
            统计lshTable里各个桶的用户数。返回统计结果字典
        '''
        detect_dic = {}
        table_num = 0
        for i in lsh_table:
            detect_dic[table_num] = {}
            for j in lsh_table[table_num].keys():
                detect_dic[table_num][j] = len(lsh_table[table_num][j]) 
            table_num += 1
        return detect_dic

    def hash_function(self,num=4,nbits=8,d=2):
        '''
            构建LSH哈希映射函数，d:数据维度，nbits:编码后的bit位数，num:哈希表个数，采用均匀分布
            注意：这里的np.random.rand产生的随机数是一样的，伪随机，建议设置随机数种子试试
        '''
        plane_norms_groups = np.empty([num,nbits,d])
        for i in range(num):
            plane_norms_groups[i] = np.random.rand(nbits, d) - .5 # 每一张hash table里的均匀分布向量都不一样
        return plane_norms_groups



    def lsh_table(self,data,plane_norms_groups,nbits,num):
        '''
            进行hash映射，hash后返回的用户索引是随机的，没有顺序的
        '''
        print("当前的nbits为:",nbits)
        value = data.values
        value_id = data.index
        # 进行hash映射
        value_dot_groups = np.empty([num,data.shape[0],nbits])
        for i in range(num):
            print(value[:,1::].shape)
            print(plane_norms_groups[i].T.shape)
            value_dot_groups[i] = np.dot(value, plane_norms_groups[i].T) # value_dot_group是按照用户顺序进行映射的，所以前面的用户可能是相同的
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

    def lsh_mae(self,testUserId,tupData): 
        '''
            计算协同过滤下的MAE误差
        '''
        # id = 0
        up_latitude,up_longitude,down = 0,0,0
        i = testUserId
        count = 0
        for tup in tupData: # (4944, 0.0054780312696541)
            sim_score = 1-tup[1]
            count += 1
            down += sim_score
            if down == 0 and count == len(tupData): # 这里会有重复的索引，会报错，i可能会和tup[0]相等,所以加个判断
                data.loc[i,'predict_latitude'] = np.mean(user_mx.loc[tup[0]]['latitude'])
                data.loc[i,'predict_longitude'] = np.mean(user_mx.loc[tup[0]]['longitude'])
                return 0
            elif down == 0 and count != len(tupData):
                data.loc[i,'predict_latitude'] = np.mean(user_mx.loc[tup[0]]['latitude'])
                data.loc[i,'predict_longitude'] = np.mean(user_mx.loc[tup[0]]['longitude'])
                continue
            up_latitude += sim_score*np.mean(user_mx.loc[tup[0]]['latitude'])
            up_longitude += sim_score*np.mean(user_mx.loc[tup[0]]['longitude'])
        if down == 0:
            return 0
        la_score = up_latitude/down
        long_score = up_longitude/down
        if data.loc[i,'predict_latitude'] == 0:
            data.loc[i,'predict_latitude'] = la_score
            data.loc[i,'predict_longitude'] = long_score
            data.loc[i,'la_MAE'] = self.sig_mae(data.loc[i,'latitude'],data.loc[i,'predict_latitude'])
            data.loc[i,'long_MAE'] = self.sig_mae(data.loc[i,'longitude'],data.loc[i,'predict_longitude'])
        
            # 这里需要修改判别准则，应该改为MAE判别
        else:
            tem = self.sig_mae(la_score,data.loc[i,'latitude'])
            if tem < data.loc[i,'la_MAE']:
                data.loc[i,'predict_latitude'] = la_score 
                data.loc[i,'la_MAE'] = tem
            tem = self.sig_mae(long_score,data.loc[i,'longitude'])
            if tem < data.loc[i,'long_MAE']:
                data.loc[i,'predict_latitude'] = long_score 
                data.loc[i,'long_MAE'] = tem   

import time
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    def train_LSH(train_data,test_data,user_mx):
        global data,lsh_cal
        testIndex = test_data.index
        topk = 3
        count = 0
        length = len(testIndex)

        print("测试集LSH为:",lsh_cal.lsh_detect(te_lshTable))
        print("训练集集LSH为:",lsh_cal.lsh_detect(lshTable))
        start_time = time.time()
        for table_num in te_lshTable.values(): # 遍历测试集的每个表,这里默认设置的一个表
            num += 1
            for buckets in table_num: # 遍历当前表的hash桶
                if buckets not in lshTable[num].keys():
                        continue
                for test_id in te_lshTable[num][buckets]: # 遍历当前表当前桶里的测试集用户和训练集用户
                    count += 1
                    print("当前进度是:",count/length)
                    sim_lis = []
                    
                    for train_id in lshTable[num][buckets]: # 遍历训练集hash映射后同一个表，同一个桶下的数据
                        sim_lis.append(lsh_cal.calSim(test_id,train_id))
                    te = sorted(sim_lis,key=lambda x:x[1],reverse=True)[1:topk]
                    lsh_cal.lsh_mae(test_id,te)

        end_time = time.time()
        la_mae = data['la_MAE'].mean()# 所有的MAE求平均最后
        long_mae = data['long_MAE'].mean() # 所有的MAE求平均最后
        mean_mae = (la_mae+long_mae)/2
        print("训练集数据规模为:",train_data.shape)
        # data = data.dropna(axis=0,how='any')
        print('测试集long_mae为%.2f'%long_mae)
        print('测试集la_mae为%.2f'%la_mae)
        print('测试集mean_mae为%.2f'%mean_mae)
        return mean_mae,end_time-start_time

    original_data = pd.read_table('./GIS_LSH_VE_CF/data/train.csv',sep=",",names=['latitude','longitude'],encoding='latin-1',engine='python')
    user_mx = pd.read_table('./GIS_LSH_VE_CF/data/user_mx.csv',sep=",",names=['latitude','longitude'],encoding='latin-1',engine='python')
    test_data = pd.read_table('./GIS_LSH_VE_CF/data/test.csv',sep=",",names=['latitude','longitude'],encoding='latin-1',engine='python')
    data = test_data.groupby(test_data.index).mean()
    data['predict_latitude'],data['predict_longitude'],data['la_MAE'],data['long_MAE'] = 0,0,0,0
    # data = pd.read_table('./GIS_LSH_VE_CF/data/predict_data.csv',sep=",",names=['latitude','longitude','predict_latitude','predict_longitude','la_MAE','long_MAE'],encoding='latin-1',engine='python')
    # print(data)
    # 对用户矩阵进行预处理，将str转换为list
    def to_list(str):
        return [ float(x) for x in str[1:-2].split(',')]
    user_mx['latitude'] = user_mx['latitude'].apply(to_list)
    user_mx['longitude'] = user_mx['longitude'].apply(to_list)
    #构造500,1000,1500,2000,2500个用户的数据采样
    data_item = [2000]
    # 参数设置
    mae,times,total_time = [],[],0
    nbits,num,d = 6,2,2
    lsh_cal = LSH(user_mx)
    hash_func = lsh_cal.hash_function(num,nbits,d) # hash映射函数
    start = 0
    for i in data_item:
        train_data = original_data.iloc[start:i]
        lshTable = lsh_cal.lsh_table(train_data,hash_func,nbits,num) # 构建hash表
        te_lshTable = lsh_cal.lsh_table(test_data,hash_func,nbits,num)
        lsh_mean_mae,lsh_time = train_LSH(train_data,test_data,user_mx)
        total_time += lsh_time
        mae.append(lsh_mean_mae)
        times.append(total_time)
        start += 2000

    print("mae:",mae)
    print("times:",times)
    data.to_csv("./GIS_LSH_VE_CF/data/predict_data.csv")
    style = ["*","o","^","s","X","<",">","p","h","1","2"]
    plt.figure(1 , figsize = (17 , 9) )
    plt.subplot(121)
    plt.plot(data_item,mae,marker=style[0],markersize=14)
    plt.xlabel("数据量")
    plt.ylabel("MAE")
    plt.title("LSH")

    plt.subplot(122)
    plt.plot(data_item,times,marker=style[1],markersize=14)
    plt.xlabel("数据量")
    plt.ylabel("时间")
    plt.title("LSH")
    fig=plt.gcf()
    fig.savefig('./GIS_LSH_VE_CF/picture/CF.jpg',dpi=500)
    plt.show()



