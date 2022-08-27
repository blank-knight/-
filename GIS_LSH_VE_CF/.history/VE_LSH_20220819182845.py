# -*- coding: utf-8 -*-
"""
    局部敏感哈希和加密局部敏感哈希的实现
"""
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
    """
        局部敏感哈希的实现
    """
    def __init__(self,user_mx,data,vec_encrypt=None) -> None:
            '''
                初始化
                Args:
                    user_mx:用户经纬度矩阵
                    data:预测矩阵
                    vec_encrypt:加密算法实例化对象
            '''
            self.user_mx = user_mx
            self.data = data
            self.vec_encrypt = vec_encrypt


    def calSim(self,userId1,userId2):
        '''
            计算相似度
            Args:
                userId1:用户1的id
                userId2:用户2的id
            return:
                返回两者的相似度
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
            计算加密后的相似度
            Args:
                userId1:用户1的id
                userId2:用户2的id
            return:
                返回两者的相似度
        '''
        # 两个用户经纬度矩阵信息
        start_time = time.time()
        user1Items,user2Items = self.splicing(userId1,userId2)
        user1Items = self.std(user1Items.astype('float'))
        user2Items = self.std(user2Items.astype('float'))
        end_time = time.time()
        m = user1Items.shape[0]
        n = m
        w = 16
        S = self.vec_encrypt.generate_key(w,m,n)
        T = self.vec_encrypt.get_T(n)
        c1,S = self.mx_encrypt(user1Items,w,m,n,T)
        c2,S = self.mx_encrypt(user2Items,w,m,n,T)
        total_time = end_time-start_time
        start_time = time.time()
        res = (userId2,1/(1+math.sqrt(np.sum(self.vec_encrypt.mx_decrypt((c1-c2),S,w)**2))/user1Items.shape[0])) 
        end_time = time.time()
        total_time += (end_time-start_time)
        return res,total_time

    def mx_encrypt(self,mx,w,m,n,T):
        '''
            对矩阵进行预处理加密
            Args:
                S 表示密钥/私钥的矩阵。用于解密。
                M 公钥。用于加密和进行数学运算。在有些算法中，不是所有数学运算都需要公钥。但这一算法非常广泛地使用公钥。
                c 加密数据向量，密文。
                x 消息，即明文。有些论文使用m作明文的变量名。
                w 单个“加权（weighting）”标量变量，用于重加权输入消息x（让它一致地更长或更短）。这一变量用于调节信噪比。加强信号后，对于给定的操作而言，
                消息较不容易受噪声影响。然而，过于加强信号，会增加完全毁坏数据的概率。这是一个平衡。
                E或e 一般指随机噪声。在某些情形下，指用公钥加密数据前添加的噪声。一般而言，噪声使解密更困难。噪声使同一消息的两次加密可以不一样，
                在让消息难以破解方面，这很重要。注意，取决于算法和实现，这可能是一个向量，也可能是一个矩阵。在其他情形下，指随操作积累的噪声，详见后文。
            return:
                c*和S*
        '''
        mx_tem = mx*1e+7
        mx_tem = mx_tem.T.astype(int)
        c,S = self.vec_encrypt.mx_encrypt(mx_tem,w,m,n,T)
        return c,S

    def std(self,mx):
        '''
            矩阵进行标准化，x1/(x1^2+y1^2)^(1/2)
            Args:
                mx:需要标准化的矩阵
            return:
                返回标准化后的矩阵
        '''
        std_mx = mx*mx
        std_mx[:,0] = np.sqrt(std_mx[:,0]+std_mx[:,1])
        std_mx[:,1] = std_mx[:,0]
        return mx/std_mx

    def splicing(self,userId1,userId2):
        '''
            将经纬度列表合并成array
            Args:
                userId1:用户1的id
                userId2:用户2的id
            return:
                合并后的np.array数组
        '''
        user1_lati = np.array(self.user_mx.loc[userId1,'latitude']) 
        user1_longi = np.array(self.user_mx.loc[userId1,'longitude']) 
        user2_lati = np.array(self.user_mx.loc[userId2,'latitude']) 
        user2_longi = np.array(self.user_mx.loc[userId2,'longitude'])
        return np.vstack((user1_lati, user1_longi)).T,np.vstack((user2_lati,user2_longi)).T

    def sig_mae(self,x,y):
        '''
            MAE是处理向量，sig_mae处理单个值
        '''
        return abs(x-y)

    def lsh_detect(self,lsh_table):
        '''
            统计lshTable里各个桶的用户数。返回统计结果字典
            Args:
                lsh_table:哈希表
            return:
                哈希表的分布情况字典
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
            构建LSH哈希映射函数,
            Args:
                d:数据维度
                nbits:编码后的bit位数
                num:哈希表个数，采用均匀分布
            return:
                哈希映射函数
        '''
        plane_norms_groups = np.empty([num,nbits,d])
        for i in range(num):
            plane_norms_groups[i] = np.random.rand(nbits, d) - .5 # 每一张hash table里的均匀分布向量都不一样
        return plane_norms_groups



    def lsh_table(self,data,plane_norms_groups,nbits,num):
        '''
            进行hash映射,构建哈希表
            Args:
                data:进行哈希映射的数据
                plane_norms_groups:哈希映射函数
                nbits:哈希编码数
                num:哈希表的表数
        '''
        # print("当前的nbits为:",nbits)
        value = data.values
        value_id = data.index
        # 进行hash映射
        value_dot_groups = np.empty([num,data.shape[0],nbits])
        for i in range(num):
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
            计算协同过滤下的MAE误差,并添加进预测矩阵self.data里面
            Args:
                testUserId:测试集用户id索引
                tupData:(用户id,相似度)元组
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
                self.data.loc[i,'predict_latitude'] = np.mean(self.user_mx.loc[tup[0]]['latitude'])
                self.data.loc[i,'predict_longitude'] = np.mean(self.user_mx.loc[tup[0]]['longitude'])
                return 0
            elif down == 0 and count != len(tupData):
                self.data.loc[i,'predict_latitude'] = np.mean(self.user_mx.loc[tup[0]]['latitude'])
                self.data.loc[i,'predict_longitude'] = np.mean(self.user_mx.loc[tup[0]]['longitude'])
                continue
            up_latitude += sim_score*np.mean(self.user_mx.loc[tup[0]]['latitude'])
            up_longitude += sim_score*np.mean(self.user_mx.loc[tup[0]]['longitude'])
        if down == 0:
            print("down is 0")
            return 0
        la_score = up_latitude/down
        long_score = up_longitude/down

        tem = self.sig_mae(la_score,self.data.loc[i,'latitude'])
        if tem < self.data.loc[i,'la_MAE']:
            self.data.loc[i,'predict_latitude'] = la_score 
            self.data.loc[i,'la_MAE'] = tem
        tem = self.sig_mae(long_score,self.data.loc[i,'longitude'])
        if tem < self.data.loc[i,'long_MAE']:
            self.data.loc[i,'predict_latitude'] = long_score 
            self.data.loc[i,'long_MAE'] = tem   

import time
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    def train_LSH(train_data,test_data,user_mx):
        global data,lsh_cal
        testIndex = test_data.index
        topk,count,length,num = 3,0,len(testIndex),-1
        # print("测试集LSH为:",lsh_cal.lsh_detect(te_lshTable))
        # print("训练集集LSH为:",lsh_cal.lsh_detect(lshTable))
        start_time = time.time()
        for table_num in te_lshTable.values(): # 遍历测试集的每个表,这里默认设置的一个表
            num += 1
            print("当前表数为:",num)
            for buckets in table_num: # 遍历当前表的hash桶
                if buckets not in lshTable[num].keys():
                        continue
                for test_id in te_lshTable[num][buckets]: # 遍历当前表当前桶里的测试集用户和训练集用户
                    count += 1
                    # print("当前进度是:",count/length)
                    sim_lis = []
                    
                    for train_id in lshTable[num][buckets]: # 遍历训练集hash映射后同一个表，同一个桶下的数据
                        sim_lis.append(lsh_cal.calSim(test_id,train_id))
                    te = sorted(sim_lis,key=lambda x:x[1],reverse=True)[0:topk]
                    lsh_cal.lsh_mae(test_id,te)
            la_mae = data['la_MAE'].mean()# 所有的MAE求平均最后
            long_mae = data['long_MAE'].mean() # 所有的MAE求平均最后
            mean_mae = (la_mae+long_mae)/2
            print('测试集long_mae为%.2f'%long_mae)
            print('测试集la_mae为%.2f'%la_mae)
            print('测试集mean_mae为%.2f'%mean_mae)
            path = "../GIS_LSH_VE_CF/data/"+str(num)
            data.to_csv(path)
        end_time = time.time()
        # la_mae = data['la_MAE'].mean()# 所有的MAE求平均最后
        # long_mae = data['long_MAE'].mean() # 所有的MAE求平均最后
        # mean_mae = (la_mae+long_mae)/2
        # print("训练集数据规模为:",train_data.shape)
        # # data = data.dropna(axis=0,how='any')
        # print('测试集long_mae为%.2f'%long_mae)
        # print('测试集la_mae为%.2f'%la_mae)
        # print('测试集mean_mae为%.2f'%mean_mae)
        return mean_mae,end_time-start_time

    original_data = pd.read_table('../GIS_LSH_VE_CF/data/train.csv',sep=",",names=['latitude','longitude'],encoding='latin-1',engine='python')
    user_mx = pd.read_table('../GIS_LSH_VE_CF/data/user_mx.csv',sep=",",names=['latitude','longitude'],encoding='latin-1',engine='python')
    test_data = pd.read_table('../GIS_LSH_VE_CF/data/test.csv',sep=",",names=['latitude','longitude'],encoding='latin-1',engine='python')
    data = test_data.groupby(test_data.index).mean()
    # data['predict_latitude'],data['predict_longitude'],data['la_MAE'],data['long_MAE'] = 0,0,0,0
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
    nbits,num_lis,d = 10,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],2
    # 实例化
    vec_encrypt = vector_encrypt()
    lsh_cal = LSH(user_mx,data,vec_encrypt)
    start = 0
    for i in data_item:
        train_data = original_data.iloc[start:i]
        for num in num_lis:
            data['predict_latitude'],data['predict_longitude'],data['la_MAE'],data['long_MAE'] = 0,0,data['latitude'],data['longitude'].abs()
            # print(data)
            hash_func = lsh_cal.hash_function(num,nbits,d) # hash映射函数
            lshTable = lsh_cal.lsh_table(train_data,hash_func,nbits,num) # 构建hash表
            te_lshTable = lsh_cal.lsh_table(test_data,hash_func,nbits,num)
            
            lsh_mean_mae,lsh_time = train_LSH(train_data,test_data,user_mx)
            mae.append(lsh_mean_mae)
            times.append(lsh_time)
            print("++++++++++++++这是分割线+++++++++++++")
        # start += 2000

    print("mae:",mae)
    print("times:",times)
    data.to_csv("../GIS_LSH_VE_CF/data/predict_data.csv")
    style = ["*","o","^","s","X","<",">","p","h","1","2"]
    plt.figure(1 , figsize = (10 , 7) )
    plt.plot(num_lis,mae,color = 'k',marker=style[0],markersize=14)
    plt.xlabel("tables")
    plt.ylabel("MAE")
    plt.title("LSH误差随tables的变化")

    plt.figure(2 , figsize = (10 , 7) )
    plt.plot(num_lis,times,color = 'k',marker=style[1],markersize=14)
    plt.xlabel("tables")
    plt.ylabel("时间")
    plt.title("LSH时间随tables的变化")
    fig=plt.gcf()
    fig.savefig('../GIS_LSH_VE_CF/bw_picture/LSH_num.jpg',dpi=500)
    plt.show()



