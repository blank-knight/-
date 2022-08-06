import numpy as np 
import pandas as pd
import math

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

    def std(self,mx):
        '''
            矩阵进行标准化，x1/(x1^2+y1^2)^(1/2)
        '''
        std_mx = mx*mx
        std_mx[:,0] = np.sqrt(std_mx[:,0]+std_mx[:,1])
        std_mx[:,1] = std_mx[:,0]
        return mx/std_mx

    def splicing(self,userId1,userId2):
        '''
            将经纬度列表合并成array
        '''
        user1_lati = np.array(self.user_mx.loc[userId1,'latitude']) 
        user1_longi = np.array(self.user_mx.loc[userId1,'longitude']) 
        user2_lati = np.array(self.user_mx.loc[userId2,'latitude']) 
        user2_longi = np.array(self.user_mx.loc[userId2,'longitude'])
        return np.vstack((user1_lati, user1_longi)).T,np.vstack((user2_lati,user2_longi)).T

        
    def cf_mae(self,testUserId,tupData): 
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

    def sig_mae(self,x,y):
        '''
            MAE是处理向量，sig_mae处理单个值
        '''
        return abs(x-y)

# 协同过滤算法的测试
import time
if __name__ == "__main__":
    from sklearn.metrics import mean_absolute_error as MAE
    import matplotlib.pyplot as plt
    
    #用来正常显示中文标签
    plt.rcParams['font.sans-serif']=['SimHei'] 
    #用来正常显示负号
    plt.rcParams['axes.unicode_minus']=False
    def train_CF(train_data,test_data,user_mx):
        global data
        trainIndex = train_data.index
        testIndex = test_data.index
       
        cf_cal = sim_cal(user_mx)
        topk = 3
        count = 0
        length = len(testIndex)
        start_time = time.time()
        for i in testIndex:
            count += 1
            # print("当前进度是:",count/length)
            sim_lis = []
            for j in trainIndex:
                if i == j:
                    continue
                sim_lis.append(cf_cal.calSim(i,j))
            te = sorted(sim_lis,key=lambda x:x[1],reverse=True)[1:topk]
            cf_cal.cf_mae(i,te)
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
    data_item = [2000,4000,6000,8000,10000]
    mae = []
    times = []
    total_time = 0
    start = 0
    for i in data_item:
        train_data = original_data.iloc[start:i]
        cf_mean_mae,cf_time = train_CF(train_data,test_data,user_mx)
        total_time += cf_time
        mae.append(cf_mean_mae)
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
    plt.title("CF")

    plt.subplot(122)
    plt.plot(data_item,times,marker=style[1],markersize=14)
    plt.xlabel("数据量")
    plt.ylabel("时间")
    plt.title("CF")
    fig=plt.gcf()
    fig.savefig('./GIS_LSH_VE_CF/picture/CF.jpg',dpi=500)
    plt.show()