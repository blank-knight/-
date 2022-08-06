import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
class K_means_trian():
    def __init__(self,data) -> None:
        self.df = data

    def KNN_n_cluseters(self,m):
        '''
            进行m次聚类中心查看，看选多少个聚类中心最合适
        '''
        plt.style.use('fivethirtyeight')
        X1 = self.df[['latitude' , 'longitude']].iloc[: , :].values
        inertia = []
        for n in range(1 , m): # 进行m次迭代，每次选择n个聚类中心查看
            self.algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                                tol=0.0001,  random_state= 111  , algorithm='elkan') )
            self.algorithm.fit(X1)
            inertia.append(self.algorithm.inertia_) # .inertia_是一种聚类评估指标，这个评价参数表示的是簇中某一点到簇中心点距离的和，
        plt.figure(1 , figsize = (15 ,8))
        plt.plot(np.arange(1 , m) , inertia , 'o')
        plt.plot(np.arange(1 , m) , inertia , '-' , alpha = 0.5)
        plt.xlabel('聚类中心个数') , plt.ylabel('样本与聚类中心距离')
        num = self.df.shape
        print("当前的数据量是",num[0])
        print(inertia)
        print("+++++++++++++++++++")
        plt.title("样本数"+str(num[0]))
        plt.show()

    def KNN(self,n):
        '''
            选n个聚类中心进行KNN聚类
        '''
        plt.style.use('fivethirtyeight')
        X1 = self.df[['latitude' , 'longitude']].iloc[: , :].values
        self.algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                            tol=0.0001,  random_state= 111  , algorithm='elkan') )
        self.algorithm.fit(X1)
        self.labels1 = self.algorithm.labels_
        self.centroids1 = self.algorithm.cluster_centers_
        h = 0.02
        x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
        y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
        self.xx, self.yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        self.Z = self.algorithm.predict(np.c_[self.xx.ravel(), self.yy.ravel()])
        
    # 生成根据k聚类得到的hash映射函数
    def getKhash(self,num,n_clusters,d=2):
        '''
            输入表的个数和数据维度，返回根据K聚类得到的hash函数与编码长度nbits，必须先有KNN聚类，才能运行
        '''
        cl = self.algorithm.cluster_centers_
        id = cl[:,1].argsort()
        row,col = cl.shape
        mid = np.zeros((row-1,col)) 
        for i in range(n_clusters-1): # 根据聚类中心点个数，返回中心直线
            mid[i,:] = (cl[id[i],:]+cl[id[i+1],:])/2
        nbits = mid.shape[0]
        plane_norms_groups = np.empty([num,nbits,d]) # k聚类hash函数
        for i in range(num):
            plane_norms_groups[i] = mid # 每一张hash table里的均匀分布向量都不一样
        return plane_norms_groups,nbits


    def nbits_plt(self,vectors,num_bits):
        '''
            绘制划分线和聚类中心点
        '''
        Z = self.Z.reshape(self.xx.shape)
        for i in vectors:
            plt.axline((0, 0), i)
        plt.figure(2 , figsize = (15 ,8))
        plt.imshow(Z , interpolation='nearest', 
                extent=(self.xx.min(), self.xx.max(), self.yy.min(), self.yy.max()),
                cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')
        plt.scatter( x = 'latitude' ,y ='longitude' , data = self.df , c = self.labels1 , 
                    s = 200 )
        plt.scatter(x = self.centroids1[: , 0] , y =  self.centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
        plt.ylabel('longtitude') , plt.xlabel('laitude')
        #plt.title(str(num_bits)+" bits")
        num = self.df.shape
        plt.title("样本数"+str(num[0]))
        plt.show()

if __name__ == "__main__":
    import pandas as pd
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    original_data = pd.read_table('./GIS_LSH_VE_CF/data/train.csv',sep=",",names=['latitude','longitude'],encoding='latin-1',engine='python')
    k_means = K_means_trian(original_data)
    k_means.KNN_n_cluseters(11)
    n_clusters,num = 4,1
    k_means.KNN(n_clusters)
    plane_norms_groups,nbits = k_means.getKhash(num,n_clusters)
    print(plane_norms_groups)
    k_means.nbits_plt(plane_norms_groups[0],nbits)