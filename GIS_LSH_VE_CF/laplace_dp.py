import numpy as np
# 参数设置
# sensitivety越大，噪声越大，误差越大
# epsilon越小，噪声越大，误差越大
# sensitivety/epsilon为满足epsilon-差分隐私，其比值越大，噪声越大，误差越大
sensitivety = 1 
epsilon = 500

class laplace_dp:
    def __init__(self) -> None:
        pass

    # 计算基于拉普拉斯分布的噪声
    def laplace_noisy(self,sensitivety,epsilon,size):
        n_value = np.random.laplace(1, sensitivety/epsilon, size)
        #print("拉普拉斯噪声矩阵：",n_value)
        return n_value

    # 基于laplace的分布函数的反函数计算
    def laplace_noisy2(self,sensitivety, epsilon):
        b = sensitivety/epsilon
        u1 = np.random.random()
        u2 = np.random.random()
        if u1 <= 0.5:
            noisy = -b*np.log(1.-u2)
        else:
            noisy = b*np.log(u2)
        return noisy

    # 计算基于拉普拉斯加噪的混淆值
    def laplace_mech(self,data, sensitivety, epsilon):
        size = data.shape
        temp_data = np.zeros(size)
        temp_data += self.laplace_noisy(sensitivety,epsilon,size)
        return temp_data
 
# 基于拉普拉斯分布的特性，如果想要分布震荡较小，需要将隐私预算epsilon的值设置较大
if __name__ =='__main__':
    import math
    user1Items1 = np.array([np.array([ 0.29908732,-0.95422575])
            ,np.array([ 0.32160848,-0.94687274])
            ,np.array([ 0.32329121,-0.94629953])
            ,np.array([ 0.3240757 ,-0.94603115])
            ,np.array([ 0.32110942,-0.9470421 ])
            ,np.array([ 0.32360346,-0.94619279])
            ,np.array([ 0.32330386,-0.9462952 ])
            ,np.array([ 0.32360346,-0.94619279])
            ,np.array([ 0.32404591,-0.94604136])
            ,np.array([ 0.32159669,-0.94687674])
            ,np.array([ 0.32319336,-0.94633295])
            ,np.array([ 0.32360346,-0.94619279])
            ,np.array([ 0.32329121,-0.94629953])
            ,np.array([ 0.2989097 ,-0.9542814 ])])
    user2Items1 = np.array([np.array([ 0.219908732,-0.95422575])
            ,np.array([ 0.3160848,-0.94687274])
            ,np.array([ 0.22329121,-0.94629953])
            ,np.array([ 0.3240757 ,-0.94603115])
            ,np.array([ 0.32110942,-0.9470421 ])
            ,np.array([ 0.32360346,-0.94619279])
            ,np.array([ 0.42330386,-0.9462952 ])
            ,np.array([ 0.32360346,-0.94619279])
            ,np.array([ 0.3404591,-0.94604136])
            ,np.array([ 0.32159669,-0.94687674])
            ,np.array([ 0.32319336,-0.94633295])
            ,np.array([ 0.32360346,-0.94619279])
            ,np.array([ 0.32329121,-0.94629953])
            ,np.array([ 0.2989097 ,-0.9542814 ])])
    test = laplace_dp()
    data_noisy1 = test.laplace_mech(user1Items1, sensitivety, epsilon)
    data_noisy2 = test.laplace_mech(user2Items1, sensitivety, epsilon)
    #print(data_noisy2)
    print("噪声值为：",np.sum(data_noisy2))
    print("未加噪声前的矩阵和：",np.sum(user2Items1))
    noisy_pre = 1/(1+math.sqrt(np.sum(((user1Items1-user2Items1)/1e+7)**2)/user2Items1.shape[1]))
    print("加入噪声前的相似度为：",noisy_pre)
    noisy_lat = 1/(1+math.sqrt(np.sum(((data_noisy1-data_noisy2)/1e+7)**2)/data_noisy2.shape[1]))
    print("加入噪声后的相似度为：",noisy_lat)
    print("相似度差值为：",noisy_lat - noisy_pre)