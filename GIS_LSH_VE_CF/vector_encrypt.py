import numpy as np
'''
    优化中间密钥传输矩阵M
'''
class vector_encrypt:
    '''
        S 表示密钥/私钥的矩阵。用于解密。
        M 公钥。
        c 加密数据向量，密文。
        x 消息，即明文。
        w 单个“加权（weighting）”标量变量，用于调节信噪比。加强信号后，对于给定的操作而言，消息较不容易受噪声影响。然而，过于加强信号，会增加完全毁坏数据的概率。
        E或e 噪声使解密更困难且让同一消息的两次加密可以不一样，
        在让消息难以破解方面,取决于算法和实现，这可能是一个向量，也可能是一个矩阵。在其他情形下，指随操作积累的噪声。
    '''
    def __init__(self) -> None:
        pass

    def generate_key(self,w,m):
        S = (np.random.rand(m,m) * w / (2 ** 16)) # 可证明 max(S) < w
        return S

    def encrypt(self,x,S,m,w):
        assert len(x) == len(S)

        e = (np.random.rand(m)) # 可证明 max(e) < w / 2
        c = np.linalg.inv(S).dot((w * x) + e)
        return c


    def generate_noise(self,m):
        # z这里有问题，还没改
        # e = (np.random.rand(mx.shape[0])) # 可证明 max(e) < w / 2
        e = (np.zeros(m)) # 可证明 max(e) < w / 2
        return e
    
    def mx_encrypt(self,mx,S,w,e):

        encrypt_mx = np.linalg.inv(S).dot((w * mx) + e)
        return encrypt_mx

    
    def mx_decrypt(self,S_prime,M,c_star,w):
        '''
            传入加密矩阵，密钥和权重，返回解密矩阵
        '''
        decrypt_mx = np.round((S_prime.dot(M).dot(c_star) / w)).astype('int')
        return decrypt_mx

    def get_c_star(self, c, m, l):
        '''
        对向量c，求解c*
        参数：
        c: 输入数组，包含m个元素的一维数组
        m: 元素数量
        l: 每个元素的二进制表示长度
        返回：
        c_star: 输出数组，包含(l * m)个元素的一维数组
        '''
        c_star = np.zeros(l * m, dtype='int')
        for i in range(m):
            # 获取c[i]的二进制表示
            binary_repr = np.array(list(np.binary_repr(int(np.abs(c[i])), width=l)), dtype='int')
            # print(c[i])
            # print(int(binary_repr,2))
            # 处理负数情况
            if c[i] < 0:
                binary_repr *= -1
            # print(binary_repr)
            # 将二进制表示添加到c_star中
            c_star[(i * l):((i + 1) * l)] += binary_repr
        return c_star

    
    def switch_key(self,S,m,T,l):
        S_star = self.get_S_star(S,m,l)
        n_prime = m + 1
        
        A = (np.random.rand(n_prime - m, m*l) * 10).astype('int')
        # E = (1 * np.random.rand(S_star.shape[0],S_star.shape[1])).astype('int')
        E = (np.random.rand(m,m*l)).astype('int')
        M = np.concatenate(((S_star - T.dot(A) + E),A))
        return M
    
    def c_star(self,c,m,l):
        c_star= self.get_c_star(c,m,l)
        # c_prime = M.dot(c_star)
        return c_star

    def get_S_prime(slef,m,T):
        S_prime = np.concatenate((np.eye(m),T.T),0).T
        return S_prime

    def test(self,S,c,m,M,S_prime):
        l = int(np.ceil(np.log2(np.max(np.abs(c)))))
        c_star = self.get_c_star(c,m,l)
        S_star = self.get_S_star(S,m,l)
        print(np.round(S_star.dot(c_star)))
        print("++++++++++++++++++")
        print(S.dot(c))
        print("++++++++++++++++++")
        print(S_prime.dot(M))
        print("++++++++++++++++++")
        print(S_star)

    def get_S_star(self,S,m,l):
        '''
            求解S*
        '''
        S_star = list()
        for i in range(l):
            S_star.append(S*2**(l-i-1))
        S_star = np.array(S_star).transpose(1,2,0).reshape(m,m*l)
        return S_star

    def get_T(self,m):
        '''
            产生n维随机向量
        '''
        n_prime = m + 1
        T = (10 * np.random.rand(m,n_prime - m)).astype('int')
        return T

    def encrypt_via_switch(self,x,w,M,m,l,T):
        '''
            返回c',S'
        '''
        c,S = self.c_S_prime(x*w,M,m,l,T)
        return c,S

if __name__ == '__main__':
    import math
    user1Items1 = np.array([np.array([ 0.219908732,0.95422575])
                ,np.array([ 0.3160848,0.94687274])
                ,np.array([ 0.22329121,0.94629953])
                ,np.array([ 0.3240757 ,0.94603115])
                ,np.array([ 0.32110942,0.9470421 ])
                ,np.array([ 0.32360346,0.94619279])
                ,np.array([ 0.42330386,0.9462952 ])
                ,np.array([ 0.32360346,0.94619279])
                ,np.array([ 0.3404591,0.94604136])
                ,np.array([ 0.32159669,0.94687674])
                ,np.array([ 0.32319336,0.94633295])
                ,np.array([ 0.32360346,0.94619279])
                ,np.array([ 0.32329121,0.94629953])
                ,np.array([ 0.2989097 ,0.9542814 ])])

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

    # user1Items1_tem = user1Items1*1e+2
    # user1Items1_tem = user1Items1_tem.T.astype(int)
    # print(user1Items1_tem)
    user1Items1_tem = user1Items1.T
    user1Items1_tem = user1Items1_tem*1e+4
    user1Items1_tem = user1Items1_tem.astype(int)

    user1Items2_tem = user2Items1.T
    user1Items2_tem = user1Items2_tem*1e+4
    user1Items2_tem = user1Items2_tem.astype(int)
    vec_encrypt = vector_encrypt()
    # 取列向量
    x = user1Items1_tem[0]
    x2 = user1Items2_tem[1]

    print("原文为：")
    print(x.shape)
    print(x)
    print("++++++++++++++++++")
    m = x.shape[0]
    w = 160
    e = vec_encrypt.generate_noise(m)
    S = vec_encrypt.generate_key(w,m)
    c = vec_encrypt.mx_encrypt(x,S,w,e)
    c2 = vec_encrypt.mx_encrypt(x2,S,w,e)
    T = vec_encrypt.get_T(m)
    l = max(int(np.ceil(np.log2(np.max(np.abs(c))))),int(np.ceil(np.log2(np.max(np.abs(c2))))))
    M = vec_encrypt.switch_key(S,m,T,l)
    c_star = vec_encrypt.c_star(c,m,l)
    S_prime = vec_encrypt.get_S_prime(m,T)
    

    # vec_encrypt.test(S,c1,m,M,S_prime)
    # print("S*c:  ")
    # print(S.dot(c1))
    # print("++++++++++++++++++++++++++++++++++")
    # print(S.dot(c1)/w)
    print("密文为：")
    print(c.shape)
    print(c)
    print("++++++++++++++++++")
    c1 = vec_encrypt.mx_decrypt(S_prime,M,c_star,w)
    print("解密后为：")
    print(c1.shape)
    print(c1)

    print("==================================")
    print("原文为：")
    print(x2.shape)
    print(x2)
    print("++++++++++++++++++")

    c_star2 = vec_encrypt.c_star(c2,m,l)
    print("密文为：")
    print(c2.shape)
    print(c2)
    print("++++++++++++++++++")
    c22 = vec_encrypt.mx_decrypt(S_prime,M,c_star2,w)

    print("解密后为：")
    print(c22.shape)
    print(c22)
    print("++++++++++++++++++")
    print("进行同态运算后为")
    print(1/(1+math.sqrt(np.sum(((user1Items1_tem[0]-user1Items2_tem[1])/1e+4)**2)/user1Items2_tem.shape[1])))

    print(1/(1+math.sqrt(np.sum((vec_encrypt.mx_decrypt(S_prime,M,(c_star2-c_star),w)/1e+4)**2)/user1Items2_tem.shape[1])))