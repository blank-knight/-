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

    def generate_key(self,w,m,n):
        S = (np.random.rand(m,n) * w / (2 ** 16)) # 可证明 max(S) < w
        return S

    def encrypt(x,S,m,n,w):
        assert len(x) == len(S)

        e = (np.random.rand(m)) # 可证明 max(e) < w / 2
        c = np.linalg.inv(S).dot((w * x) + e)
        return c

    def decrypt(self,c,S,w):
        return (S.dot(c) / w).astype('int')

    def mx_encrypt(self,mx,w,M,m,l,T):
        '''
            传入矩阵和加密算法对象，返回加密后的矩阵和密钥
        '''
        encrypt_mx = np.zeros([2,mx.shape[1]+1])
        c,S = self.encrypt_via_switch(mx[0,:],w,M,m,l,T)
        encrypt_mx[0,:] = c
        c,S = self.encrypt_via_switch(mx[1,:],w,M,m,l,T)
        encrypt_mx[1,:] = c
        return encrypt_mx,S


    def mx_decrypt(self,c,S,w):
        '''
            传入加密矩阵，密钥和权重，返回解密矩阵
        '''
        decrypt_mx = np.zeros([2,c.shape[1]-1])
        decrypt_mx[0,:] = (S.dot(c[0,:]) / w).astype('int')
        decrypt_mx[1,:] = (S.dot(c[1,:]) / w).astype('int')
        return decrypt_mx

    def get_c_star(self,c,m,l):
        '''
            求解c*
        '''
        c_star = np.zeros(l * m,dtype='int')
        for i in range(m):
            b = np.array(list(np.binary_repr(np.abs(c[i]))),dtype='int')
            if(c[i] < 0):
                b *= -1
            c_star[(i * l) + (l-len(b)): (i+1) * l] += b
        return c_star

    
    def switch_key(self,c,S,m,n,T):
        l = int(np.ceil(np.log2(np.max(np.abs(c)))))
        S_star = self.get_S_star(S,m,n,l)
        n_prime = n + 1
        
        A = (np.random.rand(n_prime - m, n*l) * 10).astype('int')
        E = (1 * np.random.rand(S_star.shape[0],S_star.shape[1])).astype('int')
        M = np.concatenate(((S_star - T.dot(A) + E),A),0)
        return M,l
    
    def c_S_prime(self,c,M,m,l,T):
        c_star = self.get_c_star(c,m,l)
        S_prime = np.concatenate((np.eye(m),T.T),0).T
        c_prime = M.dot(c_star)
        return c_prime,S_prime

    def get_S_star(self,S,m,n,l):
        '''
            求解S*
        '''
        S_star = list()
        for i in range(l):
            S_star.append(S*2**(l-i-1))
        S_star = np.array(S_star).transpose(1,2,0).reshape(m,n*l)
        return S_star

    def get_T(self,n):
        '''
            产生n维随机向量
        '''
        n_prime = n + 1
        T = (10 * np.random.rand(n,n_prime - n)).astype('int')
        return T

    def encrypt_via_switch(self,x,w,M,m,l,T):
        '''
            返回c',S'
        '''
        c,S = self.c_S_prime(x*w,M,m,l,T)
        return c,S

if __name__ == '__main__':
    import math
    user1Items1 = np.array([np.array([ 0.219908732,-0.95422575])
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

    user1Items1_tem = user1Items1*1e+2
    user1Items1_tem = user1Items1_tem.T.astype(int)
    print(user1Items1_tem)
    user1Items1_tem = user1Items1.T
    user1Items1_tem = user1Items1_tem*1e+7
    user1Items1_tem = user1Items1_tem.astype(int)

    user1Items2_tem = user2Items1.T
    user1Items2_tem = user1Items2_tem*1e+7
    user1Items2_tem = user1Items2_tem.astype(int)
    vec_encrypt = vector_encrypt()
    x = user1Items1_tem
    m = 2
    n = m
    w = 16
    S = vec_encrypt.generate_key(w,m,n)
    T = vec_encrypt.get_T(n)
    M,l = vec_encrypt.switch_key(x*w,np.eye(m),m,n,T)
    # c,S = vec_encrypt.encrypt_via_switch(x,w,m,n,T)
    c1,S = vec_encrypt.mx_encrypt(x,w,M,m,l,T)
    x = user1Items2_tem
    c2,S = vec_encrypt.mx_encrypt(x,w,M,m,l,T)

    c11 = vec_encrypt.mx_decrypt(c1,S,w)
    #print(c11)
    print(user1Items2_tem)
    c22 = vec_encrypt.mx_decrypt(c2,S,w)
    print(c22)
    # print(vec_encrypt.mx_decrypt(c,S,w))
    # print(user1Items1_tem[0])
    # vec_encrypt = vector_encrypt()
    # x = user1Items1_tem[0]
    # m = len(x)
    # n = m
    # w = 16
    # S = vec_encrypt.generate_key(w,m,n)
    # T = vec_encrypt.get_T(n)
    # c,S = vec_encrypt.encrypt_via_switch(x,w,m,n,T)
    # print(vec_encrypt.decrypt(c,S,w))
    print(1/(1+math.sqrt(np.sum(((user1Items1_tem-user1Items2_tem)/1e+7)**2)/user1Items2_tem.shape[1])))
    print(1/(1+math.sqrt(np.sum((vec_encrypt.mx_decrypt((c1-c2),S,w)/1e+7)**2)/user1Items2_tem.shape[1])))