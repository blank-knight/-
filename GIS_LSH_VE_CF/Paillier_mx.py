from random import randint
import libnum
import sys
import numpy as np
"""
    注意：Paillier不支持对负数和浮点数进行运算，这里通过将浮点数变为整数进行运算，负数可通过https://www.zhihu.com/question/521681812这个网站的方案进行操作，也可以简单处理，这里直接加了个大整数，使其变成正数，之后再转换回来
"""
class Paillier:
    def __init__(self) -> None:
        """ 为了防止不互质的情况出现，这里参数都手动设置，而不是随机值,注意：要加密的信息越长，质数就越大"""
        self.p=9967
        self.q=9973
        self.g = 1753
        self.r = 2677
        self.n = self.p*self.q

    def gcd(self,a,b):
        """ 最大公约数"""
        while b > 0:
            a, b = b, a % b
        return a
        
    def lcm(self,a, b):
        """最小公倍数"""
        return a * b // self.gcd(a, b)

    def public_key(self):
        """ 公钥(n,g) """
        return self.n,self.g

    def condition(self):
        if (self.gcd(self.g,self.n*self.n)==1):
            print("g 和 n*n 互质")
        else:
            print("WARNING: g 和 n*n 不互质!!!")
            print("不互质此时结果不对!!!")
        if (self.gcd(self.l,self.n)==1):
            print("l 和 n 互质")
        else:
            print("WARNING:l和n不互质,此时程序会报错")

    def private_key(self):
        """ 私钥(gLamda,gMu) """
        gLambda = int(self.lcm(self.p-1,self.q-1))
        self.l = (pow(self.g, gLambda, self.n*self.n)-1)//self.n # 这里是将参数带入的L(x)
        #self.condition()
        gMu = libnum.invmod(self.l, self.n) # 得到μ
        return gLambda,gMu

    # def mx_encrypt(self,mx):
    #     mx_g = np.zeros([2,mx.shape[1]])
    #     mx_n = np.zeros([2,mx.shape[1]])
    #     k1 = np.zeros([2,mx.shape[1]])
    #     k1 = 

    def encrypt(self,m):
        """ 公钥进行加密。m为明文,返回密文 """
        k1 = pow(self.g, m, self.n*self.n) # g^m%(n^2)
        self.k2 = pow(self.r, self.n, self.n*self.n)
        cipher = (k1 * self.k2) % (self.n*self.n)
        return cipher

    def decrypt(self,cipher,gLambda,gMu):
        """ 私钥进行解密。输入密文和私钥，返回明文 """
        l = (pow(cipher, gLambda, self.n*self.n)-1) // self.n
        mess= (l * gMu) % self.n
        return mess

    def add_homo(self,cipher,m1,gLambda,gMu):
        k3 = pow(self.g,m1,self.n*self.n)
        cipher2 = (k3 * self.k2) % (self.n*self.n)
        ciphertotal = (cipher* cipher2) % (self.n*self.n)
        l = (pow(ciphertotal, gLambda, self.n*self.n)-1) // self.n
        mess2= (l * gMu) % self.n
        return mess2

    def mx_encrypt(self,gLambda,gMu,user1Items1,user2Items1):
        user1Items1 = user1Items1*1e+7
        user1Items1 = user1Items1.T.astype(int)
        user2Items1 = user2Items1*1e+7
        user2Items1 = user2Items1.T.astype(int)
        for x in user1Items1:
            for y in x:
                cipher = self.encrypt(int(y))
                mess = self.decrypt(cipher,gLambda,gMu)
        for x in user2Items1:
            for y in x:
                cipher = self.encrypt(int(y))
                mess = self.decrypt(cipher,gLambda,gMu)

    def cout(self,gLambda,gMu,cipher,mess,m1):     
        print("p=",self.p,"\tq=",self.q)
        print("g=",self.g,"\tr=",self.r)
        print("================")
        print("Mu:\t\t",gMu,"\tgLambda:\t",gLambda)
        print("================")
        print("公钥 (n,g):\t\t",self.n,self.g)
        print("私钥 (lambda,mu):\t",gLambda,gMu)
        print("================")
        print("明文:\t",mess)

        print("密文:\t\t",cipher)
        print("解密后的密文:\t",mess)

        print("================")
        print("进行同态加密，原有基础上加",m1)

if __name__ == "__main__":
    test = Paillier()
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
    user1Items1 = user1Items1*1e+7+100000000
    user1Items1 = user1Items1.T.astype(int)
    user2Items1 = user2Items1*1e+7+100000000
    user2Items1 = user2Items1.T.astype(int)
    gLambda,gMu = test.private_key()
    print(user1Items1)
    print("+++++++++++++++++++++++++++")
    print(user2Items1)
    for x in user1Items1:
        for y in x:
            print("y  ",y)
    # m = 8800000
            cipher = test.encrypt(int(y))
            mess = test.decrypt(cipher,gLambda,gMu)
            print("mess  ",mess-100000000)
            # m1 = 2000000
            # test.cout(gLambda,gMu,cipher,mess,m1)
            # mess1 = test.add_homo(cipher,m1,gLambda,gMu)
            # test.condition()
            # print("Result:\t\t",mess1)