from random import randint
import libnum
import sys

def gcd(a,b):
    """ 最大公约数"""
    while b > 0:
        a, b = b, a % b
    return a
    
def lcm(a, b):
    """最小公倍数"""
    return a * b // gcd(a, b)



def L(x,n):
	return ((x-1)//n)

p=17
q=19
m=55


if (len(sys.argv)>1):
	m=int(sys.argv[1])

if (len(sys.argv)>2):
	p=int(sys.argv[2])

if (len(sys.argv)>3):
	q=int(sys.argv[3])

if (p==q):
	print("P and Q cannot be the same")
	sys.exit()

n = p*q

gLambda = lcm(p-1,q-1)


g = randint(20,150)
if (gcd(g,n*n)==1):
	print("g 和 n*n 互质")
else:
	print("WARNING: g 和 n*n 不互质!!!")
	print("不互质此时结果不对!!!")

r = randint(20,150)


l = (pow(g, gLambda, n*n)-1)//n

if (gcd(l,n)==1):
	print("l 和 n 互质")
else:
	print("l和n不互质")
gMu = libnum.invmod(l, n)



k1 = pow(g, m, n*n)
k2 = pow(r, n, n*n)


cipher = (k1 * k2) % (n*n)


l = (pow(cipher, gLambda, n*n)-1) // n

mess= (l * gMu) % n

m1=33
print("p=",p,"\tq=",q)
print("g=",g,"\tr=",r)
print("================")
print("Mu:\t\t",gMu,"\tgLambda:\t",gLambda)
print("================")
print("公钥 (n,g):\t\t",n,g)
print("私钥 (lambda,mu):\t",gLambda,gMu)
print("================")
print("明文:\t",mess)

print("密文:\t\t",cipher)
print("解密后的密文:\t",mess)

print("================")
print("进行同态加密，原有基础上加",m1)




k3 = pow(g, m1, n*n)

cipher2 = (k3 * k2) % (n*n)

ciphertotal = (cipher* cipher2) % (n*n)



l = (pow(ciphertotal, gLambda, n*n)-1) // n

mess2= (l * gMu) % n

print("Result:\t\t",mess2)