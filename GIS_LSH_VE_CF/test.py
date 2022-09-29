# mae = [2.542973146086643, 2.638896284958763, 2.532629460427218, 2.455631519531039, 2.484909689298669, 2.445172248129676, 2.489117544342532, 2.463007489301179, 2.3649925973972943, 2.5012006128137276]
# times = [42.88141107559204, 52.64333462715149, 29.315711975097656, 37.97255301475525, 30.10122847557068, 29.162851333618164, 18.302637577056885, 38.372785568237305, 23.724012851715088, 22.79667568206787]
# weight = []
# for i in range(len(mae)):
#     weight.append(times[i]/mae[i])
# print(weight)
# print(sorted(weight))

import gmpy2 as gy
import random
import time
import libnum

class Paillier(object):
    def __init__(self, pubKey=None, priKey=None):
        self.pubKey = pubKey
        self.priKey = priKey

    def __gen_prime__(self, rs):
        p = gy.mpz_urandomb(rs, 1024)
        while not gy.is_prime(p):
            p += 1
        return p
    
    def __L__(self, x, n):
        res = gy.div((x - 1), n)
        # this step is essential, directly using "/" causes bugs
        # due to the floating representation in python
        return res
    
    def __key_gen__(self):
        # generate random state
        while True:
            rs = gy.random_state(int(time.time()))
            p = self.__gen_prime__(rs)
            q = self.__gen_prime__(rs)
            n = p * q
            lmd =(p - 1) * (q - 1)
            # originally, lmd(lambda) is the least common multiple. 
            # However, if using p,q of equivalent length, then lmd = (p-1)*(q-1)
            if gy.gcd(n, lmd) == 1:
                # This property is assured if both primes are of equal length
                break
        g = n + 1
        mu = gy.invert(lmd, n)
        #Originally,
        # g would be a random number smaller than n^2, 
        # and mu = (L(g^lambda mod n^2))^(-1) mod n
        # Since q, p are of equivalent length, step can be simplified.
        self.pubKey = [n, g]
        self.priKey = [lmd, mu]
        return
        
    def decipher(self, ciphertext):
        n, g = self.pubKey
        lmd, mu = self.priKey
        m =  self.__L__(gy.powmod(ciphertext, lmd, n ** 2), n) * mu % n
        print("raw message:", m)
        plaintext = libnum.n2s(int(m))
        return plaintext

    def encipher(self, plaintext):
        m = libnum.s2n(plaintext)
        n, g = self.pubKey
        r = gy.mpz_random(gy.random_state(int(time.time())), n)
        while gy.gcd(n, r)  != 1:
            r += 1
        ciphertext = gy.powmod(g, m, n ** 2) * gy.powmod(r, n, n ** 2) % (n ** 2)
        return ciphertext

if __name__ == "__main__":
    pai = Paillier()
    pai.__key_gen__()
    pubKey = pai.pubKey
    print("Public/Private key generated.")
    plaintext = input("Enter your text: ")
    # plaintext = 'Cat is the cutest.'
    print("Original text:", plaintext)
    ciphertext = pai.encipher(plaintext)
    print("Ciphertext:", ciphertext)
    deciphertext = pai.decipher(ciphertext)
    print("Deciphertext: ", deciphertext)
    

