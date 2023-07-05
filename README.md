# 基于区块链的隐私保护推荐算法

## 基本思路
    传统的隐私保护推荐系统主要用于两个方面，一是对模型的保护，主要方式有同态加密和差分隐私（联邦学习，共享模型的时候），二是对用户数据的保护（企业间互换用户数据，数据流通，从而进行大模型训练-联盟链。或者用户单独上传，获取推荐计算结果-公链），主要方式差分隐私。为什么没有同态加密呢，一是进行密钥分配后，每个用户都是不同的密钥，从而在使用密文云端计算后，无法在本地进行有效的解密；二是同态加密速度过慢，导致时效性过差。

  对于以上两个问题：

  - 首先通过E2LSH（基于P-稳态分布的欧式距离局部敏感哈希函数）对数据进行一次映射，将其哈希键值对（key：哈希桶编号，value:哈希值）与用户id一同记录在区块链上。E2LSH的特点是高维相似的数据在低维的映射编码空间也以一定概率相似（即以很大概率会映射到同一个哈希桶里面），因此，只需要对哈希桶编号相同的用户分发相同的密钥，那么他们的数据在经过同态加密后，就能在云端进行同态运算（因为只使用同一个哈希桶里的用户进行相似度和推荐计算），然后将运算后的值返回给用户，用户使用分发的密钥解密即可得到推荐结果明文（因为都是同一个密钥，参数一致，因此能正确解密）。且E2LSH映射后，一方面，只需检索同一个哈希桶里的用户，数据量大大减小；另一方面，因为P稳态分布的特点，使其可直接用哈希值进行相似度计算，降低了计算量。因此在合理分发密钥，进行同态运算的同时，整体推荐效率也得到了提升。

  - 通过使用基于格的快速同态加密算法，其原理基于LWE困难问题和格理论，与传统基于质数分解和离散对数等困难问题的同态加密相比(Paillier，RSA等），具有更高的运算速度（格中以矩阵形式运算），且依然具有同态性质。

  - 本文采用IPFS对用户密文数据进行存储，其存储哈希值和用户E2LSH映射后的哈希值都记录在了区块链上，便与云端进行提取，同时，记录在链上还有一个好处便是能通过溯源E2LSH哈希值判断某个用户是否存在数据异常问题（历史哈希值是否存在差异过大问题）从而判断异常用户。

  本文的区块链实现过于简单，通过Dapp university上的教程编写和搭建的，使用以太坊测试网络链接IPFS网络，前端作为上传页面，Ganache分配多个以太坊账户和对应的ETH作为测试用户。手动在本地进行运算后上传测试网，IPFS会自动进行记录，之后手动通过哈希值进行数据获取等。

  注意：所有用户应使用同一个E2LSH，从而避免最后映射的值不对。

算法核心代码整理在了在另一个仓库“基于格的隐私保护推荐算法”里面，这个里也有，但比较乱。
