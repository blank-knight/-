CF
测试集long_mae为1.08
测试集la_mae为0.49
测试集mean_mae为0.79
mae: [2.6950108936050357, 1.661078180708751, 1.305728788478062, 0.959064422232398, 0.78577004196142]
times: [381.67915081977844, 757.9141619205475, 1146.3556566238403, 1511.9732131958008, 
1920.1025586128235]

LSH
测试集long_mae为1.36
测试集la_mae为0.59
测试集mean_mae为0.97
mae: [2.741702971845076, 1.7076045383346141, 1.4128751447613457, 1.1700259734973957, 0.9724947322374811]
times: [167.08440327644348, 301.912216424942, 454.7829382419586, 600.1529977321625, 735.2896223068237]

LSH_nbits
nbits = [10,20,30,40,50,60,70,80,90,100]
测试集long_mae为3.45
测试集la_mae为1.55
测试集mean_mae为2.50
mae: [2.542973146086643, 2.638896284958763, 2.532629460427218, 2.455631519531039, 2.484909689298669, 2.445172248129676, 2.489117544342532, 2.463007489301179, 2.3649925973972943, 2.5012006128137276]
times: [42.88141107559204, 52.64333462715149, 29.315711975097656, 37.97255301475525, 30.10122847557068, 29.162851333618164, 18.302637577056885, 38.372785568237305, 23.724012851715088, 22.79667568206787]
weight:[16.862707001677084, 19.948997210390225, 11.57520767769657, 15.463457246226831, 12.11361064959521, 11.926706331598918, 7.353062782694454, 15.579646320573994, 10.031326473420542, 9.114293177956139]
可以看出weight最大的为20 nbits

20 nbits时不同表数的效果
当前表数为: 0
测试集mean_mae为2.62
当前表数为: 1
测试集mean_mae为2.59
当前表数为: 2
测试集mean_mae为2.60
当前表数为: 3
测试集mean_mae为2.60
当前表数为: 4
测试集mean_mae为2.60
当前表数为: 5
测试集mean_mae为2.60
当前表数为: 6
测试集mean_mae为2.59
当前表数为: 7
测试集mean_mae为2.58
当前表数为: 8
测试集mean_mae为2.56
当前表数为: 9
测试集mean_mae为2.49
当前表数为: 10
测试集mean_mae为2.49
当前表数为: 11
测试集mean_mae为2.47
当前表数为: 12
测试集mean_mae为2.47
当前表数为: 13
测试集mean_mae为2.47
当前表数为: 14
测试集mean_mae为2.46
考虑时间成本问题，我们选择2张表

各个算法的误差为:{0: [2.6950108936050357, 1.661078180708751, 1.305728788478062, 0.959064422232398, 0.78577004196142], 1: [2.5488101880729275, 1.6738339364770343, 1.3627334053340914, 1.1102044177810622, 0.8972333170124414], 2: [2.5097956497815552, 1.67325407590191, 1.3752981360619931, 1.105247591980047, 0.9455877410157212]}
各个所用的时间为{0: [367.2615969181061, 733.6574704647064, 1111.6946697235107, 1488.874670267105, 1865.1594815254211], 1: [56.968422174453735, 109.52244162559509, 173.3467080593109, 236.44521355628967, 291.296763420105], 2: [83.4754912853241, 158.90886783599854, 253.34104895591736, 344.8364083766937, 425.42250061035156]}


各个算法的误差为:{0: [2.6950108936050357, 1.831389976139651, 1.5122078056664239, 1.3155270187922856, 1.2421340928899844], 1: [2.6494683388973495, 1.8047722936453503, 1.376751595871525, 1.164882955167775, 0.9498180984915001], 2: [2.6812975721720997, 1.820754361255375, 1.39266960691095, 1.1807903567485, 0.9657057817001752]}
各个所用的时间为{0: [412.63876700401306, 806.0924060344696, 1231.407428264618, 1546.1540579795837, 1937.8415942192078], 1: [77.90679788589478, 144.04647731781006, 227.74438452720642, 311.6672673225403, 377.8598804473877], 2: [107.58230686187744, 207.71248483657837, 325.03389406204224, 449.1197953224182, 580.3418517112732]}

各个算法的误差为:{0: [2.6950108936050357, 1.661078180708751, 1.305728788478062, 0.959064422232398, 0.78577004196142], 1: [2.656784127948325, 1.6266065549461248, 1.286687804508825, 1.090475350084425, 0.8677701116945999], 2: [2.6886133612230747, 1.64250927387285, 1.302546027911725, 1.106327935520575, 0.883603570520125]}
各个所用的时间为{0: [379.32889914512634, 764.2643761634827, 1175.7989473342896, 1559.0139877796173, 1934.7203288078308], 1: [72.31787586212158, 135.21740293502808, 214.49260234832764, 293.841189622879, 365.40583300590515], 2: [103.57593989372253, 194.49724173545837, 307.56843185424805, 520.6254773139954, 685.0465750694275]}




766,   -3.1311638547,-60.0229024887,     0.0,0.0,     0.0,0.0  0

766,    -3.1311638547,-60.0229024887,    20.810901877366653,-156.99193598825312,      23.942065732066652,96.96903349955312   1

766,-3.1311638547,-60.0229024887,20.810901877366653,-156.99193598825312,23.94206573206665,96.96903349955312,   23.94206573206665,96.96903349955312  4


1235,    54.5971603378,-5.9266128841,    34.36373999886834,-6.893833482892822,    20.23342033893166,0.9672205987928217 0

1235,    54.5971603378,-5.9266128841,    42.100695119160754,-6.893833482892822,    12.496465218639244,0.9672205987928217 1

1235,54.5971603378,-5.9266128841,42.100695119160754,-6.893833482892822,12.496465218639244,0.9672205987928216,    -7.736955120292414,0.0 4


随nums变化
nbits,num_lis,d = 10,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],2
mae: [3.205828649416741, 3.0417506977304427, 3.0662281158456284, 3.004025663877269, 3.0408865711286186, 3.023219165373059, 
3.0133234469131263, 3.014468015668421, 2.9718681478517492, 2.9811172995256565,2.94813427684252, 2.9761142430031597, 2.9667004067619027, 2.9755892894806264, 2.915668083643323] 
times: [52.69600462913513, 107.64199829101562, 130.47101545333862, 189.38578629493713, 321.3763370513916, 344.2920913696289, 
535.0574066638947, 599.1915102005005, 488.3573122024536, 686.4112279415131,641.1997437477112, 695.4912793636322, 759.8266310691833, 795.1582748889923, 892.2914493083954]  

