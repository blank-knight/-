# -*- coding:utf-8 -*-


import pandas as pd
import numpy as np
import csv
import os
import demo1



filedir = "data new/深圳/测试"
file_namelist = os.listdir(filedir)



for filename in file_namelist:
    filepath = filedir + filename

    with open(filepath, "rt", encoding="gbk") as csvfile:
        reader = csv.reader(csvfile)
        end_price = [row[5] for row in reader]  #取收盘价
        del end_price[0]
        #end_price = [i for i in end_price if i != '']  #去除空数据
        end_price1 = list(map(float, end_price)) #数据转化为浮点数
    average = demo1.differ(end_price1, 5)
    label = demo1.classify(end_price1, 5)

filedir1 = "D:\\MyApp\PyCharm Community Edition 2021.2.3\StockDirect\data new\深圳\测试结果\\"
file_namelist1 = os.listdir(filedir1)
for filename1 in file_namelist1:
    filepath1 = filedir1 + filename1
    last_doc = pd.read_csv(filepath1, encoding="gbk")
    last_doc.drop(last_doc.index[0:5-1], inplace=True)
    last_doc.drop(last_doc.index[-5:], inplace=True)
    last_doc["五日均线差"] = average
    last_doc.to_csv(filename1, index=False)
    print(last_doc)
    last_doc["标签"] = label
    last_doc.to_csv(filename1, index=False)




