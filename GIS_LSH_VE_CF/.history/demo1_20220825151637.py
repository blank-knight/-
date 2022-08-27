# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import csv
# 读取所有数据
# with open("shenzhen stockA/RESSET_DRESSTK_2016_2020_1.csv", "r") as csvfile:
#     reader = csv.reader(csvfile)
#     rows = [row for row in reader]
# print(rows)

with open("sz stockA_new/pinganyinghang.csv", "rt", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    end_price = [row[3] for row in reader]  #取收盘价
    del end_price[0]
    #end_price = [i for i in end_price if i != '']  #去除空数据
    end_price1 = list(map(float, end_price)) #数据转化为浮点数
    #print(column)
# with open("深圳资讯—粗数据.csv.csv", "rt") as massage:
#     massager = csv.reader(massage)
#     infor = [row[2] for row in massager]
#     infor = [i for i in infor if i != '']
#     del infor[0]
    # print(infor)



#前均
def moving_average(array,n):

#array为每日收盘价组成的数组,

# n可以为任意正整数，按照股市习惯，一般设为5或10或20、30、60等

    mov = np.cumsum(array, dtype=float) #逐项累加
    mov[n:] = mov[n:] - mov[:-n] #当日前N日收盘价和
    moving = mov[(n-1):]/n  #前均价
    moving = moving[:-n]
    return moving
#后均
def moving_bk(array, n):
    mov_bk = np.cumsum(array, dtype=float)  # 逐项累加
    mov_bk[:-n] = mov_bk[n:] - mov_bk[:-n]  # 当日后N日收价和
    moving_bk = mov_bk[:-n] / n  # 后均价
    moving_bk = moving_bk[n-1:]
    return moving_bk
#均线差
def differ(array, n):
    line = moving_bk(array, n) - moving_average(array, n)  # 均线差
    return line
#分类
def classify(array, n):
    data = differ(array, n)
    class_data = []
    for i in data:
        if i > 0:
            class_data.append(1)
        else:
            class_data.append(0)
    class_data = [str(i) for i in class_data]  #列表递推公式，将列表内浮点数更新为字符串
    # print(column + class_data)
    return class_data

#多列表写入文件
"""用zip()函数，接受一系列可迭代对象作为参数，
将不同对象中相对应的元素打包成一个元组（tuple），
返回由这些元组组成的list列表，如果传入的参数的长度不等，
则返回的list列表的长度和传入参数中最短对象的长度相同。"""
# def save(array, n):
#     information = infor
#     average = moving_average(array, n)
#     label = classify(array, n)
#     data_list = []
#     for i, j, k in zip(information, average, label):  #存为字典
#         x = {}
#         x["资讯正文"] = i
#         x["均线差"] = j
#         x["标签"] = k
#         data_list.append(x)
#         # csv_writer.writerow([i, j, k])
#     with open("平安银行标签.csv", 'w', newline='', encoding='GBK') as last_doc:
#         writer = csv.writer(last_doc)
#         writer.writerow(['资讯正文', '均线差', '标签'])
#         for nl in data_list:
#             writer.writerow(nl.values())

def save(array, l, m, n):
    average = differ(array, l)
    label = classify(array, l)
    last_doc = pd.read_csv("sz stockA_new/pinganyinghang1.csv", encoding="utf-8")
    last_doc.drop(last_doc.index[0:l-1], inplace=True)
    last_doc.drop(last_doc.index[-l:], inplace=True)
    last_doc["五日均线差"] = average
    last_doc.to_csv("sz stockA_new/pinganyinghang1.csv", index=False)
    last_doc["标签"] = label
    last_doc.to_csv("sz stockA_new/pinganyinghang1.csv", index=False)

    average = differ(array, m)
    label = classify(array, m)
    last_doc = pd.read_csv("sz stockA_new/pinganyinghang2.csv", encoding="utf-8")
    last_doc.drop(last_doc.index[0:m - 1], inplace=True)
    last_doc.drop(last_doc.index[-m:], inplace=True)
    last_doc["十日均线差"] = average
    last_doc.to_csv("sz stockA_new/pinganyinghang2.csv", index=False)
    last_doc["标签"] = label
    last_doc.to_csv("sz stockA_new/pinganyinghang2.csv", index=False)

    average = differ(array, n)
    label = classify(array, n)
    last_doc = pd.read_csv("sz stockA_new/pinganyinghang3.csv", encoding="utf-8")
    last_doc.drop(last_doc.index[0:n - 1], inplace=True)
    last_doc.drop(last_doc.index[-n:], inplace=True)
    last_doc["二十日均线差"] = average
    last_doc.to_csv("sz stockA_new/pinganyinghang3.csv", index=False)
    last_doc["标签"] = label
    last_doc.to_csv("sz stockA_new/pinganyinghang3.csv", index=False)

if  __name__ == "__main__":
    save(end_price1, l=5, m=10, n=20)


    # print(moving_average(column, 10))
    # print(moving_average(column, 20))


