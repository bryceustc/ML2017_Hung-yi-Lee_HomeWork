#! -*- coding=utf-8 -*-
# Project:  Machine Learning
# Date: 9/21/19
# Author: bryce

import sys
import math
import csv,os
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as scio
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def AdaGrad(X,Y,w,eta,iteration, lambdaL2):
	s_grad = np.zeros(len(X[0]))
	list_cost = []
	for i in range(iteration):
		h = np.dot(X,w)
		loss = h - Y
		cost = np.sum(loss**2)/len(X)
		list_cost.append(cost)

		grad = np.dot(X.T,loss)/len(X) + lambdaL2*w #均方误差损失函数的，梯度的向量表示
		s_grad += grad**2
		ada = np.sqrt(s_grad)
		w = w - eta*grad/ada
	return w, list_cost

def SGD(X, Y, w, eta, iteration, lambdaL2):
	list_cost = []
	for i in range(iteration):
		h = np.dot(X,w)
		loss = h - Y
		cost = np.sum(loss**2)/len(X)
		list_cost.append(cost)

		rand = np.random.randint(0, len(X))
		grad = X[rand]*loss[rand]/len(X) + lambdaL2 * w
		w = w - eta * grad
	return w, list_cost

def GD(X, Y, w, eta, iteration, lambdaL2):
	list_cost = []
	for i in range(iteration):
		h = np.dot(X,w)
		loss = h - Y
		cost = np.sum(loss**2)/len(X)
		list_cost.append(cost)

		grad = np.dot(X.T, loss)/len(X) + lambdaL2 * w
		w = w - eta*grad
	return w, list_cost





#数据读取及预处理
train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")
train_data = train_data.drop(['Date', 'stations'], axis=1)#axis 默认0 是删除行，删除列要改axis为1
features = train_data['observations'].unique()
#print (features)
new_train_data = pd.DataFrame(np.zeros([24*240,18]),columns = features)#240天24小时，18个特征的影响
#print(new_train_data)
for i in features:
	train_data1 = train_data[train_data['observations'] == i]
	train_data1.drop(['observations'],axis=1,inplace=True)#inplace参数 True不创建新对象，直接对原始对象进行修改，False对数据进行修改并返回新的对象
	train_data1 = np.array(train_data1)
	train_data1[train_data1 == 'NR'] = '0'
	train_data1 = train_data1.astype('float') #astype用于array中数值类型转换
	train_data1 = train_data1.reshape(1,24*240)
	train_data1 = train_data1.T
	new_train_data[i] = train_data1
#print (new_train_data)

label = np.array(new_train_data['PM2.5'][9:],dtype='float32')

# 探索性数据分析 EDA
# 最简单粗暴的方式就是根据 HeatMap 热力图分析各个指标之间的关联性

fig, ax = plt.subplots(figsize=(12,9))
# corr function 相关性分析，method 可选择'pearson','kendall','spearman'，默认pearson
sns.heatmap(new_train_data.corr(method ='pearson'),annot=True,fmt=".0%",linewidth=0.5) #annot为True 将数据值写入每个单元格，fmt表格显示显示数据类型'.0%'显示百分比
plt.savefig(os.path.join(os.path.dirname(__file__), "figures/FetureSlection"))
plt.show()

#模型选择
# 通过热力图分析PM2.5,PM10,SO2影响PM2.5比较大，故选取PM2.5,PM10,SO2三个特征
# 使用前九个小时的PM2.5、PM10、SO2 来预测第十个小时的PM2.5，使用线性回归模型

train_Xa = []
train_Xb = []
train_Xc = []
train_X = []

train_a = np.array(new_train_data['PM2.5'],dtype='float32')
train_b = np.array(new_train_data['PM10'],dtype='float32')
train_c = np.array(new_train_data['SO2'],dtype='float32')
for i in range (len(new_train_data)-9):
	xa = np.array(train_a[i:i+9])
	xb = np.array(train_b[i:i+9])
	xc = np.array(train_c[i:i+9])
	#print(x)
	train_Xa.append(xa)
	train_Xb.append(xb)
	train_Xc.append(xc)
train_Xa = np.array(train_Xa)
train_Xb = np.array(train_Xb)
train_Xc = np.array(train_Xc)

train_X = np.concatenate((train_Xa,train_Xb,train_Xc),axis=1)

# add bias
train_X = np.concatenate((np.ones((train_X.shape[0],1)), train_X), axis = 1)

#print(train_X)

#训练模型
w = np.zeros(len(train_X[0]))
w_sgd, cost_list_sgd = SGD(train_X, label, w, eta=0.0001, iteration=20000, lambdaL2=0)
# w_sgd50, cost_list_sgd50 = SGD(trainX, trainY, w, eta=0.0001, iteration=20000, lambdaL2=50)
w_ada, cost_list_ada = AdaGrad(train_X, label, w, eta=1, iteration=20000, lambdaL2=0)
# w_gd, cost_list_gd = SGD(trainX, trainY, w, eta=0.0001, iteration=20000, lambdaL2=0)

#close form
w_cf = inv(train_X.T.dot(train_X)).dot(train_X.T).dot(label)
cost_wcf = np.sum((train_X.dot(w_cf)-label)**2) / len(train_X)
hori = [cost_wcf for i in range(20000-5)]


# 模型评价
test_Xa = []
test_Xb = []
test_Xc = []
test_X = []

test_data = pd.read_csv("./data/test.csv")
test_Xa = test_data[test_data['AMB_TEMP'] == 'PM2.5']
test_Xa = np.array(test_Xa)
test_Xa = np.delete(test_Xa, [0,1],axis=1)
test_Xa = test_Xa.astype('float')
test_Xa = test_Xa.T

test_Xb = test_data[test_data['AMB_TEMP'] == 'PM10']
test_Xb = np.array(test_Xb)
test_Xb = np.delete(test_Xb, [0,1],axis=1)
test_Xb = test_Xb.astype('float')
test_Xb = test_Xb.T

test_Xc = test_data[test_data['AMB_TEMP'] == 'SO2']
test_Xc = np.array(test_Xc)
test_Xc = np.delete(test_Xc, [0,1],axis=1)
test_Xc = test_Xc.astype('float')
test_Xc = test_Xc.T
#print(test_Xc)

test_X = np.concatenate((test_Xa, test_Xb,test_Xc), axis=0)
test_X = test_X.T


# add bias
test_X = np.concatenate((np.ones((test_X.shape[0],1)), test_X), axis=1)

#output testdata
y_ada = np.dot(test_X, w_ada)
y_sgd = np.dot(test_X, w_sgd)
y_cf = np.dot(test_X, w_cf)

#csv format
ans = []
for i in range(len(test_X)):
    ans.append(["id_"+str(i)])
    a = np.dot(w_ada,test_X[i])
    ans[i].append(a)
ans = np.array(ans)
ans = pd.DataFrame(ans,columns=['id','value'])
#print(ans)
ans.to_csv('./result/answer.csv',index=False,header=True)  #index=False,header=False表示不保存行索引和列标题

#parse anser
ans_Y = pd.read_csv("./data/ans.csv")
ans_Y = ans_Y.drop(['id'], axis=1)
ans_Y = np.array(ans_Y)
#print(ans_Y)

#plot training data with different gradiant method
plt.plot(np.arange(len(cost_list_ada[5:])), cost_list_ada[5:], 'b', label="ada")
plt.plot(np.arange(len(cost_list_sgd[5:])), cost_list_sgd[5:], 'g', label='sgd')
# plt.plot(np.arange(len(cost_list_sgd50[3:])), cost_list_sgd50[3:], 'c', label='sgd50')
# plt.plot(np.arange(len(cost_list_gd[3:])), cost_list_gd[3:], 'r', label='gd')
plt.plot(np.arange(len(cost_list_ada[5:])), hori, 'y--', label='close-form')
plt.title('Train Process')
plt.xlabel('Iteration')
plt.ylabel('Loss Function(Quadratic)')
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), "figures/TrainProcess"))
plt.show()

#plot fianl answer
plt.figure()
plt.subplot(131)
plt.title('CloseForm')
plt.xlabel('dataset')
plt.ylabel('PM2.5')
plt.plot(np.arange((len(ans_Y))), ans_Y, 'r,')
plt.plot(np.arange(240), y_cf, 'b')
plt.subplot(132)
plt.title('AdaGrad')
plt.xlabel('dataset')
plt.ylabel('PM2.5')
plt.plot(np.arange((len(ans_Y))), ans_Y, 'r,')
plt.plot(np.arange(240), y_ada, 'g')
plt.subplot(133)
plt.title('SGD')
plt.xlabel('dataset')
plt.ylabel('PM2.5')
plt.plot(np.arange((len(ans_Y))), ans_Y, 'r,')
plt.plot(np.arange(240), y_sgd, 'y')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "figures/Compare"))
plt.show()
