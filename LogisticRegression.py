import numpy as np
import time
from sklearn.metrics import classification_report

t0=time.time()

#读取数据
def loadData():
    train_x=[]
    train_y=[]
    file=open('testData.txt')
    for line in file.readlines():
        linArry=line.strip().split()
        train_x.append([1.0,float(linArry[0]),float(linArry[1])])
        train_y.append(float(linArry[2]))
    return np.array(train_x),np.array(train_y)

#定义sigmoid函数
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

#批量梯度下降算法，参数：训练集，标记，学习率，最大迭代次数
def BGD(x,y,alpha,maxLoop):
    #将训练集和标记都转换成矩阵
    dataMatrix=np.mat(x)
    labelMatrix=np.mat(y).T
    m,n=dataMatrix.shape
    #定义参数theta
    theta=np.ones((n,1))
    for i in range(maxLoop):
        h=sigmoid(dataMatrix.dot(theta))
        #误差
        error=h-labelMatrix
        #更新参数
        theta=theta-alpha*(dataMatrix.T*error)
    return np.asarray(theta)

#预测函数
def predict(x,theta):
    return sigmoid(np.dot(x,theta)).T[0]

x,y=loadData()
m,n=np.shape(x)
theta=BGD(x,y,0.01,1000)
theta = np.atleast_2d(theta).reshape((3,1))
pre = np.round(predict(x, theta))
t1=time.time()
###############模型评测#####################
print(classification_report(y, pre))
print("Time:"+str(t1-t0))