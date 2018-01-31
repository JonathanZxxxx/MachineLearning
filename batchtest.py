import time

t0=time.time()
#x=[1,2,3,4,5,6]
#y=[13,14,20,21,25,30]
#x=[1891,1873,1612,1515,1397,994,577,254,308,388]
#y=[147,188,133,118,126,91,39,15,630,60]
x=[1,2,3,4,5,6,7,8,9,10]#y=3.82+4.87*x
y=[8.69,13.56,18.34,23.3,28.02,33.04,37.91,42.78,47.65,52.22]
#学习率
alpha=0.0001
#h(x)=theta0+theta0*x
theta0=0
theta1=0
m=len(x)
count=0
diff=[0,0]
maxCount=10000 #最大迭代次数
#梯度下降
#while count<=maxCount:
while 1:
    count+=1
    for i in range(m):
        diff[0]+=theta0+theta1*x[i]-y[i]
        diff[1]+=(theta0+theta1*x[i]-y[i])*x[i]
    theta0=theta0-alpha/m*diff[0]
    theta1=theta1-alpha/m*diff[1]
    error1=0
    for i in range(m):
        error1+=((theta0+theta1*x[i]-y[i])**2)/5
    if(error1<=0.1):
        break
    print ('count=%d'%count,"theta0=%f"%theta0,"theta1=%f"%theta1,"error%f"%error1)
t1=time.time()
print ('count=%d'%count,"theta0=%f"%theta0,"theta1=%f"%theta1,"error%f"%error1,"time"+str(t1-t0))

