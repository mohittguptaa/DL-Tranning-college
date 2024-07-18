import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    r=1/(1+np.exp(x))
    return r
    
x= np.linspace(-10,10,50)
y=sigmoid(x)
plt.plot(x,y)
plt.show()

def tanh1(i):
    h1=np.tanh(i)
    return h1

data=[-4,-3,-2,-1,0,1,2,3,4]
y=[tanh1(i) for i in data]
plt.plot(data,y)
plt.title("let the function lie [-1,1]")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

def relu(x):
    data=[max(0,i) for i in x]
    r=np.array(data,dtype=float)
    return r
x_data=np.linspace(-10,10,200)
y_data=relu(x_data)
plt.plot(x_data,y_data)

plt.title("let the function lie [-2,1]")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

e=plt.axes(projection="3d")
xdata=np.random.randint(10,50,600)
ydata=np.random.randint(10,50,600)
zdata=np.random.randint(10,50,600)
e.scatter(xdata,ydata,zdata)
plt.show()