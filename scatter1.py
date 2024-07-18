import numpy as np
import matplotlib.pyplot as plt

e=plt.axes(projection="3d")

x1=np.random.randint(0,50,50)
y1=np.random.randint(0,50,50)
z1=np.random.randint(0,50,50)
e.scatter(x1,y1,z1,label="Dataset1")

x2=np.random.randint(0,50,50)
y2=np.random.randint(0,50,50)
z2=np.random.randint(0,50,50)
e.scatter(x2,y2,z2,label="Dataset2")

x3=np.random.randint(0,50,50)
y3=np.random.randint(0,50,50)
z3=np.random.randint(0,50,50)
e.scatter(x3,y3,z3,label="Dataset3")
plt.legend()
plt.show()