import numpy as np
import matplotlib.pyplot as plt

def linear(x):
    return x+5
xdata = np.linspace(-6,6,100)
ydata = linear(xdata)
plt.plot(xdata,ydata)
plt.title("Linear Data Representation")
plt.xlabel("X_data")
plt.ylabel("Y_data")
plt.show()