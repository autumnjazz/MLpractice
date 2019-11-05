import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")
import numpy as np
from sklearn.linear_model import LinearRegression

x = 5*np.random.rand(100,1)
y = 3*x + 5*np.random.rand(100,1)

lr = LinearRegression()
lr.fit(x,y)
predicted = lr.predict(x)

fig, ax = plt.subplots(1,2, figsize=(16, 7))

ax[0].scatter(x,y)
ax[1].scatter(x,y)
ax[1].plot(x, predicted, color='b')

ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')

fig.savefig("Intro to ML/Linear Regression.png")