import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import numpy as np

x = 5*np.random.rand(100,1)
y = 3*x + 5*np.random.rand(100,1)

# Ridge Regression
# Linear least squares with l2 regularization.
ridge_reg = Ridge(alpha=0.01)
ridge_reg.fit(x,y)

# Lasso Regression
# Linear Model trained with L1 prior as regularizer (aka the Lasso)
lasso_reg = Lasso(alpha=0.01)
lasso_reg.fit(x,y)

# Elastic Net Regression
# Linear regression with combined L1 and L2 priors as regularizer.
elastic_net =  ElasticNet(alpha=1.0)
elastic_net.fit(x,y)


ridge_y_pred = ridge_reg.predict(x)
lasso_y_pred = lasso_reg.predict(x)
elastic_y_pred = elastic_net.predict(x)


plt.scatter(x, y)

plt.plot(x,ridge_y_pred, color='green')
plt.plot(x,lasso_y_pred, color='red')
plt.plot(x,elastic_y_pred, color='blue')

plt.savefig("Intro to ML/Other Linear Models.png")