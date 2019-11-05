from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")

x = 3*np.random.rand(1000, 1) + 1
y = x**2 + x + 2 +5*np.random.rand(1000,1) 

# Generate a new feature matrix consisting of all polynomial combinations
# of the features with degree less than or equal to the specified degree.
# For example, if an input sample is two dimensional and of the form [a, b],
# the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
poly_feat = PolynomialFeatures(2,False)
poly_x = poly_feat.fit_transform(x)

linear_model = LinearRegression()

linear_model.fit(poly_x, y)

def plotting_learning_curves(model, x, y):
    x_train, x_eval, y_train, y_eval = train_test_split(x,y, test_size=2)
    len_train = len(x_train)

    train_errors=[]
    eval_errors=[]
    
    for i in range(1,len_train):
        model.fit(x_train[:i], y_train[:i])
        pred_train = model.predict(x_train[:i])
        pred_eval = model.predict(x_eval)

        error_train =  mse(y_train[:i], pred_train)
        error_eval = mse(y_eval, pred_eval)
        
        train_errors.append(error_train)
        eval_errors.append(error_eval)
    
    plt.plot(np.sqrt(train_errors), 'r', label="train")
    plt.plot(np.sqrt(eval_errors), 'b', label="evaluation")
    
    plt.savefig("Intro to ML/Polynomial Features.png")

plotting_learning_curves(linear_model,x,y)