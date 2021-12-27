import numpy as np

import re

from Matrix import transpose, multiply
from GaussSeidel import gauss_seidel


class LinearRegression:
    def __init__(self):
        self.parameters = None
        self.result = None
    def fit(self,G,y):

        G_t = transpose(G)
        gtg = multiply(G_t,G)
        gty = multiply(G_t,y)
        self.parameters = gauss_seidel(gtg,gty)
        return self.parameters
    
    def to_string_predict_function(self,to_predict_feature_name,features_name):
        final_string = f"{to_predict_feature_name}({','.join(features_name[0:1])},...) = "
        #final_string = f"{to_predict_feature_name} = "
        final_string += '+'.join([f"({p:.3})*{f}" for f,p in zip(features_name,self.parameters)])
        return final_string
    def to_latex_predict_function(self,to_predict_feature_name,features_name):
        final_string = self.to_string_predict_function(to_predict_feature_name,features_name)
        final_string = re.sub(r"\*", r"\\cdot ", final_string)
        final_string = re.sub(r"_", r"", final_string)
        return final_string
    def predict(self,x):
        result = np.zeros((x.shape[0],1))
        for i in range(x.shape[0]):
#             print(Matrix([x[i]]))
#             print(self.parameters)
            result[i]=[np.sum(x[i]*self.parameters)]
        self.result = result
        return result
    def absolute_error(self,y):
        return np.abs(self.result - y)
    def relative_error(self,y):
        return np.abs(self.absolute_error(y)/y)
    def score(self,x,y): # R^2
        y_predicted = self.predict(x)
        u = ((y-y_predicted)**2).sum()
        v = ((y-y.mean())**2).sum()
        return 1 - u/v
    def adjusted_r2(self,x,y):
        r2 = self.score(x,y)
        k = self.parameters.shape[0]
        n = y.shape[0]
        return 1 - ((1-r2)*(n-1))/(n-k-1)
    def mse(self,y):
        return ((y-self.result)**2).mean()
    def rmse(self,y):
        return np.sqrt(self.mse(y))
    def rrmse(self,y):
        return np.sqrt(self.mse(y)/((y-y.mean())**2).sum())
    def mae(self,y):
        return (np.abs(self.result-y)).mean()
    def mape(self,y):
        return (np.abs((self.result-y)/y)).mean()
    def rmsle(self,y):
        return np.sqrt(((np.log(y+1)-np.log(self.result+1))**2).mean())
