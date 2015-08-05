import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math

def vectorize(M):
	temp = []
	for i in range(M.shape[0]*M.shape[1]):
		temp.append(M.T.item(i))
	V = np.asarray(temp)
	return V

def matrixize(V, C_dimension):
	temp = np.zeros(shape = (C_dimension, len(V)/C_dimension))
	for i in range(len(V)/C_dimension):
		temp.T[i] = V[i*C_dimension : (i+1)*C_dimension]
	W = temp
	return W

'''

def fun(x):
	#x = x.reshape(self.userNum, self.userNum)
	return x**2 -1
def fprime(x):
	return 2*x

cons = (
		{'type' : 'ineq',
		'fun' : lambda  x: x })
res = minimize(fun, 3, constraints = cons, method ='SLSQP', jac = fprime, options={'disp': True})
print res.fun
'''
'''
fun = lambda x: (x[0] + 1)**2 + (x[1] +2.5)**2
cons = (
	{'type': 'ineq', 'fun': lambda x: x}
	)
res = minimize(fun, (2, 0), method='SLSQP', 
               constraints=cons, options = {'disp': True})

print res.x
if res.x[1] <0:
	print 'neg'
'''







	
	


