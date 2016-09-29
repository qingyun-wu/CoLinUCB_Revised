import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.optimize import minimize
import math
from util_functions import vectorize, matrixize
import time
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from numpy import linalg as LA
from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp

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

def fun(x, y, theta):
	obj = (1/2.0)*(np.dot(np.transpose(x), theta) - click)**2
	regularization = 0
	return obj + regularization

def evaluateGradient(x,y,theta, lambda_, regu ):
	if regu == 'l1':
		grad = x*(np.dot(np.transpose(x),theta) - y) + lambda_*np.sign(theta)  #Lasso 
	elif regu == 'l2':
		grad = x*(np.dot(np.transpose(x),theta) - y) + lambda_*theta           # Ridge                      
	return grad

def getcons(dim):
	cons = []
	cons.append({'type': 'eq','fun': lambda x : np.sum(x)-1})
	for i in range(dim):
		cons.append({'type' : 'ineq','fun' : lambda  x: x[i] })
		cons.append({'type' : 'ineq','fun' : lambda x: 1-x[i]})
	return tuple(cons)
def getbounds(dim):
	bnds = []
	for i in range(dim):
		bnds.append((0,1))
	return tuple(bnds)


class WStruct_batch_Cons:
	def __init__(self, featureDimension, lambda_, eta_, userNum, windowSize =20):
		self.windowSize = windowSize
		self.counter = 0
		self.userNum = userNum
		self.lambda_ = lambda_
		# Basic stat in estimating Theta
		self.A = lambda_*np.identity(n = featureDimension*userNum)
		self.b = np.zeros(featureDimension*userNum)
		self.UserTheta = np.zeros(shape = (featureDimension, userNum))
		#self.UserTheta = np.random.random((featureDimension, userNum))
		self.AInv = np.linalg.inv(self.A)
		
		#self.W = np.random.random((userNum, userNum))
		self.W = np.identity(n = userNum)
		self.Wlong = vectorize(self.W)
		self.batchGradient = np.zeros(userNum*userNum)

		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.BigW = np.kron(np.transpose(self.W), np.identity(n=featureDimension))
		self.CCA = np.identity(n = featureDimension*userNum)
		self.BigTheta = np.kron(np.identity(n=userNum) , self.UserTheta)
		self.W_X_arr = []
		self.W_y_arr = []
		for i in range(userNum):
			self.W_X_arr.append([])
			self.W_y_arr.append([])
		
	def updateParameters(self, articlePicked, click,  userID):	
		self.counter +=1
		self.Wlong = vectorize(self.W)
		featureDimension = len(articlePicked.featureVector)
		T_X = vectorize(np.outer(articlePicked.featureVector, self.W.T[userID])) 
		self.A += np.outer(T_X, T_X)	
		self.b += click*T_X
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = matrixize(np.dot(self.AInv, self.b), len(articlePicked.featureVector)) 

		Xi_Matirx = np.zeros(shape = (featureDimension, self.userNum))
		Xi_Matirx.T[userID] = articlePicked.featureVector
		W_X = vectorize( np.dot(np.transpose(self.UserTheta), Xi_Matirx))
		W_X_current = np.dot(np.transpose(self.UserTheta), articlePicked.featureVector)

		self.W_X_arr[userID].append(W_X_current)
		self.W_y_arr[userID].append(click)

		def fun(w):
			w = np.asarray(w)
			res = np.sum((np.dot(self.W_X_arr[userID], w) - self.W_y_arr[userID])**2, axis = 0) + self.lambda_*np.linalg.norm(w)
			return res
		def fun(w,X,Y):
			w = np.asarray(w)
			res = np.sum((np.dot(X, w) - Y)**2, axis = 0) + self.lambda_*np.linalg.norm(w)
			return res

		'''	
		def fprime(w):
			w = np.asarray(w)
			res = self.W_X_arr[userID]*(np.dot(np.transpose(self.W_X_arr[userID]),w) - self.W_y_arr[userID]) + self.lambda_*w
			return res
		'''
		'''
		if self.counter%self.windowSize ==0:
			current = self.W.T[userID]
			res = minimize(fun, current, constraints = getcons(len(self.W)), method ='SLSQP', bounds=getbounds(len(self.W)), options={'disp': False})
			if res.x.any()>1 or res.x.any <0:
				print 'error'
				print res.x
			self.W.T[userID] = res.x
		'''
		if self.counter%self.windowSize ==0:
			for i in range(len(self.W)):
				if len(self.W[i]) !=0:
					def fun(w):
						w = np.asarray(w)
						res = np.sum((np.dot(self.W_X_arr[i], w) - self.W_y_arr[i])**2, axis = 0) + self.lambda_*np.linalg.norm(w)
						return res
					current = self.W.T[i]
					res = minimize(fun, current, constraints = getcons(len(self.W)), method ='SLSQP', bounds=getbounds(len(self.W)), options={'disp': False})
					self.W.T[i] = res.x
			self.windowSize = self.windowSize*2 
		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.BigW = np.kron(np.transpose(self.W), np.identity(n=len(articlePicked.featureVector)))
		self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW))

		self.BigTheta = np.kron(np.identity(n=self.userNum) , self.UserTheta)
	def getProb(self, alpha, article, userID):
		TempFeatureM = np.zeros(shape =(len(article.featureVector), self.userNum))
		TempFeatureM.T[userID] = article.featureVector
		TempFeatureV = vectorize(TempFeatureM)
		
		mean = np.dot(self.CoTheta.T[userID], article.featureVector)
		var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))
		pta = mean + alpha * var
		#pta = mean + alpha * var
		return pta


class WStruct_SGD(WStruct_batch_Cons):
	def __init__(self, featureDimension, lambda_, eta_, userNum, windowSize = 1, regu='l2'):
		WStruct_batch_Cons.__init__(self,featureDimension = featureDimension, lambda_ = lambda_, eta_ = eta_, userNum = userNum)	
		self.regu = regu

	def updateParameters(self, articlePicked, click,  userID):	
		self.counter +=1
		self.Wlong = vectorize(self.W)
		featureDimension = len(articlePicked.featureVector)
		T_X = vectorize(np.outer(articlePicked.featureVector, self.W.T[userID])) 
		self.A += np.outer(T_X, T_X)	
		self.b += click*T_X
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = matrixize(np.dot(self.AInv, self.b), len(articlePicked.featureVector)) 

		Xi_Matirx = np.zeros(shape = (featureDimension, self.userNum))
		Xi_Matirx.T[userID] = articlePicked.featureVector
		W_X = vectorize( np.dot(np.transpose(self.UserTheta), Xi_Matirx))
		self.batchGradient +=evaluateGradient(W_X, click, self.Wlong, self.lambda_, self.regu  )

		if self.counter%self.windowSize ==0:
			self.Wlong -= 1/(float(self.counter/self.windowSize)+1)*self.batchGradient
			self.W = matrixize(self.Wlong, self.userNum)
			self.W = normalize(self.W, axis=0, norm='l1')
			#print 'SVD', self.W
			self.batchGradient = np.zeros(self.userNum*self.userNum)
			# Use Ridge regression to fit W
		'''
		plt.pcolor(self.W_b)
		plt.colorbar
		plt.show()
		'''
		if self.W.T[userID].any() <0 or self.W.T[userID].any()>1:
			print self.W.T[userID]

		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.BigW = np.kron(np.transpose(self.W), np.identity(n=len(articlePicked.featureVector)))
		self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW))
		self.BigTheta = np.kron(np.identity(n=self.userNum) , self.UserTheta)

	
		
class LearnWAlgorithm:
	def __init__(self, dimension, alpha, lambda_, eta_, n):  # n is number of users
		self.USERS = WStruct_batch_Cons(dimension, lambda_, eta_, n)
		self.dimension = dimension
		self.alpha = alpha

		self.CanEstimateUserPreference = True
		self.CanEstimateCoUserPreference =  True
		self.CanEstimateW = True

	def decide(self, pool_articles, userID):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.USERS.getProb(self.alpha, x, userID)
			# pick article with highest Prob
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked
	def updateParameters(self, articlePicked, click, userID):
		self.USERS.updateParameters(articlePicked, click, userID)
		
	def getTheta(self, userID):
		return self.USERS.UserTheta.T[userID]

	def getCoTheta(self, userID):
		return self.USERS.CoTheta.T[userID]
	def getW(self, userID):
		#print self.USERS.W
		return self.USERS.W.T[userID]

	def getA(self):
		return self.USERS.A

class LearnWAlgorithm_SGD(LearnWAlgorithm):
	def __init__(self, dimension, alpha, lambda_, eta_, n):
		LearnWAlgorithm.__init__(self, dimension, alpha, lambda_, eta_, n)
		self.USERS = WStruct_SGD(dimension, lambda_, eta_, n)


